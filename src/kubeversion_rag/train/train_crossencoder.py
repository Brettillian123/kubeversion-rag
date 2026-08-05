"""Fine-tune the cross-encoder reranker on (query, passage) -> relevance.

The architectural argument for a reranker is real: the bi-encoder must compress a
passage into one vector *before* it sees the query, and two snapshots of the same
section differ by a handful of tokens, which rarely survives that compression. A
cross-encoder reads both together, so a single contradicting sentence can dominate.

**That argument is not sufficient, and this file exists partly to record why.** The
first version trained here used only mined same-family hard negatives -- the reasoning
being that positives and negatives from the same section force the model to learn the
version signal rather than topic similarity. It learned exactly that, with a 19-point
margin. Then it made end-to-end retrieval *three times worse*: nDCG@10 fell from 0.774
to 0.233.

The reason is a train/serve distribution mismatch, and it is the kind that does not
show up in training metrics at all. Every negative it had ever seen came from the same
document section as the positive; at inference it must rank against the top-50 from
dense retrieval, which are overwhelmingly other topics. It had no calibration there, so
those scores were arbitrary rather than low -- and with 49 competitors per query, an
occasional high score on an unrelated chunk is fatal.

The fix is in ``_build_pairs``: sample negatives from the whole corpus as well, so the
model sees the population it will actually rank. Keep both kinds. They teach different
things, and the mined ones alone are not enough.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from pathlib import Path

from ..models import Corpus, Example
from . import describe_device, device_kwargs, disable_model_card_widgets, warmup_kwargs

log = logging.getLogger(__name__)


def _build_pairs(
    corpus: Corpus,
    examples: Sequence[Example],
    negatives_per_positive: int,
    rng: random.Random,
    random_negatives_per_positive: int = 4,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build (query, passage, label) rows.

    **Two kinds of negative, and the second is not optional.** Training on mined
    same-family hard negatives alone produces a reranker that is excellent at the task
    it was shown and catastrophic at the task it is given.

    Measured on a version trained without random negatives: it separated correct from
    wrong-version-same-section by 19 points (+8.9 vs -10.0), exactly as intended -- but
    scored *unrelated* chunks at p90 +8.6, statistically tied with the positives. It had
    simply never been asked about a passage from another document, so its scores there
    were uncalibrated rather than low.

    That is fatal at inference. The reranker sees the top-50 from dense retrieval, which
    are mostly other topics. Beating a random chunk 92.5% of the time sounds strong until
    there are 49 of them: 0.925^49 is about 2%. The measured effect was nDCG@10 falling
    from 0.774 to 0.233 -- the reranker made retrieval three times worse.

    Random negatives fix the distribution mismatch by showing the model the population it
    will actually rank against. They are easy negatives and teach nothing about versions;
    the mined ones still carry that signal. Both are needed, for different reasons.
    """
    rows: list[dict[str, object]] = []
    counts = {"positives": 0, "negatives": 0, "random_negatives": 0, "skipped": 0}
    all_chunks = list(corpus)

    for example in examples:
        if example.unanswerable:
            continue
        positive = corpus.get(example.positive_chunk_id)
        if positive is None:
            counts["skipped"] += 1
            continue

        rows.append({"query": example.question, "passage": positive.embed_text(), "label": 1.0})
        counts["positives"] += 1

        negatives = [
            chunk
            for chunk in (corpus.get(cid) for cid in example.hard_negative_ids)
            if chunk is not None
        ]
        rng.shuffle(negatives)
        for negative in negatives[:negatives_per_positive]:
            rows.append({"query": example.question, "passage": negative.embed_text(), "label": 0.0})
            counts["negatives"] += 1

        # Sampled from the whole corpus, because that is what the top-50 from dense
        # retrieval actually looks like. Anything in the same family is skipped so a
        # legitimately relevant chunk is never mislabelled irrelevant.
        drawn = attempts = 0
        budget = random_negatives_per_positive * 8
        while drawn < random_negatives_per_positive and attempts < budget:
            attempts += 1
            candidate = all_chunks[rng.randrange(len(all_chunks))]
            if candidate.family_id == positive.family_id:
                continue
            rows.append(
                {"query": example.question, "passage": candidate.embed_text(), "label": 0.0}
            )
            counts["random_negatives"] += 1
            drawn += 1

    rng.shuffle(rows)
    return rows, counts


def train_crossencoder(
    corpus: Corpus,
    train_examples: Sequence[Example],
    dev_examples: Sequence[Example],
    base_model: str,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    negatives_per_positive: int = 4,
    random_negatives_per_positive: int = 4,
    warmup_ratio: float = 0.1,
    seed: int = 20260805,
    max_length: int = 512,
) -> Path:
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    rng = random.Random(seed)
    train_rows, counts = _build_pairs(
        corpus, train_examples, negatives_per_positive, rng, random_negatives_per_positive
    )
    if not train_rows:
        raise SystemExit(
            "no cross-encoder training pairs -- re-run `kvrag dataset build` against "
            "the current corpus."
        )
    dev_rows, _ = _build_pairs(
        corpus, dev_examples, negatives_per_positive, rng, random_negatives_per_positive
    )

    log.info(
        "cross-encoder pairs: %d positive, %d mined-hard negative, %d random negative "
        "(%d skipped), %d dev",
        counts["positives"],
        counts["negatives"],
        counts["random_negatives"],
        counts["skipped"],
        len(dev_rows),
    )
    log.info("device -- %s", describe_device())
    if counts["negatives"] + counts["random_negatives"] < counts["positives"]:
        log.warning(
            "fewer negatives (%d) than positives (%d): the reranker will be biased "
            "toward scoring everything relevant. Check hard-negative mining.",
            counts["negatives"] + counts["random_negatives"],
            counts["positives"],
        )

    # Set explicitly, and match serving/rerank.py's Reranker. A cross-encoder trained
    # at one input length and served at another underperforms in a way that is
    # genuinely hard to attribute -- the numbers simply come out lower, with no error
    # and nothing in the logs pointing at the cause.
    model = CrossEncoder(base_model, num_labels=1, max_length=max_length)
    disable_model_card_widgets(model)
    log.info("cross-encoder max_length=%d (must match the serving Reranker)", max_length)
    # Class imbalance is real here (one positive to N negatives), and an unweighted
    # BCE quietly learns to predict "irrelevant" for everything, which looks like a
    # working model until you inspect the score distribution.
    pos_weight = None
    if counts["positives"]:
        import torch

        ratio = (counts["negatives"] + counts["random_negatives"]) / counts["positives"]
        pos_weight = torch.tensor(max(ratio, 1.0))
        log.info("BCE pos_weight=%.2f", float(pos_weight))
    loss = BinaryCrossEntropyLoss(model, pos_weight=pos_weight)

    output_dir.mkdir(parents=True, exist_ok=True)
    args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        **warmup_kwargs(warmup_ratio),
        **device_kwargs(),
        eval_strategy="epoch" if dev_rows else "no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to=[],
        seed=seed,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(dev_rows) if dev_rows else None,
        loss=loss,
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    log.info("saved fine-tuned cross-encoder to %s", output_dir)
    return output_dir
