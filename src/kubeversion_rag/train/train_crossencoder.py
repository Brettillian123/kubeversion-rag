"""Fine-tune the cross-encoder reranker on (query, passage) -> relevance.

The architectural argument for a reranker is real: the bi-encoder must compress a
passage into one vector *before* it sees the query, and two snapshots of the same
section differ by a handful of tokens, which rarely survives that compression. A
cross-encoder reads both together, so a single contradicting sentence can dominate.

**That argument is not sufficient, and this file exists partly to record why.** Where the
negatives come from mattered more than the architecture did, and getting it wrong cost
more than the reranker was ever going to gain. Three rounds, measured end to end:

| negatives used                          | nDCG@10 |
|-----------------------------------------|--------:|
| no reranker (fine-tuned bi-encoder only) |   0.774 |
| same-family wrong-version only           |   0.233 |
| + uniformly sampled from the corpus      |   0.664 |
| + mined from the fine-tuned retriever    |  see docs/RESULTS.md |

Each round failed the same way for a different reason: the negatives were drawn from a
population the reranker does not meet at inference. Training loss and margin-vs-hard-
negative both looked excellent every time. Neither can see this class of bug, because
both are computed on the same skewed distribution the model was trained on.

The third round stops guessing what that population is. ``dataset mine-negatives``
retrieves with the *fine-tuned* bi-encoder under the serving version filter and keeps
what comes back, so the training candidates are the serving candidates by construction.
See ``dataset/mine_negatives.py`` for the measurements that forced each revision.
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
    random_negatives_per_positive: int = 2,
    retriever_negatives_per_positive: int = 6,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build (query, passage, label) rows from three negative sources.

    They are not interchangeable, and the proportions are the result of measurement
    rather than taste:

    **Family negatives** -- the same section at a wrong version. These carry the version
    signal the whole project is about. They are only ~2% of a retrieved top-50, far too
    rare for the model to learn the distinction from retrieved candidates alone, which is
    why they are mined separately and kept at full weight.

    **Retriever negatives** -- what the fine-tuned bi-encoder actually returns for this
    question, under the serving version filter. The dominant source, because it *is* the
    serving distribution. Two earlier versions of this function guessed at that
    distribution instead and were wrong in two different ways; the second guess put 46%
    of its ranking errors on sibling sections of the correct document, a population that
    uniform sampling produces roughly never.

    **Uniform negatives** -- kept, at reduced weight, as a floor. Retrieved negatives are
    by definition all plausible; a model trained only on hard cases drifts toward scoring
    everything highly, and these anchor the low end of the range.
    """
    rows: list[dict[str, object]] = []
    counts = {
        "positives": 0,
        "negatives": 0,
        "retriever_negatives": 0,
        "random_negatives": 0,
        "skipped": 0,
        "missing_retriever_negatives": 0,
    }
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

        retrieved = [
            chunk
            for chunk in (corpus.get(cid) for cid in example.retriever_negative_ids)
            if chunk is not None and chunk.family_id != positive.family_id
        ]
        if not retrieved:
            counts["missing_retriever_negatives"] += 1
        # Kept in retrieval order rather than shuffled: the top of the list is what the
        # reranker has to beat, so truncating takes the hardest ones.
        for negative in retrieved[:retriever_negatives_per_positive]:
            rows.append({"query": example.question, "passage": negative.embed_text(), "label": 0.0})
            counts["retriever_negatives"] += 1

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
    random_negatives_per_positive: int = 2,
    retriever_negatives_per_positive: int = 6,
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
        corpus,
        train_examples,
        negatives_per_positive,
        rng,
        random_negatives_per_positive,
        retriever_negatives_per_positive,
    )
    if not train_rows:
        raise SystemExit(
            "no cross-encoder training pairs -- re-run `kvrag dataset build` against "
            "the current corpus."
        )
    dev_rows, _ = _build_pairs(
        corpus,
        dev_examples,
        negatives_per_positive,
        rng,
        random_negatives_per_positive,
        retriever_negatives_per_positive,
    )

    log.info(
        "cross-encoder pairs: %d positive, %d family negative, %d retriever negative, "
        "%d random negative (%d skipped), %d dev",
        counts["positives"],
        counts["negatives"],
        counts["retriever_negatives"],
        counts["random_negatives"],
        counts["skipped"],
        len(dev_rows),
    )
    log.info("device -- %s", describe_device())
    # The failure this catches is silent: `dataset build` writes examples with no
    # retriever negatives, so skipping `dataset mine-negatives` produces a smaller
    # training set and a reranker trained on the wrong distribution, with no error.
    if retriever_negatives_per_positive and counts["missing_retriever_negatives"]:
        share = counts["missing_retriever_negatives"] / max(counts["positives"], 1)
        message = (
            f"{counts['missing_retriever_negatives']} of {counts['positives']} examples "
            f"({share:.0%}) have no retriever-mined negatives. Run `kvrag train biencoder` "
            "then `kvrag dataset mine-negatives` before training the reranker -- without "
            "them it is trained on a distribution it will not see at inference, which "
            "measured 0.664 nDCG@10 against 0.774 for no reranker at all."
        )
        if share > 0.5:
            raise SystemExit(message)
        log.warning("%s", message)
    total_negatives = (
        counts["negatives"] + counts["retriever_negatives"] + counts["random_negatives"]
    )
    if total_negatives < counts["positives"]:
        log.warning(
            "fewer negatives (%d) than positives (%d): the reranker will be biased "
            "toward scoring everything relevant. Check hard-negative mining.",
            total_negatives,
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

        ratio = total_negatives / counts["positives"]
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
