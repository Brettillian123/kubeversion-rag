"""Fine-tune the cross-encoder reranker on (query, passage) -> relevance.

This is where most of the measured gain comes from, for an architectural reason: the
bi-encoder must compress a passage into a single vector *before* it sees the query, and
two snapshots of the same section differ by a handful of tokens. That difference rarely
survives the compression. A cross-encoder reads both together, so one contradicting
sentence can dominate the score.

Training data is the same mined pairs as the bi-encoder, relabelled: the version-correct
chunk is 1, its wrong-version siblings are 0. Because positives and negatives come from
the *same section*, the model cannot succeed by learning topic similarity -- the only
signal that separates them is the version.
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
) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    counts = {"positives": 0, "negatives": 0, "skipped": 0}

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
    train_rows, counts = _build_pairs(corpus, train_examples, negatives_per_positive, rng)
    if not train_rows:
        raise SystemExit(
            "no cross-encoder training pairs -- re-run `kvrag dataset build` against "
            "the current corpus."
        )
    dev_rows, _ = _build_pairs(corpus, dev_examples, negatives_per_positive, rng)

    log.info(
        "cross-encoder pairs: %d positive, %d negative (%d skipped), %d dev",
        counts["positives"],
        counts["negatives"],
        counts["skipped"],
        len(dev_rows),
    )
    log.info("device -- %s", describe_device())
    if counts["negatives"] < counts["positives"]:
        log.warning(
            "fewer negatives (%d) than positives (%d): the reranker will be biased "
            "toward scoring everything relevant. Check hard-negative mining.",
            counts["negatives"],
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

        ratio = counts["negatives"] / counts["positives"]
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
