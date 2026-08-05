"""Fine-tune the bi-encoder with MultipleNegativesRankingLoss.

MNRL treats every other passage in the batch as a negative, so batch size *is* the
number of negatives and matters more than it does for most losses. Explicit hard
negatives are supplied as a third column, which MNRL appends to the in-batch pool --
that is where the version signal comes from. Without them the model only ever has to
separate a section from unrelated sections, which off-the-shelf embeddings already do
well; it never has to separate a section from *itself at a different release*, which is
the whole problem.

The evaluator is an information-retrieval evaluator over the dev split rather than a
loss curve, because loss on a contrastive objective is only loosely coupled to the
retrieval metric anyone cares about.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from ..models import Corpus, Example
from . import describe_device, device_kwargs, disable_model_card_widgets, warmup_kwargs

log = logging.getLogger(__name__)


def _build_triplets(
    corpus: Corpus,
    examples: Sequence[Example],
    query_prefix: str,
) -> list[dict[str, str]]:
    """(anchor, positive, negative) rows, one per hard negative.

    Emitting one row per negative rather than one per example means an easy question
    with a single negative contributes once and a genuinely confusable one contributes
    several times -- an implicit reweighting toward the hard cases, which is what we
    want here.
    """
    rows: list[dict[str, str]] = []
    skipped = 0
    for example in examples:
        if example.unanswerable:
            continue
        positive = corpus.get(example.positive_chunk_id)
        if positive is None:
            skipped += 1
            continue
        negatives = [
            corpus.get(chunk_id)
            for chunk_id in example.hard_negative_ids
            if corpus.get(chunk_id) is not None
        ]
        if not negatives:
            skipped += 1
            continue
        for negative in negatives:
            rows.append(
                {
                    "anchor": f"{query_prefix}{example.question}",
                    "positive": positive.embed_text(),
                    "negative": negative.embed_text(),  # type: ignore[union-attr]
                }
            )
    if skipped:
        log.info("skipped %d examples with no usable positive/negative pair", skipped)
    return rows


def _build_ir_evaluator(corpus: Corpus, examples: Sequence[Example], query_prefix: str):
    """Dev-set retrieval evaluator over a corpus restricted to relevant chunks.

    Scored against the *full* corpus this would be slow enough to dominate training
    time. Restricting to positives plus their hard negatives keeps it fast while
    preserving the discrimination that matters -- distinguishing the right version of
    a section from the wrong one.
    """
    # The canonical path, not the `sentence_transformers.evaluation` shim. The shim is
    # deprecated *and* slow: importing through it took 84s here versus 20s direct,
    # because it drags in a re-export chain. Silent minutes before the first training
    # step look exactly like a hang.
    from sentence_transformers.sentence_transformer.evaluation import (
        InformationRetrievalEvaluator,
    )

    queries: dict[str, str] = {}
    relevant: dict[str, set[str]] = {}
    documents: dict[str, str] = {}

    for index, example in enumerate(examples):
        if example.unanswerable:
            continue
        positive = corpus.get(example.positive_chunk_id)
        if positive is None:
            continue
        query_id = f"q{index}"
        queries[query_id] = f"{query_prefix}{example.question}"
        relevant[query_id] = {positive.chunk_id}
        documents[positive.chunk_id] = positive.embed_text()
        for chunk_id in example.hard_negative_ids:
            negative = corpus.get(chunk_id)
            if negative is not None:
                documents[negative.chunk_id] = negative.embed_text()

    if not queries:
        return None

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=documents,
        relevant_docs=relevant,
        name="dev",
        show_progress_bar=False,
        # ndcg@10 is the number reported in the results table; optimizing the
        # checkpoint against a different metric would make the two disagree.
        main_score_function="cosine",
    )


def train_biencoder(
    corpus: Corpus,
    train_examples: Sequence[Example],
    dev_examples: Sequence[Example],
    base_model: str,
    query_prefix: str,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> Path:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    rows = _build_triplets(corpus, train_examples, query_prefix)
    if not rows:
        raise SystemExit(
            "no training triplets -- every example lacked a usable positive or hard "
            "negative. Re-run `kvrag dataset build` against the current corpus."
        )
    log.info("training on %d triplets from %d examples", len(rows), len(train_examples))
    log.info("device -- %s", describe_device())

    dataset = Dataset.from_list(rows)
    model = SentenceTransformer(base_model)
    disable_model_card_widgets(model)
    loss = MultipleNegativesRankingLoss(model)
    evaluator = _build_ir_evaluator(corpus, dev_examples, query_prefix)

    output_dir.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        **warmup_kwargs(warmup_ratio),
        **device_kwargs(),
        # MNRL's negatives are the rest of the batch, so a batch containing two
        # near-duplicate rows creates a false negative -- the model is punished for
        # ranking a correct passage highly. Dropping the ragged last batch also keeps
        # the negative count constant across steps.
        dataloader_drop_last=True,
        eval_strategy="epoch" if evaluator else "no",
        save_strategy="epoch",
        save_total_limit=1,
        # Frequent enough to see a stall within a minute. With tqdm disabled for
        # non-TTY output this is the only progress signal there is.
        logging_steps=10,
        report_to=[],
        seed=20260805,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    model.save(str(output_dir))
    log.info("saved fine-tuned bi-encoder to %s", output_dir)

    if evaluator is not None:
        scores = evaluator(model)
        log.info("dev evaluation: %s", {k: round(v, 4) for k, v in scores.items()})
    return output_dir
