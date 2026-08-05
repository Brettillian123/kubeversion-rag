#!/usr/bin/env python
"""Run both training paths end-to-end on a tiny slice.

A full fine-tune on this dataset is hours on CPU and wants a GPU. That is fine -- but
it means the training code is otherwise only ever exercised by whoever runs it for
real, which is exactly the code you least want to discover is broken after ingesting
twelve release branches and building a 13k-example dataset.

This runs the *same* functions the real command runs, on a few hundred examples for a
handful of steps. It proves the wiring: dataset construction, loss setup, the trainer
loop, evaluation, and saving. It proves nothing about quality, and deliberately writes
to a scratch directory so its output can never be mistaken for a trained model and
picked up by the ablation.

    python scripts/smoke_train.py --examples 200
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubeversion_rag.config import load_config  # noqa: E402
from kubeversion_rag.models import Corpus, load_examples  # noqa: E402

log = logging.getLogger("smoke")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--keep", action="store_true", help="do not delete the scratch output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    warnings.filterwarnings("ignore", category=FutureWarning)

    config = load_config()
    corpus_path = config.paths.interim / "corpus.jsonl"
    train_path = config.paths.datasets / "train.jsonl"
    dev_path = config.paths.datasets / "dev.jsonl"
    for path in (corpus_path, train_path, dev_path):
        if not path.exists():
            log.error("missing %s -- run `kvrag ingest` and `kvrag dataset build` first", path)
            return 1

    corpus = Corpus.load(corpus_path)
    train = load_examples(train_path)[: args.examples]
    dev = load_examples(dev_path)[: max(args.examples // 4, 8)]
    log.info("smoke set: %d train, %d dev, over %d chunks", len(train), len(dev), len(corpus))

    # Scratch, never data/models/ -- the ablation resolves fine-tuned rows from there,
    # and a smoke-trained model sitting in that path would be silently reported as a
    # real training result.
    scratch = config.paths.root / "_smoke"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)

    failures: list[str] = []

    try:
        from kubeversion_rag.train.train_biencoder import train_biencoder

        log.info("=== bi-encoder ===")
        train_biencoder(
            corpus=corpus,
            train_examples=train,
            dev_examples=dev,
            base_model=config.retrieval.bi_encoder,
            query_prefix=config.retrieval.query_prefix,
            output_dir=scratch / "biencoder",
            epochs=1,
            batch_size=4,
            learning_rate=2e-5,
        )
        saved = scratch / "biencoder"
        assert any(saved.iterdir()), "bi-encoder produced no output"
        log.info("bi-encoder OK -> %s", saved)
    except Exception as exc:  # noqa: BLE001
        log.exception("bi-encoder smoke train failed")
        failures.append(f"biencoder: {exc}")

    try:
        from kubeversion_rag.train.train_crossencoder import train_crossencoder

        log.info("=== cross-encoder ===")
        train_crossencoder(
            corpus=corpus,
            train_examples=train,
            dev_examples=dev,
            base_model=config.retrieval.cross_encoder,
            output_dir=scratch / "crossencoder",
            epochs=1,
            batch_size=4,
            learning_rate=2e-5,
            negatives_per_positive=2,
        )
        saved = scratch / "crossencoder"
        assert any(saved.iterdir()), "cross-encoder produced no output"
        log.info("cross-encoder OK -> %s", saved)
    except Exception as exc:  # noqa: BLE001
        log.exception("cross-encoder smoke train failed")
        failures.append(f"crossencoder: {exc}")

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)

    if failures:
        for failure in failures:
            log.error("FAILED %s", failure)
        return 1
    log.info("both training paths execute end-to-end (quality untested by design)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
