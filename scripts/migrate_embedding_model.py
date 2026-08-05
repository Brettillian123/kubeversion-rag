#!/usr/bin/env python
"""Swap the embedding model without downtime or a code deploy.

Changing the embedding model invalidates every vector in the index. The naive
approaches both fail: re-embedding in place makes the collection incoherent while it
runs (old and new vectors are not comparable, so search returns nonsense), and
delete-then-rebuild is a hard outage for as long as the backfill takes.

Instead, each model gets its own collection and the API reads the active collection
name from a ConfigMap at request time:

    plan      resolve the target collection name for a model
    backfill  embed the corpus into the new collection while the old one serves
    evaluate  score both collections on the gold set
    promote   flip the ConfigMap and roll -- only if the new one wins
    rollback  flip back; the old collection was never deleted

The gate in ``promote`` is the point. A migration that cannot be blocked by a metric is
just a deploy with extra steps.

Usage:
    python scripts/migrate_embedding_model.py plan --model BAAI/bge-base-en-v1.5
    python scripts/migrate_embedding_model.py backfill --model BAAI/bge-base-en-v1.5
    python scripts/migrate_embedding_model.py evaluate --model BAAI/bge-base-en-v1.5
    python scripts/migrate_embedding_model.py promote --model BAAI/bge-base-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kubeversion_rag.config import load_config  # noqa: E402
from kubeversion_rag.models import Corpus, load_examples  # noqa: E402
from kubeversion_rag.serving.store import QdrantStore, collection_name  # noqa: E402

log = logging.getLogger("migrate")

# The metric the promotion gate reads. Deliberately the version-correctness metric
# rather than raw recall: a model that retrieves well but surfaces the wrong release
# first is a regression for this system even when recall improves.
GATE_METRIC = "version_correct@1"
GATE_TOLERANCE = 0.005


def _resolve_collection(config, model: str, revision: int) -> str:
    return collection_name(model, revision)


def cmd_plan(args, config) -> int:
    target = _resolve_collection(config, args.model, args.revision)
    current = config.serving.active_collection
    print(json.dumps({"current": current, "target": target, "model": args.model}, indent=2))
    if target == current:
        log.warning(
            "target collection equals the active one. Bump --revision, or the backfill "
            "will overwrite the collection that is currently serving traffic."
        )
        return 1
    return 0


def cmd_backfill(args, config) -> int:
    from kubeversion_rag.retrieval.dense import Embedder

    target = _resolve_collection(config, args.model, args.revision)
    if target == config.serving.active_collection and not args.force:
        log.error(
            "refusing to backfill into the live collection %s. Bump --revision, or pass "
            "--force if you really mean to rewrite what is serving traffic.",
            target,
        )
        return 1

    corpus = Corpus.load(config.paths.interim / "corpus.jsonl")
    log.info("embedding %d chunks with %s", len(corpus), args.model)

    embedder = Embedder(
        args.model,
        query_prefix=config.retrieval.query_prefix,
        batch_size=config.retrieval.embed_batch_size,
    )
    vectors = embedder.encode_chunks(list(corpus))

    store = QdrantStore(
        url=config.serving.qdrant_url,
        collection=target,
        api_key=config.serving.qdrant_api_key,
    )
    store.ensure_collection(dimension=embedder.dimension, recreate=args.recreate)
    written = store.upsert_chunks(list(corpus), vectors)

    log.info("backfilled %d points into %s", written, target)
    log.info("the live collection %s was not touched", config.serving.active_collection)
    return 0


def cmd_evaluate(args, config) -> int:
    """Score the candidate collection against the live one on the same questions.

    Runs offline against the in-memory index rather than through Qdrant: this measures
    the *model*, and going through the network would add an ANN recall ceiling to a
    comparison that is trying to isolate embedding quality.
    """
    from kubeversion_rag.eval.run import evaluate_pipeline
    from kubeversion_rag.retrieval.pipeline import PipelineConfig, build_pipeline

    gold_path = config.paths.gold / "gold.jsonl"
    examples_path = gold_path if gold_path.exists() else config.paths.datasets / "test.jsonl"
    if not examples_path.exists():
        log.error("no evaluation set at %s", examples_path)
        return 1
    log.info("evaluating on %s", examples_path)

    corpus = Corpus.load(config.paths.interim / "corpus.jsonl")
    examples = load_examples(examples_path)

    scores: dict[str, dict[str, float]] = {}
    for label, model in (("incumbent", config.retrieval.bi_encoder), ("candidate", args.model)):
        pipeline_config = PipelineConfig(
            name=label,
            description=f"{label}: {model}",
            version_filter=True,
            bi_encoder_path=model,
        )
        pipeline = build_pipeline(
            pipeline_config,
            corpus,
            models_dir=config.paths.models,
            cache_dir=config.paths.interim,
            default_bi_encoder=config.retrieval.bi_encoder,
            default_cross_encoder=config.retrieval.cross_encoder,
            query_prefix=config.retrieval.query_prefix,
            embed_batch_size=config.retrieval.embed_batch_size,
        )
        result = evaluate_pipeline(pipeline, examples, corpus)
        scores[label] = result.metrics
        log.info("%s: %s", label, {k: round(v, 4) for k, v in result.metrics.items()})

    incumbent = scores["incumbent"].get(GATE_METRIC, 0.0)
    candidate = scores["candidate"].get(GATE_METRIC, 0.0)
    verdict = {
        "metric": GATE_METRIC,
        "incumbent": round(incumbent, 4),
        "candidate": round(candidate, 4),
        "delta": round(candidate - incumbent, 4),
        "promote": candidate >= incumbent - GATE_TOLERANCE,
    }
    report = config.paths.results / f"migration__{collection_name(args.model, args.revision)}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"scores": scores, "verdict": verdict}, indent=2), encoding="utf-8"
    )

    print(json.dumps(verdict, indent=2))
    log.info("wrote %s", report)
    return 0 if verdict["promote"] else 1


def cmd_promote(args, config) -> int:
    target = _resolve_collection(config, args.model, args.revision)
    report = config.paths.results / f"migration__{target}.json"

    if not args.skip_gate:
        if not report.exists():
            log.error("no evaluation report at %s -- run `evaluate` first", report)
            return 1
        verdict = json.loads(report.read_text(encoding="utf-8"))["verdict"]
        if not verdict["promote"]:
            log.error(
                "gate failed: %s went %.4f -> %.4f (delta %.4f). Not promoting.",
                verdict["metric"],
                verdict["incumbent"],
                verdict["candidate"],
                verdict["delta"],
            )
            return 1
        log.info("gate passed: %s delta %+.4f", verdict["metric"], verdict["delta"])

    store = QdrantStore(
        url=config.serving.qdrant_url,
        collection=target,
        api_key=config.serving.qdrant_api_key,
    )
    if not store.healthy():
        log.error("target collection %s is missing or empty -- backfill first", target)
        return 1
    log.info("target collection %s has %d points", target, store.count())

    return _flip_configmap(target, args)


def cmd_rollback(args, config) -> int:
    """Point back at a previous collection.

    Fast because nothing was deleted: promotion only ever changed a ConfigMap value,
    so rolling back is the same operation in reverse and takes one rollout.
    """
    log.info("rolling back to %s", args.to)
    return _flip_configmap(args.to, args)


def _flip_configmap(collection: str, args) -> int:
    commands = [
        [
            "kubectl",
            "-n",
            args.namespace,
            "patch",
            "configmap",
            args.configmap,
            "--type",
            "merge",
            "-p",
            json.dumps({"data": {"KVRAG_COLLECTION": collection}}),
        ],
        # The pods read the value from the environment, and a ConfigMap change does
        # not restart them. Without this rollout the promote is a no-op that looks
        # like it worked.
        ["kubectl", "-n", args.namespace, "rollout", "restart", f"deployment/{args.deployment}"],
        [
            "kubectl",
            "-n",
            args.namespace,
            "rollout",
            "status",
            f"deployment/{args.deployment}",
            "--timeout",
            args.timeout,
        ],
    ]

    if args.dry_run:
        print("would run:")
        for command in commands:
            print("  " + " ".join(command))
        return 0

    for command in commands:
        log.info("$ %s", " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            log.error("command failed with exit code %d", result.returncode)
            log.error(
                "the ConfigMap may already be flipped -- check with "
                "`kubectl -n %s get configmap %s -o yaml` before retrying",
                args.namespace,
                args.configmap,
            )
            return result.returncode

    log.info("active collection is now %s", collection)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--configmap", default="kvrag-config")
    parser.add_argument("--deployment", default="kvrag-api")
    parser.add_argument("--timeout", default="5m")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revision", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, func, needs_model in (
        ("plan", cmd_plan, True),
        ("backfill", cmd_backfill, True),
        ("evaluate", cmd_evaluate, True),
        ("promote", cmd_promote, True),
        ("rollback", cmd_rollback, False),
    ):
        sp = sub.add_parser(name)
        if needs_model:
            sp.add_argument("--model", required=True, help="embedding model id or path")
        sp.set_defaults(func=func)

    sub.choices["backfill"].add_argument("--recreate", action="store_true")
    sub.choices["backfill"].add_argument("--force", action="store_true")
    sub.choices["promote"].add_argument("--skip-gate", action="store_true")
    sub.choices["rollback"].add_argument("--to", required=True, help="collection to revert to")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    return int(args.func(args, load_config()))


if __name__ == "__main__":
    raise SystemExit(main())
