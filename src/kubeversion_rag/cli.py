"""Command line entry point.

Every stage writes its output to disk and every later stage reads from disk, so any
step can be re-run in isolation. That matters most for ingestion, which is the slow,
network-bound step nobody wants to repeat while iterating on chunking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DEPRECATION_GUIDE_PATH, Config, load_config
from .models import (
    Chunk,
    Corpus,
    DeprecationFact,
    load_examples,
    read_jsonl,
    save_examples,
    write_jsonl,
)
from .versions import MinorVersion

log = logging.getLogger("kvrag")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# --- ingest -------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace, config: Config) -> int:
    from .ingest.chunk import chunk_tree, coalesce_corpus
    from .ingest.deprecation import parse_deprecation_guide
    from .ingest.fetch import iter_markdown_files, iter_version_trees

    config.paths.ensure()
    versions = config.versions()
    log.info("ingesting %d release branches: %s -> %s", len(versions), versions[0], versions[-1])

    per_version: list[Chunk] = []
    facts: list[DeprecationFact] = []
    newest_guide_version: MinorVersion | None = None

    for version, docs_root in iter_version_trees(config, versions=versions):
        files = list(iter_markdown_files(docs_root))
        chunks = list(chunk_tree(docs_root, files, version, config.chunking))
        per_version.extend(chunks)
        log.info("  %s: %d files -> %d chunks", version, len(files), len(chunks))

        # Parse the deprecation guide from the newest branch that has it. Older
        # branches carry a strict subset of the same facts, so parsing every branch
        # would just re-derive the same rows.
        # docs_root is <repo>/content/en/docs; the guide path is repo-relative.
        guide = docs_root.parents[2] / DEPRECATION_GUIDE_PATH
        if guide.is_file() and (newest_guide_version is None or version > newest_guide_version):
            parsed, report = parse_deprecation_guide(guide.read_text(encoding="utf-8"))
            log.info("  %s: %s", version, report.summary())
            if report.coverage < 0.9:
                log.warning(
                    "  deprecation-guide coverage dropped to %.0f%% -- upstream probably "
                    "reworded the page; check ingest/deprecation.py",
                    report.coverage * 100,
                )
            for skipped in report.blocks_skipped:
                log.debug("    skipped: %s", skipped)
            facts, newest_guide_version = parsed, version

    if not per_version:
        log.error("no chunks produced; nothing was ingested")
        return 1

    corpus = coalesce_corpus(per_version)
    corpus_path = config.paths.interim / "corpus.jsonl"
    corpus.save(corpus_path)
    facts_path = config.paths.interim / "deprecation_facts.jsonl"
    write_jsonl(facts_path, (fact.to_dict() for fact in facts))

    compression = len(per_version) / len(corpus) if corpus.chunks else 0
    multi_version = sum(1 for fid in corpus.families() if len(corpus.family(fid)) > 1)
    log.info(
        "%d per-version chunks -> %d coalesced (%.1fx), %d families, "
        "%d of which changed across releases",
        len(per_version),
        len(corpus),
        compression,
        len(list(corpus.families())),
        multi_version,
    )
    log.info("wrote %s and %s", corpus_path, facts_path)
    return 0


# --- dataset ------------------------------------------------------------------------


def cmd_dataset_build(args: argparse.Namespace, config: Config) -> int:
    from .dataset.build import build_dataset, split_by_family

    corpus_path = config.paths.interim / "corpus.jsonl"
    facts_path = config.paths.interim / "deprecation_facts.jsonl"
    if not corpus_path.exists():
        log.error("no corpus at %s -- run `kvrag ingest` first", corpus_path)
        return 1

    corpus = Corpus.load(corpus_path)
    facts = (
        [DeprecationFact.from_dict(row) for row in read_jsonl(facts_path)]
        if facts_path.exists()
        else []
    )
    log.info("loaded %d chunks and %d deprecation facts", len(corpus), len(facts))

    examples, stats = build_dataset(
        corpus, facts, config.versions(), seed=args.seed, unanswerable_count=args.unanswerable
    )
    log.info(stats.summary())
    if not examples:
        log.error("no examples generated")
        return 1

    splits = split_by_family(examples, seed=args.seed)
    config.paths.datasets.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        path = config.paths.datasets / f"{name}.jsonl"
        save_examples(path, split)
        answerable = sum(1 for example in split if not example.unanswerable)
        families = len({example.family_id for example in split if example.family_id})
        log.info(
            "  %-5s %5d examples (%d answerable, %d families) -> %s",
            name,
            len(split),
            answerable,
            families,
            path,
        )

    _assert_no_family_leakage(splits)
    return 0


def _assert_no_family_leakage(splits: dict[str, list]) -> None:
    """Fail loudly if a document family appears in more than one split.

    A silent leak here would inflate every number in the results table, and it is the
    single most likely way for this project to produce a dishonest headline.
    """
    seen: dict[str, str] = {}
    for name, split in splits.items():
        for example in split:
            if not example.family_id:
                continue
            previous = seen.get(example.family_id)
            if previous and previous != name:
                raise SystemExit(
                    f"LEAKAGE: family {example.family_id} appears in both {previous!r} and {name!r}"
                )
            seen[example.family_id] = name
    log.info("leakage check passed: %d families, each in exactly one split", len(seen))


def cmd_gold_resolve(args: argparse.Namespace, config: Config) -> int:
    """Bind the hand-written gold questions to chunk ids in the current corpus."""
    from .dataset.gold import load_specs, resolve

    corpus_path = config.paths.interim / "corpus.jsonl"
    spec_path = config.paths.gold / "gold_questions.yaml"
    if not corpus_path.exists():
        log.error("no corpus at %s -- run `kvrag ingest` first", corpus_path)
        return 1
    if not spec_path.exists():
        log.error("no gold questions at %s", spec_path)
        return 1

    corpus = Corpus.load(corpus_path)
    specs = load_specs(spec_path)
    examples, problems = resolve(corpus, specs)

    for problem in problems:
        log.error("unresolved: %s", problem)
    if problems:
        log.error(
            "%d of %d gold questions no longer bind to a chunk. A gold set that "
            "silently stops matching still reports a number, which is worse than "
            "having none -- fix the question or the corpus before evaluating.",
            len(problems),
            len(specs),
        )
        if not args.allow_unresolved:
            return 1

    out_path = config.paths.datasets / "gold.jsonl"
    save_examples(out_path, examples)
    answerable = sum(1 for example in examples if not example.unanswerable)
    log.info(
        "resolved %d gold questions (%d answerable, %d expected-refusal) -> %s",
        len(examples),
        answerable,
        len(examples) - answerable,
        out_path,
    )
    return 0


# --- eval ---------------------------------------------------------------------------


def cmd_eval_run(args: argparse.Namespace, config: Config) -> int:
    from .eval.run import run_ablation, write_results
    from .retrieval.pipeline import STANDARD_CONFIGS

    corpus_path = config.paths.interim / "corpus.jsonl"
    examples_path = config.paths.datasets / f"{args.split}.jsonl"
    if not corpus_path.exists() or not examples_path.exists():
        log.error(
            "need %s and %s -- run `kvrag ingest` and `kvrag dataset build`",
            corpus_path,
            examples_path,
        )
        return 1

    corpus = Corpus.load(corpus_path)
    examples = load_examples(examples_path)
    log.info("evaluating on %s: %d examples over %d chunks", args.split, len(examples), len(corpus))

    if args.configs and not args.all:
        wanted = {name.strip() for name in args.configs.split(",") if name.strip()}
        selected = [c for c in STANDARD_CONFIGS if c.name in wanted]
        unknown = wanted - {c.name for c in STANDARD_CONFIGS}
        if unknown:
            log.error("unknown configs: %s", ", ".join(sorted(unknown)))
            log.error("available: %s", ", ".join(c.name for c in STANDARD_CONFIGS))
            return 1
    else:
        selected = list(STANDARD_CONFIGS)

    run = run_ablation(
        config,
        corpus,
        examples,
        selected,
        question_set=args.split,
        force_reembed=args.force_reembed,
    )

    from .eval.metrics import markdown_table

    print()
    print(markdown_table(run.results))
    print()

    if args.write_results:
        json_path = write_results(
            run,
            config.paths.results,
            docs_path=Path(__file__).resolve().parents[2] / "docs" / "RESULTS.md",
        )
        log.info("wrote %s and docs/RESULTS.md", json_path)
    return 0


def cmd_eval_gate(args: argparse.Namespace, config: Config) -> int:
    """CI gate: fail the build if a metric regressed against the committed baseline.

    Deliberately compares against a checked-in file rather than the previous CI run:
    a baseline in version control is reviewable, and a regression has to be accepted
    by a human editing that file in a pull request.
    """
    results_path = config.paths.results / f"ablation__{args.split}.json"
    if not results_path.exists():
        log.error("no results at %s -- run `kvrag eval run --write-results` first", results_path)
        return 1

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    current = {row["config"]: row["metrics"] for row in payload["results"]}
    degraded = [row["config"] for row in payload["results"] if row.get("degraded")]
    if degraded and not args.allow_untrained:
        log.error(
            "configs %s fell back to base models; gating on these would certify an "
            "untrained system. Pass --allow-untrained only for a smoke test.",
            ", ".join(degraded),
        )
        return 1

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        log.warning("no baseline at %s; writing the current run as the new baseline", baseline_path)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for config_name, metrics in baseline.items():
        if config_name not in current:
            failures.append(f"{config_name}: missing from this run")
            continue
        for metric, expected in metrics.items():
            actual = current[config_name].get(metric)
            if actual is None:
                failures.append(f"{config_name}.{metric}: missing")
            elif actual < expected - args.tolerance:
                failures.append(
                    f"{config_name}.{metric}: {actual:.4f} < {expected:.4f} "
                    f"(tolerance {args.tolerance})"
                )

    if failures:
        log.error("EVAL GATE FAILED")
        for failure in failures:
            log.error("  %s", failure)
        return 1

    log.info("eval gate passed: no metric regressed beyond %.4f", args.tolerance)
    return 0


# --- train --------------------------------------------------------------------------


def cmd_train_biencoder(args: argparse.Namespace, config: Config) -> int:
    from .train.train_biencoder import train_biencoder

    corpus = Corpus.load(config.paths.interim / "corpus.jsonl")
    train = load_examples(config.paths.datasets / "train.jsonl")
    dev = load_examples(config.paths.datasets / "dev.jsonl")
    output = config.paths.models / "biencoder"
    train_biencoder(
        corpus=corpus,
        train_examples=train,
        dev_examples=dev,
        base_model=config.retrieval.bi_encoder,
        query_prefix=config.retrieval.query_prefix,
        output_dir=output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    return 0


def cmd_train_crossencoder(args: argparse.Namespace, config: Config) -> int:
    from .train.train_crossencoder import train_crossencoder

    corpus = Corpus.load(config.paths.interim / "corpus.jsonl")
    train = load_examples(config.paths.datasets / "train.jsonl")
    dev = load_examples(config.paths.datasets / "dev.jsonl")
    output = config.paths.models / "crossencoder"
    train_crossencoder(
        corpus=corpus,
        train_examples=train,
        dev_examples=dev,
        base_model=config.retrieval.cross_encoder,
        output_dir=output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        negatives_per_positive=args.negatives,
    )
    return 0


# --- corpus stats -------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace, config: Config) -> int:
    corpus_path = config.paths.interim / "corpus.jsonl"
    if not corpus_path.exists():
        log.error("no corpus at %s", corpus_path)
        return 1
    corpus = Corpus.load(corpus_path)

    families = list(corpus.families())
    multi = [fid for fid in families if len(corpus.family(fid)) > 1]
    lengths = sorted(len(chunk.text) for chunk in corpus)
    docs = {chunk.doc_path for chunk in corpus}

    print(f"chunks                  {len(corpus)}")
    print(f"documents               {len(docs)}")
    print(f"families                {len(families)}")
    print(f"families that changed   {len(multi)} ({len(multi) / max(len(families), 1):.1%})")
    if lengths:
        print(
            f"chunk chars  p50/p90/max {lengths[len(lengths) // 2]}/"
            f"{lengths[int(len(lengths) * 0.9)]}/{lengths[-1]}"
        )
    spans: dict[int, int] = {}
    for chunk in corpus:
        width = chunk.version_high.minor - chunk.version_low.minor + 1
        spans[width] = spans.get(width, 0) + 1
    print("version-span histogram (releases covered -> chunks):")
    for width in sorted(spans):
        print(
            f"  {width:2d}  {'#' * min(spans[width] // max(len(corpus) // 200, 1), 60)} {spans[width]}"
        )
    return 0


# --- argument parsing ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvrag", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--min-version", help="oldest release branch to ingest, e.g. 1.24")
    parser.add_argument("--max-version", help="newest release branch to ingest, e.g. 1.35")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="fetch and chunk the docs corpus")
    ingest.set_defaults(func=cmd_ingest)

    stats = subparsers.add_parser("stats", help="describe the ingested corpus")
    stats.set_defaults(func=cmd_stats)

    dataset = subparsers.add_parser("dataset", help="dataset commands")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_sub.add_parser("build", help="generate train/dev/test examples")
    dataset_build.add_argument("--seed", type=int, default=20260805)
    dataset_build.add_argument("--unanswerable", type=int, default=60)
    dataset_build.set_defaults(func=cmd_dataset_build)

    gold = subparsers.add_parser("gold", help="hand-written evaluation set")
    gold_sub = gold.add_subparsers(dest="gold_command", required=True)
    gold_resolve = gold_sub.add_parser("resolve", help="bind gold questions to chunk ids")
    gold_resolve.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="write the file even if some questions no longer bind (diagnostics only)",
    )
    gold_resolve.set_defaults(func=cmd_gold_resolve)

    evaluate = subparsers.add_parser("eval", help="evaluation commands")
    eval_sub = evaluate.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run", help="run the ablation")
    eval_run.add_argument("--split", default="test", choices=["train", "dev", "test", "gold"])
    eval_run.add_argument(
        "--configs", help="comma-separated config names; omit to run every configuration"
    )
    eval_run.add_argument(
        "--all", action="store_true", help="run every configuration, overriding --configs"
    )
    eval_run.add_argument("--write-results", action="store_true")
    eval_run.add_argument("--force-reembed", action="store_true")
    eval_run.set_defaults(func=cmd_eval_run)

    eval_gate = eval_sub.add_parser("gate", help="fail if a metric regressed")
    eval_gate.add_argument("--split", default="test")
    eval_gate.add_argument("--baseline", default="eval_baseline.json")
    eval_gate.add_argument("--tolerance", type=float, default=0.01)
    eval_gate.add_argument("--allow-untrained", action="store_true")
    eval_gate.set_defaults(func=cmd_eval_gate)

    train = subparsers.add_parser("train", help="training commands")
    train_sub = train.add_subparsers(dest="train_command", required=True)

    bi = train_sub.add_parser("biencoder", help="fine-tune the bi-encoder (MNRL)")
    bi.add_argument("--epochs", type=int, default=2)
    bi.add_argument("--batch-size", type=int, default=16)
    bi.add_argument("--learning-rate", type=float, default=2e-5)
    bi.set_defaults(func=cmd_train_biencoder)

    cross = train_sub.add_parser("crossencoder", help="fine-tune the reranker")
    cross.add_argument("--epochs", type=int, default=2)
    cross.add_argument("--batch-size", type=int, default=16)
    cross.add_argument("--learning-rate", type=float, default=2e-5)
    cross.add_argument("--negatives", type=int, default=4)
    cross.set_defaults(func=cmd_train_crossencoder)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    config = load_config()
    if args.min_version:
        config.min_version = MinorVersion.parse(args.min_version)
    if args.max_version:
        config.max_version = MinorVersion.parse(args.max_version)
    if config.min_version > config.max_version:
        log.error(
            "--min-version %s is newer than --max-version %s",
            config.min_version,
            config.max_version,
        )
        return 1

    config.paths.ensure()
    try:
        return int(args.func(args, config))
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
