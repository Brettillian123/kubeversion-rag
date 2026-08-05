"""Run the ablation.

One question set, N pipeline configurations, identical inputs and identical metric code
for every row. The output is `docs/RESULTS.md` plus a machine-readable
`data/results/*.json` so CI can gate on a number rather than on a human reading a table.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config
from ..models import Corpus, Example
from ..retrieval.pipeline import PipelineConfig, ResolvedPipeline, build_pipeline
from .metrics import EvalResult, QueryResult, aggregate, markdown_table

log = logging.getLogger(__name__)


@dataclass
class EvalRun:
    """Everything needed to reproduce a reported number."""

    results: list[EvalResult]
    question_set: str
    n_chunks: int
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "question_set": self.question_set,
            "n_chunks": self.n_chunks,
            "results": [result.to_dict() for result in self.results],
        }


def evaluate_pipeline(
    pipeline: ResolvedPipeline,
    examples: Sequence[Example],
    corpus: Corpus,
    batch_size: int = 64,
) -> EvalResult:
    """Score one configuration over the answerable examples in a question set.

    Unanswerable examples are excluded here -- there is no chunk to retrieve, so every
    retrieval metric would be a meaningless zero that drags the averages down and hides
    real movement. Refusal behaviour on those questions is measured in the serving
    tests, where a generation step actually exists to refuse.
    """
    answerable = [
        example
        for example in examples
        if not example.unanswerable and corpus.get(example.positive_chunk_id) is not None
    ]
    missing = sum(
        1
        for example in examples
        if not example.unanswerable and corpus.get(example.positive_chunk_id) is None
    )
    if missing:
        log.warning(
            "%d examples reference chunk ids absent from the corpus and were skipped; "
            "the dataset was probably built against a different ingestion run",
            missing,
        )
    if not answerable:
        return EvalResult(pipeline.config.name, pipeline.config.description, 0, {})

    query_results: list[QueryResult] = []
    for start in range(0, len(answerable), batch_size):
        batch = answerable[start : start + batch_size]
        retrieved = pipeline.retrieve_many(
            [example.question for example in batch],
            [example.target_version for example in batch],
        )
        for example, hits in zip(batch, retrieved, strict=True):
            query_results.append(
                QueryResult(
                    question=example.question,
                    target_version=example.target_version,
                    positive_chunk_id=example.positive_chunk_id,
                    ranked_chunk_ids=[hit.chunk.chunk_id for hit in hits],
                    top_chunks=[hit.chunk for hit in hits],
                    source=example.source,
                )
            )
        log.info("%s: %d/%d queries", pipeline.config.name, len(query_results), len(answerable))

    return aggregate(
        pipeline.config.name,
        pipeline.config.description,
        query_results,
        degraded=pipeline.degraded,
    )


def run_ablation(
    config: Config,
    corpus: Corpus,
    examples: Sequence[Example],
    pipeline_configs: Sequence[PipelineConfig],
    question_set: str = "test",
    force_reembed: bool = False,
) -> EvalRun:
    results: list[EvalResult] = []
    for pipeline_config in pipeline_configs:
        log.info("=== %s: %s ===", pipeline_config.name, pipeline_config.description)
        pipeline = build_pipeline(
            pipeline_config,
            corpus,
            models_dir=config.paths.models,
            cache_dir=config.paths.interim,
            default_bi_encoder=config.retrieval.bi_encoder,
            default_cross_encoder=config.retrieval.cross_encoder,
            query_prefix=config.retrieval.query_prefix,
            embed_batch_size=config.retrieval.embed_batch_size,
            force_reembed=force_reembed,
        )
        result = evaluate_pipeline(pipeline, examples, corpus)
        results.append(result)
        headline = " ".join(
            f"{key}={value:.3f}"
            for key, value in result.metrics.items()
            if key in {"recall@10", "ndcg@10", "version_correct@1"}
        )
        log.info("%s -> %s", pipeline_config.name, headline)

    return EvalRun(
        results=results,
        question_set=question_set,
        n_chunks=len(corpus),
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def write_results(run: EvalRun, results_dir: Path, docs_path: Path | None = None) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"ablation__{run.question_set}.json"
    json_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")

    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.write_text(render_results_doc(run), encoding="utf-8")
    return json_path


def render_results_doc(run: EvalRun) -> str:
    degraded_rows = [result for result in run.results if result.degraded]
    lines = [
        "# Results",
        "",
        f"Question set: **{run.question_set}** · "
        f"{run.results[0].n_queries if run.results else 0} answerable questions · "
        f"{run.n_chunks} chunks · generated {run.generated_at}",
        "",
    ]

    if degraded_rows:
        lines += [
            "> ⚠️ **These numbers are not a trained result.** "
            f"{len(degraded_rows)} configuration(s) requested a fine-tuned model that "
            "does not exist on disk and silently fell back to the base model. "
            "Run `kvrag train biencoder` and `kvrag train crossencoder`, then "
            "re-run the evaluation before quoting anything here.",
            "",
        ]

    lines += [
        "## Ablation",
        "",
        markdown_table(run.results),
        "",
        "Each row adds exactly one component to the row above it, so the delta is "
        "attributable. `version_correct@1` is the metric this project exists to move: "
        "it asks whether the top-ranked chunk actually applies to the Kubernetes "
        "version in the question.",
        "",
    ]

    if run.results and run.results[0].by_source:
        lines += ["## By question source", ""]
        for result in run.results:
            if not result.by_source:
                continue
            lines.append(f"### {result.description}")
            lines.append("")
            lines.append("| Source | recall@10 | ndcg@10 | version_correct@1 | n |")
            lines.append("|---|---:|---:|---:|---:|")
            for source, metrics in result.by_source.items():
                lines.append(
                    f"| {source} | {metrics.get('recall@10', 0):.3f} | "
                    f"{metrics.get('ndcg@10', 0):.3f} | "
                    f"{metrics.get('version_correct@1', 0):.3f} | — |"
                )
            lines.append("")

    lines += [
        "## Reading this table honestly",
        "",
        "- The question set is split **by document family**, so no section in the test "
        "set was seen during training. Splitting by question would leak.",
        "- Generated questions are templated. They isolate version-sensitivity "
        "precisely but are narrower than real user phrasing; the hand-written gold set "
        "is the check on that.",
        "- Unanswerable questions are excluded from retrieval metrics (there is nothing "
        "to retrieve) and are measured as a refusal rate in the serving tests instead.",
        "",
    ]
    return "\n".join(lines)
