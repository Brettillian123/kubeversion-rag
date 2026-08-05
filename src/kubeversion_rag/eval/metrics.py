"""Retrieval metrics.

Deliberately free of any dependency on the retrieval code: these functions take ranked
id lists and nothing else. If the metric implementation shared code with the thing
being measured, a bug in the shared part would move the numbers without anyone noticing.

There is exactly one relevant chunk per question by construction, which simplifies the
graded metrics considerably -- IDCG is always 1.0 -- but the implementations are written
for the general case anyway so a future multi-positive dataset does not silently produce
wrong numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import mean

from ..models import Chunk
from ..versions import MinorVersion


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for chunk_id in ranked_ids[:k] if chunk_id in relevant_ids)
    return hits / min(len(relevant_ids), k)


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Did anything relevant appear in the top k. With one positive this equals recall@k."""
    return 1.0 if any(chunk_id in relevant_ids for chunk_id in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str], k: int = 10) -> float:
    for position, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG.

    Distinguishes "correct chunk at rank 1" from "correct chunk at rank 9" in a way
    recall@10 cannot, which matters here because the generation step only ever sees the
    top few chunks.
    """
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def version_correct_at_1(top_chunks: Sequence[Chunk], target: MinorVersion) -> float:
    """Does the top-ranked chunk actually apply to the asked-about version.

    This is the metric the project exists to move. A system can score well on recall
    while consistently surfacing the wrong release's snapshot first -- and that failure
    is worse than returning nothing, because the answer looks authoritative and cites a
    real document.
    """
    if not top_chunks:
        return 0.0
    return 1.0 if top_chunks[0].covers(target) else 0.0


def version_correct_at_k(top_chunks: Sequence[Chunk], target: MinorVersion, k: int) -> float:
    """Fraction of the top k that apply to the target version.

    Reported alongside @1 because the generation step is fed several chunks: a top-1
    that is right while the other four are wrong-version still invites the model to
    contradict itself.
    """
    window = top_chunks[:k]
    if not window:
        return 0.0
    return sum(1.0 for chunk in window if chunk.covers(target)) / len(window)


@dataclass
class QueryResult:
    """Per-query outcome, retained so failures can be inspected rather than summarized away."""

    question: str
    target_version: MinorVersion
    positive_chunk_id: str
    ranked_chunk_ids: list[str]
    top_chunks: list[Chunk] = field(default_factory=list)
    source: str = ""

    def metrics(self, ks: Sequence[int] = (1, 5, 10, 20)) -> dict[str, float]:
        relevant = {self.positive_chunk_id}
        out: dict[str, float] = {}
        for k in ks:
            out[f"recall@{k}"] = hit_at_k(self.ranked_chunk_ids, relevant, k)
        out["mrr@10"] = reciprocal_rank(self.ranked_chunk_ids, relevant, 10)
        out["ndcg@10"] = ndcg_at_k(self.ranked_chunk_ids, relevant, 10)
        out["version_correct@1"] = version_correct_at_1(self.top_chunks, self.target_version)
        out["version_correct@5"] = version_correct_at_k(self.top_chunks, self.target_version, 5)
        return out


@dataclass
class EvalResult:
    """Aggregated metrics for one pipeline configuration on one question set."""

    config_name: str
    description: str
    n_queries: int
    metrics: dict[str, float]
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)
    refusal_rate: float | None = None
    degraded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config_name,
            "description": self.description,
            "n_queries": self.n_queries,
            "metrics": self.metrics,
            "by_source": self.by_source,
            "refusal_rate": self.refusal_rate,
            "degraded": self.degraded,
        }


def aggregate(
    config_name: str,
    description: str,
    results: Sequence[QueryResult],
    ks: Sequence[int] = (1, 5, 10, 20),
    degraded: Sequence[str] = (),
) -> EvalResult:
    if not results:
        return EvalResult(config_name, description, 0, {}, degraded=list(degraded))

    per_query = [result.metrics(ks) for result in results]
    keys = per_query[0].keys()
    overall = {key: mean(row[key] for row in per_query) for key in keys}

    by_source: dict[str, dict[str, float]] = {}
    sources = {result.source for result in results if result.source}
    for source in sorted(sources):
        rows = [result.metrics(ks) for result in results if result.source == source]
        if rows:
            by_source[source] = {key: mean(row[key] for row in rows) for key in keys}

    return EvalResult(
        config_name=config_name,
        description=description,
        n_queries=len(results),
        metrics=overall,
        by_source=by_source,
        degraded=list(degraded),
    )


HEADLINE_METRICS = ("recall@10", "mrr@10", "ndcg@10", "version_correct@1")


def markdown_table(
    results: Sequence[EvalResult],
    columns: Sequence[str] = HEADLINE_METRICS,
) -> str:
    """Render the ablation table.

    Configurations that fell back to a base model are marked, so a table produced
    before training cannot be mistaken for one produced after it.
    """
    header = "| Configuration | " + " | ".join(columns) + " |"
    divider = "|---|" + "|".join(["---:"] * len(columns)) + "|"
    lines = [header, divider]

    best = {
        column: max((r.metrics.get(column, 0.0) for r in results), default=0.0)
        for column in columns
    }

    for result in results:
        label = result.description or result.config_name
        if result.degraded:
            label += " ⚠️ *not actually fine-tuned*"
        cells = []
        for column in columns:
            value = result.metrics.get(column)
            if value is None:
                cells.append("—")
            elif abs(value - best[column]) < 1e-9 and len(results) > 1:
                cells.append(f"**{value:.3f}**")
            else:
                cells.append(f"{value:.3f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)
