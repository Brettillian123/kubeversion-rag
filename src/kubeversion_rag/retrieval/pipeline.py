"""Compose retrieval components into named, comparable configurations.

The ablation table is only meaningful if each row differs from the one above it by
exactly one thing. That constraint lives here: a configuration is a set of flags, the
pipeline honours them, and the evaluation harness runs the same query set through each.
Anything that varies between rows but is not a flag on this object is an
unattributed confound.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Chunk, Corpus, RetrievedChunk
from ..versions import MinorVersion
from .bm25 import BM25Index
from .dense import Embedder, InMemoryIndex, build_index
from .rerank import Reranker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """One row of the ablation table."""

    name: str
    description: str
    use_dense: bool = True
    use_bm25: bool = False
    version_filter: bool = False
    rerank: bool = False
    bi_encoder_path: str | None = None  # None -> the configured off-the-shelf model
    cross_encoder_path: str | None = None
    recall_k: int = 50
    # Must be at least the largest k any reported metric uses. The metrics compute
    # recall@20, so returning 10 would make recall@20 silently identical to recall@10 --
    # a column that looks informative and measures nothing.
    final_k: int = 20

    def requires_bi_encoder(self) -> bool:
        return self.use_dense

    def cache_key(self) -> str:
        """Distinguishes embedding caches. Two configs sharing a bi-encoder share a cache.

        Only a coarse filename hint -- correctness comes from the fingerprint sidecar
        written by ``build_index``, which records the *resolved* model. This key just
        keeps unrelated configurations from thrashing one file.
        """
        model = self.bi_encoder_path or "default"
        return "".join(char if char.isalnum() else "_" for char in model).strip("_")[:80]


# The ladder. Each step adds exactly one component so the table attributes cleanly.
STANDARD_CONFIGS: tuple[PipelineConfig, ...] = (
    PipelineConfig(
        name="bm25",
        description="BM25 lexical baseline, no version awareness",
        use_dense=False,
        use_bm25=True,
    ),
    PipelineConfig(
        name="dense_offtheshelf",
        description="Off-the-shelf bi-encoder, no version awareness",
    ),
    PipelineConfig(
        name="dense_filtered",
        description="Off-the-shelf bi-encoder + metadata version filter",
        version_filter=True,
    ),
    PipelineConfig(
        name="dense_ft_filtered",
        description="Fine-tuned bi-encoder + version filter",
        version_filter=True,
        bi_encoder_path="__finetuned_biencoder__",
    ),
    PipelineConfig(
        name="dense_ft_filtered_rerank",
        description="Fine-tuned bi-encoder + version filter + fine-tuned cross-encoder",
        version_filter=True,
        rerank=True,
        bi_encoder_path="__finetuned_biencoder__",
        cross_encoder_path="__finetuned_crossencoder__",
    ),
    PipelineConfig(
        name="hybrid_ft_filtered_rerank",
        description="Adds BM25 union to the full stack",
        version_filter=True,
        rerank=True,
        use_bm25=True,
        bi_encoder_path="__finetuned_biencoder__",
        cross_encoder_path="__finetuned_crossencoder__",
    ),
)

# Placeholders resolved at runtime against whatever training produced. Keeping them
# symbolic means the config list does not have to know the output directory layout.
FINETUNED_BIENCODER = "__finetuned_biencoder__"
FINETUNED_CROSSENCODER = "__finetuned_crossencoder__"


def resolve_model_path(token: str | None, models_dir: Path, default: str) -> tuple[str, bool]:
    """Map a symbolic model token to a real path.

    Returns ``(path, fell_back)``. The flag is returned rather than inferred by the
    caller comparing strings: a config that legitimately names the base model (the
    migration script's "incumbent" arm does exactly that) would otherwise be reported
    as a failed fallback, and the results doc would carry a false warning.

    Falling back rather than raising is deliberate -- the whole ablation should run
    end-to-end on a fresh clone before anything is trained. But it is logged loudly and
    surfaced in the table, because a silently-untrained row labelled "fine-tuned" is a
    fabricated result.
    """
    if token is None:
        return default, False
    if token == FINETUNED_BIENCODER:
        path = models_dir / "biencoder"
    elif token == FINETUNED_CROSSENCODER:
        path = models_dir / "crossencoder"
    else:
        return token, False

    if path.exists():
        return str(path), False
    log.warning(
        "%s not found at %s; falling back to the base model. Rows labelled fine-tuned "
        "in this run are NOT fine-tuned -- train first before quoting these numbers.",
        token,
        path,
    )
    return default, True


@dataclass
class ResolvedPipeline:
    """A PipelineConfig with its models actually loaded."""

    config: PipelineConfig
    corpus: Corpus
    dense_index: InMemoryIndex | None = None
    embedder: Embedder | None = None
    bm25_index: BM25Index | None = None
    reranker: Reranker | None = None
    # Set when a fine-tuned model was requested but not found, so the report can flag it.
    degraded: list[str] = field(default_factory=list)

    def retrieve(
        self,
        query: str,
        version: MinorVersion | None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        return self.retrieve_many([query], [version], top_k)[0]

    def retrieve_many(
        self,
        queries: Sequence[str],
        versions: Sequence[MinorVersion | None],
        top_k: int | None = None,
    ) -> list[list[RetrievedChunk]]:
        config = self.config
        final_k = top_k or config.final_k
        effective_versions: list[MinorVersion | None] = [
            version if config.version_filter else None for version in versions
        ]

        candidate_lists: list[list[tuple[Chunk, float]]] = [[] for _ in queries]

        if config.use_dense:
            if self.dense_index is None or self.embedder is None:
                raise RuntimeError(f"config {config.name} needs a dense index but none was built")
            query_vectors = self.embedder.encode_queries(list(queries), show_progress=False)
            dense_hits = self.dense_index.search_batch(
                query_vectors, config.recall_k, effective_versions
            )
            candidate_lists = [list(hits) for hits in dense_hits]

        if config.use_bm25:
            if self.bm25_index is None:
                raise RuntimeError(f"config {config.name} needs BM25 but no index was built")
            lexical = self.bm25_index.search_many(
                list(queries), config.recall_k, effective_versions
            )
            candidate_lists = [
                _union_preserving_order(dense, lex)
                for dense, lex in zip(candidate_lists, lexical, strict=True)
            ]

        if config.rerank:
            if self.reranker is None:
                raise RuntimeError(f"config {config.name} needs a reranker but none was loaded")
            reranked = self.reranker.rerank_many(list(queries), candidate_lists, final_k)
            return [
                [
                    RetrievedChunk(chunk=chunk, score=score, stage="rerank", rerank_score=score)
                    for chunk, score in hits
                ]
                for hits in reranked
            ]

        stage = (
            "hybrid"
            if (config.use_dense and config.use_bm25)
            else ("dense" if config.use_dense else "bm25")
        )
        return [
            [
                RetrievedChunk(chunk=chunk, score=score, stage=stage)
                for chunk, score in hits[:final_k]
            ]
            for hits in candidate_lists
        ]


def _union_preserving_order(
    dense: Sequence[tuple[Chunk, float]],
    lexical: Sequence[tuple[Chunk, float]],
) -> list[tuple[Chunk, float]]:
    """Merge two candidate lists, keeping each chunk once.

    Scores from BM25 and cosine are on different scales and are *not* comparable, so
    this does not attempt to fuse them numerically. It is a recall-widening union whose
    ordering only matters if no reranker follows; when one does, the reranker imposes
    the real order. Dense results keep their position, and lexical-only results are
    appended -- which is why the hybrid row is only meaningful with rerank enabled.
    """
    merged = list(dense)
    seen = {chunk.chunk_id for chunk, _ in dense}
    for chunk, score in lexical:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            merged.append((chunk, score))
    return merged


def build_pipeline(
    config: PipelineConfig,
    corpus: Corpus,
    models_dir: Path,
    cache_dir: Path,
    default_bi_encoder: str,
    default_cross_encoder: str,
    query_prefix: str,
    embed_batch_size: int = 32,
    force_reembed: bool = False,
) -> ResolvedPipeline:
    """Load exactly the components a configuration needs, and nothing more."""
    resolved = ResolvedPipeline(config=config, corpus=corpus)

    if config.use_dense:
        bi_encoder_path, fell_back = resolve_model_path(
            config.bi_encoder_path, models_dir, default_bi_encoder
        )
        if fell_back:
            resolved.degraded.append(f"bi-encoder fell back to {default_bi_encoder}")
        resolved.embedder = Embedder(
            bi_encoder_path, query_prefix=query_prefix, batch_size=embed_batch_size
        )
        cache_path = cache_dir / f"embeddings__{config.cache_key()}.npy"
        resolved.dense_index = build_index(
            corpus, resolved.embedder, cache_path=cache_path, force=force_reembed
        )

    if config.use_bm25:
        resolved.bm25_index = BM25Index(corpus)

    if config.rerank:
        cross_encoder_path, fell_back = resolve_model_path(
            config.cross_encoder_path, models_dir, default_cross_encoder
        )
        if fell_back:
            resolved.degraded.append(f"cross-encoder fell back to {default_cross_encoder}")
        resolved.reranker = Reranker(cross_encoder_path, batch_size=embed_batch_size)

    return resolved
