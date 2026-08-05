"""End-to-end retrieval composition, without loading a real model.

Component tests do not catch the interesting bug class here, which is *wiring*: a
config whose `version_filter` flag never reaches the index, a rerank stage that drops
the version-correct chunk, a hybrid union that loses results. These run the real
`retrieve_many` path with stub models so the composition itself is under test.
"""

import numpy as np
import pytest

from kubeversion_rag.models import Chunk, Corpus
from kubeversion_rag.retrieval.dense import InMemoryIndex
from kubeversion_rag.retrieval.pipeline import PipelineConfig, ResolvedPipeline
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


class StubEmbedder:
    """Returns a fixed query vector so ranking is fully determined by the index."""

    model_name_or_path = "stub"

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.array(vector, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def encode_queries(self, queries, show_progress: bool = False) -> np.ndarray:
        return np.tile(self.vector, (len(queries), 1))


class StubReranker:
    """Scores by a caller-supplied key so rerank behaviour is deterministic."""

    def __init__(self, prefer_text: str) -> None:
        self.prefer_text = prefer_text
        self.calls = 0

    def rerank_many(self, queries, candidate_lists, top_k):
        self.calls += 1
        out = []
        for candidates in candidate_lists:
            scored = [
                (chunk, 10.0 if self.prefer_text in chunk.text else 0.0) for chunk, _ in candidates
            ]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            out.append(scored[:top_k])
        return out


class StubBM25:
    def __init__(self, corpus: Corpus) -> None:
        self.chunks = list(corpus)

    def search_many(self, queries, k, versions):
        results = []
        for version in versions:
            hits = [
                (chunk, 1.0) for chunk in self.chunks if version is None or chunk.covers(version)
            ]
            results.append(hits[:k])
        return results


@pytest.fixture
def corpus() -> Corpus:
    # Two snapshots of the same section plus one unrelated chunk. The 1.24-1.27
    # snapshot is placed FIRST so an unfiltered search that ignores version returns
    # the wrong one at the top -- the exact failure the filter must fix.
    return Corpus(
        chunks=[
            Chunk("a.md", ("Doc", "Section"), "OLD guidance for PSP", v("1.24"), v("1.27")),
            Chunk("a.md", ("Doc", "Section"), "NEW guidance for PSP", v("1.28"), v("1.31")),
            Chunk("b.md", ("Doc", "Other"), "unrelated content", v("1.24"), v("1.31")),
        ]
    )


@pytest.fixture
def index(corpus) -> InMemoryIndex:
    # Chunk 0 scores highest, so unfiltered retrieval prefers the stale snapshot.
    vectors = np.array(
        [[1.0, 0.0], [0.9, 0.0], [0.1, 0.0]],
        dtype=np.float32,
    )
    return InMemoryIndex(corpus, vectors)


def build(corpus, index, config, reranker=None, bm25=None) -> ResolvedPipeline:
    return ResolvedPipeline(
        config=config,
        corpus=corpus,
        dense_index=index if config.use_dense else None,
        embedder=StubEmbedder([1.0, 0.0]) if config.use_dense else None,
        bm25_index=bm25,
        reranker=reranker,
    )


class TestVersionFilterWiring:
    def test_without_the_filter_the_stale_snapshot_wins(self, corpus, index):
        pipeline = build(corpus, index, PipelineConfig("no_filter", "d", version_filter=False))
        hits = pipeline.retrieve("PSP?", v("1.30"))
        assert hits[0].chunk.text.startswith("OLD")
        assert not hits[0].chunk.covers(v("1.30")), "this is the bug the filter fixes"

    def test_with_the_filter_only_applicable_chunks_come_back(self, corpus, index):
        pipeline = build(corpus, index, PipelineConfig("filtered", "d", version_filter=True))
        hits = pipeline.retrieve("PSP?", v("1.30"))
        assert hits
        assert all(hit.chunk.covers(v("1.30")) for hit in hits)
        assert hits[0].chunk.text.startswith("NEW")

    def test_the_filter_is_ignored_when_no_version_is_supplied(self, corpus, index):
        pipeline = build(corpus, index, PipelineConfig("filtered", "d", version_filter=True))
        assert len(pipeline.retrieve("PSP?", None)) == 3


class TestRerankWiring:
    def test_rerank_reorders_the_recall_candidates(self, corpus, index):
        reranker = StubReranker(prefer_text="NEW")
        config = PipelineConfig("rr", "d", version_filter=False, rerank=True)
        pipeline = build(corpus, index, config, reranker=reranker)
        hits = pipeline.retrieve("PSP?", v("1.30"))
        assert reranker.calls == 1
        assert hits[0].chunk.text.startswith("NEW")

    def test_rerank_score_replaces_the_retrieval_score(self, corpus, index):
        config = PipelineConfig("rr", "d", rerank=True)
        pipeline = build(corpus, index, config, reranker=StubReranker("NEW"))
        top = pipeline.retrieve("PSP?", v("1.30"))[0]
        assert top.stage == "rerank"
        assert top.rerank_score == top.final_score

    def test_a_rerank_config_without_a_reranker_fails_loudly(self, corpus, index):
        config = PipelineConfig("rr", "d", rerank=True)
        pipeline = build(corpus, index, config, reranker=None)
        with pytest.raises(RuntimeError, match="reranker"):
            pipeline.retrieve("PSP?", v("1.30"))


class TestHybrid:
    def test_union_keeps_each_chunk_once(self, corpus, index):
        config = PipelineConfig("hybrid", "d", use_dense=True, use_bm25=True)
        pipeline = build(corpus, index, config, bm25=StubBM25(corpus))
        hits = pipeline.retrieve("PSP?", None)
        ids = [hit.chunk.chunk_id for hit in hits]
        assert len(ids) == len(set(ids))

    def test_bm25_only_config_needs_no_dense_index(self, corpus):
        config = PipelineConfig("lex", "d", use_dense=False, use_bm25=True)
        pipeline = ResolvedPipeline(config=config, corpus=corpus, bm25_index=StubBM25(corpus))
        assert pipeline.retrieve("PSP?", v("1.30"))


class TestBatching:
    def test_batched_retrieval_matches_single_retrieval(self, corpus, index):
        config = PipelineConfig("filtered", "d", version_filter=True)
        pipeline = build(corpus, index, config)
        single = pipeline.retrieve("PSP?", v("1.30"))
        batched = pipeline.retrieve_many(["PSP?", "PSP?"], [v("1.30"), v("1.26")])
        assert [h.chunk.chunk_id for h in batched[0]] == [h.chunk.chunk_id for h in single]
        # The second query asks about a different version and must get the other snapshot.
        assert batched[1][0].chunk.text.startswith("OLD")

    def test_per_query_versions_are_applied_independently(self, corpus, index):
        config = PipelineConfig("filtered", "d", version_filter=True)
        pipeline = build(corpus, index, config)
        results = pipeline.retrieve_many(["q"] * 3, [v("1.25"), v("1.30"), v("1.99")])
        assert results[0][0].chunk.text.startswith("OLD")
        assert results[1][0].chunk.text.startswith("NEW")
        assert results[2] == [], "no chunk covers 1.99"


class TestConfigContract:
    def test_final_k_covers_every_reported_metric(self):
        # recall@20 is reported, so a config returning 10 would make that column
        # silently identical to recall@10.
        from kubeversion_rag.retrieval.pipeline import STANDARD_CONFIGS

        assert all(config.final_k >= 20 for config in STANDARD_CONFIGS)

    def test_the_ladder_changes_one_component_at_a_time(self):
        from kubeversion_rag.retrieval.pipeline import STANDARD_CONFIGS

        def flags(config):
            return (
                config.use_dense,
                config.use_bm25,
                config.version_filter,
                config.rerank,
                config.bi_encoder_path is not None,
                config.cross_encoder_path is not None,
            )

        for earlier, later in zip(STANDARD_CONFIGS, STANDARD_CONFIGS[1:], strict=False):
            differences = sum(
                1 for a, b in zip(flags(earlier), flags(later), strict=True) if a != b
            )
            assert differences <= 2, (
                f"{earlier.name} -> {later.name} changes {differences} things at once; "
                "the resulting delta is not attributable to any single component"
            )
