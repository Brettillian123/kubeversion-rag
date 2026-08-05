"""Guards on the embedding cache.

The bug these exist to prevent is the nastiest kind this project can have: a cached
matrix produced by the *base* model being silently reused for the *fine-tuned* row of
the ablation. Both have identical shape, so a shape-only validity check accepts it, and
the results table then reports the base model's numbers under a row labelled
"fine-tuned" with no warning anywhere.
"""

import json

import numpy as np
import pytest

from kubeversion_rag.models import Chunk, Corpus
from kubeversion_rag.retrieval.dense import InMemoryIndex, build_index
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


class FakeEmbedder:
    """Deterministic stand-in so the tests never download or run a real model."""

    def __init__(
        self, name: str, dimension: int = 8, fill: float = 1.0, max_seq_length: int = 320
    ) -> None:
        self.model_name_or_path = name
        self._dimension = dimension
        # Part of the cache fingerprint: the same model at a different input
        # length produces different vectors.
        self.max_seq_length = max_seq_length
        self.fill = fill
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_chunks(self, chunks, show_progress: bool = True) -> np.ndarray:
        self.calls += 1
        return np.full((len(chunks), self._dimension), self.fill, dtype=np.float32)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        chunks=[
            Chunk("a.md", ("A",), "one", v("1.24"), v("1.27")),
            Chunk("a.md", ("A",), "two", v("1.28"), v("1.31")),
            Chunk("b.md", ("B",), "three", v("1.24"), v("1.31")),
        ]
    )


class TestCacheValidity:
    def test_second_call_with_the_same_model_reuses_the_cache(self, corpus, tmp_path):
        cache = tmp_path / "emb.npy"
        embedder = FakeEmbedder("base")
        build_index(corpus, embedder, cache_path=cache)
        build_index(corpus, embedder, cache_path=cache)
        assert embedder.calls == 1

    def test_a_different_model_at_the_same_path_invalidates_the_cache(self, corpus, tmp_path):
        # Same dimension, same chunk count -- a shape check would happily reuse it.
        cache = tmp_path / "emb.npy"
        build_index(corpus, FakeEmbedder("base", fill=1.0), cache_path=cache)

        finetuned = FakeEmbedder("finetuned", fill=2.0)
        index = build_index(corpus, finetuned, cache_path=cache)

        assert finetuned.calls == 1, "must re-embed rather than reuse the base model's vectors"
        assert index.vectors[0][0] == 2.0

    def test_a_changed_corpus_invalidates_the_cache(self, corpus, tmp_path):
        cache = tmp_path / "emb.npy"
        embedder = FakeEmbedder("base")
        build_index(corpus, embedder, cache_path=cache)

        # Same number of chunks, different content -- the digest covers chunk ids, so
        # this is caught where a count check would not be.
        changed = Corpus(
            chunks=[
                Chunk("a.md", ("A",), "one", v("1.24"), v("1.27")),
                Chunk("a.md", ("A",), "CHANGED", v("1.28"), v("1.31")),
                Chunk("b.md", ("B",), "three", v("1.24"), v("1.31")),
            ]
        )
        build_index(changed, embedder, cache_path=cache)
        assert embedder.calls == 2

    def test_a_cache_with_no_sidecar_is_not_trusted(self, corpus, tmp_path):
        # Caches written by an older build have no provenance record. Trusting them
        # would reintroduce exactly the bug the sidecar exists to prevent.
        cache = tmp_path / "emb.npy"
        np.save(cache, np.zeros((len(corpus), 8), dtype=np.float32))
        embedder = FakeEmbedder("base")
        build_index(corpus, embedder, cache_path=cache)
        assert embedder.calls == 1

    def test_a_corrupt_sidecar_invalidates_rather_than_crashes(self, corpus, tmp_path):
        cache = tmp_path / "emb.npy"
        build_index(corpus, FakeEmbedder("base"), cache_path=cache)
        cache.with_suffix(".meta.json").write_text("{not json", encoding="utf-8")

        embedder = FakeEmbedder("base")
        build_index(corpus, embedder, cache_path=cache)
        assert embedder.calls == 1

    def test_force_bypasses_a_valid_cache(self, corpus, tmp_path):
        cache = tmp_path / "emb.npy"
        embedder = FakeEmbedder("base")
        build_index(corpus, embedder, cache_path=cache)
        build_index(corpus, embedder, cache_path=cache, force=True)
        assert embedder.calls == 2

    def test_sidecar_records_the_resolved_model(self, corpus, tmp_path):
        cache = tmp_path / "emb.npy"
        build_index(corpus, FakeEmbedder("/models/biencoder"), cache_path=cache)
        meta = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["model"] == "/models/biencoder"
        assert meta["n_chunks"] == 3


class TestVersionFiltering:
    def _index(self, corpus: Corpus) -> InMemoryIndex:
        vectors = np.eye(len(corpus), 8, dtype=np.float32)
        return InMemoryIndex(corpus, vectors)

    def test_filter_excludes_chunks_that_do_not_cover_the_version(self, corpus):
        index = self._index(corpus)
        query = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        hits = index.search(query, k=10, version=v("1.30"))
        assert all(chunk.covers(v("1.30")) for chunk, _ in hits)
        assert all(chunk.text != "one" for chunk, _ in hits)

    def test_unfiltered_search_returns_every_chunk(self, corpus):
        index = self._index(corpus)
        query = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        assert len(index.search(query, k=10)) == 3

    def test_a_version_no_chunk_covers_returns_nothing(self, corpus):
        index = self._index(corpus)
        query = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        assert index.search(query, k=10, version=v("1.99")) == []

    def test_batch_search_matches_single_search(self, corpus):
        index = self._index(corpus)
        query = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        single = index.search(query, k=3, version=v("1.30"))
        batched = index.search_batch(query[None, :], k=3, versions=[v("1.30")])[0]
        assert [c.chunk_id for c, _ in single] == [c.chunk_id for c, _ in batched]

    def test_mismatched_corpus_and_vectors_is_rejected_at_construction(self, corpus):
        with pytest.raises(ValueError, match="mismatch"):
            InMemoryIndex(corpus, np.zeros((2, 8), dtype=np.float32))
