"""Dense retrieval: the bi-encoder and an in-memory index for evaluation.

``sentence_transformers`` (and therefore torch) is imported lazily inside the class.
The RAG API image does not install torch -- it calls the embedding service over HTTP --
and a module-level import here would make the whole retrieval package unimportable in
that image.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..config import RetrievalConfig
from ..models import Chunk, Corpus
from ..versions import MinorVersion

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformers bi-encoder.

    Two asymmetries matter and are easy to get wrong:

    * ``bge`` models expect an instruction prefix on the *query* side only. Embedding
      queries without it costs several points of recall, and it is the most common way
      these models are misused.
    * Passages are embedded from ``Chunk.embed_text()``, which carries the heading path
      and version range. Embedding raw ``chunk.text`` instead would strip exactly the
      signal fine-tuning is supposed to learn to use.
    """

    def __init__(
        self,
        model_name_or_path: str,
        query_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._device = device
        self._model: SentenceTransformer | None = None

    @classmethod
    def from_config(cls, config: RetrievalConfig, model_override: str | None = None) -> Embedder:
        return cls(
            model_name_or_path=model_override or config.bi_encoder,
            query_prefix=config.query_prefix,
            batch_size=config.embed_batch_size,
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading bi-encoder %s", self.model_name_or_path)
            self._model = SentenceTransformer(self.model_name_or_path, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_queries(self, queries: Sequence[str], show_progress: bool = False) -> np.ndarray:
        prefixed = [f"{self.query_prefix}{query}" for query in queries]
        return self._encode(prefixed, show_progress)

    def encode_passages(self, passages: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self._encode(list(passages), show_progress)

    def encode_chunks(self, chunks: Sequence[Chunk], show_progress: bool = True) -> np.ndarray:
        return self.encode_passages([chunk.embed_text() for chunk in chunks], show_progress)

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            # Normalized here so every downstream similarity is a plain dot product.
            # Doing it once at encode time avoids a class of bug where one call site
            # normalizes and another does not, silently changing the score scale.
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class InMemoryIndex:
    """Brute-force cosine search over the whole corpus.

    Exact by construction, which is what the evaluation harness wants: an ANN index's
    recall ceiling would be entangled with the model's, and the ablation is trying to
    attribute changes to the *model*. At this corpus size (tens of thousands of chunks)
    a full matmul is milliseconds, so there is nothing to gain from approximation here.
    Serving uses Qdrant; evaluation uses this.
    """

    def __init__(self, corpus: Corpus, vectors: np.ndarray) -> None:
        if len(corpus) != vectors.shape[0]:
            raise ValueError(
                f"corpus/vector mismatch: {len(corpus)} chunks vs {vectors.shape[0]} vectors"
            )
        self.corpus = corpus
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunks: list[Chunk] = list(corpus)
        # Parallel arrays instead of per-chunk attribute access: the version filter runs
        # on every query over every chunk, and a Python-level loop dominates the matmul.
        self._low = np.array([c.version_low.minor for c in self.chunks], dtype=np.int16)
        self._high = np.array([c.version_high.minor for c in self.chunks], dtype=np.int16)
        self._major = np.array([c.version_low.major for c in self.chunks], dtype=np.int16)

    def version_mask(self, version: MinorVersion) -> np.ndarray:
        return (
            (self._major == version.major)
            & (self._low <= version.minor)
            & (self._high >= version.minor)
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        version: MinorVersion | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Top-k by cosine similarity, optionally restricted to chunks covering ``version``."""
        if self.vectors.shape[0] == 0:
            return []
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32).ravel()

        if version is not None:
            mask = self.version_mask(version)
            if not mask.any():
                return []
            # -inf rather than deleting rows so indices still line up with self.chunks.
            scores = np.where(mask, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in top]

    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int,
        versions: Sequence[MinorVersion | None],
    ) -> list[list[tuple[Chunk, float]]]:
        """One matmul for the whole query set, then per-query masking.

        Evaluation runs thousands of queries per configuration; batching the matmul
        turns the dominant cost from thousands of small BLAS calls into one large one.
        """
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]
        all_scores = query_vectors @ self.vectors.T

        results: list[list[tuple[Chunk, float]]] = []
        for row, version in zip(all_scores, versions, strict=True):
            scores = row
            if version is not None:
                mask = self.version_mask(version)
                if not mask.any():
                    results.append([])
                    continue
                scores = np.where(mask, row, -np.inf)
            limit = min(k, int(np.isfinite(scores).sum()))
            if limit <= 0:
                results.append([])
                continue
            top = np.argpartition(-scores, limit - 1)[:limit]
            top = top[np.argsort(-scores[top])]
            results.append([(self.chunks[int(i)], float(scores[int(i)])) for i in top])
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)

    @classmethod
    def load(cls, corpus: Corpus, path: Path) -> InMemoryIndex:
        return cls(corpus, np.load(path))


def _cache_fingerprint(corpus: Corpus, embedder: Embedder) -> dict[str, object]:
    """Identity of the (corpus, model) pair a cached matrix was produced from.

    Shape alone is not enough, and assuming it was is a silent-wrong-results bug: a
    base model and its fine-tuned descendant have identical dimensions, so a matrix
    embedded before training would sail through a shape check and be reused *as if it
    were the fine-tuned one*. The ablation would then report the base model's numbers
    under the fine-tuned row, with nothing anywhere indicating a problem.

    The corpus digest covers chunk ids, so re-ingesting changed docs invalidates the
    cache even when the chunk count happens to be unchanged.
    """
    digest = hashlib.sha256()
    for chunk in corpus:
        digest.update(chunk.chunk_id.encode("ascii"))
    return {
        "model": embedder.model_name_or_path,
        "n_chunks": len(corpus),
        "dimension": embedder.dimension,
        "corpus_digest": digest.hexdigest()[:32],
    }


def build_index(
    corpus: Corpus,
    embedder: Embedder,
    cache_path: Path | None = None,
    force: bool = False,
) -> InMemoryIndex:
    """Embed the corpus, reusing a cached matrix only when it provably still applies."""
    sidecar = cache_path.with_suffix(".meta.json") if cache_path is not None else None

    if cache_path is not None and cache_path.exists() and not force:
        expected = _cache_fingerprint(corpus, embedder)
        actual: dict[str, object] | None = None
        if sidecar is not None and sidecar.exists():
            try:
                actual = json.loads(sidecar.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("unreadable cache sidecar %s: %s", sidecar, exc)

        if actual == expected:
            log.info("reusing cached embeddings at %s (%s)", cache_path, expected["model"])
            return InMemoryIndex(corpus, np.load(cache_path))

        log.warning(
            "discarding embedding cache at %s -- it was built for %s, this run needs %s",
            cache_path,
            actual or "an unrecorded configuration",
            expected,
        )

    vectors = embedder.encode_chunks(list(corpus))
    index = InMemoryIndex(corpus, vectors)
    if cache_path is not None:
        index.save(cache_path)
        if sidecar is not None:
            sidecar.write_text(
                json.dumps(_cache_fingerprint(corpus, embedder), indent=2), encoding="utf-8"
            )
    return index


def iter_batches(items: Sequence[object], size: int) -> Iterable[Sequence[object]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
