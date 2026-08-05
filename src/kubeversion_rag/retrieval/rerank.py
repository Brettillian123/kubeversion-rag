"""Cross-encoder reranking.

The bi-encoder has to compress a passage into one vector before it ever sees the query,
which is exactly the wrong architecture for this problem: two snapshots of the same
section differ in a handful of tokens, and that difference has to survive the
compression to be usable. A cross-encoder reads the query and passage jointly, so a
single contradicting sentence can dominate the score.

That argument is sound and it is *not* where most of the measured gain came from — the
fine-tuned bi-encoder is, by a wide margin. The reranker turned out to be the most
fragile component in the stack rather than the most valuable one: what it is trained
against matters more than the architecture, and getting that wrong made retrieval worse
than having no reranker at all. `docs/RESULTS.md § What the reranker cost` has the
numbers; `dataset/mine_negatives.py` has the diagnosis.

It is only affordable at all because it runs on the top-50 from recall rather than the
whole corpus.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..models import Chunk

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)


class Reranker:
    """Wraps a sentence-transformers CrossEncoder.

    The passage side includes the version range, matching what the training pairs look
    like. If the reranker is shown bare text at serve time but version-annotated text
    during training, it will underperform in a way that is genuinely hard to debug --
    the numbers just come out lower with no obvious cause.
    """

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading cross-encoder %s", self.model_name_or_path)
            self._model = CrossEncoder(
                self.model_name_or_path,
                max_length=self.max_length,
                device=self._device,
            )
        return self._model

    @staticmethod
    def passage_text(chunk: Chunk) -> str:
        return chunk.embed_text()

    def score(self, query: str, chunks: Sequence[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, self.passage_text(chunk)) for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """Re-order recall candidates, returning ``(chunk, cross_encoder_score)``.

        The returned score replaces the retrieval score entirely rather than blending
        them. Blending sounds appealing but reintroduces the bi-encoder's version
        blindness into the final ordering, which is the failure being fixed.
        """
        if not candidates:
            return []
        chunks = [chunk for chunk, _ in candidates]
        scores = self.score(query, chunks)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    def rerank_many(
        self,
        queries: Sequence[str],
        candidate_lists: Sequence[Sequence[tuple[Chunk, float]]],
        top_k: int,
    ) -> list[list[tuple[Chunk, float]]]:
        """Score every (query, candidate) pair in one batched forward pass.

        Reranking 50 candidates for each of a few thousand eval queries is 100k+ pairs.
        Calling ``rerank`` per query would issue thousands of tiny batches; flattening
        first keeps the GPU (or the CPU's BLAS) fed.
        """
        flat_pairs: list[tuple[str, str]] = []
        offsets: list[tuple[int, int]] = []
        for query, candidates in zip(queries, candidate_lists, strict=True):
            start = len(flat_pairs)
            flat_pairs.extend((query, self.passage_text(chunk)) for chunk, _ in candidates)
            offsets.append((start, len(flat_pairs)))

        if not flat_pairs:
            return [[] for _ in queries]

        scores = self.model.predict(
            flat_pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(flat_pairs) > 5000,
            convert_to_numpy=True,
        )

        results: list[list[tuple[Chunk, float]]] = []
        for (start, end), candidates in zip(offsets, candidate_lists, strict=True):
            chunks = [chunk for chunk, _ in candidates]
            ranked = sorted(
                zip(chunks, (float(s) for s in scores[start:end]), strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            results.append(ranked[:top_k])
        return results
