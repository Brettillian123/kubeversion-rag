"""Lexical retrieval baseline.

Every ablation needs a floor that involves no learned model at all. BM25 is that floor:
if a fine-tuned bi-encoder cannot beat term matching on this corpus, the modelling work
did not pay for itself and the table should say so.

It is also genuinely competitive on part of this corpus. API group strings like
``flowcontrol.apiserver.k8s.io/v1beta3`` are rare exact tokens, and term matching finds
them reliably where a dense encoder blurs them together with their sibling versions.
The interesting failure is the opposite one: BM25 cannot tell two *versions* of the
same passage apart at all, because they share nearly every term.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from ..models import Chunk, Corpus
from ..versions import MinorVersion

log = logging.getLogger(__name__)

# Keeps dots and slashes so "networking.k8s.io/v1" survives as usable pieces rather
# than being shredded into "networking", "k8s", "io", "v1" -- those fragments match
# almost every document in the corpus and destroy the signal.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        tokens.append(token)
        # Also emit the components so a query for "flowschema" still matches a
        # document that only ever writes "flowcontrol.apiserver.k8s.io/v1beta3".
        if len(token) > 3 and any(sep in token for sep in "./-"):
            tokens.extend(part for part in re.split(r"[./-]", token) if len(part) > 2)
    return tokens


class BM25Index:
    """Okapi BM25 over the corpus, with the same version-filter semantics as dense."""

    def __init__(self, corpus: Corpus) -> None:
        from rank_bm25 import BM25Okapi

        self.corpus = corpus
        self.chunks: list[Chunk] = list(corpus)
        log.info("tokenizing %d chunks for BM25", len(self.chunks))
        tokenized = [tokenize(chunk.embed_text()) for chunk in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        k: int,
        version: MinorVersion | None = None,
    ) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: scores[i],
            reverse=True,
        )
        results: list[tuple[Chunk, float]] = []
        for index in ranked:
            chunk = self.chunks[index]
            if version is not None and not chunk.covers(version):
                continue
            results.append((chunk, float(scores[index])))
            if len(results) >= k:
                break
        return results

    def search_many(
        self,
        queries: Sequence[str],
        k: int,
        versions: Sequence[MinorVersion | None],
    ) -> list[list[tuple[Chunk, float]]]:
        return [self.search(q, k, v) for q, v in zip(queries, versions, strict=True)]
