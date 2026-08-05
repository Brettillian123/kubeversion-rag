"""Qdrant-backed vector store.

Collection naming carries the embedding model and a revision
(``chunks__bge_small_en_v15__v3``) because a collection is only meaningful alongside the
model that produced its vectors. Encoding that in the name makes the invalid
combination unnameable rather than merely discouraged, and it is what lets the
migration script keep two collections alive at once during a model swap.

The version filter is pushed into Qdrant as a range condition rather than applied after
retrieval. Post-filtering silently shrinks the result set -- ask for 50 and get 6 --
which quietly starves the reranker exactly on the queries where version matters most.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..models import Chunk, RetrievedChunk
from ..versions import MinorVersion

log = logging.getLogger(__name__)

VECTOR_NAME = "text"


def collection_name(model_id: str, revision: int = 1) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
    return f"chunks__{slug}__v{revision}"


@dataclass
class SearchFilter:
    version: MinorVersion | None = None
    doc_path_prefix: str | None = None


class QdrantStore:
    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key, timeout=int(timeout))

    # --- lifecycle ------------------------------------------------------------------

    def ensure_collection(self, dimension: int, recreate: bool = False) -> None:
        from qdrant_client.http import models as rest

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            log.warning("deleting existing collection %s", self.collection)
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            log.info("creating collection %s (dim=%d)", self.collection, dimension)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    VECTOR_NAME: rest.VectorParams(size=dimension, distance=rest.Distance.COSINE)
                },
            )

        # Range filters on an unindexed payload field force a full scan in Qdrant,
        # which shows up as p99 latency long before it shows up as an error.
        for field in ("version_low", "version_high"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=rest.PayloadSchemaType.INTEGER,
                )
            except Exception as exc:  # noqa: BLE001 - index may already exist
                log.debug("payload index %s: %s", field, exc)

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def healthy(self) -> bool:
        """Whether this store can actually serve a query right now.

        Used by the readiness probe. Checks that the *collection* exists and is
        non-empty, not merely that Qdrant is up -- a pod pointed at a collection that
        was never backfilled will happily accept traffic and return nothing.
        """
        try:
            return self.client.collection_exists(self.collection) and self.count() > 0
        except Exception as exc:  # noqa: BLE001 - probe must never raise
            log.warning("health check failed: %s", exc)
            return False

    # --- writes ---------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors,
        batch_size: int = 256,
    ) -> int:
        from qdrant_client.http import models as rest

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                rest.PointStruct(
                    # Qdrant needs an unsigned int or UUID id; chunk_id is a 16-hex
                    # sha256 prefix, so the low 63 bits are a stable, collision-safe
                    # numeric id for a corpus of this size.
                    id=int(chunk.chunk_id, 16) & ((1 << 63) - 1),
                    vector={VECTOR_NAME: [float(value) for value in vector]},
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "family_id": chunk.family_id,
                        "doc_path": chunk.doc_path,
                        "heading_path": list(chunk.heading_path),
                        "part": chunk.part,
                        "text": chunk.text,
                        "version_low": chunk.version_low.minor,
                        "version_high": chunk.version_high.minor,
                        "version_major": chunk.version_low.major,
                        "version_label": str(chunk.version_range),
                    },
                )
                for chunk, vector in zip(batch, batch_vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            written += len(points)
            log.info("upserted %d/%d", written, len(chunks))
        return written

    # --- reads ----------------------------------------------------------------------

    @staticmethod
    def _build_filter(search_filter: SearchFilter):
        from qdrant_client.http import models as rest

        conditions = []
        if search_filter.version is not None:
            version = search_filter.version
            conditions.append(
                rest.FieldCondition(key="version_low", range=rest.Range(lte=version.minor))
            )
            conditions.append(
                rest.FieldCondition(key="version_high", range=rest.Range(gte=version.minor))
            )
            conditions.append(
                rest.FieldCondition(key="version_major", match=rest.MatchValue(value=version.major))
            )
        if search_filter.doc_path_prefix:
            conditions.append(
                rest.FieldCondition(
                    key="doc_path",
                    match=rest.MatchText(text=search_filter.doc_path_prefix),
                )
            )
        return rest.Filter(must=conditions) if conditions else None

    @staticmethod
    def _to_chunk(payload: dict) -> Chunk:
        major = int(payload.get("version_major", 1))
        return Chunk(
            doc_path=payload["doc_path"],
            heading_path=tuple(payload.get("heading_path", [])),
            text=payload["text"],
            version_low=MinorVersion(major, int(payload["version_low"])),
            version_high=MinorVersion(major, int(payload["version_high"])),
            part=int(payload.get("part", 0)),
        )

    def search(
        self,
        query_vector: Sequence[float],
        limit: int,
        search_filter: SearchFilter | None = None,
    ) -> list[RetrievedChunk]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            using=VECTOR_NAME,
            limit=limit,
            query_filter=self._build_filter(search_filter) if search_filter else None,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=self._to_chunk(point.payload), score=float(point.score))
            for point in response.points
            if point.payload
        ]
