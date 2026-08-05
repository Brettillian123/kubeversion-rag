"""Embedding service.

Split out from the RAG API for two reasons that both show up under load:

* **The API image doesn't need torch.** Keeping it out means the API image is a few
  hundred MB instead of several GB, so an HPA scale-up is seconds rather than minutes --
  which matters precisely when traffic is spiking.
* **The two scale on different signals.** Embedding is CPU-saturating and batch-friendly;
  the API is mostly waiting on Qdrant and the model API. Coupling them forces one
  replica count on two workloads with unrelated bottlenecks.

The model is loaded once at startup, not per request, and readiness stays false until
that finishes -- otherwise Kubernetes routes traffic to a pod that will spend the first
30 seconds loading weights and timing out.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from ..config import load_config
from ..retrieval.dense import Embedder

log = logging.getLogger(__name__)

EMBED_REQUESTS = Counter("kvrag_embed_requests_total", "Embedding requests", ["kind", "outcome"])
EMBED_TEXTS = Counter("kvrag_embed_texts_total", "Texts embedded", ["kind"])
EMBED_LATENCY = Histogram(
    "kvrag_embed_seconds",
    "Embedding latency",
    ["kind"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

MAX_BATCH = int(os.environ.get("KVRAG_EMBED_MAX_BATCH", "256"))
MAX_CHARS = int(os.environ.get("KVRAG_EMBED_MAX_CHARS", "20000"))


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    # Queries get the bge instruction prefix, passages do not. Getting this backwards
    # silently costs several points of recall with no error anywhere.
    kind: Literal["query", "passage"] = "query"


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dimension: int
    model: str


class _State:
    embedder: Embedder | None = None
    model_name: str = ""
    ready: bool = False


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    state.model_name = os.environ.get("KVRAG_EMBED_MODEL", config.retrieval.bi_encoder)
    log.info("loading embedding model %s", state.model_name)
    started = time.monotonic()
    state.embedder = Embedder(
        state.model_name,
        query_prefix=config.retrieval.query_prefix,
        batch_size=config.retrieval.embed_batch_size,
        # Must match what the corpus was indexed with, or query vectors land in a
        # different region of the space than the passages they are meant to match.
        max_seq_length=config.retrieval.max_seq_length,
    )
    # Force the lazy load now and warm the graph, so the first real request does not
    # pay for it and readiness genuinely means ready.
    state.embedder.encode_queries(["warmup"])
    state.ready = True
    log.info("model ready in %.1fs (dim=%d)", time.monotonic() - started, state.embedder.dimension)
    yield
    state.ready = False


app = FastAPI(title="kubeversion-rag embedding service", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is running. Deliberately does not check the model."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, object]:
    """Readiness: the model is loaded and can serve. False during startup."""
    if not state.ready or state.embedder is None:
        raise HTTPException(status_code=503, detail="model still loading")
    return {"status": "ready", "model": state.model_name, "dimension": state.embedder.dimension}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    if state.embedder is None or not state.ready:
        raise HTTPException(status_code=503, detail="model still loading")

    if len(request.texts) > MAX_BATCH:
        EMBED_REQUESTS.labels(request.kind, "rejected").inc()
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(request.texts)} exceeds the limit of {MAX_BATCH}",
        )
    # An unbounded text length lets one caller pin a CPU and stall every other
    # request on the pod, since encoding is synchronous.
    oversized = [index for index, text in enumerate(request.texts) if len(text) > MAX_CHARS]
    if oversized:
        EMBED_REQUESTS.labels(request.kind, "rejected").inc()
        raise HTTPException(
            status_code=413,
            detail=f"texts at positions {oversized[:5]} exceed {MAX_CHARS} characters",
        )

    with EMBED_LATENCY.labels(request.kind).time():
        try:
            if request.kind == "query":
                vectors = state.embedder.encode_queries(request.texts)
            else:
                vectors = state.embedder.encode_passages(request.texts)
        except Exception as exc:  # noqa: BLE001
            EMBED_REQUESTS.labels(request.kind, "error").inc()
            log.exception("embedding failed")
            raise HTTPException(status_code=500, detail="embedding failed") from exc

    EMBED_REQUESTS.labels(request.kind, "ok").inc()
    EMBED_TEXTS.labels(request.kind).inc(len(request.texts))
    return EmbedResponse(
        vectors=vectors.tolist(),
        dimension=int(vectors.shape[1]),
        model=state.model_name,
    )
