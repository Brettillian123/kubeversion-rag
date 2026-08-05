"""The RAG API.

Request path: resolve the version → embed the query (over HTTP, no torch here) →
version-filtered vector search → optional rerank → cited answer or refusal.

The active Qdrant collection is read from the environment **per request**. That is what
makes the zero-downtime embedding-model migration work: flipping a ConfigMap value and
rolling the deployment is enough, with no image rebuild. Capturing it at import time
would leave every running pod serving the old collection until it happened to restart.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from ..config import load_config
from ..models import DeprecationFact, read_jsonl
from ..retrieval.query import ResolvedVersion, VersionResolver, VersionSource
from ..versions import MinorVersion
from .generate import Answer, GenerationError, Generator
from .store import QdrantStore, SearchFilter

log = logging.getLogger(__name__)

ASK_REQUESTS = Counter("kvrag_ask_total", "Questions asked", ["outcome"])
ASK_LATENCY = Histogram(
    "kvrag_ask_seconds",
    "End-to-end question latency",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
)
STAGE_LATENCY = Histogram(
    "kvrag_stage_seconds",
    "Per-stage latency",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
VERSION_SOURCE = Counter(
    "kvrag_version_source_total", "How the target version was determined", ["source"]
)
# Watched in the dashboard: a rising defaulted-version rate means users are not saying
# their version and answers are quietly assuming latest.
REFUSALS = Counter("kvrag_refusals_total", "Refusals", ["reason"])
COLLECTION_INFO = Gauge(
    "kvrag_active_collection_points", "Points in the active collection", ["collection"]
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    # An explicit version always wins over anything inferred from the text. Clients
    # that know their cluster version should send it.
    version: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str
    version: str
    version_source: str
    version_disclosure: str
    citations: list[dict]
    warnings: dict
    usage: dict
    timings_ms: dict


class _State:
    store: QdrantStore | None = None
    resolver: VersionResolver | None = None
    generator: Generator | None = None
    http: httpx.AsyncClient | None = None
    embed_url: str = ""
    collection: str = ""
    ready: bool = False


state = _State()
config = load_config()


def _active_collection() -> str:
    return config.serving.active_collection


def _store_for(collection: str) -> QdrantStore:
    """Return a store bound to ``collection``, rebuilding if the ConfigMap changed."""
    if state.store is None or state.store.collection != collection:
        log.info("binding to collection %s", collection)
        state.store = QdrantStore(
            url=config.serving.qdrant_url,
            collection=collection,
            api_key=config.serving.qdrant_api_key,
        )
        state.collection = collection
    return state.store


@asynccontextmanager
async def lifespan(app: FastAPI):
    facts_path = config.paths.interim / "deprecation_facts.jsonl"
    facts: list[DeprecationFact] = []
    if facts_path.exists():
        facts = [DeprecationFact.from_dict(row) for row in read_jsonl(facts_path)]
        log.info("loaded %d deprecation facts for version inference", len(facts))
    else:
        log.warning(
            "no deprecation facts at %s -- implicit version inference is disabled and "
            "questions without an explicit version will default to %s",
            facts_path,
            config.max_version,
        )

    state.resolver = VersionResolver(facts, config.min_version, config.max_version)
    state.generator = Generator(
        model=config.serving.generation_model,
        max_tokens=config.serving.max_output_tokens,
        min_score=config.serving.min_context_score,
        timeout=config.serving.request_timeout_s,
    )
    state.embed_url = config.serving.embed_service_url.rstrip("/")
    state.http = httpx.AsyncClient(timeout=config.serving.request_timeout_s)
    _store_for(_active_collection())
    state.ready = True
    log.info("API ready (collection=%s, embed=%s)", state.collection, state.embed_url)
    yield
    state.ready = False
    if state.http is not None:
        await state.http.aclose()


app = FastAPI(title="kubeversion-rag API", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness only. Must not depend on Qdrant: a Qdrant blip should not trigger a
    restart loop of every API pod, which would turn a degraded dependency into an
    outage."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, object]:
    """Readiness: can this pod actually answer right now.

    Checks the *collection*, not just that Qdrant responds. A pod pointed at a
    collection that was never backfilled will otherwise accept traffic and return
    nothing at all -- the failure mode this probe exists to prevent.
    """
    if not state.ready:
        raise HTTPException(status_code=503, detail="starting up")
    collection = _active_collection()
    store = _store_for(collection)
    if not store.healthy():
        raise HTTPException(status_code=503, detail=f"collection {collection} is missing or empty")
    points = store.count()
    COLLECTION_INFO.labels(collection).set(points)
    return {"status": "ready", "collection": collection, "points": points}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _embed_query(question: str) -> list[float]:
    if state.http is None:
        raise HTTPException(status_code=503, detail="starting up")
    try:
        response = await state.http.post(
            f"{state.embed_url}/embed", json={"texts": [question], "kind": "query"}
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        log.error("embedding service unreachable: %s", exc)
        raise HTTPException(status_code=502, detail="embedding service unavailable") from exc
    return response.json()["vectors"][0]


def _resolve_version(request: AskRequest) -> ResolvedVersion:
    if state.resolver is None:
        raise HTTPException(status_code=503, detail="starting up")
    if request.version:
        parsed = MinorVersion.try_parse(request.version)
        if parsed is None:
            raise HTTPException(status_code=400, detail=f"unparseable version {request.version!r}")
        clamped = min(max(parsed, config.min_version), config.max_version)
        evidence = str(parsed)
        if clamped != parsed:
            evidence = f"{parsed} (outside the indexed {config.min_version}-{config.max_version})"
        return ResolvedVersion(clamped, VersionSource.EXPLICIT, evidence=evidence)
    return state.resolver.resolve(request.question)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if not state.ready or state.generator is None:
        raise HTTPException(status_code=503, detail="starting up")

    started = time.monotonic()
    timings: dict[str, float] = {}

    resolved = _resolve_version(request)
    VERSION_SOURCE.labels(resolved.source.value).inc()

    mark = time.monotonic()
    query_vector = await _embed_query(request.question)
    timings["embed"] = (time.monotonic() - mark) * 1000
    STAGE_LATENCY.labels("embed").observe(timings["embed"] / 1000)

    mark = time.monotonic()
    store = _store_for(_active_collection())
    try:
        hits = store.search(
            query_vector,
            limit=request.top_k or config.retrieval.final_k,
            search_filter=SearchFilter(version=resolved.version),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("vector search failed")
        ASK_REQUESTS.labels("error").inc()
        raise HTTPException(status_code=502, detail="vector search unavailable") from exc
    timings["retrieve"] = (time.monotonic() - mark) * 1000
    STAGE_LATENCY.labels("retrieve").observe(timings["retrieve"] / 1000)

    mark = time.monotonic()
    try:
        answer: Answer = state.generator.answer(request.question, hits, resolved)
    except GenerationError as exc:
        ASK_REQUESTS.labels("error").inc()
        raise HTTPException(status_code=502, detail="generation failed") from exc
    timings["generate"] = (time.monotonic() - mark) * 1000
    STAGE_LATENCY.labels("generate").observe(timings["generate"] / 1000)

    total = time.monotonic() - started
    ASK_LATENCY.observe(total)
    timings["total"] = total * 1000

    if answer.refused:
        REFUSALS.labels(answer.refusal_reason).inc()
        ASK_REQUESTS.labels("refused").inc()
    else:
        ASK_REQUESTS.labels("answered").inc()

    payload = answer.to_dict()
    return AskResponse(
        answer=payload["answer"],
        refused=payload["refused"],
        refusal_reason=payload["refusal_reason"],
        version=str(resolved.version),
        version_source=resolved.source.value,
        version_disclosure=payload["version_disclosure"],
        citations=payload["citations"],
        warnings=payload["warnings"],
        usage=payload["usage"],
        timings_ms={key: round(value, 1) for key, value in timings.items()},
    )


@app.get("/debug/retrieve")
async def debug_retrieve(
    question: str = Query(..., min_length=3, max_length=2000),
    version: str | None = None,
    # Bounded like /ask. An unbounded top_k on an unauthenticated endpoint is a free
    # amplification primitive: one small request pulls arbitrarily many payloads out of
    # Qdrant and serializes them all.
    top_k: int = Query(10, ge=1, le=50),
) -> dict:
    """Retrieval without generation.

    The single most useful endpoint when an answer looks wrong: it separates "retrieval
    surfaced the wrong chunks" from "retrieval was fine and generation misread them",
    which are completely different bugs with completely different fixes.
    """
    resolved = _resolve_version(AskRequest(question=question, version=version))
    query_vector = await _embed_query(question)
    store = _store_for(_active_collection())
    hits = store.search(
        query_vector, limit=top_k, search_filter=SearchFilter(version=resolved.version)
    )
    return {
        "question": question,
        "version": str(resolved.version),
        "version_source": resolved.source.value,
        "collection": store.collection,
        "hits": [
            {
                "score": round(hit.score, 4),
                "chunk_id": hit.chunk.chunk_id,
                "versions": str(hit.chunk.version_range),
                "covers_target": hit.chunk.covers(resolved.version),
                "doc_path": hit.chunk.doc_path,
                "heading": " > ".join(hit.chunk.heading_path),
                "preview": hit.chunk.text[:240],
            }
            for hit in hits
        ],
    }


def run() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(
        "kubeversion_rag.serving.api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
