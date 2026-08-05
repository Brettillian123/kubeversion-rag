"""Runtime configuration.

Every value is overridable by environment variable so the same image runs in
docker-compose, in CI, and on a cluster with only a ConfigMap difference.

``ACTIVE_COLLECTION`` in particular is read *per request* rather than captured at
import time -- the zero-downtime embedding migration flips it via ConfigMap and a
rollout, and a module-level constant would silently keep serving the old collection
for the life of the process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .versions import MinorVersion

# --- Corpus window -----------------------------------------------------------------
# The oldest branch worth ingesting. Docs older than this describe a Kubernetes most
# people cannot run, and every extra branch is a linear cost in clone time and index
# size. 1.24 is the floor because it is where several high-traffic removals landed
# (PodSecurityPolicy deprecation, dockershim removal), which makes it a rich source
# of version-sensitive questions.
DEFAULT_MIN_VERSION = MinorVersion(1, 24)
DEFAULT_MAX_VERSION = MinorVersion(1, 35)

WEBSITE_REPO = "https://github.com/kubernetes/website.git"
DOCS_SUBTREE = "content/en/docs"
DEPRECATION_GUIDE_PATH = "content/en/docs/reference/using-api/deprecation-guide.md"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_version(name: str, default: MinorVersion) -> MinorVersion:
    raw = os.environ.get(name)
    return MinorVersion.parse(raw) if raw else default


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path = field(default_factory=lambda: _env_path("KVRAG_DATA_DIR", _REPO_ROOT / "data"))

    @property
    def raw(self) -> Path:
        """Sparse git checkouts, one worktree per release branch."""
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        """Chunked corpus and parsed deprecation facts."""
        return self.root / "interim"

    @property
    def datasets(self) -> Path:
        """Generated train/dev/test triples."""
        return self.root / "datasets"

    @property
    def gold(self) -> Path:
        """Hand-verified evaluation questions. Never trained on."""
        return self.root / "gold"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def ensure(self) -> None:
        for path in (self.raw, self.interim, self.datasets, self.gold, self.models, self.results):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkingConfig:
    # Targets are in characters, not tokens: the chunker splits on markdown structure
    # first and only uses size as a tiebreaker, so token-exactness buys nothing and
    # would drag a tokenizer into the ingestion path.
    target_chars: int = _env_int("KVRAG_CHUNK_TARGET_CHARS", 1400)
    max_chars: int = _env_int("KVRAG_CHUNK_MAX_CHARS", 2400)
    min_chars: int = _env_int("KVRAG_CHUNK_MIN_CHARS", 200)


@dataclass
class RetrievalConfig:
    bi_encoder: str = os.environ.get("KVRAG_BI_ENCODER", "BAAI/bge-small-en-v1.5")
    cross_encoder: str = os.environ.get(
        "KVRAG_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # Recall wide, rerank narrow. 50 is where recall@k flattens on this corpus; going
    # wider mostly adds cross-encoder latency for chunks that never surface.
    recall_k: int = _env_int("KVRAG_RECALL_K", 50)
    final_k: int = _env_int("KVRAG_FINAL_K", 5)
    # bge models are trained with an asymmetric query prefix; omitting it costs
    # several points of recall and is the single most common way to misuse them.
    query_prefix: str = os.environ.get(
        "KVRAG_QUERY_PREFIX", "Represent this sentence for searching relevant passages: "
    )
    embed_batch_size: int = _env_int("KVRAG_EMBED_BATCH", 32)


@dataclass
class ServingConfig:
    qdrant_url: str = os.environ.get("KVRAG_QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.environ.get("KVRAG_QDRANT_API_KEY") or None
    embed_service_url: str = os.environ.get("KVRAG_EMBED_URL", "http://localhost:8081")
    default_collection: str = os.environ.get("KVRAG_COLLECTION", "chunks__bge_small_en_v15__v1")
    generation_model: str = os.environ.get("KVRAG_MODEL", "claude-opus-5")
    max_output_tokens: int = _env_int("KVRAG_MAX_TOKENS", 2000)
    request_timeout_s: float = _env_float("KVRAG_TIMEOUT_S", 60.0)
    # Below this rerank score the pipeline refuses rather than answering. Tuned on the
    # gold set so that unanswerable questions refuse without suppressing real answers.
    min_context_score: float = _env_float("KVRAG_MIN_SCORE", -4.0)

    @property
    def active_collection(self) -> str:
        """Read per call -- see module docstring."""
        return os.environ.get("KVRAG_COLLECTION", self.default_collection)


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    min_version: MinorVersion = field(
        default_factory=lambda: _env_version("KVRAG_MIN_VERSION", DEFAULT_MIN_VERSION)
    )
    max_version: MinorVersion = field(
        default_factory=lambda: _env_version("KVRAG_MAX_VERSION", DEFAULT_MAX_VERSION)
    )

    def versions(self) -> list[MinorVersion]:
        out, current = [], self.min_version
        while current <= self.max_version:
            out.append(current)
            current = current.next()
        return out


def load_config() -> Config:
    return Config()
