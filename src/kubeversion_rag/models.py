"""Core data types shared by ingestion, training, evaluation, and serving.

These are the on-disk contract: every stage reads and writes JSONL of these shapes,
so a change here is a change to the pipeline's interchange format. Serialization is
hand-written rather than pickled so the intermediate files stay greppable and
diffable, which matters a lot when debugging why a chunk did or did not retrieve.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .versions import MinorVersion, VersionRange


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass
class Chunk:
    """One retrievable passage, valid over a contiguous span of Kubernetes versions.

    ``family_id`` groups every version of the same section of the same document. Two
    chunks sharing a family but with disjoint version ranges are exactly the
    "same document, wrong version" pair that this project treats as a hard negative.
    """

    doc_path: str
    heading_path: tuple[str, ...]
    text: str
    version_low: MinorVersion
    version_high: MinorVersion
    # Ordinal of this chunk among chunks sharing the same heading path in the same
    # document. Non-zero in two cases: an over-long section split into several pieces,
    # and a document that genuinely repeats a heading (a "Note" under two different
    # parents at the same depth). Without it those chunks would collapse into one
    # family, every one of them would claim the same (family, version) slot, and the
    # disjointness invariant that ``Corpus.covering`` and hard-negative mining both
    # depend on would be violated.
    part: int = 0

    @property
    def family_id(self) -> str:
        """Identity of the document section, independent of version or wording."""
        return _stable_id(self.doc_path, " > ".join(self.heading_path), str(self.part))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def chunk_id(self) -> str:
        """Stable across runs: same text in the same place is the same chunk."""
        return _stable_id(self.family_id, self.content_hash, str(self.version_low))

    @property
    def version_range(self) -> VersionRange:
        return VersionRange(self.version_low, self.version_high)

    def covers(self, version: MinorVersion) -> bool:
        return self.version_low <= version <= self.version_high

    @property
    def title(self) -> str:
        return self.heading_path[-1] if self.heading_path else Path(self.doc_path).stem

    def citation(self) -> str:
        """A human-checkable pointer back to the source of truth."""
        branch = self.version_high.branch
        url = f"https://github.com/kubernetes/website/blob/{branch}/{self.doc_path}"
        return f"{self.title} ({self.version_range}) — {url}"

    def embed_text(self) -> str:
        """What actually gets embedded.

        The heading path and version range are prepended deliberately: without them,
        two adjacent-version snapshots of a section embed to nearly identical vectors,
        which is the failure this project exists to fix. Giving the encoder the
        version in-band lets fine-tuning learn to use it, instead of forcing the
        metadata filter to do all the work alone.
        """
        crumbs = " > ".join(self.heading_path) if self.heading_path else self.title
        return f"[Kubernetes {self.version_range}] {crumbs}\n\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "family_id": self.family_id,
            "doc_path": self.doc_path,
            "heading_path": list(self.heading_path),
            "part": self.part,
            "text": self.text,
            "version_low": str(self.version_low),
            "version_high": str(self.version_high),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Chunk:
        return cls(
            doc_path=raw["doc_path"],
            heading_path=tuple(raw["heading_path"]),
            text=raw["text"],
            version_low=MinorVersion.parse(raw["version_low"]),
            version_high=MinorVersion.parse(raw["version_high"]),
            part=int(raw.get("part", 0)),
        )


@dataclass
class DeprecationFact:
    """One structured row parsed out of the Deprecated API Migration Guide.

    The guide is unusually regular prose -- effectively a table rendered as markdown --
    which makes it the highest-signal source of version-sensitive ground truth in the
    corpus. Each fact yields several templated questions with known correct answers.
    """

    removed_in: MinorVersion
    api_group_version: str
    resources: tuple[str, ...]
    replacement_group_version: str | None
    replacement_since: MinorVersion | None
    source_doc: str
    source_heading: tuple[str, ...]

    @property
    def fact_id(self) -> str:
        return _stable_id(str(self.removed_in), self.api_group_version, ",".join(self.resources))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "removed_in": str(self.removed_in),
            "api_group_version": self.api_group_version,
            "resources": list(self.resources),
            "replacement_group_version": self.replacement_group_version,
            "replacement_since": (str(self.replacement_since) if self.replacement_since else None),
            "source_doc": self.source_doc,
            "source_heading": list(self.source_heading),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DeprecationFact:
        since = raw.get("replacement_since")
        return cls(
            removed_in=MinorVersion.parse(raw["removed_in"]),
            api_group_version=raw["api_group_version"],
            resources=tuple(raw["resources"]),
            replacement_group_version=raw.get("replacement_group_version"),
            replacement_since=MinorVersion.parse(since) if since else None,
            source_doc=raw["source_doc"],
            source_heading=tuple(raw["source_heading"]),
        )


@dataclass
class Example:
    """A retrieval training/eval example.

    ``hard_negative_ids`` are same-family, wrong-version chunks. ``target_version`` is
    what the question is scoped to, and is what ``version-correct@1`` is judged against.
    """

    question: str
    target_version: MinorVersion
    positive_chunk_id: str
    hard_negative_ids: tuple[str, ...] = ()
    family_id: str = ""
    source: str = "generated"
    # Set on questions the system is expected to refuse: the corpus contains no
    # chunk that answers them. Kept in the same file as answerable questions on
    # purpose -- a refusal metric measured on a separate file is easy to forget to run.
    unanswerable: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "target_version": str(self.target_version),
            "positive_chunk_id": self.positive_chunk_id,
            "hard_negative_ids": list(self.hard_negative_ids),
            "family_id": self.family_id,
            "source": self.source,
            "unanswerable": self.unanswerable,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Example:
        return cls(
            question=raw["question"],
            target_version=MinorVersion.parse(raw["target_version"]),
            positive_chunk_id=raw["positive_chunk_id"],
            hard_negative_ids=tuple(raw.get("hard_negative_ids", [])),
            family_id=raw.get("family_id", ""),
            source=raw.get("source", "generated"),
            unanswerable=bool(raw.get("unanswerable", False)),
            notes=raw.get("notes", ""),
        )


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    stage: str = "dense"
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass
class Corpus:
    """The chunk collection, with the lookups every downstream stage needs."""

    chunks: list[Chunk] = field(default_factory=list)
    _by_id: dict[str, Chunk] = field(default_factory=dict, repr=False)
    _by_family: dict[str, list[Chunk]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.reindex()

    def reindex(self) -> None:
        self._by_id = {}
        self._by_family = {}
        for chunk in self.chunks:
            self._by_id[chunk.chunk_id] = chunk
            self._by_family.setdefault(chunk.family_id, []).append(chunk)
        for family in self._by_family.values():
            family.sort(key=lambda c: c.version_low)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def family(self, family_id: str) -> list[Chunk]:
        return self._by_family.get(family_id, [])

    def families(self) -> Iterable[str]:
        return self._by_family.keys()

    def covering(self, family_id: str, version: MinorVersion) -> Chunk | None:
        """The one chunk in a family valid at ``version``, if any.

        Ranges within a family are disjoint by construction (see
        ``ingest.chunk.coalesce_corpus``), so at most one can match.
        """
        for chunk in self.family(family_id):
            if chunk.covers(version):
                return chunk
        return None

    @classmethod
    def load(cls, path: Path) -> Corpus:
        return cls(chunks=[Chunk.from_dict(row) for row in read_jsonl(path)])

    def save(self, path: Path) -> None:
        write_jsonl(path, (chunk.to_dict() for chunk in self.chunks))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_examples(path: Path) -> list[Example]:
    return [Example.from_dict(row) for row in read_jsonl(path)]


def save_examples(path: Path, examples: Sequence[Example]) -> int:
    return write_jsonl(path, (example.to_dict() for example in examples))
