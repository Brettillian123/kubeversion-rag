"""Bind the hand-written gold questions to concrete chunks.

Gold questions name their expected answer as (document, required substring) rather than
a chunk id. Chunk ids move whenever chunking changes, and a gold file full of stale ids
degrades into a file of zeros that still reports a number -- the worst possible failure
for an evaluation set, because it looks like it is working.

Resolution is strict. A question that no longer matches anything is an error, not a
skip: it means either the corpus moved out from under the question or the question was
wrong, and both need a human.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..models import Chunk, Corpus, Example
from ..versions import MinorVersion

log = logging.getLogger(__name__)


@dataclass
class GoldSpec:
    question: str
    version: MinorVersion
    expect_doc: str | None = None
    expect_text: str | None = None
    unanswerable: bool = False
    rationale: str = ""


class GoldResolutionError(RuntimeError):
    """A gold question no longer matches any chunk. Never silently skipped."""


def load_specs(path: Path) -> list[GoldSpec]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: list[GoldSpec] = []
    for index, row in enumerate(payload.get("questions", []), start=1):
        if "question" not in row or "version" not in row:
            raise GoldResolutionError(f"{path}: entry {index} is missing question or version")
        specs.append(
            GoldSpec(
                question=row["question"],
                version=MinorVersion.parse(str(row["version"])),
                expect_doc=row.get("expect_doc"),
                expect_text=row.get("expect_text"),
                unanswerable=bool(row.get("unanswerable", False)),
                rationale=row.get("rationale", ""),
            )
        )
    return specs


def _candidates(corpus: Corpus, spec: GoldSpec) -> list[Chunk]:
    matches = []
    for chunk in corpus:
        if not chunk.covers(spec.version):
            continue
        if spec.expect_doc and spec.expect_doc not in chunk.doc_path:
            continue
        if spec.expect_text and spec.expect_text not in chunk.text:
            continue
        matches.append(chunk)
    return matches


def resolve(corpus: Corpus, specs: list[GoldSpec]) -> tuple[list[Example], list[str]]:
    """Turn specs into Examples, reporting anything that failed to bind."""
    examples: list[Example] = []
    problems: list[str] = []

    for spec in specs:
        if spec.unanswerable:
            examples.append(
                Example(
                    question=spec.question,
                    target_version=spec.version,
                    positive_chunk_id="",
                    source="gold-unanswerable",
                    unanswerable=True,
                    notes=spec.rationale,
                )
            )
            continue

        matches = _candidates(corpus, spec)
        if not matches:
            problems.append(
                f"{spec.question!r} (v{spec.version}): nothing matches "
                f"doc~{spec.expect_doc!r} text~{spec.expect_text!r}"
            )
            continue

        # Shortest match: among chunks that all contain the required substring, the
        # most specific section is the one a reader would actually want, and a long
        # parent section containing it is a less precise answer.
        positive = min(matches, key=lambda chunk: len(chunk.text))

        # Same-family, wrong-version snapshots -- the negatives this project is about.
        negatives = [
            chunk
            for chunk in corpus.family(positive.family_id)
            if chunk.chunk_id != positive.chunk_id and not chunk.covers(spec.version)
        ]
        # Plus sibling blocks in the same document mentioning the same API at a
        # different removal boundary (the FlowSchema v1beta1/v1beta2/v1beta3 case).
        if spec.expect_doc:
            negatives += [
                chunk
                for chunk in corpus
                if spec.expect_doc in chunk.doc_path
                and chunk.family_id != positive.family_id
                and chunk.chunk_id != positive.chunk_id
            ][:6]

        examples.append(
            Example(
                question=spec.question,
                target_version=spec.version,
                positive_chunk_id=positive.chunk_id,
                hard_negative_ids=tuple(chunk.chunk_id for chunk in negatives[:8]),
                family_id=positive.family_id,
                source="gold",
                notes=spec.rationale,
            )
        )
        if len(matches) > 5:
            log.debug(
                "%r matched %d chunks; picked the most specific. Tighten expect_text "
                "if that is the wrong one.",
                spec.question,
                len(matches),
            )

    return examples, problems
