"""Parse the Deprecated API Migration Guide into structured facts.

``content/en/docs/reference/using-api/deprecation-guide.md`` is the densest source of
version-sensitive ground truth in the corpus: effectively a table of
"API X stopped being served in version Y, use Z instead", written as prose but with a
very consistent shape:

    ### v1.32

    #### Flow control resources {#flowcontrol-resources-v132}

    The **flowcontrol.apiserver.k8s.io/v1beta3** API version of FlowSchema and
    PriorityLevelConfiguration is no longer served as of v1.32.

    * Migrate manifests and API clients to use the **flowcontrol.apiserver.k8s.io/v1**
      API version, available since v1.29.

Each parsed fact becomes several questions with a *known* correct answer and a known
correct version, which is what makes the generated training set trustworthy without
any LLM labelling.

The parser is deliberately strict: a block that does not match the expected shape is
skipped and counted rather than guessed at. A silently mis-parsed fact would poison
both training and evaluation, and a coverage count is easy to sanity-check against
the rendered page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..models import DeprecationFact
from ..versions import MinorVersion

log = logging.getLogger(__name__)

_RELEASE_HEADING_RE = re.compile(r"^###\s+v(\d+\.\d+)\s*$", re.MULTILINE)
_BLOCK_HEADING_RE = re.compile(r"^####\s+(.+?)\s*$", re.MULTILINE)
_ANCHOR_RE = re.compile(r"\s*\{#[^}]*\}\s*$")

# The guide states removals two ways. Both appear on the current page, and the second
# form covers PodSecurityPolicy -- the single most consequential version-sensitive fact
# in the corpus -- so handling only the first would drop exactly the example this
# project leads with.
#
#   1. "The **group/version** API version of A, B and C is no longer served as of v1.32."
_REMOVAL_RE = re.compile(
    r"\*\*(?P<api>[A-Za-z0-9._/-]+)\*\*\s+API\s+version[s]?\s+of\s+(?P<resources>.+?)\s+"
    r"(?:is|are)\s+no\s+longer\s+served\s+as\s+of\s+v(?P<version>\d+\.\d+)",
    re.IGNORECASE | re.DOTALL,
)
#   2. "PodSecurityPolicy in the **policy/v1beta1** API version is no longer served as of v1.25."
# The resource list is spelled out explicitly rather than with `.+?` because this form
# has no leading anchor -- a lazy wildcard would happily swallow the preceding prose.
_REMOVAL_INVERTED_RE = re.compile(
    r"(?P<resources>[A-Z][A-Za-z0-9]*(?:(?:,\s*|\s+and\s+)[A-Z][A-Za-z0-9]*)*)\s+in\s+the\s+"
    r"\*\*(?P<api>[A-Za-z0-9._/-]+)\*\*\s+API\s+version\s+(?:is|are)\s+no\s+longer\s+served\s+"
    r"as\s+of\s+v(?P<version>\d+\.\d+)",
    re.DOTALL,
)

# "...use the **group/version** API version, available since v1.29."
_REPLACEMENT_RE = re.compile(
    r"\*\*(?P<api>[A-Za-z0-9._/-]+)\*\*\s+API\s+version,\s*available\s+since\s+v(?P<version>\d+\.\d+)",
    re.IGNORECASE | re.DOTALL,
)

_RESOURCE_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+", re.IGNORECASE)
_MARKUP_RE = re.compile(r"[*`_]")


@dataclass
class ParseReport:
    """Coverage accounting, so a regression in the parser is visible, not silent."""

    blocks_seen: int = 0
    facts_parsed: int = 0
    blocks_skipped: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.blocks_skipped is None:
            self.blocks_skipped = []

    @property
    def coverage(self) -> float:
        return self.facts_parsed / self.blocks_seen if self.blocks_seen else 0.0

    def summary(self) -> str:
        return (
            f"parsed {self.facts_parsed}/{self.blocks_seen} deprecation blocks "
            f"({self.coverage:.0%} coverage)"
        )


def _clean_resource(raw: str) -> str:
    return _MARKUP_RE.sub("", raw).strip().strip(".")


def _parse_resources(raw: str) -> tuple[str, ...]:
    """Split "FlowSchema and PriorityLevelConfiguration" into individual resources.

    Resource names are CamelCase Kubernetes kinds, so anything that is not a bare
    identifier is prose the regex over-captured and is dropped.
    """
    parts = [_clean_resource(part) for part in _RESOURCE_SPLIT_RE.split(raw)]
    return tuple(part for part in parts if part and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", part))


def _iter_blocks(markdown: str) -> list[tuple[MinorVersion, str, str]]:
    """Yield ``(release, heading, body)`` for each ``####`` block under a ``### v1.x``.

    Blocks appearing before any release heading (page preamble) are ignored.
    """
    release_marks = [
        (match.start(), MinorVersion.parse(match.group(1)))
        for match in _RELEASE_HEADING_RE.finditer(markdown)
    ]
    if not release_marks:
        return []

    def release_for(offset: int) -> MinorVersion | None:
        current = None
        for start, version in release_marks:
            if start <= offset:
                current = version
            else:
                break
        return current

    block_marks = list(_BLOCK_HEADING_RE.finditer(markdown))
    blocks: list[tuple[MinorVersion, str, str]] = []
    for index, match in enumerate(block_marks):
        release = release_for(match.start())
        if release is None:
            continue
        end = block_marks[index + 1].start() if index + 1 < len(block_marks) else len(markdown)
        # A block also ends at the next release heading, whichever comes first.
        for start, _ in release_marks:
            if match.end() < start < end:
                end = start
                break
        heading = _ANCHOR_RE.sub("", match.group(1)).strip()
        blocks.append((release, heading, markdown[match.end() : end]))
    return blocks


def parse_deprecation_guide(
    markdown: str,
    source_doc: str = "content/en/docs/reference/using-api/deprecation-guide.md",
) -> tuple[list[DeprecationFact], ParseReport]:
    """Extract every removal fact from the guide.

    Returns the facts plus a coverage report. Callers should log the report --
    coverage dropping is the earliest signal that upstream reworded the page.
    """
    report = ParseReport()
    facts: list[DeprecationFact] = []
    seen: set[str] = set()

    for release, heading, body in _iter_blocks(markdown):
        report.blocks_seen += 1

        removal = _REMOVAL_RE.search(body) or _REMOVAL_INVERTED_RE.search(body)
        if not removal:
            report.blocks_skipped.append(f"v{release}/{heading}: no removal sentence")
            continue

        resources = _parse_resources(removal.group("resources"))
        if not resources:
            report.blocks_skipped.append(f"v{release}/{heading}: no parseable resources")
            continue

        stated_version = MinorVersion.parse(removal.group("version"))
        if stated_version != release:
            # The in-sentence version is authoritative -- the heading grouping is only
            # a navigational convenience and has been wrong in the past.
            log.debug(
                "block under v%s states removal in v%s; trusting the sentence",
                release,
                stated_version,
            )

        replacement = _REPLACEMENT_RE.search(body)
        fact = DeprecationFact(
            removed_in=stated_version,
            api_group_version=removal.group("api").strip(),
            resources=resources,
            replacement_group_version=replacement.group("api").strip() if replacement else None,
            replacement_since=(
                MinorVersion.parse(replacement.group("version")) if replacement else None
            ),
            source_doc=source_doc,
            source_heading=("Deprecated API Migration Guide", "Removed APIs by release", heading),
        )
        if fact.fact_id in seen:
            continue
        seen.add(fact.fact_id)
        facts.append(fact)
        report.facts_parsed += 1

    facts.sort(key=lambda f: (f.removed_in, f.api_group_version))
    return facts, report
