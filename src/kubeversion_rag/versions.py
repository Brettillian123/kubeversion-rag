"""Kubernetes minor-version arithmetic.

Kubernetes documentation is branched per minor release (``release-1.31``), and every
question in this system is scoped to one. Patch versions are irrelevant — the docs do
not branch on them — so the whole system speaks in ``MinorVersion``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

_VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)(?:\.\d+)?$")

# Matches "1.31", "v1.31", "1.31.4" when surrounded by non-version characters.
# Requires the major to be a single digit or more but rejects things like "0.31"
# only at the semantic layer -- Kubernetes has never shipped a major other than 1,
# but hard-coding that here would make the type useless for testing.
_INLINE_VERSION_RE = re.compile(r"(?<![\w.])v?(\d+)\.(\d+)(?:\.\d+)?(?![\w.])")


@dataclass(frozen=True, order=True)
class MinorVersion:
    """A Kubernetes minor version, e.g. 1.31.

    Ordered, hashable, and cheap to compare -- it is used as a dict key and sorted
    in hot paths during chunk coalescing.
    """

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    @classmethod
    def parse(cls, raw: str) -> MinorVersion:
        """Parse ``1.31``/``v1.31``/``1.31.4``. Raises ValueError on anything else."""
        match = _VERSION_RE.match(raw.strip())
        if not match:
            raise ValueError(f"not a Kubernetes version: {raw!r}")
        return cls(int(match.group(1)), int(match.group(2)))

    @classmethod
    def try_parse(cls, raw: str) -> MinorVersion | None:
        try:
            return cls.parse(raw)
        except ValueError:
            return None

    @property
    def branch(self) -> str:
        """The kubernetes/website git branch holding this version's docs."""
        return f"release-{self.major}.{self.minor}"

    def next(self) -> MinorVersion:
        return MinorVersion(self.major, self.minor + 1)

    def previous(self) -> MinorVersion:
        if self.minor == 0:
            raise ValueError(f"no minor version before {self}")
        return MinorVersion(self.major, self.minor - 1)


@dataclass(frozen=True)
class VersionRange:
    """An inclusive, contiguous span of minor versions.

    Ranges are only ever built over a known, contiguous ingestion window, so a range
    is fully described by its endpoints -- there are no holes. ``coalesce_versions``
    is responsible for splitting a non-contiguous set into multiple ranges.
    """

    low: MinorVersion
    high: MinorVersion

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"inverted version range: {self.low} > {self.high}")

    def __str__(self) -> str:
        return str(self.low) if self.low == self.high else f"{self.low}-{self.high}"

    def contains(self, version: MinorVersion) -> bool:
        return self.low <= version <= self.high

    def overlaps(self, other: VersionRange) -> bool:
        return self.low <= other.high and other.low <= self.high

    def __iter__(self) -> Iterator[MinorVersion]:
        current = self.low
        while current <= self.high:
            yield current
            current = current.next()


def coalesce_versions(versions: Iterable[MinorVersion]) -> list[VersionRange]:
    """Collapse a set of versions into the minimal list of contiguous ranges.

    ``{1.24, 1.25, 1.27}`` becomes ``[1.24-1.25, 1.27]``. Used when the same chunk
    text appears on several release branches: adjacent branches collapse into one
    range, and a gap (text changed, then changed back) yields two.

    Assumes a *single* major version stream, which is true for Kubernetes. Versions
    across different majors are never treated as adjacent.
    """
    ordered = sorted(set(versions))
    if not ordered:
        return []

    ranges: list[VersionRange] = []
    start = previous = ordered[0]
    for version in ordered[1:]:
        is_adjacent = version.major == previous.major and version.minor == previous.minor + 1
        if not is_adjacent:
            ranges.append(VersionRange(start, previous))
            start = version
        previous = version
    ranges.append(VersionRange(start, previous))
    return ranges


def extract_versions(text: str) -> list[MinorVersion]:
    """Pull every plausible Kubernetes version mentioned in free text, in order.

    Deliberately permissive at the regex layer and strict at the semantic layer:
    Kubernetes minors are two-digit-ish, so ``2.5`` (a Helm chart version, say) and
    ``1.999`` are rejected. Duplicates are removed while preserving first-seen order,
    because query parsing cares about which version was mentioned *first*.
    """
    seen: dict[MinorVersion, None] = {}
    for match in _INLINE_VERSION_RE.finditer(text):
        major, minor = int(match.group(1)), int(match.group(2))
        if major != 1 or not (0 <= minor <= 99):
            continue
        seen.setdefault(MinorVersion(major, minor), None)
    return list(seen)
