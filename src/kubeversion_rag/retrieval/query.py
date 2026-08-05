"""Work out which Kubernetes version a question is really about.

Three cases, in descending order of confidence:

1. **Explicit** — the question names a version ("on 1.28", "v1.31.4").
2. **Implicit** — the question names an API version that was removed at a known
   release, which bounds the asker's cluster below it. Someone asking about
   ``policy/v1beta1`` PodSecurityPolicy is, almost by construction, not on 1.25+.
3. **Absent** — nothing to go on. Default to the newest ingested version, and *say so
   in the answer*. Silently assuming latest is how a version-aware system quietly
   becomes a version-blind one.

The implicit table is derived from the parsed deprecation facts rather than
hand-written, so it stays correct as the corpus advances. That also makes this a real
baseline to beat: the deliberate ordering here is rules first, learned model second,
so any classifier has to earn its place against a table that costs nothing to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ..models import DeprecationFact
from ..versions import MinorVersion, extract_versions

# "1.28" inside a version-ish context. Guards against matching unrelated decimals
# ("scale to 1.5x the replicas") by requiring a nearby cue word or a v-prefix.
_CUE_RE = re.compile(
    r"(?:kubernetes|k8s|cluster|version|release|upgrad\w*|running|on|to|eks|gke|aks)\W{0,12}"
    r"v?(\d\.\d{1,2})(?:\.\d+)?",
    re.IGNORECASE,
)
_V_PREFIXED_RE = re.compile(r"(?<![\w.])v(\d\.\d{1,2})(?:\.\d+)?(?![\w.])")


class VersionSource(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    DEFAULTED = "defaulted"


@dataclass
class ResolvedVersion:
    version: MinorVersion
    source: VersionSource
    evidence: str = ""

    @property
    def is_confident(self) -> bool:
        return self.source is not VersionSource.DEFAULTED

    def disclosure(self) -> str:
        """A sentence the answer can carry so the user can catch a wrong inference."""
        if self.source is VersionSource.EXPLICIT:
            return f"Answering for Kubernetes {self.version}."
        if self.source is VersionSource.INFERRED:
            return (
                f"No version given; inferred Kubernetes {self.version} from "
                f"{self.evidence}. Say your version if that's wrong."
            )
        return (
            f"No version given, so this answers for Kubernetes {self.version} "
            f"(the newest indexed). Behaviour may differ on older clusters."
        )


class VersionResolver:
    """Rule-based version extraction. The baseline a learned model must beat."""

    def __init__(
        self,
        facts: list[DeprecationFact],
        min_version: MinorVersion,
        max_version: MinorVersion,
    ) -> None:
        self.min_version = min_version
        self.max_version = max_version
        # api_group_version -> earliest release that stopped serving it. Earliest,
        # because an API removed in several stages bounds the cluster by the first.
        self._removed_in: dict[str, MinorVersion] = {}
        for fact in facts:
            key = fact.api_group_version.lower()
            existing = self._removed_in.get(key)
            if existing is None or fact.removed_in < existing:
                self._removed_in[key] = fact.removed_in

    def _explicit(self, query: str) -> ResolvedVersion | None:
        candidates: list[str] = [m.group(1) for m in _V_PREFIXED_RE.finditer(query)]
        candidates += [m.group(1) for m in _CUE_RE.finditer(query)]
        for raw in candidates:
            version = MinorVersion.try_parse(raw)
            if version and version.major == 1:
                clamped = self._clamp(version)
                return ResolvedVersion(clamped, VersionSource.EXPLICIT, evidence=raw)

        # Last resort: a bare version anywhere in the string. Lower confidence than the
        # cued forms above, but still explicit -- the user typed a version.
        for version in extract_versions(query):
            return ResolvedVersion(
                self._clamp(version), VersionSource.EXPLICIT, evidence=str(version)
            )
        return None

    def _inferred(self, query: str) -> ResolvedVersion | None:
        lowered = query.lower()
        best: tuple[MinorVersion, str] | None = None
        for api, removed_in in self._removed_in.items():
            if api not in lowered:
                continue
            if removed_in <= self.min_version:
                continue
            bound = removed_in.previous()
            if best is None or bound < best[0]:
                best = (bound, api)
        if best is None:
            return None
        version, api = best
        return ResolvedVersion(
            self._clamp(version),
            VersionSource.INFERRED,
            evidence=f"the mention of {api} (removed in {self._removed_in[api]})",
        )

    def _clamp(self, version: MinorVersion) -> MinorVersion:
        """Pin to the ingested window.

        A question about 1.19 cannot be answered from a corpus starting at 1.24.
        Clamping and disclosing beats returning nothing, but the caller can detect the
        clamp by comparing against the evidence string.
        """
        if version < self.min_version:
            return self.min_version
        if version > self.max_version:
            return self.max_version
        return version

    def resolve(self, query: str) -> ResolvedVersion:
        return (
            self._explicit(query)
            or self._inferred(query)
            or ResolvedVersion(self.max_version, VersionSource.DEFAULTED)
        )
