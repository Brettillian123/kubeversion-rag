"""Question templates.

Every template must produce a question whose *correct answer changes with the target
version*. A question like "what is a PodSecurityPolicy" is useless here: the same chunk
answers it at every version, so it cannot distinguish a version-aware retriever from a
naive one and contributes nothing but noise to both training and evaluation.

Templates are deliberately varied in surface form (first person, imperative, indirect)
so the fine-tuned encoder does not learn a single lexical pattern. They are still
templates, and the README says so -- the hand-written gold set exists precisely
because templated phrasing is narrower than real user questions.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ..versions import MinorVersion

# Questions where the API in question has already been removed at the target version.
# The correct answer is "no, use the replacement" -- and crucially, the *same question
# text asked about an earlier version* has the opposite answer.
REMOVED_TEMPLATES: Sequence[str] = (
    "I'm on Kubernetes {version}. Can I still use {api} for {resource}?",
    "Is {api} still served for {resource} in Kubernetes {version}?",
    "My {resource} manifest uses apiVersion {api} and my cluster is {version}. Will it apply?",
    "Which apiVersion should I use for {resource} on Kubernetes {version}?",
    "We're upgrading to {version} — what happens to our {resource} resources on {api}?",
    "kubectl rejects my {resource} on a {version} cluster. It uses {api}. Why?",
    "What replaced {api} for {resource} by Kubernetes {version}?",
)

# Questions asked at a version where the API still exists but is on its way out.
STILL_SERVED_TEMPLATES: Sequence[str] = (
    "On Kubernetes {version}, is {api} still available for {resource}?",
    "Do I need to migrate {resource} off {api} before running Kubernetes {version}?",
    "Is it safe to keep using {api} for {resource} on a {version} cluster?",
)

# Questions about the removal boundary itself.
BOUNDARY_TEMPLATES: Sequence[str] = (
    "In which Kubernetes release did {api} stop being served for {resource}?",
    "When was {api} removed for {resource}?",
    "Up to which Kubernetes version can I use {api} for {resource}?",
)

# Generic templates over any documentation section whose text changed between
# releases. These carry the volume; the deprecation-derived ones carry the precision.
SECTION_TEMPLATES: Sequence[str] = (
    "How does {topic} work in Kubernetes {version}?",
    "What does the Kubernetes {version} documentation say about {topic}?",
    "{topic} on Kubernetes {version} — what's the current guidance?",
    "I'm running Kubernetes {version}. What do I need to know about {topic}?",
    "Has {topic} changed in Kubernetes {version}?",
    "Explain {topic} as of Kubernetes {version}.",
)

# Plausible-sounding questions with no answer anywhere in the corpus. The system is
# expected to refuse these rather than retrieve the nearest topical chunk and
# confabulate. Kept alongside answerable questions so the refusal metric is measured
# on every eval run rather than in a file someone forgets to pass.
UNANSWERABLE_TEMPLATES: Sequence[str] = (
    "What is the default value of --max-quantum-pods in Kubernetes {version}?",
    "How do I enable the HolographicScheduler feature gate on Kubernetes {version}?",
    "In Kubernetes {version}, what does the apiVersion warp.k8s.io/v2 provide?",
    "Which Kubernetes {version} flag controls the PodTelepathy admission plugin?",
    "How do I configure the quantum-entangled storage class in Kubernetes {version}?",
)


def _pick(templates: Sequence[str], rng: random.Random) -> str:
    return templates[rng.randrange(len(templates))]


def render_removal_question(
    api: str,
    resource: str,
    version: MinorVersion,
    removed_in: MinorVersion,
    rng: random.Random,
) -> str:
    """Choose a template appropriate to where ``version`` sits relative to removal."""
    pool = REMOVED_TEMPLATES if version >= removed_in else STILL_SERVED_TEMPLATES
    return _pick(pool, rng).format(api=api, resource=resource, version=version)


def render_boundary_question(api: str, resource: str, rng: random.Random) -> str:
    return _pick(BOUNDARY_TEMPLATES, rng).format(api=api, resource=resource)


def render_section_question(topic: str, version: MinorVersion, rng: random.Random) -> str:
    return _pick(SECTION_TEMPLATES, rng).format(topic=topic, version=version)


def render_unanswerable_question(version: MinorVersion, rng: random.Random) -> str:
    return _pick(UNANSWERABLE_TEMPLATES, rng).format(version=version)


def topic_from_heading(heading_path: tuple[str, ...]) -> str:
    """Turn a heading breadcrumb into a natural noun phrase for a question.

    Uses the two most specific headings: the leaf alone is often too generic
    ("Overview", "Example"), and the whole breadcrumb reads like a file path.
    """
    meaningful = [part for part in heading_path if part and part.lower() not in _GENERIC_HEADINGS]
    if not meaningful:
        meaningful = list(heading_path)
    tail = meaningful[-2:] if len(meaningful) >= 2 else meaningful
    return " — ".join(tail) if len(tail) > 1 else (tail[0] if tail else "this topic")


_GENERIC_HEADINGS = {
    "overview",
    "example",
    "examples",
    "notes",
    "summary",
    "introduction",
    "what's next",
    "whats next",
    "before you begin",
    "see also",
    "next steps",
    "feedback",
}
