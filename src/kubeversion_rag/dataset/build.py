"""Generate labelled retrieval examples from the corpus's own structure.

No LLM labelling anywhere in this file, deliberately. Positives and hard negatives are
derived from facts the corpus states about itself:

* a chunk **covers** a version (it was present on that release branch), and
* two chunks in the same **family** with disjoint version ranges are the same section
  of the same document at different points in time.

That second relationship is the hard negative this project is built around: two texts
that a sentence encoder scores as near-identical, where exactly one is correct for the
asked version. Mining negatives is usually the expensive part of building a retrieval
training set; here the corpus generates them for free.

An LLM-labelled set would also make the evaluation circular -- the thing being measured
would have produced the yardstick.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass

from ..models import Chunk, Corpus, DeprecationFact, Example
from ..versions import MinorVersion
from .templates import (
    render_boundary_question,
    render_removal_question,
    render_section_question,
    render_unanswerable_question,
    topic_from_heading,
)

log = logging.getLogger(__name__)

MAX_HARD_NEGATIVES = 6


@dataclass
class BuildStats:
    from_deprecations: int = 0
    from_changed_sections: int = 0
    unanswerable: int = 0
    dropped_no_positive: int = 0
    dropped_no_negative: int = 0

    @property
    def total(self) -> int:
        return self.from_deprecations + self.from_changed_sections + self.unanswerable

    def summary(self) -> str:
        return (
            f"{self.total} examples "
            f"({self.from_deprecations} deprecation, "
            f"{self.from_changed_sections} changed-section, "
            f"{self.unanswerable} unanswerable); "
            f"dropped {self.dropped_no_positive} without a positive, "
            f"{self.dropped_no_negative} without a hard negative"
        )


def _deprecation_chunks(corpus: Corpus) -> list[Chunk]:
    return [chunk for chunk in corpus if "deprecation-guide" in chunk.doc_path]


def _mentions(chunk: Chunk, needles: tuple[str, ...]) -> bool:
    haystack = f"{' > '.join(chunk.heading_path)}\n{chunk.text}"
    return any(needle in haystack for needle in needles)


def _same_family_negatives(corpus: Corpus, positive: Chunk, version: MinorVersion) -> list[Chunk]:
    """The wrong-version snapshots of the same document section.

    The single strongest negative type available: same doc, same heading, near-identical
    prose, wrong answer for the asked version.
    """
    return [
        chunk
        for chunk in corpus.family(positive.family_id)
        if chunk.chunk_id != positive.chunk_id and not chunk.covers(version)
    ]


def _sibling_removal_negatives(
    corpus: Corpus,
    positive: Chunk,
    fact: DeprecationFact,
    deprecation_chunks: list[Chunk],
) -> list[Chunk]:
    """Blocks about the *same resource* but a *different* removal version.

    FlowSchema is the canonical case: ``v1beta1`` was removed in 1.26, ``v1beta2`` in
    1.29, ``v1beta3`` in 1.32. Three blocks, nearly identical wording, three different
    correct answers. A retriever that ignores version gets these wrong constantly, and
    they are far harder than an unrelated topical negative.
    """
    negatives = []
    for chunk in deprecation_chunks:
        if chunk.family_id == positive.family_id:
            continue
        if fact.api_group_version in chunk.text:
            continue  # would arguably also be correct; not a clean negative
        if _mentions(chunk, fact.resources):
            negatives.append(chunk)
    return negatives


def build_deprecation_examples(
    corpus: Corpus,
    facts: list[DeprecationFact],
    versions: list[MinorVersion],
    rng: random.Random,
    stats: BuildStats,
) -> list[Example]:
    """Questions whose answer is pinned by a parsed removal fact."""
    examples: list[Example] = []
    guide_chunks = _deprecation_chunks(corpus)
    if not guide_chunks:
        log.warning("no deprecation-guide chunks in corpus; skipping deprecation questions")
        return examples

    for fact in facts:
        # The block that states this removal, at each version where the guide carries it.
        candidates = [
            chunk
            for chunk in guide_chunks
            if fact.api_group_version in chunk.text and _mentions(chunk, fact.resources)
        ]
        if not candidates:
            stats.dropped_no_positive += 1
            continue

        for version in versions:
            # Only ask about versions at or after the removal: before it, the guide
            # does not yet carry the block, so there is no correct chunk to retrieve.
            if version < fact.removed_in:
                continue
            positive = next((chunk for chunk in candidates if chunk.covers(version)), None)
            if positive is None:
                continue

            negatives = _same_family_negatives(corpus, positive, version)
            negatives += _sibling_removal_negatives(corpus, positive, fact, guide_chunks)
            if not negatives:
                stats.dropped_no_negative += 1
                continue

            resource = fact.resources[rng.randrange(len(fact.resources))]
            question = render_removal_question(
                api=fact.api_group_version,
                resource=resource,
                version=version,
                removed_in=fact.removed_in,
                rng=rng,
            )
            examples.append(
                Example(
                    question=question,
                    target_version=version,
                    positive_chunk_id=positive.chunk_id,
                    hard_negative_ids=tuple(
                        chunk.chunk_id for chunk in negatives[:MAX_HARD_NEGATIVES]
                    ),
                    family_id=positive.family_id,
                    source="deprecation",
                    notes=f"{fact.api_group_version} removed in {fact.removed_in}",
                )
            )
            stats.from_deprecations += 1

        # One boundary question per fact, asked at the removal version itself.
        boundary_positive = next(
            (chunk for chunk in candidates if chunk.covers(fact.removed_in)), None
        )
        if boundary_positive is not None:
            negatives = _sibling_removal_negatives(corpus, boundary_positive, fact, guide_chunks)
            if negatives:
                examples.append(
                    Example(
                        question=render_boundary_question(
                            fact.api_group_version, fact.resources[0], rng
                        ),
                        target_version=fact.removed_in,
                        positive_chunk_id=boundary_positive.chunk_id,
                        hard_negative_ids=tuple(
                            chunk.chunk_id for chunk in negatives[:MAX_HARD_NEGATIVES]
                        ),
                        family_id=boundary_positive.family_id,
                        source="deprecation-boundary",
                        notes=f"boundary for {fact.api_group_version}",
                    )
                )
                stats.from_deprecations += 1

    return examples


def build_changed_section_examples(
    corpus: Corpus,
    rng: random.Random,
    stats: BuildStats,
    max_per_family: int = 3,
) -> list[Example]:
    """Questions over any section whose text changed between releases.

    A family with a single chunk never changed, so no version-sensitive question can be
    asked about it and no same-family hard negative exists. Families with two or more
    chunks are exactly the version-sensitive surface of the corpus, and they supply the
    bulk of the training signal.
    """
    examples: list[Example] = []

    for family_id in corpus.families():
        members = corpus.family(family_id)
        if len(members) < 2:
            continue

        # Sample versions across different snapshots so a family does not contribute
        # three near-duplicate questions all pinned to the same chunk.
        sampled: list[tuple[Chunk, MinorVersion]] = []
        for chunk in members:
            span = list(chunk.version_range)
            sampled.append((chunk, span[rng.randrange(len(span))]))
        rng.shuffle(sampled)

        for chunk, version in sampled[:max_per_family]:
            negatives = _same_family_negatives(corpus, chunk, version)
            if not negatives:
                stats.dropped_no_negative += 1
                continue
            topic = topic_from_heading(chunk.heading_path)
            examples.append(
                Example(
                    question=render_section_question(topic, version, rng),
                    target_version=version,
                    positive_chunk_id=chunk.chunk_id,
                    hard_negative_ids=tuple(
                        negative.chunk_id for negative in negatives[:MAX_HARD_NEGATIVES]
                    ),
                    family_id=family_id,
                    source="changed-section",
                    notes=f"{len(members)} snapshots of this section",
                )
            )
            stats.from_changed_sections += 1

    return examples


def build_unanswerable_examples(
    versions: list[MinorVersion],
    rng: random.Random,
    stats: BuildStats,
    count: int = 60,
) -> list[Example]:
    """Questions the corpus cannot answer. Correct behaviour is refusal.

    These carry no positive chunk. They are excluded from retrieval metrics (there is
    nothing to retrieve) and drive the refusal rate reported alongside them.
    """
    examples: list[Example] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < count and attempts < count * 20:
        attempts += 1
        version = versions[rng.randrange(len(versions))]
        question = render_unanswerable_question(version, rng)
        if question in seen:
            continue
        seen.add(question)
        examples.append(
            Example(
                question=question,
                target_version=version,
                positive_chunk_id="",
                family_id="",
                source="unanswerable",
                unanswerable=True,
                notes="no supporting chunk exists; correct behaviour is refusal",
            )
        )
        stats.unanswerable += 1
    return examples


def build_dataset(
    corpus: Corpus,
    facts: list[DeprecationFact],
    versions: list[MinorVersion],
    seed: int = 20260805,
    unanswerable_count: int = 60,
) -> tuple[list[Example], BuildStats]:
    """Assemble the full generated dataset. Deterministic for a given seed."""
    rng = random.Random(seed)
    stats = BuildStats()

    examples = build_deprecation_examples(corpus, facts, versions, rng, stats)
    examples += build_changed_section_examples(corpus, rng, stats)
    examples += build_unanswerable_examples(versions, rng, stats, count=unanswerable_count)

    # Deduplicate on the question text: different templates occasionally collide, and
    # a duplicated question inflates whichever split it lands in.
    deduped: dict[str, Example] = {}
    for example in examples:
        deduped.setdefault(example.question.lower().strip(), example)

    ordered = sorted(deduped.values(), key=lambda e: (e.source, e.question))
    rng.shuffle(ordered)
    return ordered, stats


def split_by_family(
    examples: list[Example],
    dev_fraction: float = 0.1,
    test_fraction: float = 0.15,
    seed: int = 20260805,
) -> dict[str, list[Example]]:
    """Split train/dev/test **by document family**, never by question.

    Splitting by question would leak: two questions about the same section share the
    same positive chunk, so a model that memorized it in training would be rewarded at
    test time for retrieval it never had to learn. Family-level splitting makes the
    test set genuinely unseen sections.

    Unanswerable examples have no family and are distributed independently -- they
    cannot leak because they have no positive to memorize.
    """
    rng = random.Random(seed)

    by_family: dict[str, list[Example]] = defaultdict(list)
    unattached: list[Example] = []
    for example in examples:
        if example.family_id:
            by_family[example.family_id].append(example)
        else:
            unattached.append(example)

    families = sorted(by_family)
    rng.shuffle(families)
    dev_cut = int(len(families) * dev_fraction)
    test_cut = dev_cut + int(len(families) * test_fraction)

    splits: dict[str, list[Example]] = {"train": [], "dev": [], "test": []}
    for index, family in enumerate(families):
        name = "dev" if index < dev_cut else "test" if index < test_cut else "train"
        splits[name].extend(by_family[family])

    rng.shuffle(unattached)
    dev_cut = int(len(unattached) * dev_fraction)
    test_cut = dev_cut + int(len(unattached) * test_fraction)
    splits["dev"].extend(unattached[:dev_cut])
    splits["test"].extend(unattached[dev_cut:test_cut])
    splits["train"].extend(unattached[test_cut:])

    for split in splits.values():
        rng.shuffle(split)
    return splits
