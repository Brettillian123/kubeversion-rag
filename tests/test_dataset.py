import random

import pytest

from kubeversion_rag.dataset.build import (
    BuildStats,
    build_changed_section_examples,
    build_unanswerable_examples,
    split_by_family,
)
from kubeversion_rag.models import Chunk, Corpus, Example
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


def chunk(section: str, low: str, high: str, text: str, part: int = 0) -> Chunk:
    return Chunk(
        doc_path="docs/a.md",
        heading_path=("Doc", section),
        text=text,
        version_low=v(low),
        version_high=v(high),
        part=part,
    )


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        chunks=[
            # A section that changed at 1.28: two snapshots, disjoint ranges.
            chunk("Changed", "1.24", "1.27", "old guidance " * 20),
            chunk("Changed", "1.28", "1.31", "new guidance " * 20),
            # A section that never changed: no version-sensitive question is possible.
            chunk("Static", "1.24", "1.31", "unchanging " * 20),
        ]
    )


class TestChangedSectionExamples:
    def test_only_changed_sections_produce_examples(self, corpus):
        stats = BuildStats()
        examples = build_changed_section_examples(corpus, random.Random(1), stats)
        families = {example.family_id for example in examples}
        static_family = corpus.chunks[2].family_id
        assert static_family not in families, "an unchanged section cannot be version-sensitive"
        assert families

    def test_hard_negatives_are_the_same_section_at_a_different_version(self, corpus):
        stats = BuildStats()
        examples = build_changed_section_examples(corpus, random.Random(1), stats)
        for example in examples:
            positive = corpus.get(example.positive_chunk_id)
            assert example.hard_negative_ids, "an example with no hard negative teaches nothing"
            for negative_id in example.hard_negative_ids:
                negative = corpus.get(negative_id)
                assert negative.family_id == positive.family_id, "same section"
                assert not negative.covers(example.target_version), "wrong version"

    def test_target_version_is_covered_by_the_positive(self, corpus):
        stats = BuildStats()
        for example in build_changed_section_examples(corpus, random.Random(7), stats):
            assert corpus.get(example.positive_chunk_id).covers(example.target_version)

    def test_generation_is_deterministic_for_a_seed(self, corpus):
        first = build_changed_section_examples(corpus, random.Random(42), BuildStats())
        second = build_changed_section_examples(corpus, random.Random(42), BuildStats())
        assert [e.question for e in first] == [e.question for e in second]


class TestUnanswerable:
    def test_carries_no_positive_and_is_flagged(self):
        stats = BuildStats()
        examples = build_unanswerable_examples([v("1.30")], random.Random(3), stats, count=5)
        assert examples
        for example in examples:
            assert example.unanswerable
            assert example.positive_chunk_id == ""

    def test_does_not_emit_duplicates(self):
        stats = BuildStats()
        examples = build_unanswerable_examples([v("1.30")], random.Random(3), stats, count=200)
        questions = [e.question for e in examples]
        assert len(questions) == len(set(questions))


class TestSplits:
    def _examples(self, n_families: int = 40) -> list[Example]:
        return [
            Example(
                question=f"q{family}-{index}",
                target_version=v("1.30"),
                positive_chunk_id=f"c{family}",
                family_id=f"fam{family}",
            )
            for family in range(n_families)
            for index in range(3)
        ]

    def test_a_family_never_spans_two_splits(self):
        # The leakage this guards is the most likely way for this project to report a
        # dishonest headline: two questions about one section share a positive chunk,
        # so a model that memorized it in training gets free credit at test time.
        splits = split_by_family(self._examples())
        placement: dict[str, str] = {}
        for name, split in splits.items():
            for example in split:
                assert placement.setdefault(example.family_id, name) == name

    def test_every_example_lands_in_exactly_one_split(self):
        examples = self._examples()
        splits = split_by_family(examples)
        total = sum(len(split) for split in splits.values())
        assert total == len(examples)

    def test_all_three_splits_are_populated(self):
        splits = split_by_family(self._examples(60))
        assert all(splits[name] for name in ("train", "dev", "test"))

    def test_unanswerable_examples_are_distributed_not_dropped(self):
        examples = self._examples(20) + [
            Example(f"u{i}", v("1.30"), "", family_id="", unanswerable=True) for i in range(20)
        ]
        splits = split_by_family(examples)
        found = sum(1 for split in splits.values() for e in split if e.unanswerable)
        assert found == 20
