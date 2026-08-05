"""Guards on the hand-written evaluation set.

The failure mode these prevent: gold questions specify their expected answer by
(document, substring), and if a re-chunk stops matching, a lenient resolver would drop
the question and the metric would keep reporting a plausible-looking number computed
over fewer questions than anyone thinks. Silent shrinkage of an eval set is worse than
a crash, so resolution is strict and the CLI exits non-zero on any unresolved question.
"""

import pytest

from kubeversion_rag.dataset.gold import GoldResolutionError, GoldSpec, load_specs, resolve
from kubeversion_rag.models import Chunk, Corpus
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        chunks=[
            Chunk(
                doc_path="content/en/docs/reference/using-api/deprecation-guide.md",
                heading_path=("Guide", "PodSecurityPolicy"),
                text="PodSecurityPolicy in the policy/v1beta1 API version is no longer served as of v1.25.",
                version_low=v("1.25"),
                version_high=v("1.35"),
            ),
            Chunk(
                doc_path="content/en/docs/reference/using-api/deprecation-guide.md",
                heading_path=("Guide", "Flow control"),
                text="The flowcontrol.apiserver.k8s.io/v1beta3 API version is no longer served as of v1.32.",
                version_low=v("1.32"),
                version_high=v("1.35"),
            ),
            Chunk(
                doc_path="content/en/docs/concepts/security/psp.md",
                heading_path=("PSP", "Overview"),
                text="A longer document that also mentions PodSecurityPolicy somewhere in the body text.",
                version_low=v("1.24"),
                version_high=v("1.24"),
            ),
        ]
    )


class TestResolution:
    def test_binds_a_question_to_the_expected_chunk(self, corpus):
        specs = [
            GoldSpec(
                question="Is PSP supported on 1.30?",
                version=v("1.30"),
                expect_doc="deprecation-guide",
                expect_text="PodSecurityPolicy",
            )
        ]
        examples, problems = resolve(corpus, specs)
        assert problems == []
        assert corpus.get(examples[0].positive_chunk_id).doc_path.endswith("deprecation-guide.md")

    def test_only_considers_chunks_covering_the_target_version(self, corpus):
        # The concepts doc mentions PodSecurityPolicy but only exists at 1.24, so a
        # question scoped to 1.30 must not bind to it.
        specs = [
            GoldSpec(
                question="PSP on 1.30?",
                version=v("1.30"),
                expect_text="PodSecurityPolicy",
            )
        ]
        examples, _ = resolve(corpus, specs)
        assert corpus.get(examples[0].positive_chunk_id).covers(v("1.30"))

    def test_an_unresolvable_question_is_reported_not_dropped_silently(self, corpus):
        specs = [
            GoldSpec(
                question="Something about a topic that does not exist",
                version=v("1.30"),
                expect_text="QuantumScheduler",
            )
        ]
        examples, problems = resolve(corpus, specs)
        assert examples == []
        assert len(problems) == 1
        assert "QuantumScheduler" in problems[0]

    def test_unanswerable_questions_bind_with_no_positive(self, corpus):
        specs = [GoldSpec(question="Fake flag?", version=v("1.30"), unanswerable=True)]
        examples, problems = resolve(corpus, specs)
        assert problems == []
        assert examples[0].unanswerable
        assert examples[0].positive_chunk_id == ""

    def test_sibling_blocks_become_hard_negatives(self, corpus):
        # The Flow control block is the near-identical sibling that a version-blind
        # retriever confuses with the PSP one.
        specs = [
            GoldSpec(
                question="PSP on 1.32?",
                version=v("1.32"),
                expect_doc="deprecation-guide",
                expect_text="PodSecurityPolicy",
            )
        ]
        examples, _ = resolve(corpus, specs)
        assert examples[0].hard_negative_ids


class TestSpecLoading:
    def test_loads_a_well_formed_file(self, tmp_path):
        path = tmp_path / "gold.yaml"
        path.write_text(
            "questions:\n"
            "  - question: Does X work on 1.30?\n"
            "    version: '1.30'\n"
            "    expect_text: X\n",
            encoding="utf-8",
        )
        specs = load_specs(path)
        assert len(specs) == 1
        assert specs[0].version == v("1.30")

    def test_a_malformed_entry_is_an_error(self, tmp_path):
        path = tmp_path / "gold.yaml"
        path.write_text("questions:\n  - version: '1.30'\n", encoding="utf-8")
        with pytest.raises(GoldResolutionError, match="missing question or version"):
            load_specs(path)

    def test_an_empty_file_yields_no_specs(self, tmp_path):
        path = tmp_path / "gold.yaml"
        path.write_text("questions: []\n", encoding="utf-8")
        assert load_specs(path) == []


class TestShippedGoldFile:
    """The real file must stay parseable, because CI runs it."""

    def test_ships_and_parses(self):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "data" / "gold" / "gold_questions.yaml"
        specs = load_specs(path)
        assert len(specs) >= 10, "the gold set should not shrink silently"
        assert any(spec.unanswerable for spec in specs), "must include expected refusals"
        assert all(spec.rationale for spec in specs), (
            "every gold question needs a stated rationale, or nobody can tell whether "
            "it is still correct a year from now"
        )
