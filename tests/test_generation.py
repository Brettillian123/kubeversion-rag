from kubeversion_rag.models import Chunk, RetrievedChunk
from kubeversion_rag.retrieval.query import ResolvedVersion, VersionSource
from kubeversion_rag.serving.generate import Generator, build_context_block
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


def hit(low: str, high: str, text: str = "body text", score: float = 5.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            doc_path="content/en/docs/a.md",
            heading_path=("Doc", "Section"),
            text=text,
            version_low=v(low),
            version_high=v(high),
        ),
        score=score,
    )


class TestContextBlocks:
    def test_numbers_blocks_from_one_to_match_citation_markers(self):
        context, citations = build_context_block([hit("1.28", "1.31"), hit("1.24", "1.27")])
        assert "[1]" in context and "[2]" in context
        assert [c.marker for c in citations] == [1, 2]

    def test_each_block_states_the_versions_it_applies_to(self):
        # The model can only warn about a version mismatch if it can see one.
        context, _ = build_context_block([hit("1.28", "1.31")])
        assert "1.28-1.31" in context

    def test_citations_carry_a_resolvable_source_url(self):
        _, citations = build_context_block([hit("1.28", "1.31")])
        assert citations[0].url.startswith("https://github.com/kubernetes/website/blob/")

    def test_no_hits_yields_empty_context(self):
        context, citations = build_context_block([])
        assert context == ""
        assert citations == []


class TestCitationValidation:
    def test_accepts_markers_that_exist(self):
        used, invalid = Generator._validate_citations("PSP was removed in 1.25 [1].", 3)
        assert used == {1}
        assert invalid == set()

    def test_flags_markers_the_model_invented(self):
        # "Cite your sources" is followed most of the time; the residue is
        # indistinguishable from a correct answer to anyone who does not check.
        used, invalid = Generator._validate_citations("As shown [1] and [7].", 3)
        assert used == {1}
        assert invalid == {7}

    def test_zero_is_not_a_valid_marker(self):
        used, invalid = Generator._validate_citations("see [0]", 3)
        assert used == set()
        assert invalid == {0}

    def test_an_uncited_answer_is_detectable(self):
        used, _ = Generator._validate_citations("PodSecurityPolicy was removed.", 3)
        assert used == set()


class TestRefusal:
    def _generator(self) -> Generator:
        return Generator(min_score=0.0)

    def test_refuses_when_nothing_was_retrieved(self):
        resolved = ResolvedVersion(v("1.30"), VersionSource.EXPLICIT)
        answer = self._generator().answer("anything?", [], resolved)
        assert answer.refused
        assert answer.refusal_reason == "no_results"

    def test_refuses_when_the_best_hit_is_below_the_score_floor(self):
        resolved = ResolvedVersion(v("1.30"), VersionSource.EXPLICIT)
        answer = self._generator().answer("anything?", [hit("1.28", "1.31", score=-9.0)], resolved)
        assert answer.refused
        assert answer.refusal_reason == "low_confidence"

    def test_a_refusal_still_discloses_the_version_it_was_scoped_to(self):
        resolved = ResolvedVersion(v("1.30"), VersionSource.DEFAULTED)
        answer = self._generator().answer("anything?", [], resolved)
        assert "1.30" in answer.text

    def test_refusal_does_not_call_the_model(self):
        # A Generator with no API key would raise on client construction. Reaching a
        # refusal without raising proves the short-circuit happens first, which is what
        # keeps an unanswerable question from costing a model call.
        generator = Generator(api_key="", min_score=0.0)
        resolved = ResolvedVersion(v("1.30"), VersionSource.EXPLICIT)
        assert generator.answer("q", [], resolved).refused
