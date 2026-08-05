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


class TestScoreScaleGuard:
    """The refusal floor and the scores it is compared against must be on one scale.

    They were not: `min_score` was -4.0, a cross-encoder logit, while the serving path
    passes cosine similarities in [0, 1]. `best < -4.0` is never true, so the
    low-confidence refusal was unreachable. Nothing about that is visible in operation --
    a gate that never fires looks exactly like a gate that is merely generous.
    """

    def _hits(self, score: float):
        from kubeversion_rag.models import Chunk, RetrievedChunk
        from kubeversion_rag.versions import MinorVersion

        chunk = Chunk(
            "a.md", ("A",), "text", MinorVersion.parse("1.24"), MinorVersion.parse("1.31")
        )
        return [RetrievedChunk(chunk=chunk, score=score)]

    def test_a_crossencoder_floor_against_cosine_scores_is_reported(self, caplog):
        import logging

        generator = Generator(api_key="", min_score=-4.0)
        with caplog.at_level(logging.ERROR):
            generator._warn_once_if_scale_looks_wrong(0.72)
        assert "never fire" in caplog.text

    def test_a_cosine_floor_against_logit_scores_is_reported(self, caplog):
        import logging

        generator = Generator(api_key="", min_score=0.35)
        with caplog.at_level(logging.ERROR):
            generator._warn_once_if_scale_looks_wrong(9.4)
        assert "wider scale" in caplog.text

    def test_a_matched_scale_is_silent(self, caplog):
        import logging

        generator = Generator(api_key="", min_score=0.35)
        with caplog.at_level(logging.ERROR):
            generator._warn_once_if_scale_looks_wrong(0.72)
        assert caplog.text == ""

    def test_it_warns_once_not_per_request(self, caplog):
        import logging

        generator = Generator(api_key="", min_score=-4.0)
        with caplog.at_level(logging.ERROR):
            for _ in range(5):
                generator._warn_once_if_scale_looks_wrong(0.72)
        assert caplog.text.count("never fire") == 1

    def test_the_shipped_default_is_on_the_cosine_scale(self):
        # The serving path does not rerank, so the default has to be a cosine floor.
        # If a reranking stage is ever added to serving, this is the assertion that
        # should be updated deliberately rather than the constant edited quietly.
        from kubeversion_rag.config import load_config

        assert 0.0 <= load_config().serving.min_context_score <= 1.0

    def test_the_floor_sits_below_the_lowest_answerable_gold_question(self):
        # Measured: answerable gold questions score 0.563-0.818, expected-refusal ones
        # 0.532-0.592. They overlap, so no cosine floor separates them. Given that, the
        # floor must sit below the answerable minimum -- a floor inside the overlap
        # buys a few refusals by silently suppressing real answers.
        from kubeversion_rag.config import load_config

        assert load_config().serving.min_context_score < 0.563
