import math

from kubeversion_rag.eval.metrics import (
    EvalResult,
    QueryResult,
    aggregate,
    hit_at_k,
    markdown_table,
    ndcg_at_k,
    reciprocal_rank,
    version_correct_at_1,
    version_correct_at_k,
)
from kubeversion_rag.models import Chunk
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


def chunk(low: str, high: str) -> Chunk:
    return Chunk("d.md", ("Doc",), "text", v(low), v(high))


class TestRankMetrics:
    def test_hit_at_k_respects_the_cutoff(self):
        ranked = ["a", "b", "c"]
        assert hit_at_k(ranked, {"c"}, 3) == 1.0
        assert hit_at_k(ranked, {"c"}, 2) == 0.0

    def test_reciprocal_rank_is_one_over_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
        assert reciprocal_rank(["a", "b"], {"z"}) == 0.0

    def test_ndcg_discounts_lower_positions(self):
        assert ndcg_at_k(["a", "b"], {"a"}) == 1.0
        assert math.isclose(ndcg_at_k(["b", "a"], {"a"}), 1 / math.log2(3))

    def test_ndcg_beyond_the_cutoff_is_zero(self):
        assert ndcg_at_k([f"x{i}" for i in range(20)] + ["a"], {"a"}, k=10) == 0.0

    def test_metrics_on_an_empty_ranking(self):
        assert hit_at_k([], {"a"}, 10) == 0.0
        assert reciprocal_rank([], {"a"}) == 0.0
        assert ndcg_at_k([], {"a"}) == 0.0


class TestVersionCorrectness:
    def test_top_1_covering_the_target_scores_one(self):
        assert version_correct_at_1([chunk("1.28", "1.31")], v("1.30")) == 1.0

    def test_top_1_from_the_wrong_release_scores_zero(self):
        # The exact failure this project exists to fix: a plausible, real, cited
        # document that does not apply to the asked-about version.
        assert version_correct_at_1([chunk("1.24", "1.26")], v("1.30")) == 0.0

    def test_empty_results_score_zero_not_an_error(self):
        assert version_correct_at_1([], v("1.30")) == 0.0

    def test_at_k_is_the_fraction_of_the_window_that_applies(self):
        chunks = [chunk("1.28", "1.31"), chunk("1.24", "1.26"), chunk("1.30", "1.30")]
        assert version_correct_at_k(chunks, v("1.30"), 3) == 2 / 3

    def test_at_k_uses_the_actual_window_when_fewer_results_exist(self):
        assert version_correct_at_k([chunk("1.28", "1.31")], v("1.30"), 5) == 1.0


class TestAggregate:
    def _results(self):
        return [
            QueryResult("q1", v("1.30"), "a", ["a", "b"], [chunk("1.28", "1.31")], "deprecation"),
            QueryResult(
                "q2", v("1.30"), "z", ["a", "b"], [chunk("1.24", "1.26")], "changed-section"
            ),
        ]

    def test_averages_across_queries(self):
        result = aggregate("cfg", "desc", self._results())
        assert result.n_queries == 2
        assert result.metrics["recall@10"] == 0.5
        assert result.metrics["version_correct@1"] == 0.5

    def test_breaks_out_by_question_source(self):
        result = aggregate("cfg", "desc", self._results())
        assert set(result.by_source) == {"deprecation", "changed-section"}
        assert result.by_source["deprecation"]["recall@10"] == 1.0
        assert result.by_source["changed-section"]["recall@10"] == 0.0

    def test_no_queries_does_not_divide_by_zero(self):
        assert aggregate("cfg", "desc", []).n_queries == 0


class TestMarkdownTable:
    def test_bolds_the_winner_per_column(self):
        rows = [
            EvalResult("a", "Config A", 10, {"ndcg@10": 0.5}),
            EvalResult("b", "Config B", 10, {"ndcg@10": 0.9}),
        ]
        table = markdown_table(rows, columns=["ndcg@10"])
        assert "**0.900**" in table
        assert "| 0.500 |" in table

    def test_flags_a_row_that_silently_fell_back_to_the_base_model(self):
        # Without this marker a results table generated before training looks exactly
        # like one generated after it -- a fabricated result.
        rows = [EvalResult("ft", "Fine-tuned", 10, {"ndcg@10": 0.5}, degraded=["fell back"])]
        assert "not actually fine-tuned" in markdown_table(rows, columns=["ndcg@10"])


class TestRoundTrip:
    """Stored results have to survive the trip back, so tables can be re-rendered.

    Some ablation rows cost half an hour of GPU time. Re-running them because the
    markdown needed a column is the kind of waste that quietly discourages fixing the
    markdown at all.
    """

    def test_eval_result_survives_a_round_trip(self):
        original = EvalResult(
            "dense_ft_filtered",
            "Fine-tuned bi-encoder + version filter",
            1811,
            {"ndcg@10": 0.774, "recall@1": 0.593},
            by_source={"deprecation": {"ndcg@10": 0.744}},
            degraded=["cross-encoder fell back"],
        )
        restored = EvalResult.from_dict(original.to_dict())
        assert restored == original

    def test_eval_run_survives_a_round_trip(self, tmp_path):
        import json

        from kubeversion_rag.eval.run import EvalRun

        run = EvalRun(
            results=[EvalResult("bm25", "BM25", 10, {"ndcg@10": 0.351})],
            question_set="test",
            n_chunks=23018,
            generated_at="2026-08-05T18:20:59+00:00",
        )
        path = tmp_path / "ablation__test.json"
        path.write_text(json.dumps(run.to_dict()), encoding="utf-8")
        assert EvalRun.load(path) == run


class TestGeneratedDocsStayOutOfTheHandWrittenOne:
    def test_the_writer_never_targets_docs_results_md(self):
        # docs/RESULTS.md carries the analysis of *why* the numbers came out this way,
        # which no generator can reproduce. Pointing the generator at it meant running
        # `eval run --split gold --write-results` silently replaced the test-split
        # table and every word of prose around it.
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "src" / "kubeversion_rag" / "cli.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        assert "RESULTS.md" not in literals, (
            "cli.py names RESULTS.md as an output path again; generated tables belong "
            "under docs/results/ so the hand-written analysis survives an eval run"
        )
