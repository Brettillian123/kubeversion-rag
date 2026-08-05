"""Guards on what makes the ablation table mean anything.

A results table is only evidence if each row differs from the one above it by exactly
one thing. These assert the properties that keep that true — the ones that are easy to
break with an innocuous-looking change and impossible to notice afterwards, because a
confounded table looks exactly like a clean one.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "kubeversion_rag"


class TestSequenceLengthIsPinned:
    def test_every_embedder_construction_passes_max_seq_length(self):
        # A fine-tuned model saves whatever length it trained at. If the base-model
        # rows then embed at the model default, those rows differ from the fine-tuned
        # rows by *two* things, and the delta stops being attributable to fine-tuning.
        # Serving matters for a different reason: query and passage vectors computed at
        # different lengths land in different regions of the space.
        offenders = []
        for path in list(SRC.rglob("*.py")) + list((SRC.parents[1] / "scripts").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name != "Embedder":
                    continue
                kwargs = {kw.arg for kw in node.keywords}
                if "max_seq_length" not in kwargs:
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            f"Embedder built without max_seq_length at {offenders}. Every construction "
            "must pin it or the ablation rows are not comparable."
        )

    def test_the_cache_fingerprint_includes_sequence_length(self):
        # The same model at two lengths produces different vectors. Without this in the
        # fingerprint, one ablation run could silently mix them.
        source = (SRC / "retrieval" / "dense.py").read_text(encoding="utf-8")
        fingerprint = source.split("def _cache_fingerprint")[1].split("\ndef ")[0]
        assert "max_seq_length" in fingerprint

    def test_config_exposes_a_single_source_of_truth(self):
        from kubeversion_rag.config import load_config

        assert load_config().retrieval.max_seq_length > 0


class TestLadderShape:
    def test_the_dense_ladder_adds_one_component_at_a_time(self):
        from kubeversion_rag.retrieval.pipeline import STANDARD_CONFIGS

        def flags(config):
            return {
                "dense": config.use_dense,
                "bm25": config.use_bm25,
                "filter": config.version_filter,
                "rerank": config.rerank,
                "ft_bi": config.bi_encoder_path is not None,
                "ft_ce": config.cross_encoder_path is not None,
            }

        # The BM25 row is excluded deliberately. It is a *floor* -- a different
        # retrieval family, present to show what no learned model achieves -- not a
        # rung on the incremental ladder. Its transition to the first dense row swaps
        # two flags at once, and pretending otherwise would either weaken this
        # assertion or misdescribe the table.
        ladder = [config for config in STANDARD_CONFIGS if config.use_dense]
        assert len(ladder) >= 2, "the ladder needs at least two dense rows to compare"

        for earlier, later in zip(ladder, ladder[1:], strict=False):
            a, b = flags(earlier), flags(later)
            changed = [key for key in a if a[key] != b[key]]
            # Rerank and its fine-tuned model arrive together; one conceptual
            # component expressed as two flags.
            if set(changed) == {"rerank", "ft_ce"}:
                continue
            assert len(changed) <= 1, (
                f"{earlier.name} -> {later.name} changes {changed}; the delta cannot be "
                "attributed to any single component"
            )

    def test_the_table_includes_a_no_learned_model_floor(self):
        from kubeversion_rag.retrieval.pipeline import STANDARD_CONFIGS

        floors = [
            config for config in STANDARD_CONFIGS if not config.use_dense and not config.rerank
        ]
        assert floors, (
            "the table needs a row with no learned model at all. Without one there is "
            "nothing to establish that the modelling work paid for itself."
        )

    def test_config_names_are_unique(self):
        from kubeversion_rag.retrieval.pipeline import STANDARD_CONFIGS

        names = [config.name for config in STANDARD_CONFIGS]
        assert len(names) == len(set(names))
