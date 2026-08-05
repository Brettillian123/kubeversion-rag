"""Assert the architectural claims the docs make.

A design document that drifts from the code is worse than none, because people trust
it. These are the claims in docs/ARCHITECTURE.md that can be checked mechanically.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "kubeversion_rag"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            # Relative imports: ast records level separately, so reconstruct enough to
            # spot a cross-package dependency.
            if node.level:
                names.add("." * node.level + node.module)
    return names


class TestMetricsIndependence:
    def test_metrics_do_not_import_the_retrieval_code(self):
        # docs/ARCHITECTURE.md: "Metrics ... with no dependency on the retrieval code".
        # If they shared code, a bug in the shared part would move the numbers without
        # anyone noticing -- the measuring instrument and the thing measured would fail
        # together and agree with each other.
        imports = imported_modules(SRC / "eval" / "metrics.py")
        offenders = [name for name in imports if "retrieval" in name or "train" in name]
        assert not offenders, f"eval/metrics.py imports {offenders}"


class TestServingIsTorchFree:
    def test_the_api_module_never_imports_torch_or_sentence_transformers(self):
        # The API image installs neither. A module-level import would make the whole
        # module unimportable in that image; a lazy one inside a function would blow up
        # at request time instead, which is worse.
        for module in ("api.py", "store.py", "generate.py"):
            source = (SRC / "serving" / module).read_text(encoding="utf-8")
            assert "import torch" not in source, f"serving/{module} imports torch"
            assert "sentence_transformers" not in source, (
                f"serving/{module} references sentence_transformers"
            )

    def test_the_embedding_service_is_the_only_serving_module_that_loads_a_model(self):
        source = (SRC / "serving" / "embed_service.py").read_text(encoding="utf-8")
        assert "Embedder" in source, "the embedding service should be the one that loads a model"


class TestLazyModelImports:
    def test_heavy_imports_are_deferred_in_the_retrieval_package(self):
        # dense.py and rerank.py are imported by the serving package's dependency
        # graph, so a top-level sentence_transformers import there would drag torch
        # into the API image by transitivity.
        for module in ("dense.py", "rerank.py", "bm25.py"):
            tree = ast.parse((SRC / "retrieval" / module).read_text(encoding="utf-8"))
            for node in tree.body:  # module level only
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    text = ast.unparse(node)
                    assert "sentence_transformers" not in text or "TYPE_CHECKING" in text, (
                        f"retrieval/{module} imports sentence_transformers at module level"
                    )
                    assert "rank_bm25" not in text, (
                        f"retrieval/{module} imports rank_bm25 at module level"
                    )


class TestNoLlmInTheLabellingPath:
    def test_dataset_construction_never_calls_a_model(self):
        # docs/ARCHITECTURE.md: "No LLM-generated training labels ... an LLM-labelled
        # set would make the eval circular." This is the check on that claim.
        for path in (SRC / "dataset").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("anthropic", "openai", "Generator("):
                assert forbidden not in source, (
                    f"dataset/{path.name} references {forbidden!r}; labels must come "
                    "from the corpus structure, not from a model"
                )
