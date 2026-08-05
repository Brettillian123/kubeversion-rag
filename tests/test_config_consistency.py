"""Cross-file consistency checks.

Every assertion here guards a mismatch whose symptom is *silence*: a collection nobody
wrote to returns an empty result set rather than an error, and an env var the chart
sets but the code never reads simply does nothing. Both look like a working system
right up until someone notices the answers are wrong.
"""

import os
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Config reads the environment; a stray var would make these tests lie."""
    for key in list(os.environ):
        if key.startswith("KVRAG_"):
            monkeypatch.delenv(key, raising=False)


class TestCollectionNaming:
    def test_config_default_matches_the_generated_name(self):
        # The backfill Job names its collection with collection_name(); the API reads
        # the default from config. If they disagree the API queries a collection that
        # was never written, and every answer is an empty refusal with no error anywhere.
        from kubeversion_rag.config import load_config
        from kubeversion_rag.serving.store import collection_name

        config = load_config()
        assert config.serving.default_collection == collection_name(config.retrieval.bi_encoder)

    def test_helm_default_matches_too(self):
        from kubeversion_rag.serving.store import collection_name

        values = yaml.safe_load((REPO / "deploy/helm/kubeversion-rag/values.yaml").read_text())
        assert values["activeCollection"] == collection_name(values["embed"]["model"])

    def test_compose_default_matches_too(self):
        from kubeversion_rag.serving.store import collection_name

        text = (REPO / "deploy/docker-compose.yml").read_text(encoding="utf-8")
        collection = re.search(r"KVRAG_COLLECTION:\s*\$\{KVRAG_COLLECTION:-([^}]+)\}", text)
        model = re.search(r"KVRAG_EMBED_MODEL:\s*\$\{KVRAG_EMBED_MODEL:-([^}]+)\}", text)
        assert collection and model, "compose no longer declares these defaults"
        assert collection.group(1).strip() == collection_name(model.group(1).strip())

    def test_names_are_valid_qdrant_identifiers(self):
        from kubeversion_rag.serving.store import collection_name

        for model in (
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "/models/biencoder",
        ):
            name = collection_name(model)
            assert re.fullmatch(r"[a-z0-9_]+", name), name
            assert not name.endswith("_")

    def test_revision_changes_the_name(self):
        # The migration flow depends on this: two revisions must be able to coexist.
        from kubeversion_rag.serving.store import collection_name

        assert collection_name("m", 1) != collection_name("m", 2)


class TestChartCodeContract:
    def test_every_env_var_the_chart_sets_is_read_by_the_code(self):
        configmap = (REPO / "deploy/helm/kubeversion-rag/templates/configmap.yaml").read_text()
        declared = set(re.findall(r"^  (KVRAG_[A-Z_]+):", configmap, re.MULTILINE))

        sources = "\n".join(
            path.read_text(encoding="utf-8") for path in (REPO / "src").rglob("*.py")
        )
        read = set(re.findall(r"KVRAG_[A-Z_]+", sources))

        orphans = declared - read
        assert not orphans, (
            f"the chart sets {sorted(orphans)} but no code reads them -- either dead "
            "config or a rename that only landed on one side"
        )

    def test_the_chart_never_takes_the_api_key_as_a_value(self):
        # A key passed via --set lands in the Helm release object, shell history, and
        # CI logs. It must come from an existing Secret referenced by name.
        values = (REPO / "deploy/helm/kubeversion-rag/values.yaml").read_text()
        parsed = yaml.safe_load(values)
        assert "apiKey" not in parsed.get("anthropic", {})
        assert "existingSecret" in parsed["anthropic"]


class TestVersionWindow:
    def test_chart_and_code_agree_on_the_corpus_window(self):
        from kubeversion_rag.config import load_config

        values = yaml.safe_load((REPO / "deploy/helm/kubeversion-rag/values.yaml").read_text())
        config = load_config()
        assert values["corpus"]["minVersion"] == str(config.min_version)
        assert values["corpus"]["maxVersion"] == str(config.max_version)
