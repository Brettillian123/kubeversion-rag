import pytest

from kubeversion_rag.models import DeprecationFact
from kubeversion_rag.retrieval.query import VersionResolver, VersionSource
from kubeversion_rag.versions import MinorVersion


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


@pytest.fixture
def resolver() -> VersionResolver:
    facts = [
        DeprecationFact(
            removed_in=v("1.25"),
            api_group_version="policy/v1beta1",
            resources=("PodSecurityPolicy",),
            replacement_group_version=None,
            replacement_since=None,
            source_doc="guide.md",
            source_heading=("Guide",),
        ),
        DeprecationFact(
            removed_in=v("1.32"),
            api_group_version="flowcontrol.apiserver.k8s.io/v1beta3",
            resources=("FlowSchema",),
            replacement_group_version="flowcontrol.apiserver.k8s.io/v1",
            replacement_since=v("1.29"),
            source_doc="guide.md",
            source_heading=("Guide",),
        ),
    ]
    return VersionResolver(facts, v("1.24"), v("1.35"))


class TestExplicit:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("I'm on Kubernetes 1.28, can I use PSP?", "1.28"),
            ("does this work on v1.31?", "1.31"),
            ("upgrading my cluster to 1.30.4", "1.30"),
            ("k8s 1.27 question", "1.27"),
        ],
    )
    def test_finds_a_stated_version(self, resolver, query, expected):
        resolved = resolver.resolve(query)
        assert resolved.version == v(expected)
        assert resolved.source is VersionSource.EXPLICIT

    def test_explicit_beats_inference(self, resolver):
        # The query mentions policy/v1beta1 (which would infer <=1.24) but also states
        # 1.30. What the user typed wins.
        resolved = resolver.resolve("On 1.30, what replaced policy/v1beta1?")
        assert resolved.version == v("1.30")
        assert resolved.source is VersionSource.EXPLICIT

    def test_clamps_to_the_ingested_window(self, resolver):
        # A corpus starting at 1.24 cannot answer for 1.19. Clamping and disclosing
        # beats silently returning nothing.
        assert resolver.resolve("on kubernetes 1.19").version == v("1.24")
        assert resolver.resolve("on kubernetes 1.99").version == v("1.35")


class TestInferred:
    def test_infers_an_upper_bound_from_a_removed_api(self, resolver):
        resolved = resolver.resolve("How do I fix my policy/v1beta1 PodSecurityPolicy?")
        assert resolved.source is VersionSource.INFERRED
        assert resolved.version == v("1.24"), "removed in 1.25, so the cluster is at most 1.24"

    def test_takes_the_tightest_bound_when_several_apply(self, resolver):
        resolved = resolver.resolve(
            "we use policy/v1beta1 and flowcontrol.apiserver.k8s.io/v1beta3"
        )
        assert resolved.version == v("1.24")

    def test_the_inference_is_disclosed_to_the_user(self, resolver):
        # An inference the reader cannot see is an inference they cannot correct.
        resolved = resolver.resolve("my policy/v1beta1 manifest fails")
        assert "policy/v1beta1" in resolved.disclosure()


class TestDefaulted:
    def test_falls_back_to_the_newest_indexed_version(self, resolver):
        resolved = resolver.resolve("How do I create a Deployment?")
        assert resolved.version == v("1.35")
        assert resolved.source is VersionSource.DEFAULTED
        assert not resolved.is_confident

    def test_the_default_is_stated_rather_than_assumed_silently(self, resolver):
        # Silently assuming latest is exactly how a version-aware system becomes a
        # version-blind one that looks like it is working.
        disclosure = resolver.resolve("How do I create a Deployment?").disclosure()
        assert "1.35" in disclosure
        assert "no version given" in disclosure.lower()


class TestFalsePositives:
    @pytest.mark.parametrize(
        "query",
        [
            "scale my deployment to 1.5x capacity",
            "my container uses nginx 1.21 as a base image",
        ],
    )
    def test_does_not_treat_unrelated_decimals_as_cluster_versions(self, resolver, query):
        resolved = resolver.resolve(query)
        # Either it defaults (best) or it explicitly found something, but it must not
        # silently answer for a version the user never mentioned as their cluster.
        if resolved.source is VersionSource.EXPLICIT:
            assert resolved.version != v("1.5")
