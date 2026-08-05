from kubeversion_rag.versions import (
    MinorVersion,
    VersionRange,
    coalesce_versions,
    extract_versions,
)


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


class TestMinorVersion:
    def test_parses_common_forms(self):
        assert v("1.31") == MinorVersion(1, 31)
        assert v("v1.31") == MinorVersion(1, 31)
        assert v("1.31.4") == MinorVersion(1, 31), "patch version is dropped, not rejected"

    def test_rejects_nonsense(self):
        assert MinorVersion.try_parse("banana") is None
        assert MinorVersion.try_parse("") is None
        assert MinorVersion.try_parse("1") is None

    def test_orders_numerically_not_lexically(self):
        # The bug this guards: "1.9" > "1.10" as strings.
        assert v("1.9") < v("1.10")
        assert sorted([v("1.34"), v("1.9"), v("1.24")]) == [v("1.9"), v("1.24"), v("1.34")]

    def test_branch_name_matches_the_repo_convention(self):
        assert v("1.31").branch == "release-1.31"


class TestVersionRange:
    def test_contains_is_inclusive_at_both_ends(self):
        span = VersionRange(v("1.24"), v("1.28"))
        assert span.contains(v("1.24"))
        assert span.contains(v("1.28"))
        assert not span.contains(v("1.29"))

    def test_rejects_inverted_range(self):
        import pytest

        with pytest.raises(ValueError):
            VersionRange(v("1.30"), v("1.24"))

    def test_iterates_every_version_in_span(self):
        assert list(VersionRange(v("1.24"), v("1.26"))) == [v("1.24"), v("1.25"), v("1.26")]


class TestCoalesce:
    def test_merges_adjacent_versions(self):
        spans = coalesce_versions([v("1.24"), v("1.25"), v("1.26")])
        assert spans == [VersionRange(v("1.24"), v("1.26"))]

    def test_splits_on_a_gap(self):
        # A section that changed at 1.26 and changed back at 1.28 must become two
        # ranges, not one span that wrongly claims to cover 1.26-1.27.
        spans = coalesce_versions([v("1.24"), v("1.25"), v("1.28")])
        assert spans == [
            VersionRange(v("1.24"), v("1.25")),
            VersionRange(v("1.28"), v("1.28")),
        ]

    def test_is_order_and_duplicate_insensitive(self):
        assert coalesce_versions([v("1.26"), v("1.24"), v("1.25"), v("1.24")]) == [
            VersionRange(v("1.24"), v("1.26"))
        ]

    def test_empty_input(self):
        assert coalesce_versions([]) == []


class TestExtractVersions:
    def test_finds_versions_in_prose(self):
        assert extract_versions("upgrading from 1.28 to v1.31") == [v("1.28"), v("1.31")]

    def test_rejects_non_kubernetes_majors(self):
        # Helm chart versions, image tags, and similar decimals must not be mistaken
        # for a cluster version.
        assert extract_versions("chart 2.5 and app 3.1") == []

    def test_deduplicates_preserving_first_seen_order(self):
        assert extract_versions("1.31, then 1.28, then 1.31 again") == [v("1.31"), v("1.28")]

    def test_does_not_match_inside_longer_numbers(self):
        assert extract_versions("build 11.245.3") == []
