from kubeversion_rag.ingest.deprecation import parse_deprecation_guide
from kubeversion_rag.versions import MinorVersion

# Both sentence forms the real guide uses, plus a multi-resource block.
GUIDE = """\
---
title: "Deprecated API Migration Guide"
---

## Removed APIs by release

### v1.32

#### Flow control resources {#flowcontrol-resources-v132}

The **flowcontrol.apiserver.k8s.io/v1beta3** API version of FlowSchema and
PriorityLevelConfiguration is no longer served as of v1.32.

* Migrate manifests and API clients to use the **flowcontrol.apiserver.k8s.io/v1** API
  version, available since v1.29.
* All existing persisted objects are accessible via the new API

### v1.25

#### PodSecurityPolicy {#psp-v125}

PodSecurityPolicy in the **policy/v1beta1** API version is no longer served as of v1.25,
and the PodSecurityPolicy admission controller will be removed.

Migrate to [Pod Security Admission](/docs/concepts/security/pod-security-admission/).

#### RuntimeClass {#runtimeclass-v125}

RuntimeClass in the **node.k8s.io/v1beta1** API version is no longer served as of v1.25.

* Migrate manifests and API clients to use the **node.k8s.io/v1** API version,
  available since v1.20.
"""


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


class TestParseDeprecationGuide:
    def test_parses_every_block(self):
        facts, report = parse_deprecation_guide(GUIDE)
        assert report.blocks_seen == 3
        assert report.facts_parsed == 3
        assert report.coverage == 1.0
        assert len(facts) == 3

    def test_standard_sentence_form_with_multiple_resources(self):
        facts, _ = parse_deprecation_guide(GUIDE)
        flow = next(
            f for f in facts if f.api_group_version == "flowcontrol.apiserver.k8s.io/v1beta3"
        )
        assert flow.removed_in == v("1.32")
        assert set(flow.resources) == {"FlowSchema", "PriorityLevelConfiguration"}
        assert flow.replacement_group_version == "flowcontrol.apiserver.k8s.io/v1"
        assert flow.replacement_since == v("1.29")

    def test_inverted_sentence_form(self):
        # "PodSecurityPolicy in the **policy/v1beta1** API version is no longer
        # served..." -- the form that covers the corpus's single most consequential
        # version-sensitive fact. An earlier parser missed it entirely.
        facts, _ = parse_deprecation_guide(GUIDE)
        psp = next(f for f in facts if "PodSecurityPolicy" in f.resources)
        assert psp.removed_in == v("1.25")
        assert psp.api_group_version == "policy/v1beta1"

    def test_a_removal_with_no_api_replacement_parses_with_none(self):
        # PSP was replaced by an admission controller, not another API version.
        # Guessing a replacement here would put a false fact into training data.
        facts, _ = parse_deprecation_guide(GUIDE)
        psp = next(f for f in facts if "PodSecurityPolicy" in f.resources)
        assert psp.replacement_group_version is None
        assert psp.replacement_since is None

    def test_the_removal_sentence_wins_over_the_heading_grouping(self):
        facts, _ = parse_deprecation_guide(GUIDE)
        assert all(f.removed_in in {v("1.32"), v("1.25")} for f in facts)

    def test_unparseable_blocks_are_reported_not_guessed(self):
        broken = GUIDE + "\n### v1.99\n\n#### Mystery\n\nSomething happened.\n"
        facts, report = parse_deprecation_guide(broken)
        assert report.blocks_seen == 4
        assert report.facts_parsed == 3
        assert any("Mystery" in note for note in report.blocks_skipped)

    def test_empty_document_is_not_an_error(self):
        facts, report = parse_deprecation_guide("# Nothing here")
        assert facts == []
        assert report.blocks_seen == 0

    def test_fact_ids_are_stable_and_unique(self):
        facts, _ = parse_deprecation_guide(GUIDE)
        ids = [fact.fact_id for fact in facts]
        assert len(ids) == len(set(ids))
        again, _ = parse_deprecation_guide(GUIDE)
        assert [f.fact_id for f in again] == ids
