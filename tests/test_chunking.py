from kubeversion_rag.config import ChunkingConfig
from kubeversion_rag.ingest.chunk import (
    chunk_document,
    coalesce_corpus,
    normalize,
    split_sections,
    strip_frontmatter,
)
from kubeversion_rag.versions import MinorVersion

CONFIG = ChunkingConfig(target_chars=600, max_chars=900, min_chars=40)


def v(text: str) -> MinorVersion:
    return MinorVersion.parse(text)


class TestFrontmatter:
    def test_extracts_title_and_strips_block(self):
        body, title = strip_frontmatter("---\ntitle: Pod Security\nweight: 10\n---\n\nHello.")
        assert title == "Pod Security"
        assert body.strip() == "Hello."

    def test_handles_quoted_titles(self):
        _, title = strip_frontmatter('---\ntitle: "Deprecated API Migration Guide"\n---\nx')
        assert title == "Deprecated API Migration Guide"

    def test_document_without_frontmatter_is_untouched(self):
        body, title = strip_frontmatter("# Heading\n\ntext")
        assert title is None
        assert body == "# Heading\n\ntext"


class TestNormalize:
    def test_strips_hugo_shortcodes_but_keeps_inner_text(self):
        result = normalize("{{< note >}}\nThis matters.\n{{< /note >}}")
        assert "This matters." in result
        assert "{{<" not in result

    def test_removes_html_comments(self):
        assert "body" not in normalize("<!-- body -->").lower()

    def test_collapses_whitespace_deterministically(self):
        # Two branches differing only in trailing whitespace must hash identically, or
        # coalescing silently fails and the index fills with near-duplicates.
        assert normalize("a   \n\n\n\nb") == normalize("a\n\nb")


class TestSectionSplitting:
    def test_builds_a_heading_breadcrumb(self):
        sections = split_sections("## Parent\n\nintro\n\n### Child\n\ndetail", "Doc")
        paths = [section.heading_path for section in sections]
        assert ("Doc", "Parent") in paths
        assert ("Doc", "Parent", "Child") in paths

    def test_hash_inside_a_code_fence_is_not_a_heading(self):
        # The failure this prevents: a "# comment" in a shell example becomes a
        # spurious heading, shattering the family on branches where the example exists.
        markdown = "## Real\n\n```bash\n# not a heading\nkubectl get pods\n```\n\nafter"
        paths = [section.heading_path for section in split_sections(markdown, "Doc")]
        assert ("Doc", "Real") in paths
        assert not any("not a heading" in " ".join(path) for path in paths)

    def test_explicit_anchors_are_stripped_from_headings(self):
        # Anchors in this corpus embed the release number
        # ({#flowcontrol-resources-v132}); leaving them in makes every heading
        # branch-specific and breaks coalescing outright.
        sections = split_sections("#### Flow control {#flowcontrol-resources-v132}\n\nbody", "Doc")
        assert any(section.heading_path[-1] == "Flow control" for section in sections)

    def test_deeper_heading_resets_when_returning_to_a_shallower_level(self):
        markdown = "## A\n\nx\n\n### A1\n\ny\n\n## B\n\nz"
        paths = [section.heading_path for section in split_sections(markdown, "Doc")]
        assert ("Doc", "B") in paths
        assert ("Doc", "B", "A1") not in paths


class TestChunkDocument:
    def test_short_sections_are_dropped(self):
        chunks = chunk_document("doc.md", "---\ntitle: T\n---\n\n## Tiny\n\nx", v("1.30"), CONFIG)
        assert chunks == []

    def test_repeated_heading_paths_get_distinct_parts(self):
        # Without distinct parts these collapse into one family, every one of them
        # claims the same (family, version) slot, and the disjointness invariant that
        # hard-negative mining relies on is violated.
        body = "\n\n".join(["## Notes\n\n" + "detail " * 40] * 2)
        chunks = chunk_document("doc.md", f"---\ntitle: T\n---\n\n{body}", v("1.30"), CONFIG)
        assert len(chunks) >= 2
        assert len({chunk.family_id for chunk in chunks}) == len(chunks)

    def test_long_section_splits_without_breaking_a_code_fence(self):
        fence = "```yaml\n" + "key: value\n" * 60 + "```"
        markdown = f"---\ntitle: T\n---\n\n## Big\n\n{'prose ' * 200}\n\n{fence}"
        chunks = chunk_document("doc.md", markdown, v("1.30"), CONFIG)
        for chunk in chunks:
            assert chunk.text.count("```") % 2 == 0, "a code fence was split in half"


class TestCoalescing:
    def _chunks(self, texts_by_version):
        from kubeversion_rag.models import Chunk

        return [
            Chunk(
                doc_path="doc.md",
                heading_path=("Doc", "Section"),
                text=text,
                version_low=v(version),
                version_high=v(version),
            )
            for version, text in texts_by_version
        ]

    def test_identical_text_across_branches_collapses_to_one_chunk(self):
        corpus = coalesce_corpus(
            self._chunks([("1.24", "same"), ("1.25", "same"), ("1.26", "same")])
        )
        assert len(corpus) == 1
        assert str(corpus.chunks[0].version_range) == "1.24-1.26"

    def test_changed_text_produces_disjoint_adjacent_ranges(self):
        corpus = coalesce_corpus(
            self._chunks([("1.24", "old"), ("1.25", "old"), ("1.26", "new"), ("1.27", "new")])
        )
        assert len(corpus) == 2
        family = corpus.family(corpus.chunks[0].family_id)
        assert len(family) == 2
        earlier, later = family
        assert earlier.version_high < later.version_low, "ranges must not overlap"

    def test_covering_returns_the_right_snapshot_for_each_version(self):
        corpus = coalesce_corpus(self._chunks([("1.24", "old"), ("1.25", "old"), ("1.26", "new")]))
        family_id = corpus.chunks[0].family_id
        assert corpus.covering(family_id, v("1.24")).text == "old"
        assert corpus.covering(family_id, v("1.26")).text == "new"
        assert corpus.covering(family_id, v("1.31")) is None

    def test_chunk_ids_are_stable_across_runs(self):
        first = coalesce_corpus(self._chunks([("1.24", "same")]))
        second = coalesce_corpus(self._chunks([("1.24", "same")]))
        assert first.chunks[0].chunk_id == second.chunks[0].chunk_id
