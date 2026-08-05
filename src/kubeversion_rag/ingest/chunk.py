"""Turn Hugo-flavoured markdown into version-tagged, heading-aware chunks.

Two properties matter more than chunk size here:

1. **Chunks must align to document sections.** The version filter and the hard-negative
   construction both key on ``family_id = (doc_path, heading_path)``. Fixed-width
   windows would put the same family boundary in different places on different release
   branches, and the "same section, different version" relationship would dissolve.

2. **Chunk text must be stable across branches when the content did not change.**
   Anything that varies incidentally between branches -- shortcode arguments carrying
   the release number, trailing whitespace -- would defeat coalescing and inflate the
   index with near-duplicates. Normalization is therefore aggressive and deterministic.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..config import ChunkingConfig
from ..models import Chunk, Corpus
from ..versions import MinorVersion, coalesce_versions

log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_ANCHOR_RE = re.compile(r"\s*\{#[^}]*\}\s*$")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Hugo shortcodes. Self-closing and paired forms both appear; the paired form's inner
# text is real content and must survive, so only the delimiters are removed.
_SHORTCODE_RE = re.compile(r"\{\{[<%]\s*/?\s*.*?\s*[>%]\}\}", re.DOTALL)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def strip_frontmatter(text: str) -> tuple[str, str | None]:
    """Split YAML frontmatter off the body, returning ``(body, title)``.

    A hand-rolled regex rather than a YAML parse: the frontmatter in this corpus
    occasionally contains constructs that strict YAML rejects, and the only field
    needed is ``title``. Failing to parse a whole document over a malformed
    reviewers list would be a poor trade.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text, None
    title_match = _TITLE_RE.search(match.group(1))
    title = title_match.group(1).strip().strip("\"'") if title_match else None
    return text[match.end() :], title


def normalize(text: str) -> str:
    """Deterministic cleanup so identical content hashes identically across branches."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HTML_COMMENT_RE.sub("", text)
    text = _SHORTCODE_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _clean_heading(raw: str) -> str:
    """Strip the trailing ``{#explicit-anchor}`` Hugo/markdown attaches to headings.

    Those anchors embed the release number in this corpus
    (``{#flowcontrol-resources-v132}``), so leaving them in would make every heading
    branch-specific and break coalescing outright.
    """
    return _ANCHOR_RE.sub("", _SHORTCODE_RE.sub("", raw)).strip()


class _Section:
    __slots__ = ("heading_path", "lines")

    def __init__(self, heading_path: tuple[str, ...]) -> None:
        self.heading_path = heading_path
        self.lines: list[str] = []

    def body(self) -> str:
        return normalize("\n".join(self.lines))


def split_sections(body: str, doc_title: str | None) -> list[_Section]:
    """Split a markdown body into sections keyed by their heading breadcrumb.

    Tracks fenced code blocks so a ``# comment`` inside a shell example is never
    mistaken for a heading -- a mistake that silently shatters families, because the
    spurious heading appears on some branches and not others.
    """
    root_path: tuple[str, ...] = (doc_title,) if doc_title else ()
    sections = [_Section(root_path)]
    # stack[i] is the heading text at level i+1, or None if that level is unused.
    stack: list[str | None] = [None] * 6
    in_fence = False
    fence_marker: str | None = None

    for line in body.split("\n"):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, None
            sections[-1].lines.append(line)
            continue

        heading_match = None if in_fence else _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = _clean_heading(heading_match.group(2))
            if not title:
                sections[-1].lines.append(line)
                continue
            stack[level - 1] = title
            for deeper in range(level, 6):
                stack[deeper] = None
            crumbs = root_path + tuple(part for part in stack[:level] if part)
            sections.append(_Section(crumbs))
        else:
            sections[-1].lines.append(line)

    return [section for section in sections if section.body()]


def _split_oversized(text: str, config: ChunkingConfig) -> list[str]:
    """Break an over-long section on paragraph boundaries, never inside a code fence.

    Splitting a fence in half produces two chunks that are each syntactically broken
    and semantically useless, so a fence is kept whole even when that pushes a chunk
    past ``max_chars``.
    """
    if len(text) <= config.max_chars:
        return [text]

    # Group paragraphs, treating a whole fenced block as one atomic paragraph.
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker: str | None = None

    for paragraph in text.split("\n\n"):
        current.append(paragraph)
        for line in paragraph.split("\n"):
            fence_match = _FENCE_RE.match(line)
            if not fence_match:
                continue
            marker = fence_match.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, None
        if not in_fence:
            blocks.append("\n\n".join(current))
            current = []
    if current:
        blocks.append("\n\n".join(current))

    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for block in blocks:
        block_size = len(block) + 2
        if buffer and size + block_size > config.target_chars:
            chunks.append("\n\n".join(buffer))
            buffer, size = [], 0
        buffer.append(block)
        size += block_size
    if buffer:
        chunks.append("\n\n".join(buffer))
    return [chunk for chunk in chunks if chunk.strip()]


def chunk_document(
    doc_path: str,
    raw_text: str,
    version: MinorVersion,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Chunk one markdown file as seen on one release branch."""
    body, title = strip_frontmatter(raw_text)
    if title is None:
        title = Path(doc_path).stem.replace("-", " ").replace("_", " ").title()

    chunks: list[Chunk] = []
    # Chunks sharing a heading path within one document must still be distinguishable,
    # or they collapse into a single family and every one of them claims the same
    # (family, version) slot. Counting occurrences in document order gives a stable
    # ordinal: the same section split the same way on two branches yields matching
    # parts, so coalescing still works.
    part_counter: dict[tuple[str, ...], int] = {}

    for section in split_sections(body, title):
        section_body = section.body()
        if len(section_body) < config.min_chars:
            # Short sections are navigational stubs ("see also", a lone image) far
            # more often than they are answers. Indexing them dilutes the top-k.
            continue
        for piece in _split_oversized(section_body, config):
            if len(piece.strip()) < config.min_chars:
                continue
            part = part_counter.get(section.heading_path, 0)
            part_counter[section.heading_path] = part + 1
            chunks.append(
                Chunk(
                    doc_path=doc_path,
                    heading_path=section.heading_path,
                    text=piece.strip(),
                    version_low=version,
                    version_high=version,
                    part=part,
                )
            )
    return chunks


def chunk_tree(
    docs_root: Path,
    files: Iterable[Path],
    version: MinorVersion,
    config: ChunkingConfig,
    path_prefix: str = "content/en/docs",
) -> Iterator[Chunk]:
    """Chunk every markdown file in one branch's docs tree."""
    for file_path in files:
        try:
            raw = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            log.warning("skipping %s: %s", file_path, exc)
            continue
        relative = file_path.relative_to(docs_root).as_posix()
        doc_path = f"{path_prefix}/{relative}"
        yield from chunk_document(doc_path, raw, version, config)


def coalesce_corpus(per_version_chunks: Iterable[Chunk]) -> Corpus:
    """Collapse identical chunk text across adjacent versions into version ranges.

    Input is single-version chunks from every branch. Output is one chunk per
    ``(family, exact text)`` carrying the contiguous version span it was observed on.
    A section untouched from 1.24 to 1.35 collapses twelve-to-one; a section that
    changed at 1.29 becomes two chunks with adjacent, disjoint ranges.

    That disjointness is what lets ``Corpus.covering`` assume at most one match per
    family per version, and what makes "same family, does not cover V" a sound
    definition of a hard negative.
    """
    # (family_id, content_hash) -> the versions it was seen on
    grouped: dict[tuple[str, str], list[MinorVersion]] = {}
    exemplar: dict[tuple[str, str], Chunk] = {}

    for chunk in per_version_chunks:
        key = (chunk.family_id, chunk.content_hash)
        grouped.setdefault(key, []).append(chunk.version_low)
        exemplar.setdefault(key, chunk)

    coalesced: list[Chunk] = []
    for key, versions in grouped.items():
        source = exemplar[key]
        for span in coalesce_versions(versions):
            coalesced.append(
                Chunk(
                    doc_path=source.doc_path,
                    heading_path=source.heading_path,
                    text=source.text,
                    version_low=span.low,
                    version_high=span.high,
                    part=source.part,
                )
            )

    coalesced.sort(key=lambda c: (c.doc_path, c.heading_path, c.part, c.version_low))
    corpus = Corpus(chunks=coalesced)
    _warn_on_overlaps(corpus)
    return corpus


def _warn_on_overlaps(corpus: Corpus) -> None:
    """Assert the disjointness invariant that downstream code relies on.

    An overlap means two different texts claim the same (family, version), which can
    only happen if the chunker produced two sections with an identical heading path
    within one document. Downstream code would then silently pick whichever came
    first, so this is worth surfacing loudly.
    """
    overlaps = 0
    for family_id in corpus.families():
        members = corpus.family(family_id)
        # Intentionally ragged: this walks consecutive pairs, so the two sequences
        # differ in length by one by construction.
        for earlier, later in zip(members, members[1:], strict=False):
            if earlier.version_high >= later.version_low:
                overlaps += 1
    if overlaps:
        log.warning(
            "%d overlapping version ranges detected (duplicate heading paths within a "
            "document); Corpus.covering will return the earliest match",
            overlaps,
        )
