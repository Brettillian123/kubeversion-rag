"""Turn retrieved chunks into a cited answer, or refuse.

Two rules the generation step enforces mechanically rather than by asking nicely:

* **Cite or refuse.** Every claim carries a ``[n]`` marker keyed to a numbered context
  block. After generation, citations are parsed and validated against the blocks that
  were actually supplied; an answer citing ``[7]`` when six blocks were provided is a
  fabrication, and it is caught here rather than by the reader.
* **Say which version you answered for.** The whole system exists to be
  version-correct, so an answer that does not state its version is not fit to ship even
  when it is right -- the reader has no way to notice when the inference was wrong.

Refusal is a first-class outcome, not an error path. Retrieval below the score floor
means the corpus does not cover the question, and saying so is the correct answer.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..models import RetrievedChunk
from ..retrieval.query import ResolvedVersion

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You answer questions about Kubernetes using only the numbered context blocks supplied \
with each question. The context has already been filtered to the Kubernetes version the \
user is asking about.

Rules:

1. Use only the supplied context. If it does not contain the answer, say so plainly and \
stop. Do not fall back on general Kubernetes knowledge — the whole point of this system \
is that the answer is version-specific, and your general knowledge is not version-scoped.

2. Cite every factual claim with the bracketed number of the block it came from, like \
[2]. A sentence stating a fact with no citation is a bug. Never cite a number that was \
not supplied.

3. Lead with the direct answer. Supporting detail comes after. If the user asked whether \
something works on their version, the first sentence answers yes or no.

4. Version-specific behaviour is the reason this system exists. When the context says an \
API was removed or changed in a particular release, state the release explicitly.

5. If the context blocks disagree with each other, say so and quote both rather than \
silently picking one. Disagreement usually means the retrieval returned two different \
releases, and the user needs to know.

Keep answers to the length the question needs. A yes/no question about API availability \
deserves a short paragraph, not a tutorial."""


@dataclass
class Citation:
    marker: int
    chunk_id: str
    title: str
    version_label: str
    url: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    version_disclosure: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    invalid_citations: list[int] = field(default_factory=list)
    uncited: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.text,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "version_disclosure": self.version_disclosure,
            "citations": [
                {
                    "marker": c.marker,
                    "chunk_id": c.chunk_id,
                    "title": c.title,
                    "versions": c.version_label,
                    "url": c.url,
                }
                for c in self.citations
            ],
            "warnings": {
                "invalid_citations": self.invalid_citations,
                "uncited_answer": self.uncited,
            },
            "usage": {
                "model": self.model,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
        }


def build_context_block(hits: Sequence[RetrievedChunk]) -> tuple[str, list[Citation]]:
    """Render retrieved chunks as numbered blocks and the citation table for them."""
    parts: list[str] = []
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.title
        parts.append(
            f"[{index}] Applies to Kubernetes {chunk.version_range}\n"
            f"Source: {chunk.doc_path}\n"
            f"Section: {heading}\n\n{chunk.text}"
        )
        citations.append(
            Citation(
                marker=index,
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                version_label=str(chunk.version_range),
                url=chunk.citation().split(" — ")[-1],
            )
        )
    return "\n\n---\n\n".join(parts), citations


def _refusal(reason: str, disclosure: str, text: str) -> Answer:
    return Answer(text=text, refused=True, refusal_reason=reason, version_disclosure=disclosure)


class Generator:
    """Wraps the Anthropic client with this system's answer contract."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        max_tokens: int = 2000,
        min_score: float = -4.0,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        # Deliberately far below the usual default: answers here are a short cited
        # paragraph, and a large ceiling on a non-streaming call only buys the risk of
        # a long generation hitting the request timeout.
        self.max_tokens = max_tokens
        self.min_score = min_score
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            # Zero-arg construction when no explicit key: the SDK also resolves
            # ANTHROPIC_AUTH_TOKEN and an `ant auth login` profile, and passing
            # api_key=None explicitly would defeat that.
            self._client = (
                anthropic.Anthropic(api_key=self._api_key, timeout=self.timeout)
                if self._api_key
                else anthropic.Anthropic(timeout=self.timeout)
            )
        return self._client

    def answer(
        self,
        question: str,
        hits: Sequence[RetrievedChunk],
        resolved_version: ResolvedVersion,
    ) -> Answer:
        disclosure = resolved_version.disclosure()

        if not hits:
            return _refusal(
                "no_results",
                disclosure,
                f"I don't have documentation covering that for Kubernetes "
                f"{resolved_version.version}. {disclosure}",
            )

        best = max(hit.final_score for hit in hits)
        if best < self.min_score:
            log.info("refusing: best score %.3f below floor %.3f", best, self.min_score)
            return _refusal(
                "low_confidence",
                disclosure,
                "I found documentation that is topically related but nothing that "
                "actually answers this question, so I'd rather not guess. "
                f"{disclosure}",
            )

        context, citations = build_context_block(hits)
        user_content = (
            f"Kubernetes version in scope: {resolved_version.version}\n"
            f"(How that was determined: {resolved_version.source.value})\n\n"
            f"Question: {question}\n\n"
            f"Context blocks:\n\n{context}"
        )

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # The system prompt is byte-identical on every request, so it is the
                # natural cache prefix. The context blocks vary per request and sit
                # after it, which is the placement that actually gets cache hits.
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                # Safety classifiers can decline a request; without a fallback the
                # request simply stops. Routing by refusal category recovers it
                # server-side inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 502
            log.exception("generation failed")
            raise GenerationError(str(exc)) from exc

        # A refusal is an HTTP 200 with an empty or partial content list, so this must
        # be checked before indexing into content.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            log.warning("model declined the request (category=%s)", category)
            return _refusal(
                "model_declined",
                disclosure,
                "I wasn't able to answer that one. Rephrasing the question usually helps.",
            )

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        used, invalid = self._validate_citations(text, len(citations))

        return Answer(
            text=text,
            citations=[c for c in citations if c.marker in used],
            version_disclosure=disclosure,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            invalid_citations=sorted(invalid),
            uncited=not used,
        )

    @staticmethod
    def _validate_citations(text: str, n_blocks: int) -> tuple[set[int], set[int]]:
        """Split cited markers into ones that exist and ones the model invented.

        Checked programmatically because "cite your sources" is exactly the kind of
        instruction that is followed 95% of the time, and the 5% is indistinguishable
        from a correct answer to anyone who does not check.
        """
        cited = {int(match.group(1)) for match in _CITATION_RE.finditer(text)}
        valid = {marker for marker in cited if 1 <= marker <= n_blocks}
        invalid = cited - valid
        if invalid:
            log.warning("model cited non-existent blocks: %s", sorted(invalid))
        return valid, invalid


class GenerationError(RuntimeError):
    """The upstream model call failed. Distinct from a refusal, which is a valid answer."""
