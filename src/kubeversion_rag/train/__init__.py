"""Fine-tuning the bi-encoder and the cross-encoder reranker."""

from __future__ import annotations


def warmup_kwargs(ratio: float) -> dict[str, float]:
    """Spell the warmup ratio the way the installed Transformers expects.

    Transformers v5 deprecated ``warmup_ratio`` in favour of a float ``warmup_steps``;
    v4 wants ``warmup_ratio`` and treats ``warmup_steps`` as an integer count. Passing
    the wrong one is not fatal on either, but on v4 a float step count is silently
    nonsense rather than a ratio -- so this resolves it rather than picking one and
    hoping.
    """
    try:
        import transformers

        major = int(transformers.__version__.split(".")[0])
    except (ImportError, ValueError):  # pragma: no cover - defensive
        major = 4
    return {"warmup_steps": ratio} if major >= 5 else {"warmup_ratio": ratio}
