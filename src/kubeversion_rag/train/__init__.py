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


def device_kwargs() -> dict[str, bool]:
    """Enable mixed precision when there is a CUDA device to use it on.

    Roughly a 2x speedup on consumer NVIDIA hardware, and these models are small
    enough that the reduced precision costs nothing measurable in retrieval quality.
    Guarded because ``fp16=True`` on CPU is not merely slow -- it raises.
    """
    try:
        import torch

        return {"fp16": bool(torch.cuda.is_available())}
    except ImportError:  # pragma: no cover - torch is a hard dep of this module
        return {}


def describe_device() -> str:
    """Human-readable accelerator summary, logged before a run starts.

    Worth surfacing loudly: the default ``pip install torch`` on Windows is the
    CPU-only wheel, so a machine with a perfectly good GPU trains an order of
    magnitude slower with nothing to indicate why.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "torch not installed"
    if not torch.cuda.is_available():
        return (
            f"CPU only (torch {torch.__version__}). If this machine has an NVIDIA GPU, "
            "the CPU-only wheel is installed -- reinstall from the CUDA index."
        )
    properties = torch.cuda.get_device_properties(0)
    return (
        f"CUDA: {properties.name}, {properties.total_memory / 1e9:.1f} GB "
        f"(torch {torch.__version__})"
    )
