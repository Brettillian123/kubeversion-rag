"""Fine-tuning the bi-encoder and the cross-encoder reranker."""

from __future__ import annotations

import sys


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


def device_kwargs(dataloader_workers: int | None = None) -> dict[str, object]:
    """Mixed precision, and a dataloader worker count that is safe on this platform.

    ``fp16`` is roughly a 2x speedup on consumer NVIDIA hardware, and these models are
    small enough that reduced precision costs nothing measurable in retrieval quality.
    Guarded because ``fp16=True`` on CPU raises rather than merely being slow.

    **Dataloader workers default to 0 on Windows, and that is deliberate.** The obvious
    optimization -- overlap tokenization with compute by adding workers -- backfires
    badly here. Windows has no ``fork``, so every worker is a fresh interpreter that
    re-imports torch and sentence-transformers; that import chain is ~20s at best and
    ~85s through the deprecated re-export shims. With two workers the run sat for
    minutes before step 1 with the GPU near idle, looking exactly like a hang, and each
    orphaned worker held multiple GB of RSS.

    On Linux, ``fork`` makes workers nearly free and they are worth having, so the
    default is platform-conditional rather than simply off.
    """
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a hard dep of this module
        return {}

    if dataloader_workers is None:
        dataloader_workers = 0 if sys.platform == "win32" else 2

    return {
        "fp16": cuda,
        "dataloader_num_workers": dataloader_workers if cuda else 0,
        # Only meaningful once workers are producing batches ahead of time.
        "dataloader_pin_memory": cuda and dataloader_workers > 0,
        # tqdm writes carriage-return progress bars to stderr, which several shells
        # (PowerShell among them) buffer until the process exits. A training run you
        # cannot observe is one you cannot tell apart from a hung one -- so emit
        # progress through the logging module, which flushes per line.
        "disable_tqdm": not sys.stderr.isatty(),
    }


def disable_model_card_widgets(model: object) -> None:
    """Skip sentence-transformers' model-card widget example generation.

    It computes example outputs for a Hub model card, for a model that is never going
    to the Hub. Measured at well under a second here, so this is tidiness rather than
    a fix -- but it runs during Trainer *construction*, before any step counter moves,
    which makes it a confusing place to be when you are trying to work out whether a
    run has started.

    The setting lives on the model rather than in the training arguments, so it has to
    be applied to the model object before the Trainer is built.
    """
    card = getattr(model, "model_card_data", None)
    if card is not None and hasattr(card, "generate_widget_examples"):
        card.generate_widget_examples = False


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
