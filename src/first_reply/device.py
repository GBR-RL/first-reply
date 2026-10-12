"""Where the embedding and reranking models run: a CUDA GPU when one is present, else Apple's
MPS, else the CPU. ``FIRST_REPLY_DEVICE`` (``auto``, ``cpu``, ``cuda``, ``cuda:1``, ``mps``)
overrides the choice."""

from __future__ import annotations

import os


def pick_device(wanted: str | None = None) -> str:
    wanted = (wanted or os.environ.get("FIRST_REPLY_DEVICE") or "auto").strip().lower()
    if wanted != "auto":
        return wanted
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
