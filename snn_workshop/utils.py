"""Small reproducibility / device helpers (not a participant task)."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, else CPU.

    The workshop runs comfortably on CPU (RacketSports is tiny); a GPU mostly
    helps the ``torch.compile`` speed-up demo in Chapter 2.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
