"""Shared helpers for the BISCCITS SNN workshop.

These modules hold *plumbing* that is **not** part of the participant tasks
(plotting, dataset loading, seeding). The actual neuroscience / SNN code that
participants implement lives in the chapter scripts under ``scripts/``.

When the chapter scripts are assembled into the participant notebook, the few
helpers used from here are pasted into a single "utilities" setup cell.
"""

from .utils import set_seed, get_device

__all__ = ["set_seed", "get_device"]
