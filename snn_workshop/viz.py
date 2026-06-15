"""Plotting helpers for the workshop (not a participant task).

Kept deliberately framework-light: every function takes plain numpy arrays so it
works the same whether called from a script or a notebook cell.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_membrane_dynamics(
    current: np.ndarray,
    mem: np.ndarray,
    spikes: np.ndarray,
    threshold: float,
    dt: float = 1.0,
    title: str = "LIF neuron dynamics",
):
    """Three stacked panels: input current, membrane potential, output spikes.

    All inputs are 1-D arrays of length T (a single neuron over time).
    """
    T = len(mem)
    t = np.arange(T) * dt
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t, current, color="tab:blue")
    axes[0].set_ylabel("input\ncurrent  I(t)")
    axes[0].set_title(title)

    axes[1].plot(t, mem, color="tab:purple", label="V(t)")
    axes[1].axhline(threshold, color="tab:red", ls="--", lw=1, label="threshold")
    axes[1].set_ylabel("membrane\npotential  V(t)")
    axes[1].legend(loc="upper right", fontsize=8)

    spike_times = t[spikes > 0.5]
    axes[2].vlines(spike_times, 0, 1, color="black")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("spikes  S(t)")
    axes[2].set_xlabel("time")

    fig.tight_layout()
    return fig, axes


def plot_spike_raster(
    spikes: np.ndarray,
    dt: float = 1.0,
    title: str = "Spike raster",
    ax=None,
):
    """Raster of a population over time.

    ``spikes`` has shape (T, N): T timesteps, N neurons.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    T, N = spikes.shape
    t = np.arange(T) * dt
    for n in range(N):
        st = t[spikes[:, n] > 0.5]
        ax.vlines(st, n + 0.5, n + 1.5, color="black", lw=0.8)
    ax.set_xlim(0, T * dt)
    ax.set_ylim(0.5, N + 0.5)
    ax.set_xlabel("time")
    ax.set_ylabel("neuron index")
    ax.set_title(title)
    return ax


def plot_firing_rates(rates_per_layer: list[np.ndarray], title: str = "Firing rates"):
    """Histogram of per-neuron firing probability for each layer.

    ``rates_per_layer[i]`` is a 1-D array of per-neuron mean firing rates (in [0, 1]).
    """
    n = len(rates_per_layer)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), squeeze=False)
    for i, rates in enumerate(rates_per_layer):
        ax = axes[0, i]
        ax.hist(rates, bins=20, range=(0, 1), color="tab:green", alpha=0.8)
        ax.set_title(f"layer {i + 1}  (mean={rates.mean():.2f})")
        ax.set_xlabel("firing rate (spikes / step)")
        if i == 0:
            ax.set_ylabel("# neurons")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
