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


def plot_training_curves(history: dict, title: str = "Training curves"):
    """Two panels (loss, accuracy), each with train (solid) and test (dashed) curves.

    ``history`` has keys ``train_loss``, ``test_loss``, ``train_acc``, ``test_acc``,
    each a list of per-epoch values.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    a1.plot(epochs, history["train_loss"], color="tab:blue", label="train")
    a1.plot(epochs, history["test_loss"], color="tab:orange", ls="--", label="test")
    a1.set_xlabel("epoch")
    a1.set_ylabel("cross-entropy loss")
    a1.set_title("Loss")
    a1.legend()

    a2.plot(epochs, history["train_acc"], color="tab:blue", label="train")
    a2.plot(epochs, history["test_acc"], color="tab:orange", ls="--", label="test")
    a2.set_xlabel("epoch")
    a2.set_ylabel("accuracy")
    a2.set_ylim(0, 1.02)
    a2.set_title("Accuracy")
    a2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_metric_comparison(histories: dict, metric: str, ylabel: str,
                           title: str, ax=None):
    """Overlay one metric (e.g. ``train_acc``) across models on a single axis.

    ``histories`` maps model name -> history dict.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, h in histories.items():
        ax.plot(range(1, len(h[metric]) + 1), h[metric], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if "acc" in metric:
        ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return ax


def plot_method_grid(histories: dict, title: str = "Method comparison"):
    """2x2 grid comparing all models: rows = train / test, cols = loss / accuracy."""
    layout = {
        (0, 0): ("train_loss", "Train — loss", "cross-entropy loss"),
        (0, 1): ("train_acc", "Train — accuracy", "accuracy"),
        (1, 0): ("test_loss", "Test — loss", "cross-entropy loss"),
        (1, 1): ("test_acc", "Test — accuracy", "accuracy"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for (r, c), (key, panel_title, ylabel) in layout.items():
        ax = axes[r, c]
        for name, h in histories.items():
            ax.plot(range(1, len(h[key]) + 1), h[key], label=name)
        ax.set_title(panel_title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        if "acc" in key:
            ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_runtime_bar(times: dict, title: str = "Training wall-clock time"):
    """Bar chart of per-model training time (seconds). ``times`` maps name -> seconds."""
    names = list(times)
    vals = [times[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, vals, color="tab:purple")
    ax.set_ylabel("training time (s)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(cms, class_names, model_names, normalize=True):
    """Side-by-side confusion matrices (one per model). ``cms[i]`` is (C, C) counts."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    for i, (cm, name) in enumerate(zip(cms, model_names)):
        ax = axes[0, i]
        M = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
        im = ax.imshow(M, vmin=0, vmax=1 if normalize else None, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("predicted")
        if i == 0:
            ax.set_ylabel("true")
        ax.set_title(name)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center",
                        fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices (test set)")
    fig.tight_layout()
    return fig


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
