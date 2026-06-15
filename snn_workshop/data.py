"""RacketSports dataset loading (not a participant task).

RacketSports (UEA multivariate time-series archive): university students playing
badminton or squash while wearing a smartwatch. Each 3-second trial is sampled at
10 Hz -> 30 timesteps, with 6 channels (3-axis accelerometer + 3-axis gyroscope).
The task is to identify the sport + stroke (4 classes). 151 train / 152 test trials.

``aeon`` returns arrays shaped ``(n_cases, n_channels, n_timepoints)``. For the SNN
we want a *time-major-per-sample* layout ``(n_cases, n_timesteps, n_channels)`` so
that we can iterate over timesteps cleanly in the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Human-readable names for the raw RacketSports labels (aeon returns these strings).
RACKET_SPORTS_CLASS_NAMES = {
    "badminton_clear": "Badminton: Clear",
    "badminton_smash": "Badminton: Smash",
    "squash_forehandboast": "Squash: Forehand Boast",
    "squash_backhandboast": "Squash: Backhand Boast",
}


@dataclass
class RacketSports:
    """Container for the loaded dataset (numpy)."""

    X_train: np.ndarray  # (N_train, T, C) float32
    y_train: np.ndarray  # (N_train,) int64
    X_test: np.ndarray   # (N_test,  T, C) float32
    y_test: np.ndarray   # (N_test,)  int64
    class_names: list[str]

    @property
    def n_timesteps(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_channels(self) -> int:
        return self.X_train.shape[2]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def load_racket_sports(normalize: bool = True) -> RacketSports:
    """Download + load RacketSports via ``aeon`` and return tidy numpy arrays.

    Parameters
    ----------
    normalize:
        If True, z-score each channel using statistics computed on the training
        split only (then applied to test) -- standard practice to avoid leakage.
    """
    from aeon.datasets import load_classification

    X_train, y_train = load_classification("RacketSports", split="train")
    X_test, y_test = load_classification("RacketSports", split="test")

    # (N, C, T) -> (N, T, C)
    X_train = np.transpose(X_train, (0, 2, 1)).astype(np.float32)
    X_test = np.transpose(X_test, (0, 2, 1)).astype(np.float32)

    # Encode string labels -> contiguous ints using a stable sorted order.
    raw_labels = sorted(np.unique(y_train).tolist())
    label_to_idx = {lab: i for i, lab in enumerate(raw_labels)}
    y_train = np.array([label_to_idx[v] for v in y_train], dtype=np.int64)
    y_test = np.array([label_to_idx[v] for v in y_test], dtype=np.int64)
    class_names = [RACKET_SPORTS_CLASS_NAMES.get(lab, str(lab)) for lab in raw_labels]

    if normalize:
        # Per-channel mean/std over (samples, timesteps) of the training split.
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    return RacketSports(X_train, y_train, X_test, y_test, class_names)
