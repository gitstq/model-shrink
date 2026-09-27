"""Built-in, zero-download, linearly *non-separable* classification data.

Latent 2-D points are sampled uniformly in a disk and labelled by the
combination of an **annulus** (inner vs outer ring) and an angular **sector**,
giving ``2 * n_sectors`` classes. The pattern needs both a circular and several
wedge-shaped decision boundaries, so a linear (or very narrow) model cannot fit
it. The 2-D pattern is then embedded into ``n_features`` dimensions with a
fixed random matrix (a linear embedding preserves linear non-separability).

Natural confusable groups are adjacent sectors, and the inner/outer ring over
the same sector; the soft probabilities of a good teacher encode exactly these
relationships, which is useful signal for distillation.
"""
from __future__ import annotations

import numpy as np


def make_disk_patterns(
    n_samples: int = 1300,
    n_sectors: int = 5,
    r_split: float = 0.62,
    noise: float = 0.09,
    n_features: int = 16,
    seed: int = 0,
):
    """Generate the ring × sector dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) with values in [0, 2*n_sectors)
    """
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n_samples))          # uniform radius in [0, 1)
    theta = rng.random(n_samples) * 2 * np.pi  # uniform angle
    z = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    z = z + rng.normal(scale=noise, size=z.shape)

    rr = np.sqrt((z ** 2).sum(axis=1))
    ang = np.arctan2(z[:, 1], z[:, 0]) % (2 * np.pi)
    ring = (rr > r_split).astype(np.int64)
    sector = (np.floor(ang / (2 * np.pi / n_sectors)) % n_sectors).astype(np.int64)
    y = ring * n_sectors + sector

    embed = rng.standard_normal((2, n_features))
    X = z @ embed
    return X.astype(np.float64), y.astype(np.int64)


def _standardize(X_train, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


def load_dataset(
    n_train: int = 900,
    n_test: int = 400,
    n_sectors: int = 5,
    r_split: float = 0.62,
    noise: float = 0.09,
    n_features: int = 16,
    label_noise: float = 0.0,
    noise_seed: int = 77,
    seed: int = 0,
):
    """Return a standardized train/test split of the ring × sector dataset.

    Parameters
    ----------
    label_noise : float in [0, 1)
        Fraction of *training* labels randomly flipped to another class. The
        test labels are always kept clean. This creates the classic setting in
        which a capacity-rich teacher can "see through" the label noise and a
        student benefits from learning the teacher's de-noised soft targets.
    """
    X, y = make_disk_patterns(
        n_samples=n_train + n_test, n_sectors=n_sectors, r_split=r_split,
        noise=noise, n_features=n_features, seed=seed,
    )
    X_train, X_test = X[:n_train], X[n_train:]
    y_train_clean, y_test = y[:n_train], y[n_train:]
    X_train, X_test = _standardize(X_train, X_test)

    y_train = y_train_clean.copy()
    if label_noise > 0:
        n_classes = 2 * n_sectors
        nrng = np.random.default_rng(noise_seed)
        flip = nrng.random(n_train) < label_noise
        y_train[flip] = nrng.integers(0, n_classes, size=int(flip.sum()))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_clean": y_train_clean,
        "X_test": X_test,
        "y_test": y_test,
        "n_classes": 2 * n_sectors,
        "n_features": n_features,
        "label_noise": label_noise,
    }


# Backward-compatible / convenience alias.
load_synthetic = load_dataset
