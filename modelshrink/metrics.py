"""Compression and error metrics, all computed with plain NumPy."""
from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------- #
# Tensor error (original vs reconstructed weights)
# --------------------------------------------------------------------------- #
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 1.0


def signal_to_noise_db(a: np.ndarray, b: np.ndarray) -> float:
    """SNR in dB between the original signal ``a`` and reconstruction ``b``."""
    signal = np.mean(a ** 2)
    noise = np.mean((a - b) ** 2)
    if noise <= 0:
        return float("inf")
    return float(10.0 * math.log10(signal / noise))


def tensor_error(a: np.ndarray, b: np.ndarray) -> dict:
    """A bundle of reconstruction-error metrics for two weight tensors."""
    return {
        "mse": mean_squared_error(a, b),
        "max_abs": max_abs_error(a, b),
        "cosine": cosine_similarity(a, b),
        "snr_db": signal_to_noise_db(a, b),
    }


# --------------------------------------------------------------------------- #
# Sparsity
# --------------------------------------------------------------------------- #
def sparsity(w: np.ndarray) -> float:
    """Fraction of exactly-zero entries."""
    return float(np.mean(w == 0))


def model_sparsity(model) -> float:
    total = sum(lin.W.size for lin in model.linears)
    zeros = sum(int(np.sum(lin.W == 0)) for lin in model.linears)
    return zeros / total if total else 0.0


# --------------------------------------------------------------------------- #
# Size / parameter accounting
# --------------------------------------------------------------------------- #
def count_parameters(model) -> int:
    return int(sum(lin.num_params() for lin in model.linears))


def dense_bytes(n_params: int, bits: int = 32) -> float:
    """Bytes needed to store ``n_params`` values at ``bits`` each."""
    return n_params * bits / 8.0


def sparse_coo_bytes(
    n_nonzero: int,
    n_total: int,
    value_bits: int = 8,
    index_bits: int | None = None,
) -> float:
    """Estimate COO sparse storage: one value + one coordinate per nonzero.

    Coordinate width defaults to the number of bits needed to index the tensor.
    """
    if n_nonzero == 0:
        return 0.0
    if index_bits is None:
        index_bits = max(1, int(math.ceil(math.log2(max(2, n_total)))))
    return n_nonzero * (value_bits + index_bits) / 8.0


def compression_ratio(original_bytes: float, compressed_bytes: float) -> float:
    if compressed_bytes <= 0:
        return float("inf")
    return float(original_bytes / compressed_bytes)


def layer_table(model) -> list:
    """Per-layer shape / parameter description for reporting."""
    rows = []
    for i, lin in enumerate(model.linears):
        rows.append(
            {
                "layer": i,
                "weight_shape": list(lin.W.shape),
                "bias_shape": list(lin.b.shape),
                "params": lin.num_params(),
                "sparsity": sparsity(lin.effective_W),
            }
        )
    return rows
