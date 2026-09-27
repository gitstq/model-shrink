"""Weight pruning: magnitude (unstructured) and channel (structured) masks.

Pruning sets selected weights to exactly zero and attaches a binary mask to
each layer. The mask is enforced in the forward pass, on the gradient and after
every optimizer update (see :class:`modelshrink.nn.Linear`), so zeros stay zero
during optional fine-tuning.
"""
from __future__ import annotations

import numpy as np


def _magnitude_mask(w: np.ndarray, threshold: float) -> np.ndarray:
    return (np.abs(w) > threshold).astype(np.float64)


def magnitude_masks(weights, sparsity: float, scope: str = "global") -> list:
    """Build 0/1 masks for a list of weight matrices.

    Parameters
    ----------
    sparsity : float in [0, 1)
        Target fraction of weights set to zero.
    scope : {"global", "layer"}
        ``global`` picks one threshold across all layers; ``layer`` picks a
        threshold independently per layer.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    weights = [np.asarray(w) for w in weights]
    masks = []
    if scope == "global":
        flat = np.concatenate([np.abs(w).reshape(-1) for w in weights])
        threshold = np.quantile(flat, sparsity)
        for w in weights:
            masks.append(_magnitude_mask(w, threshold))
    elif scope == "layer":
        for w in weights:
            threshold = np.quantile(np.abs(w).reshape(-1), sparsity)
            masks.append(_magnitude_mask(w, threshold))
    else:
        raise ValueError(f"unknown scope {scope!r}")
    return masks


def channel_prune_mask(w: np.ndarray, prune_fraction: float, channel_axis: int = 1) -> np.ndarray:
    """Structured pruning: remove whole output channels with smallest L2 norm.

    The mask keeps the original shape (whole columns become zero) so it stays
    compatible with the adjacent layers.
    """
    ca = channel_axis % w.ndim
    reduce_axes = tuple(i for i in range(w.ndim) if i != ca)
    norms = np.linalg.norm(w, axis=reduce_axes)
    n_channels = norms.shape[0]
    n_prune = int(round(prune_fraction * n_channels))
    mask = np.ones_like(w, dtype=np.float64)
    if n_prune == 0:
        return mask
    kill = np.argsort(norms)[:n_prune]
    # Build a boolean channel mask then broadcast to the weight shape.
    channel_keep = np.ones(n_channels, dtype=bool)
    channel_keep[kill] = False
    return np.broadcast_to(
        np.expand_dims(channel_keep.astype(np.float64), axis=reduce_axes), w.shape
    ).copy()


def apply_masks(model, masks) -> None:
    if len(masks) != len(model.linears):
        raise ValueError("number of masks must match number of linear layers")
    for lin, mask in zip(model.linears, masks):
        lin.mask = mask.astype(np.float64)
        lin.W *= lin.mask


def reset_masks(model) -> None:
    for lin in model.linears:
        lin.mask = None


def actual_sparsity(mask: np.ndarray) -> float:
    return float(np.mean(mask == 0))


def prune_model(
    model,
    sparsity: float = 0.5,
    scope: str = "global",
    structured: bool = False,
) -> dict:
    """Prune a model in place and return a summary.

    For structured pruning ``sparsity`` is interpreted as the fraction of
    output channels removed in every layer.
    """
    weights = [lin.W for lin in model.linears]
    if structured:
        masks = [channel_prune_mask(w, sparsity, channel_axis=1) for w in weights]
        method = "structured/channel"
    else:
        masks = magnitude_masks(weights, sparsity, scope=scope)
        method = f"unstructured/magnitude ({scope})"
    apply_masks(model, masks)

    per_layer = [
        {"layer": i, "sparsity": actual_sparsity(m), "zeros": int(np.sum(m == 0)),
         "entries": int(m.size)}
        for i, m in enumerate(masks)
    ]
    total_entries = sum(m.size for m in masks)
    total_zeros = sum(int(np.sum(m == 0)) for m in masks)
    return {
        "method": method,
        "target_sparsity": sparsity,
        "achieved_sparsity": total_zeros / total_entries,
        "layers": per_layer,
        "masks": masks,
    }
