"""Post-training quantization (PTQ) via simulated / "fake" quantization.

We quantize weights to a low-bit integer grid and immediately dequantize back
to float32/64 so the existing NumPy network can run unchanged. The difference
between the original and dequantized weights is the quantization error, which
we measure directly. Real integer packing size is estimated in
:mod:`modelshrink.metrics`.

Both symmetric and asymmetric schemes and per-tensor / per-channel
granularity are supported, at any bit-width.
"""
from __future__ import annotations

import numpy as np

from . import metrics


# --------------------------------------------------------------------------- #
# Quantization parameter computation
# --------------------------------------------------------------------------- #
def _symmetric_params(w: np.ndarray, bits: int):
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    amax = np.max(np.abs(w))
    scale = np.where(amax > 0, amax / qmax, 1e-8)
    return qmin, qmax, scale, np.zeros_like(scale)


def _asymmetric_params(w: np.ndarray, bits: int):
    qmin = 0
    qmax = 2 ** bits - 1
    # Affine unsigned quantization always spans zero (so it can represent 0
    # exactly); extend the range accordingly.
    wmin = min(0.0, float(np.min(w)))
    wmax = max(0.0, float(np.max(w)))
    scale = np.where(wmax > wmin, (wmax - wmin) / (qmax - qmin), 1e-8)
    zp = np.round(qmin - wmin / scale)
    zp = np.clip(zp, qmin, qmax)
    return qmin, qmax, scale, zp


def _reduce_keepdims(w: np.ndarray, channel_axis: int):
    axes = tuple(i for i in range(w.ndim) if i != channel_axis)
    if not axes:  # degenerates to per-tensor
        return w, lambda u: u.reshape((1,) * w.ndim)
    return axes, None


def quantize_tensor(
    w: np.ndarray,
    bits: int = 8,
    scheme: str = "symmetric",
    granularity: str = "per-tensor",
    channel_axis: int = -1,
) -> dict:
    """Quantize then dequantize a single tensor.

    Parameters
    ----------
    bits : int
        Target bit-width (2..16).
    scheme : {"symmetric", "asymmetric"}
    granularity : {"per-tensor", "per-channel"}
    channel_axis : int
        Axis holding channels when ``per-channel`` is used.

    Returns
    -------
    dict with ``dequant`` (float weights), ``scale``, ``zero_point``,
    ``qmin``, ``qmax`` and ``error``.
    """
    if bits < 2 or bits > 16:
        raise ValueError("bits must be in [2, 16]")
    w = np.asarray(w, dtype=np.float64)

    if granularity == "per-channel":
        ca = channel_axis % w.ndim
        axes = tuple(i for i in range(w.ndim) if i != ca)
        amax = np.max(np.abs(w), axis=axes, keepdims=True)
        wmin = np.minimum(0.0, np.min(w, axis=axes, keepdims=True))
        wmax = np.maximum(0.0, np.max(w, axis=axes, keepdims=True))
        if scheme == "symmetric":
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
            scale = np.where(amax > 0, amax / qmax, 1e-8)
            zp = np.zeros_like(scale)
        elif scheme == "asymmetric":
            qmin, qmax = 0, 2 ** bits - 1
            scale = np.where(wmax > wmin, (wmax - wmin) / (qmax - qmin), 1e-8)
            zp = np.clip(np.round(qmin - wmin / scale), qmin, qmax)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
    elif granularity == "per-tensor":
        if scheme == "symmetric":
            qmin, qmax, scale, zp = _symmetric_params(w, bits)
        elif scheme == "asymmetric":
            qmin, qmax, scale, zp = _asymmetric_params(w, bits)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        scale = np.asarray(scale)
        zp = np.asarray(zp)
    else:
        raise ValueError(f"unknown granularity {granularity!r}")

    q = np.round(w / scale + zp)
    q = np.clip(q, qmin, qmax)
    deq = (q - zp) * scale
    return {
        "dequant": deq,
        "scale": np.asarray(scale),
        "zero_point": np.asarray(zp),
        "qmin": int(qmin),
        "qmax": int(qmax),
        "bits": bits,
        "scheme": scheme,
        "granularity": granularity,
        "error": metrics.tensor_error(w, deq),
    }


def quantize_model(
    model,
    bits: int = 8,
    scheme: str = "symmetric",
    granularity: str = "per-tensor",
    inplace: bool = True,
) -> dict:
    """Fake-quantize every weight matrix of an :class:`~modelshrink.nn.MLP`.

    Returns a dict with per-layer results and the aggregate cosine / MSE over
    all weights. When ``inplace`` is True the model weights are replaced by
    their dequantized values.
    """
    layers = []
    all_orig, all_deq = [], []
    for i, lin in enumerate(model.linears):
        orig = lin.W.copy()
        res = quantize_tensor(
            lin.W, bits=bits, scheme=scheme, granularity=granularity, channel_axis=1
        )
        if inplace:
            lin.W = res["dequant"]
        layers.append({"layer": i, **res})
        all_orig.append(orig.reshape(-1))
        all_deq.append(res["dequant"].reshape(-1))

    aggregate = metrics.tensor_error(
        np.concatenate(all_orig), np.concatenate(all_deq)
    )
    return {"bits": bits, "scheme": scheme, "granularity": granularity,
            "layers": layers, "error": aggregate}
