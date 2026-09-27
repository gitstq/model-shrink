"""Human-readable reporting: terminal tables, JSON and optional plots."""
from __future__ import annotations

import json

import numpy as np


# --------------------------------------------------------------------------- #
# Terminal rendering
# --------------------------------------------------------------------------- #
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def _delta_str(value: float, baseline: float, color: bool) -> str:
    d = value - baseline
    txt = f"{value:.4f} ({d:+.4f})"
    if color:
        col = C.GREEN if d >= 0 else C.RED
        return f"{col}{txt}{C.END}"
    return txt


def render_comparison(rows: list, baseline_accuracy: float, color: bool = True) -> str:
    """Render the scenario comparison table.

    Each row is a dict with keys: method, accuracy, params, bytes, ratio, note.
    """
    headers = ["Method", "Accuracy (Δ)", "Params", "Est. bytes", "Ratio", "Note"]
    data = []
    for r in rows:
        data.append(
            [
                r["method"],
                _delta_str(r["accuracy"], r.get("base", baseline_accuracy), color),
                f"{r['params']:,}",
                f"{r['bytes']:,.0f}",
                ("∞" if not np.isfinite(r["ratio"]) else f"{r['ratio']:.2f}x"),
                r.get("note", ""),
            ]
        )
    widths = [
        max(len(headers[i]), max(_plain_len(row[i]) for row in data))
        for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [C.BOLD + line + C.END if color else line, "-" * len(line)]
    for row in data:
        cells = []
        for i, cell in enumerate(row):
            pad = widths[i] + (len(cell) - _plain_len(cell))
            cells.append(cell.ljust(pad))
        out.append("  ".join(cells))
    return "\n".join(out)


def _plain_len(s: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", str(s)))


def section(title: str, color: bool = True) -> str:
    return f"\n{C.BOLD}{C.CYAN}{title}{C.END}" if color else f"\n{title}"


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Optional matplotlib plots
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def accuracy_bar_chart(labels, values, path: str, title: str = "Accuracy by method"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#4C72B0" if i == 0 else "#55A868" for i in range(len(labels))]
    bars = ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def weight_histogram(orig, deq, path: str):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, w, name in ((axes[0], orig, "Original (FP32)"), (axes[1], deq, "INT8 dequantized")):
        ax.hist(w.reshape(-1), bins=60, color="#4C72B0", alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("weight value")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def curves_chart(curves: dict, path: str, ylabel: str = "Accuracy", title: str = ""):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, vals in curves.items():
        ax.plot(range(1, len(vals) + 1), vals, label=name, linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
