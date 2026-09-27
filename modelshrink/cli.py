"""Command-line interface for ModelShrink.

Run the full pipeline with::

    modelshrink demo

Individual techniques are available via ``modelshrink quant``,
``modelshrink prune`` and ``modelshrink distill``.
"""
from __future__ import annotations

import argparse
import copy
import os

import numpy as np

from . import __version__, metrics, report
from .data import load_dataset
from .distill import distill
from .nn import MLP
from .prune import prune_model
from .quantize import quantize_model


# --------------------------------------------------------------------------- #
# Size helpers
# --------------------------------------------------------------------------- #
def _weight_counts(model):
    Wp = sum(lin.W.size for lin in model.linears)
    Bp = sum(lin.b.size for lin in model.linears)
    return int(Wp), int(Bp)


def _pruned_bytes(model, value_bits=32, bias_bits=32):
    Wp, Bp = _weight_counts(model)
    nz = sum(int(lin.mask.sum()) for lin in model.linears)
    return (metrics.sparse_coo_bytes(nz, Wp, value_bits=value_bits)
            + metrics.dense_bytes(Bp, bias_bits))


def _quant_bytes(model, bits):
    Wp, Bp = _weight_counts(model)
    return metrics.dense_bytes(Wp + Bp, bits)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(seed=0, teacher_epochs=200, student_epochs=260, sparsity=0.5,
                 hi_sparsity=0.7, bits=(8, 4), student_hidden=12,
                 temperature=2.0, alpha=0.5, json_path=None, plots_dir=None,
                 color=True):
    data = load_dataset(seed=seed)
    f, k = data["n_features"], data["n_classes"]
    X, y, Xt, yt = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    teacher = MLP([f, 64, 64, k], seed=1)
    teacher.fit(X, y, epochs=teacher_epochs, X_val=Xt, y_val=yt)
    Wp, Bp = _weight_counts(teacher)
    fp_bytes = metrics.dense_bytes(Wp + Bp, 32)
    base_acc = teacher.accuracy(Xt, yt)

    rows = []

    def add(name, model, size_bytes, note, base=None):
        rows.append({
            "method": name,
            "accuracy": model.accuracy(Xt, yt),
            "params": model.count_parameters(),
            "bytes": float(size_bytes),
            "ratio": metrics.compression_ratio(fp_bytes, size_bytes),
            "note": note,
            "base": base_acc if base is None else base,
        })

    add("Teacher FP32", teacher, fp_bytes, "baseline")

    quant_models = {}
    for b in bits:
        mq = copy.deepcopy(teacher)
        qres = quantize_model(mq, bits=b, scheme="symmetric")
        quant_models[b] = (mq, qres)
        add(f"Quantized INT{b}", mq, _quant_bytes(mq, b),
            f"cos={qres['error']['cosine']:.5f}")

    for sp in (sparsity, hi_sparsity):
        mp = copy.deepcopy(teacher)
        pres = prune_model(mp, sparsity=sp, scope="global")
        add(f"Pruned {sp:.0%}", mp, _pruned_bytes(mp, 32, 32),
            f"sparse {pres['achieved_sparsity']:.1%} (COO)")

    mpq = copy.deepcopy(teacher)
    prune_model(mpq, sparsity=sparsity, scope="global")
    quantize_model(mpq, bits=8, scheme="symmetric")
    add("Pruned 50% + INT8", mpq, _pruned_bytes(mpq, 8, 8), "combined")

    # Tiny student: hard-only vs distilled from the teacher.
    student = MLP([f, student_hidden, k], seed=7)
    s_hist = student.fit(X, y, epochs=student_epochs, X_val=Xt, y_val=yt)
    sa = student.accuracy(Xt, yt)
    s_bytes = metrics.dense_bytes(student.count_parameters(), 32)
    add(f"Student [{student_hidden}] hard", student, s_bytes,
        f"{fp_bytes / s_bytes:.1f}x smaller", base=base_acc)

    distilled = MLP([f, student_hidden, k], seed=7)
    d_hist = distill(teacher, distilled, X, y, temperature=temperature,
                     alpha=alpha, epochs=student_epochs, X_val=Xt, y_val=yt)
    da = distilled.accuracy(Xt, yt)
    add("Student (distilled)", distilled, s_bytes,
        f"vs hard {da - sa:+.4f}", base=sa)

    print(report.section(f"ModelShrink v{__version__} — compression demo", color))
    print(f"dataset: {X.shape[0]} train / {Xt.shape[0]} test, "
          f"{f} features, {k} classes")
    print(f"teacher MLP {teacher.sizes}  params={teacher.count_parameters():,}  "
          f"FP32 test acc={base_acc:.4f}")

    print(report.section("Quantization weight error", color))
    for b, (_, qr) in quant_models.items():
        e = qr["error"]
        print(f"INT{b}: cosine={e['cosine']:.6f}  mse={e['mse']:.3e}  snr={e['snr_db']:.1f} dB")

    print(report.section("Pruning", color))
    print("Global magnitude pruning at 50% / 70% — see comparison table.")

    print(report.section("Distillation into a tiny student", color))
    print(f"tiny student [{student_hidden}] hard-only : acc={sa:.4f}  "
          f"({fp_bytes / s_bytes:.1f}x fewer bytes than teacher)")
    print(f"tiny student distilled from teacher : acc={da:.4f}  "
          f"(change vs hard {da - sa:+.4f})")
    print(f"-> a ~{fp_bytes / s_bytes:.0f}x smaller model keeps "
          f"{da / base_acc:.0%} of the teacher's accuracy.")

    print(report.section("Comparison", color))
    print(report.render_comparison(rows, base_acc, color=color))

    result = {
        "version": __version__,
        "seed": seed,
        "teacher_accuracy": base_acc,
        "quantization_error": {b: qr["error"] for b, (_, qr) in quant_models.items()},
        "tiny_student": {"hard_acc": sa, "distilled_acc": da,
                         "size_ratio": fp_bytes / s_bytes},
        "rows": rows,
    }
    if json_path:
        report.write_json(result, json_path)
        print(f"\nJSON report written to {json_path}")

    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        report.accuracy_bar_chart(
            [r["method"] for r in rows], [r["accuracy"] for r in rows],
            os.path.join(plots_dir, "accuracy.png"))
        orig = np.concatenate([l.W.reshape(-1) for l in teacher.linears])
        q8 = quant_models[8][1]
        deq = np.concatenate([lyr["dequant"].reshape(-1) for lyr in q8["layers"]])
        report.weight_histogram(orig, deq, os.path.join(plots_dir, "weight_hist.png"))
        report.curves_chart(
            {"student hard": s_hist["val_acc"],
             "student distilled": d_hist["val_acc"]},
            os.path.join(plots_dir, "distill_curves.png"),
            title="Validation accuracy during training")
        print(f"plots written to {plots_dir}/ (accuracy.png, weight_hist.png, distill_curves.png)")
    return result


# --------------------------------------------------------------------------- #
# Single-technique commands
# --------------------------------------------------------------------------- #
def cmd_quant(a):
    data = load_dataset(seed=a.seed)
    teacher = MLP([data["n_features"], 64, 64, data["n_classes"]], seed=1)
    teacher.fit(data["X_train"], data["y_train"], epochs=a.epochs)
    res = quantize_model(teacher, bits=a.bits, scheme=a.scheme,
                         granularity=a.granularity)
    print(f"INT{a.bits} {a.scheme} {a.granularity}: "
          f"acc={teacher.accuracy(data['X_test'], data['y_test']):.4f}  "
          f"cos={res['error']['cosine']:.6f}  mse={res['error']['mse']:.3e}")
    if a.json:
        report.write_json({k: v for k, v in res.items() if k != "layers"}, a.json)


def cmd_prune(a):
    data = load_dataset(seed=a.seed)
    teacher = MLP([data["n_features"], 64, 64, data["n_classes"]], seed=1)
    teacher.fit(data["X_train"], data["y_train"], epochs=a.epochs)
    res = prune_model(teacher, sparsity=a.sparsity, scope=a.scope,
                      structured=a.structured)
    print(f"{res['method']}: target={a.sparsity:.0%} "
          f"achieved={res['achieved_sparsity']:.2%} "
          f"acc={teacher.accuracy(data['X_test'], data['y_test']):.4f}")
    if a.json:
        report.write_json({k: v for k, v in res.items() if k != "masks"}, a.json)


def cmd_distill(a):
    data = load_dataset(seed=a.seed)
    f, k = data["n_features"], data["n_classes"]
    teacher = MLP([f, 64, 64, k], seed=1)
    teacher.fit(data["X_train"], data["y_train"], epochs=a.epochs)
    hard = MLP([f, a.hidden, k], seed=7)
    hard.fit(data["X_train"], data["y_train"], epochs=a.epochs)
    student = MLP([f, a.hidden, k], seed=7)
    distill(teacher, student, data["X_train"], data["y_train"],
            temperature=a.temperature, alpha=a.alpha, epochs=a.epochs)
    ha = hard.accuracy(data["X_test"], data["y_test"])
    da = student.accuracy(data["X_test"], data["y_test"])
    print(f"tiny student: hard={ha:.4f}  distilled={da:.4f}  change={da - ha:+.4f}")
    if a.json:
        report.write_json({"hard_acc": ha, "distilled_acc": da, "change": da - ha}, a.json)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(prog="modelshrink",
                                description="Lightweight neural-network compression toolbox")
    p.add_argument("--version", action="version", version=f"modelshrink {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="full quantization/pruning/distillation pipeline")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--teacher-epochs", type=int, default=200)
    d.add_argument("--student-epochs", type=int, default=260)
    d.add_argument("--sparsity", type=float, default=0.5)
    d.add_argument("--student-hidden", type=int, default=12)
    d.add_argument("--temperature", type=float, default=2.0)
    d.add_argument("--alpha", type=float, default=0.5)
    d.add_argument("--json", dest="json_path")
    d.add_argument("--plots", dest="plots_dir")
    d.add_argument("--no-color", action="store_true")

    q = sub.add_parser("quant", help="quantize a trained model")
    q.add_argument("--bits", type=int, default=8)
    q.add_argument("--scheme", choices=["symmetric", "asymmetric"], default="symmetric")
    q.add_argument("--granularity", choices=["per-tensor", "per-channel"], default="per-tensor")
    q.add_argument("--epochs", type=int, default=200)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--json")

    pr = sub.add_parser("prune", help="prune a trained model")
    pr.add_argument("--sparsity", type=float, default=0.5)
    pr.add_argument("--scope", choices=["global", "layer"], default="global")
    pr.add_argument("--structured", action="store_true")
    pr.add_argument("--epochs", type=int, default=200)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--json")

    di = sub.add_parser("distill", help="distill into a tiny student")
    di.add_argument("--hidden", type=int, default=12)
    di.add_argument("--temperature", type=float, default=2.0)
    di.add_argument("--alpha", type=float, default=0.5)
    di.add_argument("--epochs", type=int, default=200)
    di.add_argument("--seed", type=int, default=0)
    di.add_argument("--json")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        run_pipeline(
            seed=args.seed, teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs, sparsity=args.sparsity,
            student_hidden=args.student_hidden, temperature=args.temperature,
            alpha=args.alpha, json_path=args.json_path, plots_dir=args.plots_dir,
            color=not args.no_color)
    elif args.command == "quant":
        cmd_quant(args)
    elif args.command == "prune":
        cmd_prune(args)
    elif args.command == "distill":
        cmd_distill(args)


if __name__ == "__main__":
    main()
