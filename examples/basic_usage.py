"""End-to-end library example (no CLI needed).

Run from the repository root::

    python examples/basic_usage.py
"""
from modelshrink.data import load_dataset
from modelshrink.nn import MLP
from modelshrink.prune import prune_model
from modelshrink.quantize import quantize_model
from modelshrink.distill import distill


def main():
    data = load_dataset(seed=0)
    f, k = data["n_features"], data["n_classes"]

    # 1) A capacity-rich teacher.
    teacher = MLP([f, 64, 64, k], seed=1)
    teacher.fit(data["X_train"], data["y_train"], epochs=200)
    print("teacher accuracy:", teacher.accuracy(data["X_test"], data["y_test"]))

    # 2) Post-training INT8 quantization (fake quant).
    q8 = quantize_model(teacher, bits=8)
    print("INT8 weight cosine:", q8["error"]["cosine"])

    # 3) Magnitude pruning.
    prune_model(teacher, sparsity=0.5, scope="global")
    print("pruned accuracy:", teacher.accuracy(data["X_test"], data["y_test"]))

    # 4) Distill into a tiny student.
    big = MLP([f, 64, 64, k], seed=1)
    big.fit(data["X_train"], data["y_train"], epochs=200)
    student = MLP([f, 12, k], seed=7)
    distill(big, student, data["X_train"], data["y_train"],
            temperature=2.0, alpha=0.5, epochs=260)
    print("distilled tiny student:", student.accuracy(data["X_test"], data["y_test"]))


if __name__ == "__main__":
    main()
