"""Knowledge distillation (teacher → student) in pure NumPy.

The student is trained on a weighted mix of:

* **soft targets** — the teacher's temperature-softened class probabilities,
  which carry inter-class similarity ("dark knowledge"); and
* **hard targets** — the true labels.

The combined loss is::

    L = alpha * T^2 * KL(teacher_soft || student_soft) + (1 - alpha) * CE(y, student)

The factor ``T^2`` keeps the magnitude of the soft and hard gradients
comparable as the temperature changes. We derive the gradient analytically and
feed it straight into the student's existing backward pass.
"""
from __future__ import annotations

import numpy as np

from .nn import cross_entropy, log_softmax, one_hot, softmax


def distill(
    teacher,
    student,
    X: np.ndarray,
    y: np.ndarray,
    temperature: float = 4.0,
    alpha: float = 0.5,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    lr_min: float = 1e-4,
    grad_clip: float = 1.0,
    shuffle_seed: int = 12345,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    verbose: bool = False,
) -> dict:
    """Distill a frozen ``teacher`` into ``student``. Returns a history dict."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    T = float(temperature)
    n = X.shape[0]
    # Same shuffle seed as MLP.fit -> fair, reproducible hard-vs-soft comparison.
    rng = np.random.default_rng(shuffle_seed)
    history = {"loss": [], "soft_loss": [], "hard_loss": [], "lr": [], "val_acc": []}

    for epoch in range(epochs):
        idx = rng.permutation(n)
        losses, softs, hards = [], [], []
        lr_t = student._cosine_lr(epoch, epochs, lr, lr_min)
        for start in range(0, n, batch_size):
            bidx = idx[start : start + batch_size]
            xb, yb = X[bidx], y[bidx]
            bn = xb.shape[0]

            # Frozen teacher soft targets.
            t_logits = teacher.forward(xb, training=False)
            q = softmax(t_logits / T)

            # Student forward at temperature T and at T=1.
            s_logits = student.forward(xb, training=True)
            p = softmax(s_logits / T)
            hard_p = softmax(s_logits)

            log_p_t = log_softmax(s_logits / T)
            soft_loss = float(-np.mean(np.sum(q * log_p_t, axis=1)))
            hard_loss = cross_entropy(s_logits, yb)
            loss = alpha * T * T * soft_loss + (1 - alpha) * hard_loss

            # Analytic gradient w.r.t. student logits.
            d_soft = T * (p - q) / bn
            d_hard = (hard_p - one_hot(yb, hard_p.shape[1])) / bn
            d_logits = alpha * d_soft + (1 - alpha) * d_hard

            student.backward(d_logits)
            student._clip_grads(grad_clip)
            student.step(lr_t)

            losses.append(loss)
            softs.append(soft_loss)
            hards.append(hard_loss)

        history["loss"].append(float(np.mean(losses)))
        history["soft_loss"].append(float(np.mean(softs)))
        history["hard_loss"].append(float(np.mean(hards)))
        history["lr"].append(lr_t)
        if X_val is not None:
            history["val_acc"].append(student.accuracy(X_val, y_val))
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            msg = f"distill epoch {epoch:4d}  loss={history['loss'][-1]:.4f}"
            if X_val is not None:
                msg += f"  val_acc={history['val_acc'][-1]:.4f}"
            print(msg)
    return history
