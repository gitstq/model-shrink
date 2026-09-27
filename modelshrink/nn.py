"""A minimal, fully trainable MLP implemented in pure NumPy.

The network is intentionally small and explicit (forward / backward / Adam
step) so that ModelShrink can manipulate its weights directly for pruning,
fake-quantization and distillation. It supports per-layer binary prune masks
that are enforced both in the forward pass and on the gradient/update.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Numerical helpers
# --------------------------------------------------------------------------- #
def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=axis, keepdims=True))


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> float:
    n = logits.shape[0]
    return float(-log_softmax(logits)[np.arange(n), y].mean())


def cross_entropy_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of mean softmax cross-entropy w.r.t. the logits."""
    n = logits.shape[0]
    return (softmax(logits) - one_hot(y, logits.shape[1])) / n


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
class Linear:
    """A fully connected layer ``y = x @ W + b`` with an optional prune mask."""

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        self.in_features = in_features
        self.out_features = out_features
        # He / Kaiming initialisation (works well with ReLU).
        scale = np.sqrt(2.0 / in_features)
        self.W = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float64)
        self.b = np.zeros(out_features, dtype=np.float64)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.mask: np.ndarray | None = None  # 0/1, same shape as W

        # Adam state.
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        self._x: np.ndarray | None = None

    @property
    def effective_W(self) -> np.ndarray:
        return self.W if self.mask is None else self.W * self.mask

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self._x = x
        return x @ self.effective_W + self.b

    def backward(self, dz: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward called before a training forward pass")
        x = self._x
        # ``dz`` is the gradient of the MEAN loss, so it already carries the
        # 1/batch-size factor; accumulate over the batch without dividing again.
        self.dW = x.T @ dz
        self.db = dz.sum(axis=0)
        if self.mask is not None:
            self.dW *= self.mask
        return dz @ self.effective_W.T

    def adam_step(self, t: int, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        for p, g, m, v in (
            (self.W, self.dW, self.mW, self.vW),
            (self.b, self.db, self.mb, self.vb),
        ):
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
        # Keep pruned connections pinned to exactly zero.
        if self.mask is not None:
            self.W *= self.mask

    def num_params(self) -> int:
        return int(self.W.size + self.b.size)


class ReLU:
    def __init__(self):
        self._keep: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self._keep = (x > 0)
        return np.maximum(x, 0.0)

    def backward(self, dz: np.ndarray) -> np.ndarray:
        return dz * self._keep


class Tanh:
    def __init__(self):
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = np.tanh(x)
        if training:
            self._out = out
        return out

    def backward(self, dz: np.ndarray) -> np.ndarray:
        return dz * (1.0 - self._out ** 2)


_ACTIVATIONS = {"relu": ReLU, "tanh": Tanh}


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class MLP:
    """A simple feed-forward classifier with linear output logits."""

    def __init__(self, sizes, hidden_activation: str = "relu", seed: int = 0):
        if len(sizes) < 2:
            raise ValueError("sizes must contain at least an input and output size")
        if hidden_activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {hidden_activation!r}")
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.linears: list[Linear] = []
        self.layers: list = []
        act_cls = _ACTIVATIONS[hidden_activation]
        for i in range(len(sizes) - 1):
            lin = Linear(sizes[i], sizes[i + 1], rng)
            self.linears.append(lin)
            self.layers.append(lin)
            if i < len(sizes) - 2:
                self.layers.append(act_cls())
        self._t = 0

    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, dlogits: np.ndarray):
        for layer in reversed(self.layers):
            dlogits = layer.backward(dlogits)

    def step(self, lr: float):
        self._t += 1
        for lin in self.linears:
            lin.adam_step(self._t, lr)

    # ------------------------------------------------------------------ #
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x, training=False), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    @staticmethod
    def _cosine_lr(epoch: int, epochs: int, lr_max: float, lr_min: float) -> float:
        return lr_min + 0.5 * (lr_max - lr_min) * (
            1.0 + np.cos(np.pi * epoch / max(1, epochs))
        )

    def _clip_grads(self, grad_clip: float):
        """Clip all parameter gradients by their global L2 norm."""
        if grad_clip is None or grad_clip <= 0:
            return
        norm2 = 0.0
        for lin in self.linears:
            norm2 += float((lin.dW ** 2).sum()) + float((lin.db ** 2).sum())
        norm = np.sqrt(norm2)
        if norm > grad_clip:
            scale = grad_clip / norm
            for lin in self.linears:
                lin.dW *= scale
                lin.db *= scale

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
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
        """Train with mini-batch Adam, cosine LR decay and gradient clipping.

        Returns a history dict. The cosine schedule makes convergence stable
        and largely insensitive to the mini-batch order.
        """
        n = X.shape[0]
        rng = np.random.default_rng(shuffle_seed)
        history = {"loss": [], "lr": [], "val_acc": []}
        for epoch in range(epochs):
            idx = rng.permutation(n)
            losses = []
            lr_t = self._cosine_lr(epoch, epochs, lr, lr_min)
            for start in range(0, n, batch_size):
                bidx = idx[start : start + batch_size]
                xb, yb = X[bidx], y[bidx]
                logits = self.forward(xb, training=True)
                losses.append(cross_entropy(logits, yb))
                self.backward(cross_entropy_grad(logits, yb))
                self._clip_grads(grad_clip)
                self.step(lr_t)
            mean_loss = float(np.mean(losses))
            history["loss"].append(mean_loss)
            history["lr"].append(lr_t)
            if X_val is not None:
                history["val_acc"].append(self.accuracy(X_val, y_val))
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                msg = f"epoch {epoch:4d}  loss={mean_loss:.4f}  lr={lr_t:.2e}"
                if X_val is not None:
                    msg += f"  val_acc={history['val_acc'][-1]:.4f}"
                print(msg)
        return history

    # ------------------------------------------------------------------ #
    def count_parameters(self) -> int:
        return int(sum(lin.num_params() for lin in self.linears))
