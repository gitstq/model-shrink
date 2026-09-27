import unittest

import numpy as np

from modelshrink.data import load_dataset
from modelshrink.nn import (MLP, cross_entropy, cross_entropy_grad, softmax)


class TestNN(unittest.TestCase):
    def test_softmax_normalizes(self):
        z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        p = softmax(z)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(2), atol=1e-12)
        self.assertTrue(np.all(p >= 0))
        np.testing.assert_allclose(p[1], np.ones(3) / 3)

    def test_network_learns(self):
        d = load_dataset(n_train=400, n_test=200, seed=0)
        model = MLP([16, 48, 32, 10], seed=1)
        model.fit(d["X_train"], d["y_train"], epochs=120)
        self.assertGreater(model.accuracy(d["X_test"], d["y_test"]), 0.85)

    def test_backprop_matches_numerical_gradient(self):
        model = MLP([3, 5, 4], seed=2)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 3))
        y = rng.integers(0, 4, size=6)

        def compute_loss():
            logits = model.forward(X, training=True)
            return cross_entropy(logits, y)

        # Analytic gradients.
        logits = model.forward(X, training=True)
        model.backward(cross_entropy_grad(logits, y))

        eps = 1e-6
        for lin in model.linears:
            for pname, gname in (("W", "dW"), ("b", "db")):
                P = getattr(lin, pname)
                G = getattr(lin, gname)
                for idx in np.ndindex(P.shape):
                    saved = P[idx]
                    P[idx] = saved + eps
                    loss_plus = compute_loss()
                    P[idx] = saved - eps
                    loss_minus = compute_loss()
                    P[idx] = saved
                    numerical = (loss_plus - loss_minus) / (2 * eps)
                    analytic = G[idx]
                    scale = max(1e-8, abs(numerical), abs(analytic))
                    self.assertLess(abs(numerical - analytic) / scale, 1e-4,
                                    f"grad mismatch at {pname}{idx}")


if __name__ == "__main__":
    unittest.main()
