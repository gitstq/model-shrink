import unittest

import numpy as np

from modelshrink.data import load_dataset
from modelshrink.nn import MLP
from modelshrink.distill import distill


class TestDistill(unittest.TestCase):
    def setUp(self):
        self.d = load_dataset(seed=0)
        self.f = self.d["n_features"]
        self.k = self.d["n_classes"]

    def test_alpha_zero_reproduces_hard_training(self):
        d, f, k = self.d, self.f, self.k
        teacher = MLP([f, 64, 64, k], seed=1)
        teacher.fit(d["X_train"], d["y_train"], epochs=80)
        hard = MLP([f, 12, k], seed=7)
        hard.fit(d["X_train"], d["y_train"], epochs=80)
        via_distill = MLP([f, 12, k], seed=7)
        distill(teacher, via_distill, d["X_train"], d["y_train"],
                alpha=0.0, epochs=80)
        # Same init, shuffle and schedule -> identical trajectory.
        for lh, ld in zip(hard.linears, via_distill.linears):
            np.testing.assert_allclose(lh.W, ld.W, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(lh.b, ld.b, atol=1e-10, rtol=1e-10)

    def test_distilled_tiny_student_not_worse(self):
        d, f, k = self.d, self.f, self.k
        teacher = MLP([f, 64, 64, k], seed=1)
        teacher.fit(d["X_train"], d["y_train"], epochs=160)
        hard = MLP([f, 12, k], seed=7)
        hard.fit(d["X_train"], d["y_train"], epochs=200)
        student = MLP([f, 12, k], seed=7)
        distill(teacher, student, d["X_train"], d["y_train"],
                temperature=2.0, alpha=0.5, epochs=200)
        ha = hard.accuracy(d["X_test"], d["y_test"])
        sa = student.accuracy(d["X_test"], d["y_test"])
        # Distillation must not degrade the tiny student; it typically lets a
        # ~18x smaller model keep ~97% of the teacher's accuracy.
        self.assertGreaterEqual(sa, ha - 0.02, f"hard={ha}, distill={sa}")
        self.assertGreaterEqual(sa, teacher.accuracy(d["X_test"], d["y_test"]) - 0.06)


if __name__ == "__main__":
    unittest.main()
