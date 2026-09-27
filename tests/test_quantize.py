import unittest

import numpy as np

from modelshrink.nn import MLP
from modelshrink.quantize import quantize_model, quantize_tensor


class TestQuantize(unittest.TestCase):
    def setUp(self):
        self.w = np.random.default_rng(1).normal(size=(40, 20))

    def test_int8_high_fidelity(self):
        res = quantize_tensor(self.w, bits=8)
        self.assertGreater(res["error"]["cosine"], 0.999)
        self.assertLess(res["error"]["mse"], 1e-3)

    def test_fewer_bits_more_error(self):
        c8 = quantize_tensor(self.w, bits=8)["error"]["cosine"]
        c4 = quantize_tensor(self.w, bits=4)["error"]["cosine"]
        c2 = quantize_tensor(self.w, bits=2)["error"]["cosine"]
        self.assertGreaterEqual(c8, c4)
        self.assertGreaterEqual(c4, c2)

    def test_symmetric_zero_preserved_and_bounded(self):
        w = self.w.copy()
        w[0, 0] = 0.0
        res = quantize_tensor(w, bits=8, scheme="symmetric")
        self.assertAlmostEqual(res["dequant"][0, 0], 0.0, places=12)
        # Dequantized values stay within the extended integer grid.
        self.assertTrue(np.all(res["dequant"] <= np.abs(self.w).max() + 1e-6))

    def test_asymmetric_shifted_distribution(self):
        wp = np.abs(self.w) + 3.0  # strictly positive
        res = quantize_tensor(wp, bits=8, scheme="asymmetric")
        self.assertGreater(res["error"]["cosine"], 0.999)
        self.assertGreaterEqual(res["zero_point"].item(), 0)

    def test_per_channel(self):
        res = quantize_tensor(self.w, bits=8, granularity="per-channel", channel_axis=1)
        self.assertGreater(res["error"]["cosine"], 0.999)
        # One scale per output channel (columns of (40, 20)).
        self.assertEqual(res["scale"].shape[-1], 20)

    def test_quantize_model_int8_preserves_accuracy(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(300, 10)), rng.integers(0, 5, size=300)
        model = MLP([10, 32, 5], seed=1)
        model.fit(X, y, epochs=80)
        before = model.accuracy(X, y)
        out = quantize_model(model, bits=8)
        after = model.accuracy(X, y)
        self.assertEqual(len(out["layers"]), 2)
        self.assertGreaterEqual(after, before - 0.03)

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            quantize_tensor(self.w, bits=1)


if __name__ == "__main__":
    unittest.main()
