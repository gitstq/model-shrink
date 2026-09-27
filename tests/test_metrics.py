import unittest

import numpy as np

from modelshrink import metrics


class TestMetrics(unittest.TestCase):
    def test_identical_tensors(self):
        w = np.random.default_rng(0).normal(size=(20, 10))
        err = metrics.tensor_error(w, w)
        self.assertAlmostEqual(err["mse"], 0.0, places=12)
        self.assertAlmostEqual(err["cosine"], 1.0, places=10)
        self.assertEqual(err["max_abs"], 0.0)

    def test_cosine_known(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(metrics.cosine_similarity(a, b), 0.0)
        self.assertAlmostEqual(metrics.cosine_similarity(a, a), 1.0)

    def test_sparsity(self):
        w = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]])
        self.assertAlmostEqual(metrics.sparsity(w), 0.5)

    def test_dense_and_ratio(self):
        self.assertAlmostEqual(metrics.dense_bytes(100, 32), 400.0)
        self.assertAlmostEqual(metrics.dense_bytes(100, 8), 100.0)
        self.assertAlmostEqual(metrics.compression_ratio(400.0, 100.0), 4.0)

    def test_sparse_storage_monotone(self):
        # Fewer nonzeros -> less storage; indices add overhead vs packed ints.
        full = metrics.sparse_coo_bytes(100, 100, value_bits=8)
        half = metrics.sparse_coo_bytes(50, 100, value_bits=8)
        self.assertLess(half, full)
        self.assertGreater(half, 0)


if __name__ == "__main__":
    unittest.main()
