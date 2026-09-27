import unittest

import numpy as np

from modelshrink.data import load_dataset
from modelshrink.nn import MLP
from modelshrink.prune import (channel_prune_mask, prune_model, reset_masks)


class TestPrune(unittest.TestCase):
    def setUp(self):
        self.weights = [np.random.default_rng(i).normal(size=(30, 20)) for i in range(3)]

    def test_layer_scope_sparsity(self):
        model = MLP([10, 20, 5], seed=0)
        res = prune_model(model, sparsity=0.6, scope="layer")
        for layer in res["layers"]:
            self.assertAlmostEqual(layer["sparsity"], 0.6, places=6)

    def test_global_scope_sparsity(self):
        model = MLP([10, 20, 20, 5], seed=0)
        res = prune_model(model, sparsity=0.5, scope="global")
        self.assertAlmostEqual(res["achieved_sparsity"], 0.5, places=6)

    def test_masks_are_binary(self):
        model = MLP([10, 20, 5], seed=0)
        prune_model(model, sparsity=0.4)
        for lin in model.linears:
            unique = np.unique(lin.mask)
            self.assertTrue(set(unique.tolist()).issubset({0.0, 1.0}))

    def test_zeros_survive_training(self):
        d = load_dataset(n_train=300, n_test=100, seed=0)
        model = MLP([16, 32, 10], seed=1)
        model.fit(d["X_train"], d["y_train"], epochs=60)
        prune_model(model, sparsity=0.5)
        # Further training must keep pruned connections pinned to zero.
        model.fit(d["X_train"], d["y_train"], epochs=10)
        for lin in model.linears:
            self.assertTrue(np.all(lin.W[lin.mask == 0] == 0.0))

    def test_structured_channel_mask(self):
        W = self.weights[0]
        mask = channel_prune_mask(W, prune_fraction=0.25, channel_axis=1)
        # 25% of the 20 output columns (=5) are fully zero.
        column_alive = mask[0]
        self.assertEqual(int(np.sum(column_alive == 0)), 5)
        # Every removed column is zero for all input rows.
        self.assertTrue(np.all(mask[:, column_alive == 0] == 0.0))

    def test_reset_masks(self):
        model = MLP([10, 20, 5], seed=0)
        prune_model(model, sparsity=0.5)
        reset_masks(model)
        self.assertTrue(all(lin.mask is None for lin in model.linears))


if __name__ == "__main__":
    unittest.main()
