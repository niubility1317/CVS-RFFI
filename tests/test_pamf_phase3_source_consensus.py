import copy
import importlib
import unittest

import numpy as np

from experiments.pamf_phase3_open_world import source_open_readout as core


class SourceConsensusTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_consensus')

    def geometry(self):
        branches = {}
        for name, dimension in core.DIMENSIONS.items():
            prototypes = np.zeros((6, dimension)); prototypes[np.arange(6), np.arange(6)] = 1.
            branches[name] = dict(mean=np.zeros(dimension), scale=np.ones(dimension), prototypes=prototypes, whitener=np.eye(dimension))
        return dict(schema=core.GEOMETRY_SCHEMA, class_count=6, branches=branches)

    def branches(self, count=3):
        rng = np.random.default_rng(5)
        branches = {name: rng.normal(size=(count, dimension)).astype(np.float32) for name, dimension in core.DIMENSIONS.items()}
        branches['reference29'][:, 27] = 0.
        return branches

    def test_class_joint_permutation_and_ties_are_exactly_invariant(self):
        module = self.module(); geometry = self.geometry(); branches = self.branches()
        logits = np.array([[2, 2, 0, -1, -1, 2], [0]*6, [-1000, 1000, 0, 0, 0, 1000]], dtype=np.float32)
        expected = module.descriptor(branches, logits, geometry)
        permutation = np.array([4, 2, 5, 0, 3, 1]); changed = copy.deepcopy(geometry)
        for value in changed['branches'].values(): value['prototypes'] = value['prototypes'][permutation]
        actual = module.descriptor(branches, logits[:, permutation], changed)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (3, 100)); self.assertEqual(actual.dtype, np.dtype('float32'))

    def test_same_sorted_evidence_but_different_head_support_is_distinguishable(self):
        module = self.module(); geometry = self.geometry()
        branches = {name: np.zeros((2, dimension), dtype=np.float32) for name, dimension in core.DIMENSIONS.items()}
        for value in branches.values(): value[0, 0] = 1.; value[1, 1] = 1.
        logits = np.tile(np.array([3., -1, -1, -1, -1, -1], dtype=np.float32), (2, 1))
        old = core.descriptor(branches, logits, geometry)
        np.testing.assert_array_equal(old[0], old[1])
        new = module.descriptor(branches, logits, geometry)
        self.assertNotEqual(new[0, 4], new[1, 4]); self.assertNotEqual(new[0, 5], new[1, 5])
        probability = np.exp(logits[0].astype(np.float64) - 3); probability /= probability.sum()
        np.testing.assert_allclose(new[0, :6], [1., 0., 1/6, np.std([1., 0, 0, 0, 0, 0]), probability[0], 1-probability[0]], atol=1e-7)

    def test_population_stats_globals_and_row_locality(self):
        module = self.module(); branches = self.branches(4); geometry = self.geometry()
        logits = np.arange(24, dtype=np.float32).reshape(4, 6) / 3
        together = module.descriptor(branches, logits, geometry)
        old = core.descriptor(branches, logits, geometry)
        np.testing.assert_array_equal(together[:, -4:], old[:, -4:])
        for row in range(4):
            alone = module.descriptor({key: value[row:row+1] for key, value in branches.items()}, logits[row:row+1], geometry)
            np.testing.assert_allclose(alone[0], together[row], atol=1e-6, rtol=1e-6)

    def test_zero_branch_nonfinite_shape_and_cfo_boundaries(self):
        module = self.module(); geometry = self.geometry()
        branches = {name: np.zeros((2, dimension), dtype=np.float32) for name, dimension in core.DIMENSIONS.items()}
        result = module.descriptor(branches, np.zeros((2, 6), dtype=np.float32), geometry)
        self.assertTrue(np.isfinite(result).all())
        for fault in ['shape', 'nonfinite', 'CFO']:
            with self.subTest(fault=fault):
                changed = copy.deepcopy(branches)
                if fault == 'shape': changed['fused160'] = changed['fused160'][:, :-1]
                elif fault == 'nonfinite': changed['time160'][0, 0] = np.nan
                else: changed['reference29'][0, 27] = .1
                with self.assertRaises(ValueError): module.descriptor(changed, np.zeros((2, 6), dtype=np.float32), geometry)


if __name__ == '__main__': unittest.main()
