from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import time
import unittest

import numpy as np

from experiments.pamf_phase3_open_world import source_open_readout as source
from experiments.pamf_phase3_open_world.source_risk_calibration import METHODS


class TargetOrthogonalAlignmentTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/target_orthogonal_alignment.py'
        self.assertTrue(path.is_file(), 'Bounded target orthogonal alignment is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.target_orthogonal_alignment')

    def fixture(self, counts=(8, 8, 8), classes=(0, 1, 2), *, angle=.22, seed=7):
        rng = np.random.default_rng(seed)
        labels = np.concatenate([np.full(count, label, np.int64) for count, label in zip(counts, classes)])
        geometry = dict(schema=source.GEOMETRY_SCHEMA, class_count=6, branches={})
        branches = {}
        for name, dimension in source.DIMENSIONS.items():
            mean = np.zeros(dimension) if name == 'fused160' else np.round(rng.uniform(-.3, .3, dimension) * 16) / 16
            scale = np.ones(dimension) if name == 'fused160' else np.round(rng.uniform(.7, 1.4, dimension) * 16) / 16
            centers = np.eye(6, dimension, dtype=np.float64)
            q = np.eye(dimension)
            q[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            unit = centers[labels] @ q + rng.normal(0., .002, (len(labels), dimension))
            unit /= np.linalg.norm(unit, axis=1, keepdims=True)
            raw = mean + scale * unit * rng.uniform(.5, 2., (len(labels), 1))
            if name == 'reference29':
                mean[27] = 0.
                scale[27] = 1.
                raw[:, 27] = 0.
            branches[name] = raw.astype(np.float32)
            geometry['branches'][name] = dict(mean=mean, scale=scale, prototypes=centers,
                whitener=np.diag(np.linspace(1., 3., dimension)))
        logits = np.zeros((len(labels), 6), np.float32)
        logits[np.arange(len(labels)), labels] = 4.
        scores = np.zeros((len(labels), 7), np.float64)
        scores[:, 0] = rng.uniform(.85, .99, len(labels))
        scores[:, 4] = .95
        scores[:, 6] = -1e6
        thresholds = np.asarray([.8, 1e6, 1e6, 1e6, .5, 1e6, 1e6], np.float64)
        return branches, logits, scores, thresholds, geometry, labels

    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, np.ndarray):
            self.assertTrue(np.array_equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected: self.assert_tree_equal(actual[key], expected[key])
        else:
            self.assertEqual(actual, expected)

    def test_core_uses_static_gates_head_fused_and_two_votes_without_MLP(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture()
        # Six additional class0 rows independently fail each stated gate.
        for name in branches: branches[name] = np.concatenate((branches[name], np.repeat(branches[name][:1], 6, axis=0)))
        logits = np.concatenate((logits, np.repeat(logits[:1], 6, axis=0)))
        scores = np.concatenate((scores, np.repeat(scores[:1], 6, axis=0)))
        logits[24, 1] = logits[24, 0]  # head tie
        branches['fused160'][25] = 0.
        branches['fused160'][25, :2] = 1.  # fused nearest tie
        for name in ('time160', 'frequency160'):
            g = geometry['branches'][name]
            branches[name][26] = g['mean'] + g['scale'] * g['prototypes'][1]
        scores[27, 0] = thresholds[0] - 1e-4
        scores[28, 4] = thresholds[4] - 1e-4
        for name in ('frequency160', 'pa160'):
            g = geometry['branches'][name]
            branches[name][29] = g['mean'] + g['scale'] * (g['prototypes'][0] + g['prototypes'][1])
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        self.assertTrue(state['applied'])
        self.assertTrue(state['core_mask'][:24].all())
        self.assertFalse(state['core_mask'][24:].any())
        self.assertEqual(state['selected_indices'].tolist(), list(range(24)))
        self.assertTrue(bool((scores[:, METHODS.index('mlp16_oe')] < thresholds[METHODS.index('mlp16_oe')]).all()))
        self.assertEqual(state['pseudo_label_counts'].tolist(), [8, 8, 8, 0, 0, 0])

    def test_present_classes_equal_weight_without_six_class_quota(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture((8, 10, 9), (0, 2, 5))
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        self.assertTrue(state['applied'])
        self.assertEqual(state['selected_class_counts'].tolist(), [8, 0, 10, 0, 0, 9])
        self.assertTrue(np.array_equal(state['selected_indices'], np.arange(27)))
        self.assertTrue(np.allclose(state['class_weights'], [1/3, 0, 1/3, 0, 0, 1/3]))

    def test_rotation_matches_weighted_regularized_polar_with_row_vector_direction(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture((8, 9, 10))
        # One PA disagreement still leaves two valid votes, reducing this
        # sample's weight without excluding it from class0.
        g = geometry['branches']['pa160']
        branches['pa160'][0] = g['mean'] + g['scale'] * g['prototypes'][1]
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        unit = source._unit(branches['fused160'].astype(np.float64))
        weights = scores[:, 0].copy()
        weights[0] *= 2 / 3
        means = np.asarray([np.sum(unit[labels == label] * (weights[labels == label] / weights[labels == label].sum())[:, None], axis=0)
                            for label in (0, 1, 2)])
        means = source._unit(means)
        prototypes = geometry['branches']['fused160']['prototypes'][:3]
        c = means.T @ prototypes / 3
        u, singular, vt = np.linalg.svd(np.eye(160) + .5 * c, full_matrices=False)
        expected = u @ vt
        r = state['rotations']['fused160']
        self.assertTrue(np.allclose(r, expected, rtol=0., atol=2e-12))
        self.assertLess(np.linalg.norm(means @ r - prototypes), np.linalg.norm(means - prototypes))
        self.assertGreater(np.linalg.norm(r - r.T), .001)
        self.assertTrue(np.all(singular > .49))

    def test_class_column_permutation_preserves_core_and_rotations(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture((8, 10, 9), (0, 2, 5))
        first = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        permutation = np.asarray([5, 0, 3, 2, 1, 4])
        changed = deepcopy(geometry)
        for branch in changed['branches'].values(): branch['prototypes'] = branch['prototypes'][permutation].copy()
        second = module.fit_alignment(branches, logits[:, permutation].copy(), scores, thresholds, changed)
        self.assertTrue(np.array_equal(first['core_mask'], second['core_mask']))
        self.assertTrue(np.array_equal(first['selected_indices'], second['selected_indices']))
        self.assertTrue(np.array_equal(second['pseudo_label_counts'], first['pseudo_label_counts'][permutation]))
        for name in source.DIMENSIONS:
            self.assertTrue(np.allclose(first['rotations'][name], second['rotations'][name], atol=3e-12, rtol=0.))

    def test_apply_preserves_standardized_norm_cosine_distance_and_masks_CFO(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture()
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        # Query can have a different count from the fit role.
        query = {name: value[:7].copy() for name, value in branches.items()}
        adapted = module.apply_alignment(query, state, geometry)
        mahalanobis_changed = False
        for name, dimension in source.DIMENSIONS.items():
            r = state['rotations'][name]
            self.assertTrue(np.allclose(r.T @ r, np.eye(dimension), atol=5e-12, rtol=0.))
            g = geometry['branches'][name]
            before = (query[name].astype(np.float64) - g['mean']) / g['scale']
            after = (adapted[name].astype(np.float64) - g['mean']) / g['scale']
            self.assertEqual(adapted[name].dtype, np.float32)
            self.assertTrue(np.allclose(np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1), atol=2e-6))
            self.assertTrue(np.allclose(source._unit(before) @ source._unit(before).T,
                                        source._unit(after) @ source._unit(after).T, atol=2e-6))
            self.assertTrue(np.allclose(np.linalg.norm(before[1:] - before[0], axis=1),
                                        np.linalg.norm(after[1:] - after[0], axis=1), atol=2e-6))
            old_white = (before[1:] - before[0]) @ g['whitener'].T
            new_white = (after[1:] - after[0]) @ g['whitener'].T
            mahalanobis_changed |= not np.allclose(np.linalg.norm(old_white, axis=1), np.linalg.norm(new_white, axis=1), atol=1e-6)
        self.assertTrue(mahalanobis_changed, 'Whitened Mahalanobis preservation must not be claimed')
        self.assertTrue(bool((adapted['reference29'][:, 27] == 0).all()))
        expected = np.eye(29)[27]
        self.assertTrue(np.array_equal(state['rotations']['reference29'][27], expected))
        self.assertTrue(np.array_equal(state['rotations']['reference29'][:, 27], expected))
        self.assertTrue(np.allclose(adapted['fused160'], query['fused160'] @ state['rotations']['fused160'], atol=1e-7))

    def test_insufficient_core_or_classes_returns_exact_copies(self):
        module = self.module()
        for counts, classes in [((7, 8, 8), (0, 1, 2)), ((12, 12), (0, 5)), ((3, 4, 7), (0, 1, 2))]:
            with self.subTest(counts=counts):
                branches, logits, scores, thresholds, geometry, labels = self.fixture(counts, classes)
                state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
                self.assertFalse(state['applied'])
                adapted = module.apply_alignment(branches, state, geometry)
                for name in source.DIMENSIONS:
                    self.assertTrue(np.array_equal(adapted[name], branches[name]))
                    self.assertFalse(np.shares_memory(adapted[name], branches[name]))
                    self.assertTrue(np.array_equal(state['rotations'][name], np.eye(source.DIMENSIONS[name])))

    def test_degenerate_auxiliary_branch_stays_identity_and_fused_failure_disables_all(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture()
        g = geometry['branches']['pa160']
        branches['pa160'][:] = g['mean']
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        self.assertTrue(state['applied'])
        self.assertFalse(state['diagnostics']['branches']['pa160']['fitted'])
        self.assertEqual(state['diagnostics']['branches']['pa160']['valid_core_classes'], 0)
        self.assertTrue(np.array_equal(state['rotations']['pa160'], np.eye(160)))
        adapted = module.apply_alignment(branches, state, geometry)
        self.assertTrue(np.array_equal(adapted['pa160'], branches['pa160']))
        branches, logits, scores, thresholds, geometry, labels = self.fixture()
        geometry['branches']['fused160']['prototypes'][0] = 0.
        branches['fused160'][labels == 0] = 0.
        branches['fused160'][labels == 0, 1:6] = -1.
        thresholds[4] = 0.
        scores[labels == 0, 4] = 0.
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        self.assertEqual(state['selected_class_counts'][:3].tolist(), [8, 8, 8])
        self.assertFalse(state['applied'])
        self.assertEqual(state['reason'], 'insufficient_nondegenerate_fused_classes')
        self.assertTrue(all(np.array_equal(value, np.eye(value.shape[0])) for value in state['rotations'].values()))

    def test_near_ties_do_not_vote_and_never_enter_head_or_fused_core(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture()
        logits[0, 1] = logits[0, 0]  # FP32 exact tie
        branches['fused160'][1] = 0.
        branches['fused160'][1, :2] = 1.
        # A tied third branch contributes no vote but the two clean votes
        # remain sufficient; these records still count toward class1.
        for name in ('pa160',):
            g = geometry['branches'][name]
            branches[name][8:16] = g['mean'] + g['scale'] * (g['prototypes'][1] + g['prototypes'][3])
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        self.assertFalse(state['core_mask'][0])
        self.assertFalse(state['core_mask'][1])
        self.assertTrue(state['core_mask'][8:16].all())
        self.assertEqual(state['selected_class_counts'][0], 0)

    def test_input_contracts_no_truth_and_no_mutation(self):
        module = self.module()
        self.assertEqual(list(inspect.signature(module.fit_alignment).parameters),
                         ['branches', 'logits', 'scores', 'thresholds', 'geometry'])
        branches, logits, scores, thresholds, geometry, labels = self.fixture()
        original = deepcopy(dict(branches=branches, logits=logits, scores=scores, thresholds=thresholds, geometry=geometry))
        for value in branches.values(): value.flags.writeable = False
        for branch in geometry['branches'].values():
            for value in branch.values(): value.flags.writeable = False
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        saved = deepcopy(state)
        module.apply_alignment(branches, state, geometry)
        self.assert_tree_equal(state, saved)
        self.assert_tree_equal(dict(branches=branches, logits=logits, scores=scores, thresholds=thresholds, geometry=geometry), original)
        mutations = [dict(logits=logits[:, :5]), dict(scores=scores[:, :6]), dict(thresholds=thresholds.astype(np.float32)),
            dict(branches=dict(branches, y=labels)), dict(branches=dict(branches, fused160=branches['fused160'].astype(np.float64)))]
        for change in mutations:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                module.fit_alignment(**dict(original, **change))
        bad = scores.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError): module.fit_alignment(branches, logits, bad, thresholds, geometry)
        with self.assertRaises(TypeError): module.fit_alignment(branches, logits, scores, thresholds, geometry, y=labels)

    def test_aggregate_state_has_only_rotations_and_aggregate_diagnostics(self):
        module = self.module()
        branches, logits, scores, thresholds, geometry, labels = self.fixture()
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        aggregate = module.aggregate_state(state)
        for key in ('core_mask', 'selected_indices', 'ids', 'features', 'logits', 'scores'):
            self.assertNotIn(key, aggregate)
        self.assertEqual(sum(value.nbytes for value in aggregate['rotations'].values()), 1878600)
        self.assertTrue(all(not np.shares_memory(value, state['rotations'][name]) for name, value in aggregate['rotations'].items()))
        full = module.apply_alignment(branches, state, geometry)
        compact = module.apply_alignment(branches, aggregate, geometry)
        self.assert_tree_equal(full, compact)

    def test_real_eight_branch_geometry_API_on_small_synthetic_source_and_U(self):
        module = self.module()
        rng = np.random.default_rng(101)
        labels = np.repeat(np.arange(6), 8)
        branches = {}
        for name, dimension in source.DIMENSIONS.items():
            value = (5 * np.eye(6, dimension)[labels] + rng.normal(0., .03, (48, dimension))).astype(np.float32)
            if name == 'reference29': value[:, 27] = 0.
            branches[name] = value
        geometry = source.fit_geometry(branches, labels)
        logits = np.eye(6, dtype=np.float32)[labels] * 5
        scores = np.zeros((48, 7), np.float64); scores[:, 0] = .99; scores[:, 4] = .99
        thresholds = np.zeros(7, np.float64); thresholds[0] = .8; thresholds[4] = .5
        started = time.perf_counter()
        state = module.fit_alignment(branches, logits, scores, thresholds, geometry)
        adapted = module.apply_alignment(branches, state, geometry)
        elapsed = time.perf_counter() - started
        self.assertTrue(state['applied'])
        self.assertEqual(state['selected_class_counts'].tolist(), [8] * 6)
        source._branches(adapted)
        source._geometry(geometry)
        self.assertTrue(all(state['diagnostics']['branches'][name]['fitted'] for name in source.DIMENSIONS))
        print('SYNTHETIC_CPU_ALIGNMENT', dict(rows=48, dimensions=dict(source.DIMENSIONS),
            rotation_bytes=sum(value.nbytes for value in state['rotations'].values()), elapsed_seconds=elapsed))


if __name__ == '__main__':
    unittest.main()
