import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np


D = 160


class StreamGeometryTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.stream_geometry')

    def bundle(self):
        prototypes = np.zeros((2, D))
        prototypes[:, 0] = [1., -1.]
        return dict(schema='phase3_stream_geometry_v1', feature_dim=D, class_count=2,
            source_count=40, source_counts=np.array([20, 20]), calibration_count=20,
            prototypes=prototypes, covariance=np.eye(D), chol=np.eye(D),
            whitener=np.eye(D), anchors=prototypes.copy(), lambda_det=.04, lambda_cluster=.01)

    def stream(self, *, new=True, contaminants=0):
        rng = np.random.default_rng(100)
        old = np.repeat(self.bundle()['prototypes'], 48, axis=0)
        old[:, 1] += .08
        old += rng.normal(0., .0003, old.shape)
        pseudo = np.repeat([0, 1], 48)
        rows = [old]
        if new:
            discovered = np.zeros((32, D)); discovered[:, 2] = 1.; discovered[:, 1] = .08
            discovered += rng.normal(0., .0003, discovered.shape)
            rows.append(discovered)
            pseudo = np.r_[pseudo, np.zeros(32, dtype=int)]
        if contaminants:
            unknown = np.zeros((contaminants, D)); unknown[:, 3] = 1.
            rows.append(unknown)
            pseudo = np.r_[pseudo, np.zeros(contaminants, dtype=int)]
        features = np.concatenate(rows)
        logits = np.zeros((len(features), 2)); logits[np.arange(len(features)), pseudo] = 1.
        return features, logits, np.asarray(['opaque-%04d' % i for i in range(len(features))])

    def test_source_aggregates_use_only_train_and_label_permutation(self):
        module = self.module()
        rng = np.random.default_rng(7)
        centers = self.bundle()['prototypes']
        train = np.repeat(centers, 30, axis=0) + rng.normal(0., .02, (60, D))
        y = np.repeat([0, 1], 30)
        calibration = np.repeat(centers, 20, axis=0) + rng.normal(0., .025, (40, D))
        original = train.copy()
        a = module.fit_source(train, y, calibration)
        b = module.fit_source(train, y, calibration * 3 + .1)
        for key in ('prototypes', 'covariance', 'chol', 'whitener', 'anchors', 'source_counts'):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertNotEqual(a['lambda_det'], b['lambda_det'])
        self.assertEqual(a['lambda_cluster'], a['lambda_det'] / 4)
        self.assertTrue(np.array_equal(train, original))
        self.assertNotIn('train_features', a)
        self.assertNotIn('calibration_features', a)
        permuted = module.fit_source(train, 1 - y, calibration)
        np.testing.assert_allclose(permuted['anchors'], a['anchors'][::-1], atol=1e-10)
        np.testing.assert_allclose(permuted['covariance'], a['covariance'], atol=1e-14)
        self.assertAlmostEqual(permuted['lambda_det'], a['lambda_det'], places=8)

    def test_common_shift_discovery_registration_and_future_unknown(self):
        module = self.module(); bundle = self.bundle()
        features, logits, ids = self.stream()
        state = module.fit_stream(bundle, features, logits, ids, adapt_domain=True)
        self.assertTrue(state['converged'])
        self.assertFalse(state['cap_reached'])
        self.assertTrue(state['promotion_eligible'])
        self.assertGreater(state['b'][1], .06)
        self.assertLess(state['b'][1], .081)
        self.assertEqual(int(state['qualified'].sum()), 1)
        np.testing.assert_array_equal(state['old_centers'], bundle['anchors'])
        query = np.zeros((3, D)); query[0, 0] = 1.; query[1, 2] = 1.; query[2, 4] = 1.; query[:, 1] = .08
        closed = np.array([1, 0, 0])  # Old winner must retain the actual head prediction.
        r0 = module.predict_geometry(query, closed, bundle, state, registered=False)
        r1 = module.predict_geometry(query, closed, bundle, state, registered=True)
        np.testing.assert_array_equal(r0['prediction'], [1, -1, -1])
        np.testing.assert_array_equal(r0['accepted'], [True, False, False])
        np.testing.assert_array_equal(r1['prediction'], [1, 2, -1])
        np.testing.assert_array_equal(r1['accepted'], [True, True, False])
        self.assertTrue(r1['nearest_is_new'][1])
        f = module.predict_geometry(query, closed, bundle, registered=False)
        self.assertEqual(f['prediction'][1], -1)
        with self.assertRaises(ValueError):
            module.predict_geometry(query, closed, bundle, registered=True)

    def test_u_permutation_and_source_label_permutation_are_equivariant(self):
        module = self.module(); bundle = self.bundle()
        f, logits, ids = self.stream()
        a = module.fit_stream(bundle, f, logits, ids, adapt_domain=True)
        permutation = np.random.default_rng(19).permutation(len(f))
        b = module.fit_stream(bundle, f[permutation], logits[permutation], ids[permutation], adapt_domain=True)
        for key in ('b', 'free_centers', 'free_ids', 'qualified', 'anonymous_ids', 'assignments', 'U_ids'):
            np.testing.assert_array_equal(a[key], b[key])
        relabeled = copy.deepcopy(bundle)
        for key in ('anchors', 'prototypes', 'source_counts'):
            relabeled[key] = relabeled[key][::-1].copy()
        c = module.fit_stream(relabeled, f, logits[:, ::-1], ids, adapt_domain=True)
        np.testing.assert_allclose(c['b'], a['b'], atol=1e-14)
        np.testing.assert_allclose(c['free_centers'], a['free_centers'], atol=1e-14)
        np.testing.assert_array_equal(c['anonymous_ids'], a['anonymous_ids'])

    def test_small_unknown_contamination_cannot_make_large_shift(self):
        module = self.module(); bundle = self.bundle()
        a = module.fit_stream(bundle, *self.stream(new=False), adapt_domain=True)
        b = module.fit_stream(bundle, *self.stream(new=False, contaminants=5), adapt_domain=True)
        np.testing.assert_allclose(a['b'], b['b'], atol=1e-12)
        self.assertLessEqual(np.linalg.norm(b['b']), .5)
        self.assertEqual(b['shift_iterations'], 10)
        self.assertEqual(b['shift_core_counts'].shape, (10, 2))

    def test_corrected_head_inverse_whitening_zero_b_and_row_independence(self):
        module = self.module(); bundle = self.bundle()
        bundle['chol'][1, 1] = 3.
        f = np.zeros((3, D), dtype=np.float32); f[0, 0] = 2.; f[1, 3] = 4.
        state = dict(b=np.zeros(D))
        corrected = module.corrected_head_features(f, bundle, state)
        np.testing.assert_array_equal(corrected, f)
        self.assertEqual(corrected.dtype, f.dtype)
        state['b'][1] = .2
        corrected = module.corrected_head_features(f, bundle, state)
        self.assertAlmostEqual(corrected[0, 1], -1.2, places=6)
        self.assertAlmostEqual(corrected[1, 1], -2.4, places=6)
        np.testing.assert_array_equal(corrected[2], f[2])
        for i in range(len(f)):
            np.testing.assert_array_equal(module.corrected_head_features(f[i:i+1], bundle, state)[0], corrected[i])

    def test_predictions_never_update_state_or_depend_on_other_query_rows(self):
        module = self.module(); bundle = self.bundle()
        state = module.fit_stream(bundle, *self.stream(), adapt_domain=True)
        snapshot = copy.deepcopy(state)
        rng = np.random.default_rng(81); queries = rng.normal(size=(5, D)); closed = np.zeros(5, dtype=int)
        together = module.predict_geometry(queries, closed, bundle, state, registered=True)
        for i in range(5):
            alone = module.predict_geometry(queries[i:i+1], closed[i:i+1], bundle, state, registered=True)
            for key in together:
                np.testing.assert_allclose(alone[key][0], together[key][i], rtol=0, atol=1e-12)
        for key in state:
            if isinstance(state[key], np.ndarray):
                np.testing.assert_array_equal(state[key], snapshot[key])
            else:
                self.assertEqual(state[key], snapshot[key])

    def test_d0_is_exact_zero_and_unqualified_fragments_do_not_register(self):
        module = self.module(); bundle = self.bundle()
        f = np.zeros((24, D)); f[:8, 2] = 1.; f[8:16, 3] = 1.; f[16:, 4] = 1.
        logits = np.tile([1., 0.], (len(f), 1)); ids = np.asarray(['p%03d' % i for i in range(len(f))])
        state = module.fit_stream(bundle, f, logits, ids, adapt_domain=False)
        np.testing.assert_array_equal(state['b'], np.zeros(D))
        self.assertTrue(state['converged'])
        self.assertFalse(state['qualified'].any())
        self.assertFalse(state['promotion_eligible'])
        decision = module.predict_geometry(f, np.zeros(len(f), dtype=int), bundle, state, registered=True)
        self.assertFalse(decision['accepted'].any())

    def test_cap_and_fixed_round_budget_cannot_promote(self):
        module = self.module(); bundle = self.bundle()
        f = np.zeros((40, D)); f[np.arange(40), np.arange(2, 42)] = 1.
        logits = np.tile([1., 0.], (40, 1)); ids = np.asarray(['p%03d' % i for i in range(40)])
        state = module.fit_stream(bundle, f, logits, ids, adapt_domain=False)
        self.assertTrue(state['cap_reached'])
        self.assertTrue(state['limit_reached'])
        self.assertFalse(state['converged'])
        self.assertFalse(state['promotion_eligible'])
        self.assertEqual(len(state['free_centers']), 32)
        with patch.object(module, 'MAX_ROUNDS', 1):
            unfinished = module.fit_stream(bundle, *self.stream(), adapt_domain=False)
        self.assertFalse(unfinished['converged'])
        self.assertFalse(unfinished['promotion_eligible'])
        self.assertEqual(unfinished['rounds'], 1)

    def test_zero_nonfinite_shapes_and_hidden_input_are_checked(self):
        module = self.module(); bundle = self.bundle()
        zeros = np.zeros((2, D)); result = module.predict_geometry(zeros, np.zeros(2, dtype=int), bundle)
        self.assertTrue(np.isfinite(result['knownness']).all())
        self.assertFalse(result['accepted'].any())
        for invalid in (np.zeros((2, 159)), np.full((2, D), np.nan), np.full((2, D), np.inf)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                module.predict_geometry(invalid, np.zeros(2, dtype=int), bundle)
        f, logits, ids = self.stream()
        with self.assertRaises(ValueError):
            module.fit_stream(bundle, f, logits, np.repeat('same-id', len(f)), adapt_domain=False)
        with self.assertRaises(ValueError):
            module.fit_stream(bundle, f, logits[:, :1], ids, adapt_domain=False)
        with self.assertRaises(ValueError):
            module.fit_source(f[:10], np.array([0] * 5 + [2] * 5), f[:3])
        with self.assertRaises(TypeError):
            module.fit_stream(bundle, f, logits, ids, adapt_domain=False, unknown_count=1)

    def test_positive_gain_merges_preserve_fixed_old_anchors(self):
        module = self.module(); anchors = self.bundle()['anchors']
        old = anchors.copy()
        q = np.zeros((2, D)); q[:, 2] = 1.; q[1, 3] = .1
        free, merges = module._merge_components(q, anchors, q.copy(), np.array(['a', 'b']), .01)
        self.assertEqual(merges, 1)
        self.assertEqual(len(free), 1)
        np.testing.assert_allclose(free[0], q.mean(axis=0), atol=1e-15)
        near = anchors[:1].copy(); near[0, 1] = .01
        free, merges = module._merge_components(near, anchors, near.copy(), np.array(['a']), .01)
        self.assertEqual(merges, 1)
        self.assertEqual(len(free), 0)
        np.testing.assert_array_equal(anchors, old)

    def test_qualification_uses_global_id_parity_and_higher_radius(self):
        module = self.module(); bundle = self.bundle()
        q = np.zeros((32, D)); q[:, 2] = 1.
        ids = np.array(['x%02d' % i for i in range(32)])
        # Sixteen members can still all have the same global-rank parity.
        assignments = np.where(np.arange(32) % 2 == 0, 2, 0)
        free = q[:1].copy()
        _, qualified, anon, diagnostics = module._qualify(q, bundle['anchors'], free, assignments, ids, .01)
        self.assertFalse(qualified[0])
        self.assertEqual(anon[0], -1)
        self.assertEqual(diagnostics[0]['count'], 16)
        self.assertFalse(diagnostics[0]['checks']['both_halves'])
        # Exactly half of two well-populated groups with separated means fails stability.
        q[1::2, 3] = .05
        assignments[:] = 2
        _, qualified, _, diagnostics = module._qualify(q, bundle['anchors'], free, assignments, ids, .01)
        self.assertFalse(qualified[0])
        self.assertFalse(diagnostics[0]['checks']['half_means'])


if __name__ == '__main__':
    unittest.main()
