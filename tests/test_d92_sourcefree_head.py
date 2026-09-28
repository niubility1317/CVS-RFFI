"""Synthetic-only numerical and data-boundary checks."""
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi.stage2_d92_sourcefree_head import fit_sourcefree_head, _objective


def fixture(c=8, k=5, old=6):
    rng = np.random.default_rng(723)
    centers = rng.normal(size=(c, 160))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat(np.arange(c), k)
    x = centers[labels] + rng.normal(scale=.13, size=(len(labels), 160))
    logits = rng.normal(size=(len(labels), old))
    return x, labels, logits, tuple(f'class-{i}' for i in range(c))


class SourceFreeHeadTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference_including_bias_and_kd(self):
        rng = np.random.default_rng(12)
        c, d, k, old = 3, 4, 2, 2
        x = rng.normal(size=(c*k, d))
        y = np.repeat(np.arange(c), k)
        teacher = rng.normal(size=(c*k, old))
        teacher -= logsumexp(teacher, axis=1, keepdims=True)
        w0 = rng.normal(size=(c, d))
        parameters = rng.normal(size=c*(d+1))
        value, gradient, metrics = _objective(parameters, x, y, teacher, old, k, w0)
        numeric = np.empty_like(parameters)
        for j in range(len(parameters)):
            high, low = parameters.copy(), parameters.copy()
            high[j] += 1e-6; low[j] -= 1e-6
            numeric[j] = (_objective(high, x, y, teacher, old, k, w0)[0]
                          - _objective(low, x, y, teacher, old, k, w0)[0]) / 2e-6
        np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-8)
        self.assertAlmostEqual(value, metrics['support_ce'] + metrics['kd_weight'] * metrics['conditional_old_kd'] + metrics['regularization'])

    def test_training_records_actual_accepted_steps_and_is_immutable(self):
        x, y, t, classes = fixture()
        state = fit_sourcefree_head(x, y, t, classes)
        self.assertTrue(state.audit['converged'])
        self.assertLess(state.audit['final']['loss'], state.audit['initial']['loss'])
        self.assertEqual(len(state.audit['steps']), state.audit['optimizer_steps'])
        self.assertGreater(len(state.audit['steps']), 0)
        self.assertEqual(state.audit['steps'][-1]['loss'], state.audit['final']['loss'])
        self.assertTrue(all(step['gradient_l2'] >= 0 for step in state.audit['steps']))
        self.assertEqual(state.audit['persistent_state_bytes'], 8 * len(classes) * 161)
        with self.assertRaises(ValueError): state.W[0, 0] = 1
        with self.assertRaises(ValueError): state.W.setflags(write=True)
        with self.assertRaises(TypeError): state.audit['steps'][0]['loss'] = 0
        copied = state.audit_dict()
        json.dumps(copied, allow_nan=False)
        copied['steps'][0]['loss'] = -1
        self.assertNotEqual(state.audit['steps'][0]['loss'], -1)

    def test_support_permutation_and_query_partition_are_invariant(self):
        x, y, t, classes = fixture(c=4, k=5, old=2)
        state = fit_sourcefree_head(x, y, t, classes, old_count=2)
        order = np.random.default_rng(6).permutation(len(x))
        shuffled = fit_sourcefree_head(x[order], y[order], t[order], classes, old_count=2)
        np.testing.assert_array_equal(state.W, shuffled.W)
        np.testing.assert_array_equal(state.b, shuffled.b)
        scores = state.score(x)
        np.testing.assert_array_equal(scores, np.concatenate([state.score(x[:3]), state.score(x[3:])]))
        np.testing.assert_array_equal(scores[order], state.score(x[order]))
        self.assertEqual(state.score(x[:0]).shape, (0, len(classes)))
        for dimension in (256, 288):
            expanded = np.pad(x, ((0, 0), (0, dimension-160)))
            np.testing.assert_array_equal(scores, state.score(expanded))
        self.assertFalse(any('query' in name for name in inspect.signature(fit_sourcefree_head).parameters))

    def test_class_relabeling_equivariance_within_old_and_new_groups(self):
        x, y, t, classes = fixture(c=4, k=5, old=2)
        original = fit_sourcefree_head(x, y, t, classes, old_count=2)
        permutation = np.array([1, 0, 3, 2])
        changed = fit_sourcefree_head(x, permutation[y], t[:, [1, 0]],
                                     tuple(classes[i] for i in permutation), old_count=2)
        np.testing.assert_allclose(original.score(x), changed.score(x)[:, permutation], atol=2e-5, rtol=2e-5)

    def test_26_classes_k1_and_old_only_are_supported(self):
        for c, old, k in ((26, 6, 1), (6, 6, 1), (8, 6, 10), (8, 6, 20), (2, 1, 1)):
            with self.subTest(c=c, k=k):
                x, y, t, classes = fixture(c=c, k=k, old=old)
                state = fit_sourcefree_head(x, y, t, classes, old_count=old)
                self.assertEqual(state.score(x).shape, (c*k, c))
                self.assertEqual(state.audit['k'], k)
                self.assertEqual(state.audit['selection'], 'fixed_preregistered_formula')
                self.assertTrue(np.isfinite(state.score(x)).all())

    def test_iteration_limit_is_explicit_without_fallback(self):
        x, y, t, classes = fixture()
        state = fit_sourcefree_head(x, y, t, classes, max_iter=1)
        self.assertFalse(state.audit['converged'])
        self.assertEqual(state.audit['status'], 'NOT_CONVERGED')
        self.assertEqual(state.audit['optimizer_steps'], 1)

    def test_invalid_inputs_fail_closed(self):
        x, y, t, classes = fixture()
        cases = [dict(support_labels=y.astype(float)), dict(support_labels=y.astype(bool)),
                 dict(support_labels=y.reshape(-1, 1)), dict(support_labels=y+1),
                 dict(support_features=x[:, :159]), dict(support_features=np.zeros_like(x)),
                 dict(support_features=x.astype(str)), dict(support_features=x.astype(complex)),
                 dict(support_features=np.full_like(x, np.nan)),
                 dict(support_logits=t[:, :-1]), dict(support_logits=np.full_like(t, np.inf)),
                 dict(support_logits=t.astype(str)), dict(classes=classes[:-1]+(classes[0],)),
                 dict(old_count=0), dict(old_count=True), dict(max_iter=0), dict(max_iter=1.5)]
        defaults = dict(support_features=x, support_labels=y, support_logits=t, classes=classes)
        for changes in cases:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                fit_sourcefree_head(**dict(defaults, **changes))
        with self.assertRaises(ValueError):
            fit_sourcefree_head(x[:-1], y[:-1], t[:-1], classes)
        with self.assertRaises(FloatingPointError):
            _objective(np.full(4*161, np.nan), x[:4]/np.linalg.norm(x[:4], axis=1, keepdims=True),
                       np.arange(4), np.zeros((4, 2)), 2, 1, np.zeros((4, 160)))


if __name__ == '__main__':
    unittest.main()
