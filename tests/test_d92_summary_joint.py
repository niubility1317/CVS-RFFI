"""Synthetic-only SGJoint mathematics, permutation and access-boundary checks."""
import inspect
import json
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi import phase1_center_lowrank_prototype_bundle as codec
from cvsrffi.d92_ground_summary import GroundSummary
from cvsrffi.stage2_d92_summary_joint import (
    FROZEN_CONFIG, FrozenSummaryOperator, build_summary_operator, fit_summary_joint,
    _statistics, _fit_from_statistics, _row_affine,
)


def summary_fixture(class_order=(0, 1), domain_order=(0, 1, 2, 3), flat=False):
    rng = np.random.default_rng(933)
    c = len(class_order)
    core = rng.integers(-30, 31, (c, 160), dtype=np.int8)
    basis = rng.integers(-12, 13, (c, 3, 160), dtype=np.int8)
    coeff = rng.integers(-8, 9, (3, c, 3), dtype=np.int8)
    if flat:
        coeff.fill(0)
    radii = rng.integers(1, 10, (4, c), dtype=np.int8)
    cp = np.asarray(class_order)
    residual_domains = tuple(index for index in domain_order if index != 0)
    component = codec.CenterLowRankPrototypeComponent(
        core_q=core[cp], core_scale=np.full(c, .01, dtype=np.float16),
        residual_basis_q=basis[cp], residual_basis_scale=np.full((c, 3), .01, dtype=np.float16),
        residual_coeff_q=coeff[np.asarray(residual_domains)-1][:, cp],
        residual_coeff_scale=np.full((3, c), .01, dtype=np.float16),
        radius_q=radii[np.asarray(domain_order)][:, cp], radius_scale=np.full(c, .01, dtype=np.float16),
        domain_registry=tuple(f'domain-{i}' for i in domain_order),
        residual_domain_registry=tuple(f'domain-{i}' for i in residual_domains),
        class_registry=tuple(f'old-{i}' for i in class_order), center_domain_handle='domain-0',
        manifest=MappingProxyType(dict(schema=codec.SCHEMA, feature_dim=160, formal_phase2_eligible=True)))
    return GroundSummary(component, MappingProxyType(dict(incremental_transfer_bytes=0, total_file_bytes=123)))


def support(c=4, k=5):
    rng = np.random.default_rng(771)
    labels = np.repeat(np.arange(c), k)
    centers = rng.normal(size=(c, 256))
    x = centers[labels] + rng.normal(scale=.9, size=(c*k, 256))
    classes = tuple(['old-0', 'old-1'] + [f'new-{i}' for i in range(c-2)])
    ids = tuple(f'physical-{i:05d}' for i in range(c*k))
    return x, labels, ids, classes


class SummaryJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=1)
        cls.operator = build_summary_operator(summary_fixture())

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def fit(self, fixture=None, **changes):
        x, y, ids, classes = support() if fixture is None else fixture
        return fit_summary_joint(**dict(dict(support_features=x, support_labels=y, support_ids=ids,
            classes=classes, old_classes=('old-0', 'old-1'), summary_operator=self.operator), **changes))

    def test_summary_is_fixed_class_domain_permutation_invariant(self):
        original = summary_fixture()
        before = original.component.core_q.tobytes()
        with patch('builtins.open', side_effect=AssertionError('unexpected file read')):
            operator = build_summary_operator(original)
            reordered = build_summary_operator(summary_fixture((1, 0), (3, 1, 0, 2)))
        np.testing.assert_array_equal(operator.matrix, reordered.matrix)
        self.assertEqual(original.component.core_q.tobytes(), before)
        self.assertEqual(operator.audit['virtual_samples_generated'], 0)
        self.assertEqual(operator.audit['matrix_bytes'], 160*160*8)
        with self.assertRaises(ValueError): operator.matrix.setflags(write=True)
        with self.assertRaises(TypeError): operator.audit['geometry_trace'] = 0
        flat = build_summary_operator(summary_fixture(flat=True))
        # Tiny floating residuals after averaging identical centers must not
        # create a nontrivial geometry direction.
        np.testing.assert_allclose(flat.matrix, np.eye(160), atol=1e-10)

    def test_lda_matches_mahalanobis_equal_prior_scores(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(12, 7)); y = np.repeat(np.arange(3), 4)
        statistics = _statistics(x, y, 3)
        means, values, vectors, variance = statistics
        coef, bias, audit = _fit_from_statistics(x, *statistics, .1)
        covariance = (vectors * (.9*values + .1*variance)) @ vectors.T
        precision = np.linalg.inv(covariance)
        direct = np.stack([-.5*np.einsum('ij,jk,ik->i', x-m, precision, x-m) for m in means], axis=1)
        direct -= direct.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(_row_affine(x, coef, bias), direct/audit['training_scale'], atol=1e-12)

    def test_fold_counts_nll_trace_and_json_accounting(self):
        state = self.fit()
        audit = state.audit_dict()
        json.dumps(audit, allow_nan=False)
        self.assertEqual(audit['algorithm'], FROZEN_CONFIG)
        self.assertEqual(len(audit['candidate_trace']), 16)
        self.assertEqual(audit['fold_count'], 3)
        self.assertEqual(audit['eigendecompositions'], 25)
        held = [identifier for fold in audit['folds'] for identifier in fold['held_ids']]
        self.assertEqual(sorted(held), sorted(support()[2]))
        for candidate in audit['candidate_trace']:
            self.assertTrue(.1 <= candidate['temperature'] <= 10)
            self.assertEqual([fold['train_per_class'] for fold in candidate['folds']], [3, 3, 4])
            self.assertEqual([fold['held_per_class'] for fold in candidate['folds']], [2, 2, 1])
            self.assertTrue(all(fold['training_scale'] > 0 for fold in candidate['folds']))
        self.assertEqual(audit['head_bytes'], state.coefficient.nbytes+state.intercept.nbytes)
        self.assertEqual(audit['persistent_state_bytes'], audit['head_bytes']+state.operator.matrix.nbytes)
        with self.assertRaises(ValueError): state.coefficient.setflags(write=True)
        with self.assertRaises(TypeError): state.audit['selected']['alpha'] = 2

    def test_registry_and_support_permutations_and_query_partition(self):
        x, y, ids, classes = support()
        state = self.fit((x, y, ids, classes))
        rng = np.random.default_rng(44)
        order = rng.permutation(len(x))
        permutation = np.array([3, 0, 2, 1])
        new_classes = tuple(classes[i] for i in permutation)
        inverse = np.argsort(permutation)
        other = self.fit((x[order], inverse[y[order]], tuple(ids[i] for i in order), new_classes))
        np.testing.assert_array_equal(state.score(x)[:, permutation], other.score(x))
        self.assertEqual(dict(state.audit['selected']), dict(other.audit['selected']))
        np.testing.assert_array_equal(state.score(x), np.concatenate([state.score(x[:2]), state.score(x[2:])]))
        np.testing.assert_array_equal(state.score(x)[order], state.score(x[order]))
        self.assertEqual(state.score(x[:0]).shape, (0, 4))
        self.assertEqual(state.predict(x[:0]).shape, (0,))
        np.testing.assert_array_equal(state.score(x), state.score(np.pad(x, ((0, 0), (0, 32)))))

    def test_k1_has_no_holdout_and_stable_physical_class_tie(self):
        fixture = support(k=1)
        with patch('cvsrffi.stage2_d92_summary_joint._temperature', side_effect=AssertionError('K1 validation')):
            state = self.fit(fixture)
        self.assertEqual(state.audit['selection'], 'fixed_K1')
        self.assertEqual(state.audit['candidate_trace'], ())
        self.assertEqual(dict(state.audit['selected']), dict(alpha=1.0, beta=1, shrinkage=1.0, temperature=1.0))
        x, y, ids, classes = fixture
        x[:] = x[0]
        tied = self.fit((x, y, ids, classes))
        self.assertTrue(np.all(tied.predict(x) == classes.index(min(classes))))

    def test_k2_10_20_old_only_26_classes_and_zero_fft(self):
        for c, k in ((4, 2), (4, 10), (4, 20), (2, 5), (26, 1)):
            with self.subTest(c=c, k=k):
                fixture = support(c=c, k=k)
                fixture[0][:, 160:256] = 0
                state = self.fit(fixture)
                self.assertTrue(np.isfinite(state.score(fixture[0])).all())
                self.assertEqual(state.audit['zero_fft_support_rows'], c*k)
                if c == 2:
                    self.assertIsNone(state.audit['selected_oof_risk']['new_nll'])

    def test_identical_support_lowrank_geometry_and_norm_failure(self):
        x, y, ids, classes = support()
        x[:] = x[0]
        state = self.fit((x, y, ids, classes))
        self.assertTrue(np.isfinite(state.score(x)).all())
        self.assertTrue(np.all(state.predict(x) == classes.index(min(classes))))
        bad = x.copy(); bad[0, :160] = 0
        with self.assertRaises(ValueError): self.fit((bad, y, ids, classes))

    def test_input_validation_and_no_query_teacher_interface(self):
        x, y, ids, classes = support()
        changes = [dict(support_features=x[:, :160]), dict(support_features=np.full_like(x, np.nan)),
                   dict(support_features=x.astype(str)), dict(support_labels=y.astype(float)),
                   dict(support_labels=y.reshape(-1, 1)), dict(support_labels=y+1),
                   dict(support_ids=ids[:-1]), dict(support_ids=(ids[0],)*len(ids)),
                   dict(classes=classes[:-1]+(classes[0],)), dict(old_classes=('missing',)),
                   dict(summary_operator=np.eye(160))]
        for change in changes:
            with self.subTest(fields=list(change)), self.assertRaises(ValueError): self.fit(**change)
        with self.assertRaises(ValueError): self.fit((x[:-1], y[:-1], ids[:-1], classes))
        parameters = inspect.signature(fit_summary_joint).parameters
        self.assertFalse(any('query' in name or 'teacher' in name for name in parameters))
        with self.assertRaises(ValueError): build_summary_operator(dict(schema=codec.SCHEMA))
        with self.assertRaises(ValueError): FrozenSummaryOperator(np.zeros((160, 160)), ('old-0',), codec.FEATURE_SCHEMA, {})


if __name__ == '__main__':
    unittest.main()
