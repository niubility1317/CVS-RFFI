"""Synthetic independent-oracle tests; direct execution is static only."""

if __name__ == '__main__':
    import ast
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    checked = []
    for path in (root/'tools/d92_affine_analysis_math.py', Path(__file__),
                 root/'docs/D92_AFFINE_ANALYSIS_CENTER_VJP_20261001.md'):
        raw = path.read_bytes()
        source = raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source and '\r' not in source
        if path.suffix == '.py':
            ast.parse(source, filename=str(path))
        checked.append(str(path))
    print(json.dumps(dict(status='STATIC_CHECKS_ONLY', checked=checked), ensure_ascii=False))
    raise SystemExit(0)

import ast
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import d92_affine_analysis_math as math_helper


def dense_oracle(A, B, q, gamma):
    """Deliberately materialize P only in this independent small oracle."""
    n = len(q)
    P = np.eye(n)-np.ones((n, 1))*q[None, :]
    raw = P.T@A@P-q[:, None]*(np.ones(len(B))@B@P)[None, :]
    return gamma*.5*(raw+raw.T), gamma*B@P


def assert_archive_tolerance(test, actual, expected):
    """Use the existing analyzer's norm/shape tolerance, not a new looser one."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    test.assertEqual(actual.shape, expected.shape)
    tolerance = 128*np.finfo(np.float64).eps*max((1, *actual.shape))*max(1., float(np.linalg.norm(expected)))
    test.assertLessEqual(float(np.linalg.norm(actual-expected)), tolerance)


class TestCenterKernelVJP(unittest.TestCase):
    def test_dense_P_oracle_shapes_fixed_nonuniform_measures_and_no_mutation(self):
        rng = np.random.default_rng(421)
        for n, h in ((1, 0), (1, 4), (2, 1), (7, 0), (7, 5), (31, 11)):
            A = rng.normal(size=(n, n)); A = .5*(A+A.T)
            B = rng.normal(size=(h, n))
            uniform = np.full(n, 1/n)
            onehot = np.zeros(n); onehot[n//2] = 1.
            sparse = np.zeros(n); sparse[::2] = np.arange(1, len(sparse[::2])+1)
            sparse /= sparse.sum()
            for q in (uniform, onehot, sparse):
                for gamma in (.7, -1.3, 0.):
                    before = [x.copy() for x in (A, B, q)]
                    expected = dense_oracle(A, B, q, gamma)
                    with self.subTest(n=n, h=h, gamma=gamma, q=q.tolist()):
                        got = math_helper.center_kernel_vjp(A, B, q, gamma)
                        self.assertEqual(got[0].dtype, np.dtype('float64'))
                        self.assertEqual(got[1].dtype, np.dtype('float64'))
                        for left, right in zip(got, expected): assert_archive_tolerance(self, left, right)
                        np.testing.assert_array_equal(got[0], got[0].T)
                        for left, right in zip((A, B, q), before): np.testing.assert_array_equal(left, right)

    def test_small_large_signed_adjoint_values_with_existing_tolerance(self):
        rng = np.random.default_rng(618)
        A = rng.normal(size=(9, 9)); A = .5*(A+A.T)
        B = rng.normal(size=(6, 9)); q = np.array([.125, 0, .25, 0, 0, .5, 0, .125, 0])
        for multiplier in (1e-100, 1., 1e100, -1e100):
            with self.subTest(multiplier=multiplier):
                expected = dense_oracle(multiplier*A, multiplier*B, q, .83)
                got = math_helper.center_kernel_vjp(multiplier*A, multiplier*B, q, .83)
                for left, right in zip(got, expected): assert_archive_tolerance(self, left, right)
                if multiplier:
                    unit = math_helper.center_kernel_vjp(A, B, q, .83)
                    for left, right in zip(got, unit):
                        np.testing.assert_allclose(left/multiplier, right, rtol=2e-13, atol=2e-13)

    def test_complete_reference_terms_match_forward_finite_difference(self):
        rng = np.random.default_rng(86); n, h = 8, 5
        R = rng.normal(size=(n, n)); R = .5*(R+R.T)
        Q = rng.normal(size=(h, n)); A = rng.normal(size=(n, n)); A = .5*(A+A.T)
        B = rng.normal(size=(h, n)); dR = rng.normal(size=(n, n)); dR = .5*(dR+dR.T)
        dQ = rng.normal(size=(h, n)); gamma = .71
        q = np.array([.2, 0, 0, .3, 0, .5, 0, 0])
        P = np.eye(n)-np.ones((n, 1))*q[None, :]
        def value(r, cross):
            K = gamma*P@r@P.T
            L = gamma*(cross-np.ones((h, 1))*(q@r)[None, :])@P.T
            return float(np.sum(A*K)+np.sum(B*L))
        br, bq = math_helper.center_kernel_vjp(A, B, q, gamma)
        step = 1e-4
        for dr, dq in ((dR, np.zeros_like(dQ)), (np.zeros_like(dR), dQ), (dR, dQ)):
            fd = (value(R+step*dr, Q+step*dq)-value(R-step*dr, Q-step*dq))/(2*step)
            analytic = float(np.sum(br*dr)+np.sum(bq*dq))
            with self.subTest(train=bool(np.any(dr)), cross=bool(np.any(dq))):
                self.assertAlmostEqual(fd, analytic, delta=2e-8*max(1., abs(analytic)))
        wrong = gamma*P.T@A@P
        self.assertGreater(float(np.linalg.norm(br-wrong)), 1e-2)

    def test_affine_saddle_cancellation_retains_nonzero_intercept_adjoint(self):
        rng = np.random.default_rng(172); n, h, c = 9, 6, 4
        alpha = rng.normal(size=(n, c)); alpha[-1] = -alpha[:-1].sum(axis=0)
        G = rng.normal(size=(h, c)); g_b = G.sum(axis=0)
        T = rng.normal(size=(n, c)); T[-1] = g_b-T[:-1].sum(axis=0)
        A = -.5*(T@alpha.T+alpha@T.T); B = G@alpha.T
        self.assertGreater(float(np.linalg.norm(g_b)), .1)
        for q in (np.full(n, 1/n), np.array([.2, 0, .3, 0, 0, 0, .5, 0, 0]), np.eye(n)[4]):
            br, bq = math_helper.center_kernel_vjp(A, B, q, .8)
            oracle = dense_oracle(A, B, q, .8)
            for left, right in zip((br, bq), oracle): assert_archive_tolerance(self, left, right)
            assert_archive_tolerance(self, br, .8*A)
            assert_archive_tolerance(self, bq, .8*B)
            P = np.eye(n)-np.ones((n, 1))*q[None, :]
            wrong = .8*P.T@A@P
            self.assertGreater(float(np.linalg.norm(br-wrong)), 1e-2)

    def test_near_cancellation_and_exact_one_sample_boundary(self):
        rng = np.random.default_rng(201); n = 7
        q = np.array([.125, 0, .375, 0, .5, 0, 0])
        v = rng.normal(size=n); small = rng.normal(size=(n, n)); small = .5*(small+small.T)
        gauge = q[:, None]*v[None, :]+v[:, None]*q[None, :]
        A = gauge+1e-12*small; B = np.zeros((3, n))
        br, bq = math_helper.center_kernel_vjp(A, B, q, 1.)
        expected = dense_oracle(A, B, q, 1.)
        for left, right in zip((br, bq), expected): assert_archive_tolerance(self, left, right)
        self.assertLess(float(np.linalg.norm(br)), 1e-10)
        # Cancellation has an absolute rounding boundary; relative error to
        # an arbitrarily small exact gradient is not a meaningful guarantee.
        for val in (1e-100, -2., 1e100):
            br, bq = math_helper.center_kernel_vjp(np.array([[val]]), np.array([[val], [-val], [3*val]]), np.ones(1), .4)
            np.testing.assert_array_equal(br, np.zeros((1, 1)))
            np.testing.assert_array_equal(bq, np.zeros((3, 1)))

    def test_general_incoming_matrix_and_unnormalized_fixed_weights_not_rewritten(self):
        # The algebra does not silently symmetrize input A or renormalize q.
        A = np.array([[1., 2., -1.], [5., 3., 4.], [2., -6., .5]])
        B = np.array([[3., -2., 7.], [-1., 5., 2.]])
        q = np.array([.2, 0, .4])
        expected = dense_oracle(A, B, q, .9)
        for left, right in zip(math_helper.center_kernel_vjp(A, B, q, .9), expected):
            assert_archive_tolerance(self, left, right)

    def test_no_dense_P_allocation_or_matrix_multiplication_in_helper(self):
        tree = ast.parse(inspect.getsource(math_helper.center_kernel_vjp))
        self.assertFalse(any(isinstance(node, ast.MatMult) for node in ast.walk(tree)))
        A = np.array([[2., -1.], [-1., 3.]]); B = np.array([[1., 2.]]); q = np.array([.25, .75])
        expected = dense_oracle(A, B, q, .8)
        with patch.object(math_helper.np, 'eye', side_effect=AssertionError('Dense P allocation forbidden')):
            got = math_helper.center_kernel_vjp(A, B, q, .8)
        for left, right in zip(got, expected): assert_archive_tolerance(self, left, right)

    def test_invalid_shapes_and_nonfinite_inputs_rejected(self):
        good = (np.eye(2), np.ones((3, 2)), np.array([.5, .5]), 1.)
        invalid = ((np.empty((0, 0)), np.empty((0, 0)), np.empty(0), 1.),
                   (np.ones((2, 3)), good[1], good[2], 1.),
                   (good[0], np.ones((3, 1)), good[2], 1.),
                   (good[0], good[1], np.ones((1, 2)), 1.),
                   (good[0], good[1], good[2], np.ones(1)),
                   (good[0]*np.nan, good[1], good[2], 1.),
                   (good[0], good[1]*np.inf, good[2], 1.),
                   (good[0], good[1], np.array([np.nan, 1.]), 1.),
                   (good[0], good[1], good[2], np.inf))
        for args in invalid:
            with self.subTest(shapes=[np.shape(v) for v in args]), self.assertRaises(ValueError):
                math_helper.center_kernel_vjp(*args)
