"""Synthetic-only MVKME formula, physical-fold and immutable inference tests."""
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi import stage2_d92_mv_kme as m


def fixture(c=4, k=5):
    rng = np.random.default_rng(907)
    labels = np.repeat(np.arange(c), k)
    centers = rng.normal(size=(c, 3, 256))
    blocks = (centers[labels] + rng.normal(scale=.5, size=(c*k, 3, 256))) / 20
    ids = tuple(f'id-{i:04}' for i in range(c*k))
    classes = tuple(f'tx-{i:02}' for i in range(c))
    return dict(support_blocks=blocks, support_labels=labels, support_ids=ids,
                classes=classes, old_classes=classes[:min(c, 2)])


class MVKMETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.threads.restore_original_limits()

    def test_views_are_exact_and_not_extra_physical_samples(self):
        iq = np.arange(2*2*256).reshape(2, 2, 256)
        views = m.make_received_views(iq)
        self.assertEqual(views.shape, (2, 4, 4, 2, 256))
        np.testing.assert_array_equal(views[:, 0, 0], iq)
        for j, start in enumerate((0, 32, 64), 1):
            np.testing.assert_array_equal(views[:, j, 0, :, 32:224], iq[:, :, start:start+192])
            self.assertFalse(views[:, j, 0, :, :32].any())
            self.assertFalse(views[:, j, 0, :, 224:].any())
        np.testing.assert_array_equal(views[:, :, 1, 0], -views[:, :, 0, 1])
        np.testing.assert_array_equal(views[:, :, 3, 1], -views[:, :, 0, 0])
        self.assertFalse(views.flags.writeable)

    def test_fft_matches_existing_and_is_phase_invariant(self):
        from cvsrffi.stage2_diag_cosine_exploration import spectral_logmag_sketch
        iq = np.random.default_rng(18).normal(size=(3, 2, 256))
        np.testing.assert_array_equal(m.received_fft96(iq), spectral_logmag_sketch(iq))
        np.testing.assert_array_equal(m.received_fft96(iq*1e-11), spectral_logmag_sketch(iq*1e-11))
        views = m.make_received_views(iq)
        np.testing.assert_allclose(m.received_fft96(views[:, 0, 1]), m.received_fft96(iq), atol=1e-6)
        np.testing.assert_array_equal(m.received_fft96(np.zeros((1, 2, 256))), np.zeros((1, 96)))

    def test_fourier_unit_norm_and_nonlinearity(self):
        x = np.zeros((3, 256)); x[1, 0] = .8; x[2, 0] = -.8
        mapped = m.fourier_map(x)
        np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1, atol=1e-14)
        # This map violates the midpoint identity on this constructed line.
        self.assertGreater(np.linalg.norm(mapped[0]-(mapped[1]+mapped[2])/2), .01)
        np.testing.assert_array_equal(mapped, np.concatenate([m.fourier_map(x[:1]), m.fourier_map(x[1:])]))

    def test_blocks_formula_phase_orbit_and_batch_independence(self):
        rng = np.random.default_rng(86)
        iq = rng.normal(size=(2, 2, 256)); z = rng.normal(size=(2, 4, 4, 160))
        blocks = m.build_fourier_blocks(received_iq=iq, identity_views=z)
        rotated = m.make_received_views(iq)[:, 0, 1]
        shifted_z = np.roll(z, -1, axis=2)
        other = m.build_fourier_blocks(received_iq=rotated, identity_views=shifted_z)
        np.testing.assert_allclose(blocks[:, 1:], other[:, 1:], atol=2e-8)
        chunks = [m.build_fourier_blocks(received_iq=iq[i:i+1], identity_views=z[i:i+1]) for i in range(2)]
        np.testing.assert_array_equal(blocks, np.concatenate(chunks))
        self.assertFalse(blocks.flags.writeable)
        degenerate = m.build_fourier_blocks(received_iq=np.zeros_like(iq), identity_views=np.zeros_like(z))
        self.assertTrue(np.isfinite(degenerate).all())

    def test_ridge_matches_direct_solve_and_stationarity(self):
        rng = np.random.default_rng(124)
        phi = rng.normal(size=(8, 768)); y = rng.normal(size=(8, 3)); k = 2; gamma = .1
        w = m._ridge(phi, y, k, gamma)
        direct = phi.T @ np.linalg.solve(phi @ phi.T + k*gamma*np.eye(8), y)
        np.testing.assert_allclose(w, direct, atol=1e-13)
        np.testing.assert_allclose(phi.T @ (phi @ w-y) + k*gamma*w, 0, atol=1e-12)

    def test_physical_folds_actual_k_and_complete_oof(self):
        args = fixture(k=5); state = m.fit_mv_kme(**args); a = state.audit_dict()
        self.assertEqual(a['candidate_count'], 9); self.assertEqual(a['fold_count'], 3)
        steps = [s for s in a['steps'] if s['event']=='PHYSICAL_SUPPORT_FOLD']
        self.assertEqual(len(steps), 27)
        self.assertEqual({s['train_k'] for s in steps}, {3, 4})
        for step in steps:
            self.assertEqual(step['regularizer'], step['train_k']*step['gamma'])
            self.assertEqual(step['train_physical_count']+step['heldout_physical_count'], 20)
        for eta in (0, .5, 1):
            for gamma in (.01, .1, 1):
                self.assertEqual(sum(s['heldout_physical_count'] for s in steps if s['eta']==eta and s['gamma']==gamma), 20)
        self.assertEqual(len(a['physical_fold_assignment']), 20)
        json.dumps(a, allow_nan=False)

    def test_permutations_and_query_chunks_do_not_change_state(self):
        args = fixture(); first = m.fit_mv_kme(**args)
        rng = np.random.default_rng(77); p = rng.permutation(len(args['support_labels']))
        cp = np.array([2, 0, 3, 1]); inv = np.argsort(cp)
        altered = dict(args, support_blocks=args['support_blocks'][p], support_labels=inv[args['support_labels'][p]],
                       support_ids=tuple(args['support_ids'][i] for i in p),
                       classes=tuple(args['classes'][i] for i in cp))
        other = m.fit_mv_kme(**altered)
        self.assertEqual(first.audit_dict()['selected'], other.audit_dict()['selected'])
        q = rng.normal(size=(7, 3, 256))/20
        score = first.score(q)
        np.testing.assert_allclose(other.score(q)[:, inv], score, atol=1e-11)
        np.testing.assert_array_equal(score, np.concatenate([first.score(q[:2]), first.score(q[2:])]))
        before = first.coefficient.tobytes(); first.predict(q[::-1])
        self.assertEqual(before, first.coefficient.tobytes())
        with self.assertRaises(ValueError): first.coefficient.setflags(write=True)
        with self.assertRaises(TypeError): first.audit['k'] = 10
        self.assertEqual(first.score(np.empty((0, 3, 256))).shape, (0, 4))

    def test_k1_26_classes_no_cv_and_stable_tie(self):
        args = fixture(c=26, k=1)
        with patch.object(m, '_losses', side_effect=AssertionError('K1 must not hold out')):
            state = m.fit_mv_kme(**args)
        a = state.audit_dict()
        self.assertEqual(a['candidate_count'], 0); self.assertEqual(a['fold_count'], 0)
        self.assertEqual(a['selected'], dict(eta=.5, gamma=.1))
        self.assertEqual(a['head_bytes'], state.coefficient.nbytes)
        self.assertEqual(a['persistent_state_bytes'], state.coefficient.nbytes + m._OMEGA.nbytes)
        args['support_blocks'].fill(0)
        tied = m.fit_mv_kme(**args)
        self.assertEqual(tied.predict(np.zeros((1, 3, 256)))[0], min(args['classes']))
        self.assertTrue(np.isfinite(tied.score(args['support_blocks'])).all())
        one = m.fit_mv_kme(**fixture(c=1,k=1))
        self.assertEqual(one.predict(np.zeros((1,3,256)))[0], 'tx-00')

    def test_invalid_inputs_and_no_query_training_interface(self):
        base = fixture()
        changes = [dict(support_labels=np.ones(20)), dict(support_labels=np.ones(20,dtype=bool)),
            dict(support_labels=np.full(20, 99)), dict(support_ids=['same']*20),
            dict(classes=['x']*4), dict(old_classes=['missing']),
            dict(support_blocks=np.zeros((20, 256))), dict(support_blocks=np.zeros((20,3,256),dtype=bool)),
            dict(support_blocks=np.full((20,3,256), np.nan)),
            dict(support_blocks=base['support_blocks'][:-1],support_labels=base['support_labels'][:-1],support_ids=base['support_ids'][:-1])]
        for change in changes:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError): m.fit_mv_kme(**dict(base, **change))
        for shape in ((2,256), (1,2,128)):
            with self.assertRaises(ValueError): m.make_received_views(np.zeros(shape))
        with self.assertRaises(ValueError): m.make_received_views(np.full((1,2,256),np.inf))
        with self.assertRaises(ValueError): m.build_fourier_blocks(received_iq=np.zeros((1,2,256)),identity_views=np.zeros((2,4,4,160)))
        self.assertEqual(set(inspect.signature(m.fit_mv_kme).parameters),
            {'support_blocks','support_labels','support_ids','classes','old_classes'})

    def test_frozen_config_file(self):
        path = Path(__file__).resolve().parents[1]/'configs/d92_mv_kme_frozen_20260928.json'
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'algorithm':m.FROZEN_CONFIG})


if __name__ == '__main__':
    unittest.main()
