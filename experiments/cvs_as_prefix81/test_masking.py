"""Focused synthetic checks; no dataset, checkpoint, model or target access."""
import copy
import io
import json
import random
import time
import unittest

import numpy as np

from experiments.cvs_as_prefix81 import masking as m


def records(n=64, length=61):
    return [dict(sample_id=(1, 3, 2, i, 1000+i),
                 residual_kernel_length=length+1,
                 propagation_kernel_length=length+1,
                 numerical_filter_latency_samples=24,
                 channel_equalization_applied=False,
                 quality_snr_db=10., TX=i % 6) for i in range(n)]


class MaskingChecks(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(64*2*256, dtype=np.float32).reshape(64, 2, 256)+1
        self.meta = records()

    def apply(self, x=None, metadata=None, **kw):
        args = dict(mode='PREFIX', epoch=21, step=4441)
        args.update(kw)
        return m.apply_mask(self.x if x is None else x,
                            self.meta if metadata is None else metadata, **args)

    def test_prefix_exact_active_inactive_copy_and_power(self):
        before = self.x.copy()
        oldmeta = copy.deepcopy(self.meta)
        out, detail = self.apply()
        active = [i for i, r in enumerate(detail) if r['enabled']]
        inactive = [i for i, r in enumerate(detail) if not r['enabled']]
        self.assertTrue(active and inactive)
        for i, rec in enumerate(detail):
            expected = before[i].copy()
            if rec['enabled']:
                expected[:, :61] = 0
            np.testing.assert_array_equal(out[i], expected)
            self.assertEqual((rec['begin'], rec['length'], rec['end']), (0, 61, 61))
            original = np.square(before[i].astype(np.float64)).sum()/256
            removed = np.square(before[i, :, :61].astype(np.float64)).sum()/256 if rec['enabled'] else 0.
            self.assertEqual(rec['original_power'], original)
            self.assertEqual(rec['removed_power'], removed)
            self.assertAlmostEqual(rec['output_power'], original-removed)
        np.testing.assert_array_equal(self.x, before)
        self.assertEqual(self.meta, oldmeta)
        self.assertIsNot(out, self.x)
        self.assertEqual(out.dtype, self.x.dtype)

    def test_random_equal_count_exact_enable_reference_and_bounds(self):
        prefix, p = self.apply()
        out, r = self.apply(mode='RANDOM')
        self.assertEqual([v['enable_draw'] for v in p], [v['enable_draw'] for v in r])
        self.assertEqual([v['enabled'] for v in p], [v['enabled'] for v in r])
        seen_left = seen_right = False
        for i, rec in enumerate(r):
            start, end = rec['begin'], rec['end']
            self.assertTrue(0 <= start <= 19 or 160 <= start <= 195)
            self.assertEqual(end-start, 61)
            seen_left |= start < 80
            seen_right |= start >= 160
            expected = self.x[i].copy()
            if rec['enabled']:
                expected[:, start:end] = 0
            np.testing.assert_array_equal(out[i], expected)
            self.assertEqual(np.count_nonzero(prefix[i] == 0), np.count_nonzero(out[i] == 0))
        self.assertTrue(seen_left and seen_right)
        np.testing.assert_array_equal(out[:, :, 80:160], self.x[:, :, 80:160])

    def test_deterministic_reorder_duplicate_and_no_TX_dependency(self):
        out, detail = self.apply(mode='RANDOM')
        changed = copy.deepcopy(self.meta)
        for i, rec in enumerate(changed):
            rec['sample_id'] = (*rec['sample_id'][:4], 90000+i)
            rec['TX'] = (rec['TX']+1) % 6
            rec['label'] = 'unreadable TX truth'
        after, changed_detail = self.apply(metadata=changed, mode='RANDOM')
        np.testing.assert_array_equal(after, out)
        self.assertEqual(detail, changed_detail)
        rev, rd = self.apply(self.x[::-1], self.meta[::-1], mode='RANDOM')
        np.testing.assert_array_equal(rev, out[::-1])
        self.assertEqual(rd, detail[::-1])
        repeated = np.stack([self.x[0], self.x[0]])
        dup, dd = self.apply(repeated, [self.meta[0], self.meta[0]], mode='RANDOM')
        np.testing.assert_array_equal(dup[0], dup[1])
        self.assertEqual(dd[0], dd[1])

    def test_epoch_step_physical_refresh(self):
        _, first = self.apply()
        _, next_step = self.apply(step=4442)
        _, next_epoch = self.apply(epoch=22, step=4663)
        for other in (next_step, next_epoch):
            self.assertNotEqual([v['enable_draw'] for v in first], [v['enable_draw'] for v in other])
        fields = ('eq_i', 'rx_i', 'day_i', 'sig_i')
        explicit = [dict(zip(fields, rec['sample_id'][:4]), residual_kernel_length=62) for rec in self.meta]
        _, same = self.apply(metadata=explicit)
        self.assertEqual([v['enable_draw'] for v in first], [v['enable_draw'] for v in same])
        explicit[0]['rx_i'] = 4
        _, changed = self.apply(metadata=explicit)
        self.assertNotEqual(first[0]['enable_draw'], changed[0]['enable_draw'])
        self.assertEqual(first[1:], changed[1:])

    def test_all_global_RNG_states_unchanged(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = m.torch.random.get_rng_state().clone() if m.torch is not None else None
        self.apply(mode='RANDOM')
        self.assertEqual(random.getstate(), python_before)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_before[0])
        np.testing.assert_array_equal(after[1], numpy_before[1])
        self.assertEqual(after[2:], numpy_before[2:])
        if m.torch is not None:
            self.assertTrue(m.torch.equal(m.torch.random.get_rng_state(), torch_before))

    def test_native_length_fallback_and_no_delay_substitution(self):
        _, native = self.apply()
        self.assertEqual(native[0]['length'], 61)
        prop = copy.deepcopy(self.meta)
        for rec in prop:
            del rec['residual_kernel_length']
        out, pr = self.apply(metadata=prop)
        np.testing.assert_array_equal(out, self.apply()[0])
        self.assertEqual(pr[0]['length_source'], 'native_EQoff_propagation_kernel_length_minus_one')
        light = [dict(sample_id=rec['sample_id'], numerical_filter_latency_samples=24) for rec in self.meta]
        explicit, er = self.apply(metadata=light, lengths=[61]*64)
        np.testing.assert_array_equal(explicit, out)
        self.assertEqual(er[0]['length_source'], 'wrapper_actual_kernel_length_minus_one')
        for length in (1, 80):
            small = records(length=length)
            _, checks = self.apply(metadata=small, mode='RANDOM', lengths=length)
            self.assertTrue(all(v['end'] <= 80 or v['begin'] >= 160 for v in checks))
        zero, zr = self.apply(np.zeros_like(self.x))
        self.assertTrue(np.isfinite(zero).all())
        self.assertTrue(all(v['removed_power_fraction'] == 0. for v in zr))

    @unittest.skipIf(m.torch is None, 'torch unavailable')
    def test_tensor_type_device_dtype_and_numpy_parity(self):
        t = m.torch.frombuffer(bytearray(self.x.tobytes()), dtype=m.torch.float32).reshape(self.x.shape)
        old = t.clone()
        out, detail = self.apply(t, mode='RANDOM')
        npout, npdetail = self.apply(mode='RANDOM')
        self.assertIsInstance(out, m.torch.Tensor)
        self.assertEqual((out.device, out.dtype), (t.device, t.dtype))
        self.assertNotEqual(out.data_ptr(), t.data_ptr())
        self.assertTrue(m.torch.equal(t, old))
        np.testing.assert_array_equal(np.asarray(out.tolist(), dtype=np.float32), npout)
        self.assertEqual(detail, npdetail)

    def test_reject_invalid_inputs_and_no_mutation(self):
        bad = [dict(mode='other'), dict(epoch=20), dict(epoch=31), dict(step=4440),
               dict(step=6661), dict(step=4663), dict(epoch=True), dict(seed=-1),
               dict(seed=1.5), dict(lengths=24), dict(lengths=[61]), dict(lengths=True),
               dict(lengths=np.array(61)), dict(lengths=np.ones((64, 1), dtype=np.int64))]
        for kw in bad:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.apply(**kw)
        arrays = [self.x.astype(np.float64), self.x[0], self.x[:, :, :255], self.x[:0], self.x.tolist()]
        nonfinite = self.x.copy(); nonfinite[0, 0, 0] = np.nan; arrays.append(nonfinite)
        for value in arrays:
            with self.subTest(shape=np.shape(value)), self.assertRaises(ValueError):
                self.apply(value)
        for meta in ([], self.meta[:-1], [None]*64, [{}]*64):
            with self.assertRaises(ValueError):
                self.apply(metadata=meta)
        cases = [dict(residual_kernel_length=1), dict(residual_kernel_length=82),
                 dict(residual_kernel_length=62.), dict(sample_id=(1, 2, 2, 1, 1)),
                 dict(sample_id=(2, 3, 2, 1, 1)), dict(sample_id=(1, 3, 4, 1, 1)),
                 dict(sample_id=(1, 3, 2, -1, 1)), dict(sample_id='opaque-only'),
                 dict(eq_i=1), dict(eq_i=1, rx_i=4, day_i=2, sig_i=63)]
        for fields in cases:
            meta = copy.deepcopy(self.meta); meta[-1].update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.apply(metadata=meta)
        for eq in (True, None):
            meta = copy.deepcopy(self.meta)
            del meta[-1]['residual_kernel_length']
            meta[-1]['channel_equalization_applied'] = eq
            with self.assertRaises(ValueError):
                self.apply(metadata=meta)
        light = [dict(sample_id=rec['sample_id'], numerical_filter_latency_samples=24) for rec in self.meta]
        with self.assertRaises(ValueError):
            self.apply(metadata=light)
        np.testing.assert_array_equal(self.x, np.arange(64*2*256, dtype=np.float32).reshape(64, 2, 256)+1)


def run():
    """Small callable used by the experiment's protocol-negative checks."""
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MaskingChecks)
    names = [test._testMethodName for test in suite]
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError(output.getvalue())
    return dict(status='PASS', checks=result.testsRun,
                passed=names, skipped=len(result.skipped),
                wall_seconds=time.perf_counter()-start,
                NN=0, CP=0, real_IQ=0, updates=0, target_access=False)


if __name__ == '__main__':
    print(json.dumps(run(), allow_nan=False))
