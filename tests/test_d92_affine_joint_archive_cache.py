"""Synthetic archive LRU/byte budgets without caching mathematical verdicts."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import evaluate_d92_affine_joint_probe as probe
import summarize_d92_affine_joint_probe as summary
from test_evaluate_d92_affine_joint_probe import synthetic, scope
from test_summarize_d92_affine_joint_probe import as_record


def archive(root, sizes=(8, 12, 6)):
    writer = probe.StateArchive(root)
    namespace = json.dumps(dict(state='B_AFFINE'))
    refs = [writer(namespace+'/array_'+str(i), dict(vector=np.arange(n, dtype=np.float64)))
        for i, n in enumerate(sizes)]
    writer.finalize('COMPLETE'); return refs


class CacheStorageTests(unittest.TestCase):
    def test_byte_budget_pre_evicts_LRU_before_np_load_using_actual_numeric_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            refs = archive(root); resolver = summary.StateResolver(root, cache_budget_bytes=160)
            original = np.load; bytes_at_load = []
            def loading(*args, **kwargs):
                bytes_at_load.append(resolver.cache_numeric_bytes); return original(*args, **kwargs)
            with patch.object(summary.np, 'load', side_effect=loading) as loads:
                resolver(refs[0]); resolver(refs[1]); resolver(refs[0]); resolver(refs[2])
            self.assertEqual(loads.call_count, 3); self.assertEqual(bytes_at_load, [0, 64, 64])
            self.assertEqual(list(resolver.cache), [refs[0]['path'], refs[2]['path']])
            stats = resolver.cache_statistics()
            self.assertEqual(stats['resident_numeric_bytes'], 112); self.assertEqual(stats['peak_numeric_bytes'], 160)
            self.assertEqual(stats['hit_count'], 1); self.assertEqual(stats['eviction_count'], 1)
            self.assertEqual(stats['evicted_numeric_bytes'], 96); self.assertEqual(stats['loaded_numeric_bytes'], 208)
            self.assertTrue(all(not a.flags.writeable for entry in resolver.cache.values() for a in entry.values()))

    def test_entry_cap_evicts_even_when_byte_budget_has_space(self):
        with tempfile.TemporaryDirectory() as root:
            refs = archive(root); resolver = summary.StateResolver(root, cache_budget_bytes=1024, max_entries=1)
            resolver(refs[0]); resolver(refs[1])
            stats = resolver.cache_statistics()
            self.assertEqual(stats['resident_entry_count'], 1); self.assertEqual(stats['peak_entries'], 1)
            self.assertEqual(stats['resident_numeric_bytes'], 96); self.assertEqual(stats['eviction_count'], 1)
            self.assertEqual(list(resolver.cache), [refs[1]['path']])

    def test_zero_budget_disables_residency_but_reloads_and_validates_arrays(self):
        with tempfile.TemporaryDirectory() as root:
            ref = archive(root, (8,))[0]; resolver = summary.StateResolver(root, cache_budget_bytes=0)
            with patch.object(summary.np, 'load', wraps=np.load) as loads:
                first = resolver(ref); second = resolver(ref)
            self.assertEqual(loads.call_count, 2); self.assertIsNot(first['vector'], second['vector'])
            self.assertFalse(first['vector'].flags.writeable); self.assertFalse(second['vector'].flags.writeable)
            stats = resolver.cache_statistics()
            self.assertEqual(stats['resident_numeric_bytes'], 0); self.assertEqual(stats['resident_entry_count'], 0)
            self.assertEqual(stats['hit_count'], 0); self.assertEqual(stats['uncached_load_count'], 2)
            self.assertEqual(stats['oversized_load_count'], 0); resolver.finalize()

    def test_oversized_archive_is_fully_verified_without_residency_and_preevicts(self):
        with tempfile.TemporaryDirectory() as root:
            refs = archive(root, (8, 12)); resolver = summary.StateResolver(root, cache_budget_bytes=80)
            resolver(refs[0]); original = np.load; bytes_at_load = []
            def loading(*args, **kwargs):
                bytes_at_load.append(resolver.cache_numeric_bytes); return original(*args, **kwargs)
            with patch.object(summary.np, 'load', side_effect=loading) as loads:
                first = resolver(refs[1]); second = resolver(refs[1])
            self.assertEqual(bytes_at_load, [0, 0]); self.assertEqual(loads.call_count, 2)
            np.testing.assert_array_equal(first['vector'], np.arange(12, dtype=float))
            self.assertFalse(second['vector'].flags.writeable); self.assertEqual(resolver.verified, {r['path'] for r in refs})
            stats = resolver.cache_statistics()
            self.assertEqual(stats['oversized_load_count'], 2); self.assertEqual(stats['uncached_load_count'], 2)
            self.assertEqual(stats['eviction_count'], 1); self.assertEqual(stats['resident_numeric_bytes'], 0)
            resolver.finalize()

    def test_lane_clear_releases_arrays_but_retains_inventory_and_first_file_mtime(self):
        with tempfile.TemporaryDirectory() as root:
            refs = archive(root, (8, 12)); resolver = summary.StateResolver(root)
            for ref in refs: resolver(ref)
            resolver.finalize(); refs_before = resolver.refs; used = resolver.used.copy(); verified = resolver.verified.copy()
            mtimes = resolver.file_mtimes.copy(); resolver.clear_cache()
            self.assertIs(resolver.refs, refs_before); self.assertEqual(resolver.used, used); self.assertEqual(resolver.verified, verified)
            self.assertEqual(resolver.file_mtimes, mtimes); self.assertFalse(resolver.cache)
            stats = resolver.cache_statistics()
            self.assertEqual(stats['cleared_entry_count'], 2); self.assertEqual(stats['cleared_numeric_bytes'], 160)
            self.assertEqual(stats['resident_numeric_bytes'], 0); self.assertEqual(stats['clear_count'], 1)
            resolver(refs[0]); self.assertEqual(resolver.cache_statistics()['load_count'], 3); resolver.finalize()

    def test_cache_hit_and_post_clear_reload_reject_file_mtime_change(self):
        for clear in (False, True):
            with self.subTest(clear=clear), tempfile.TemporaryDirectory() as root:
                ref = archive(root, (8,))[0]; resolver = summary.StateResolver(root); resolver(ref)
                if clear: resolver.clear_cache()
                path = Path(root)/ref['path']; info = path.stat()
                os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns+10_000_000))
                self.assertEqual(path.stat().st_size, ref['file_bytes'])
                with patch.object(summary.np, 'load') as loads:
                    with self.assertRaisesRegex(ValueError, 'mtime'): resolver(ref)
                    loads.assert_not_called()

    def test_warm_cache_rejects_changed_size_and_tampered_exact_reference(self):
        with tempfile.TemporaryDirectory() as root:
            ref = archive(root, (8,))[0]; resolver = summary.StateResolver(root); resolver(ref)
            for field in ('path', 'shape', 'norm'):
                bad = deepcopy(ref)
                if field == 'path': bad['path'] = '../escaped.npz'
                if field == 'shape': bad['arrays']['vector']['shape'] = [1, 8]
                if field == 'norm': bad['array_summaries']['vector']['norm'] += 1
                with self.subTest(field=field), self.assertRaises(ValueError): resolver(bad)
            with (Path(root)/ref['path']).open('ab') as stream: stream.write(b'changed')
            with patch.object(summary.np, 'load') as loads:
                with self.assertRaisesRegex(ValueError, 'file size'): resolver(ref)
                loads.assert_not_called()


class CompleteVerificationCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); cls.inputs = synthetic(k=3)
        writer = probe.StateArchive(cls.directory.name)
        result = probe.probe_affine_joint(**cls.inputs, context=scope(), state_callback=writer)
        cls.record, cls.split = as_record(result, cls.inputs, 3); writer.finalize('COMPLETE')

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def verify(self, budget):
        resolver = summary.StateResolver(self.directory.name, cache_budget_bytes=budget)
        with ExitStack() as stack:
            math_calls = {name:stack.enter_context(patch.object(summary, name, wraps=getattr(summary, name)))
                for name in ('verify_head', 'verify_companion', 'verify_objective', 'verify_coordinate_map')}
            loads = stack.enter_context(patch.object(summary.np, 'load', wraps=np.load))
            result = summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
            resolver.finalize(); counts = {name:call.call_count for name, call in math_calls.items()}
            load_count = loads.call_count
        return result, counts, load_count, resolver

    def test_complete_cached_and_uncached_verification_outputs_and_math_call_counts_match(self):
        plain, plain_math, plain_loads, plain_resolver = self.verify(0)
        cached, cached_math, cached_loads, cached_resolver = self.verify(64*1024*1024)
        self.assertEqual(plain, cached); self.assertEqual(plain_math, cached_math)
        self.assertTrue(all(count > 0 for count in plain_math.values()))
        self.assertLess(cached_loads, plain_loads)
        self.assertEqual(plain_resolver.cache_statistics()['load_count'], plain_loads)
        self.assertEqual(cached_resolver.cache_statistics()['load_count'], cached_loads)
        self.assertGreater(cached_resolver.cache_statistics()['hit_count'], 0)
        self.assertLessEqual(cached_resolver.cache_statistics()['peak_numeric_bytes'], 64*1024*1024)
        self.assertLessEqual(cached_resolver.cache_statistics()['peak_entries'], 64)
        cached_resolver.clear_cache(); plain_resolver.clear_cache()
        self.assertEqual(cached_resolver.cache_statistics()['resident_numeric_bytes'], 0)
        self.assertEqual(cached_resolver.used, plain_resolver.used)

    def test_prewarmed_array_cache_does_not_hide_changed_mathematical_evidence(self):
        resolver = summary.StateResolver(self.directory.name)
        summary.verify_record(self.record, self.split, self.inputs['old_classes'], resolver)
        record = deepcopy(self.record)
        record['folds'][0]['candidate_stages'][0]['initial_objective']['class_ce_means'][0] += .1
        with patch.object(summary, 'verify_objective', wraps=summary.verify_objective) as objectives:
            with self.assertRaises(ValueError): summary.verify_record(record, self.split, self.inputs['old_classes'], resolver)
            self.assertGreater(objectives.call_count, 0)


if __name__ == '__main__': unittest.main()
