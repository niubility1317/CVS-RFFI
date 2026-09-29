import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'code')]
import evaluate_d92_branch_interaction_probe as entry
from test_evaluate_d92_branch_support_probe import fixture as cache_fixture, mutate


def fixture(root, ks=(1,2,5)):
    args = cache_fixture(root,ks=ks)
    args['config']['algorithm'] = entry.FROZEN_CONFIG
    return args


class InteractionEntryTests(unittest.TestCase):
    def test_real_core_reads_only_bound_support_cache_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp)); original = Path.open; reads = []
            def guarded(path,*a,**kw):
                mode = a[0] if a else kw.get('mode','r')
                if 'r' in mode:
                    reads.append(path)
                    allowed = path.parent == args['support_features'] or path == args['capsule']/'manifest.json'
                    self.assertTrue(allowed,str(path))
                return original(path,*a,**kw)
            real = entry.probe_branch_interaction; physical = []
            def probe(**kw):
                self.assertEqual(set(kw),set(entry.BRANCHES)|{'support_labels','support_ids','classes','old_classes'})
                physical.append(len(kw['z_id']))
                return real(**kw)
            with patch.object(Path,'open',guarded),patch.object(entry,'probe_branch_interaction',side_effect=probe),contextlib.redirect_stdout(io.StringIO()):
                marker = entry.evaluate(**args)
            self.assertEqual(physical,[2,4,10]); self.assertTrue(reads)
            self.assertEqual(marker['factorization_count'],15)
            self.assertEqual((marker['episodes'],marker['k1_episodes'],marker['oof_episodes']),(3,1,2))
            self.assertEqual(marker['payload_audit']['native_physical_forward_count_this_run'],0)
            self.assertTrue(marker['payload_audit']['feature_cache_reused'])
            out = args['output']
            compact = [json.loads(line) for line in (out/'compact.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertIsNone(compact[0]['oof'])
            self.assertNotIn('rows',compact[1]['oof']['interaction'])
            with (out/'fit_stages.csv').open(encoding='utf-8',newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),15)
            with self.assertRaises(FileExistsError): entry.evaluate(**args)

    def test_new_algorithm_and_old_cache_contract_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp)); args['config']['algorithm'] = entry.CACHE_VALIDATION_CONFIG
            with self.assertRaises(ValueError): entry.evaluate(**args)
            self.assertFalse(args['output'].exists())
        for name,change in [('features_complete.json',lambda d:d.update(checkpoint_sha256='b'*64)),
                            ('support_splits.json',lambda d:d['splits'][0].update(query_indices=[0])),
                            ('startup.json',lambda d:d.update(source_data_access=True))]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                args = fixture(Path(tmp)); mutate(args['support_features']/name,change)
                with self.assertRaises(ValueError): entry.evaluate(**args)
                self.assertFalse(args['output'].exists())

    def test_failure_preserves_partial_without_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            with patch.object(entry,'probe_branch_interaction',side_effect=RuntimeError('synthetic')),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError): entry.evaluate(**args)
            self.assertTrue((args['output']/'startup.json').exists())
            self.assertFalse((args['output']/'probe_complete.json').exists())

    def test_k1_has_no_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp),ks=(1,))
            with contextlib.redirect_stdout(io.StringIO()): marker = entry.evaluate(**args)
            self.assertEqual(marker['factorization_count'],0)
            self.assertEqual(marker['oof_episodes'],0)


if __name__ == '__main__': unittest.main()
