import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing  # Load dependency metadata before the artifact-read boundary guard.
import scipy.linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import evaluate_d92_branch_local_margin_probe as entry
from export_d92_branch_support_features import CACHE_SCHEMA, CACHE_NAME, FEATURE_CONTRACT


def fixture(root, ks=(1, 2, 5)):
    """Write synthetic bound caches directly: no exporter, encoder, IQ or Torch."""
    cache = root / 'cache'
    capsule = root / 'capsule'
    cache.mkdir()
    capsule.mkdir()
    sha, cid = 'a' * 64, 'synthetic-capsule'
    classes, old = ['TX9', 'TX2'], ['TX9']
    n = 2 * max(ks)
    ids = np.array([f'physical-{i:03}' for i in range(n)])
    indices = np.arange(n, dtype=np.int64) + 10
    labels = np.array([classes[i // max(ks)] for i in range(n)])
    rng = np.random.default_rng(61)
    arrays = {key: rng.normal(size=(n, 96 if key == 'fft' else 160)).astype(np.float32)
              for key in entry.BRANCHES}
    np.savez(cache / CACHE_NAME, **arrays, ids=ids, indices=indices, labels=labels,
             checkpoint_sha256=np.array(sha), capsule_id=np.array(cid),
             feature_contract_json=np.array(json.dumps(FEATURE_CONTRACT)))
    splits = []
    for k in ks:
        positions = list(range(k)) + list(range(max(ks), max(ks) + k))
        splits.append(dict(split_id=f'split-k{k}', receiver='RX', scenario='scene', k=k,
            support_seed=7, registered_classes=classes, support_indices=indices[positions].tolist(),
            support_ids=ids[positions].tolist(), support_labels=[0] * k + [1] * k))
    provenance = dict(checkpoint_sha256=sha, classes=old, model_seed=3,
        verdict='MATCHED_SOURCE_ONLY_SCRATCH', source_role_comparison='EXACT_MATCH',
        checkpoint_epoch=200, checkpoint_inheritance=[], target_access_before_freeze=False,
        model_file_bytes=100)
    common = dict(schema=CACHE_SCHEMA, checkpoint_sha256=sha, capsule_id=cid, model_seed=3,
        feature_contract=FEATURE_CONTRACT, query_iq_access=False, query_rows_read=0,
        source_data_access=False, truth_read=False, adapted_state_inherited=False,
        native_eval=True, new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        model_file_bytes=100)
    startup = dict(common, provenance=provenance, query_fit_access=False)
    marker = dict(common, status='BRANCH_SUPPORT_FEATURES_COMPLETE', classes=old,
        dtype='float32', view_count_per_observation=1, query_used_for_fitting=False,
        encoder_updated=False, native_parameters_unchanged=True, native_buffers_unchanged=True,
        count=n, split_count=len(splits), feature_array_bytes=sum(v.nbytes for v in arrays.values()),
        index_bytes=indices.nbytes, registry_array_bytes=ids.nbytes + labels.nbytes,
        feature_file_bytes=(cache / CACHE_NAME).stat().st_size,
        shapes={key: list(value.shape) for key, value in arrays.items()},
        native_physical_forward_count=n, support_iq_rows_read=n, identity_reference_checks=n,
        identity_check_additional_encoder_forwards=0)
    for name, value in [('features_complete.json', marker), ('startup.json', startup),
                        ('checkpoint_provenance.json', provenance),
                        ('support_splits.json', dict(schema=CACHE_SCHEMA, capsule_id=cid,
                                                    checkpoint_sha256=sha, splits=splits))]:
        entry.write(cache / name, value)
    entry.write(capsule / 'manifest.json', dict(protocol_schema='p2_min_v1',
        phase2_data_status='VALIDATED_ONCE', capsule_id=cid, split_count=len(splits)))
    return dict(support_features=cache, capsule=capsule, output=root / 'out',
        expected_capsule_id=cid, expected_checkpoint_sha256=sha, expected_model_seed=3,
        config=dict(algorithm=entry.FROZEN_CONFIG, matrix=dict(receivers=['RX'],
            scenarios=['scene'], ks=list(ks), new_counts=[1], support_seeds=[7])))


def mutate(path, function):
    value = entry.read(path)
    function(value)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class LocalMarginEntryTests(unittest.TestCase):
    def test_bound_cache_only_complete_schema_and_measured_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            original_open, original_load = Path.open, np.load
            reads, physical = [], []

            def guarded_open(path, *a, **kw):
                mode = a[0] if a else kw.get('mode', 'r')
                if 'r' in mode:
                    reads.append(path)
                    allowed = (path.parent == args['support_features']
                               or path == args['capsule'] / 'manifest.json'
                               or path in (args['output'] / 'fit_stages.jsonl', args['output'] / 'solver_sweeps.jsonl'))
                    self.assertTrue(allowed, str(path))
                return original_open(path, *a, **kw)

            def guarded_load(path, *a, **kw):
                self.assertEqual(Path(path), args['support_features'] / CACHE_NAME)
                return original_load(path, *a, **kw)

            real = entry.probe_branch_local_margin

            def probe(**kw):
                self.assertEqual(set(kw), set(entry.BRANCHES)
                                 | {'support_labels', 'support_ids', 'classes', 'old_classes', 'log_callback'})
                physical.append(len(kw['z_id']))
                return real(**kw)

            with patch.object(Path, 'open', guarded_open), patch.object(np, 'load', guarded_load), \
                 patch.object(entry, 'probe_branch_local_margin', side_effect=probe), \
                 contextlib.redirect_stdout(io.StringIO()):
                marker = entry.evaluate(**args)
            self.assertEqual(physical, [2, 4, 10])
            self.assertTrue(reads)
            self.assertEqual(marker['factorization_count'], 36)
            self.assertEqual(marker['standard_factorization_count'], 15)
            self.assertEqual(marker['proxy_anchor_count'], 7)
            self.assertEqual((marker['episodes'], marker['k1_episodes'], marker['oof_episodes']), (3, 1, 2))
            self.assertEqual(marker['payload_audit']['native_physical_forward_count_this_run'], 0)
            self.assertTrue(marker['payload_audit']['feature_cache_reused'])
            out = args['output']
            compact, traces = lines(out / 'compact.jsonl'), lines(out / 'fit_trace.jsonl')
            self.assertIsNone(compact[0]['oof'])
            self.assertIsNone(compact[0]['oneshot_proxy'])
            self.assertNotIn('trials', compact[1]['oneshot_proxy'])
            for trace in traces[1:]:
                self.assertEqual(set(trace['oof']), set(entry.ARMS))
                self.assertEqual(set(trace['paired']), set(entry.PAIRS))
                self.assertNotIn('ncm_equivalence', json.dumps(trace))
                for fold in trace['folds']:
                    self.assertFalse(set(fold['training_ids']) & set(fold['held_ids']))
                for trial in trace['oneshot_proxy']['trials']:
                    self.assertEqual(len(trial['training_ids']), 2)
                    self.assertEqual(len(trial['held_ids']), 2 * (trace['k'] - 1))
                    self.assertFalse(set(trial['training_ids']) & set(trial['held_ids']))
                    for arm in trial['oof'].values():
                        self.assertNotIn('rows', arm)
                        self.assertEqual(arm['record_count'], len(trial['held_ids']))
            stages = lines(out / 'fit_stages.jsonl')
            self.assertEqual(len(stages), 48)
            self.assertGreater(marker['optimizer_steps'], 0)
            self.assertEqual(marker['optimizer_steps'], sum(s['optimizer_steps'] for s in stages))
            sweeps = lines(out / 'solver_sweeps.jsonl')
            self.assertTrue(sweeps)
            with (out / 'solver_sweeps.csv').open(encoding='utf-8', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(sweeps))
            self.assertIn('LR=N/A source_validation=N/A', (out / 'solver_sweeps.log').read_text(encoding='utf-8'))
            for stage in stages:
                if stage['arm'] == 'local_margin':
                    self.assertGreater(stage['optimizer_steps'], 0)
                    self.assertIn('relative_duality_gap', stage)
                else:
                    self.assertEqual(stage['optimizer_steps'], 0)
                if stage['arm'] == 'local_ridge':
                    for key in ('bandwidth_tau', 'trace_scale', 'interaction_centered_trace',
                                'radial_centered_trace', 'normal_equation_residual',
                                'trace_relative_error', 'degeneracy_reason'):
                        self.assertIn(key, stage)
            with (out / 'fit_stages.csv').open(encoding='utf-8', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(stages))
            self.assertFalse((out / 'probe_failed.json').exists())
            with self.assertRaises(FileExistsError):
                entry.evaluate(**args)

    def test_cache_binding_and_configuration_fail_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            args['config']['algorithm'] = entry.CACHE_VALIDATION_CONFIG
            with self.assertRaises(ValueError):
                entry.evaluate(**args)
            self.assertFalse(args['output'].exists())
        edits = [('features_complete.json', lambda d: d.update(checkpoint_sha256='b' * 64)),
                 ('support_splits.json', lambda d: d['splits'][0].update(query_indices=[0])),
                 ('startup.json', lambda d: d.update(source_data_access=True)),
                 ('checkpoint_provenance.json', lambda d: d.update(target_access_before_freeze=True))]
        for name, change in edits:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                args = fixture(Path(tmp))
                mutate(args['support_features'] / name, change)
                with self.assertRaises(ValueError):
                    entry.evaluate(**args)
                self.assertFalse(args['output'].exists())

    def test_failure_preserves_completed_stages_and_physical_context(self):
        completed = dict(arm='branch_ridge', factorization_count=1, optimizer_steps=0,
                         fit_seconds=.01, fold=0, solver='cholesky')
        context = dict(arm='local_margin', training_physical_ids=['physical-000', 'physical-002'],
                       scope='standard_oof', parent_k=2, train_k=1, fold=0, trial=None,
                       completed_stages=[completed], solver_state={'beta': [[1.]], 'alpha': [[1., -1.]], 'F': [[0., 0.]]}, optimizer_steps=2, failure_context={'trace_relative_error': 1.})

        class SyntheticFailure(RuntimeError):
            def audit_dict(self):
                return context

        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), ks=(2,))
            with patch.object(entry, 'probe_branch_local_margin', side_effect=SyntheticFailure('synthetic')), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SyntheticFailure):
                    entry.evaluate(**args)
            out = args['output']
            failure = entry.read(out / 'probe_failed.json')
            self.assertEqual(failure['completed_stages'], [completed])
            self.assertEqual(failure['solver_state'], context['solver_state'])
            self.assertEqual(failure['optimizer_steps'], 2)
            self.assertEqual(failure['failure_context'], context)
            self.assertEqual(failure['training_physical_ids'], context['training_physical_ids'])
            self.assertEqual(lines(out / 'fit_stages.jsonl'), [dict(completed, split_id='split-k2')])
            self.assertFalse((out / 'probe_complete.json').exists())
            with (out / 'fit_stages.csv').open(encoding='utf-8', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_true_k1_does_not_fit_or_fabricate_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), ks=(1,))
            with contextlib.redirect_stdout(io.StringIO()):
                marker = entry.evaluate(**args)
            self.assertEqual(marker['factorization_count'], 0)
            self.assertEqual(marker['oof_episodes'], 0)
            trace = lines(args['output'] / 'fit_trace.jsonl')[0]
            self.assertEqual(trace['folds'], [])
            self.assertTrue(all(trace[key] is None for key in ('oof', 'paired', 'oneshot_proxy')))
            self.assertEqual(lines(args['output'] / 'fit_stages.jsonl'), [])

    def test_import_requires_no_torch(self):
        script = '''import importlib.abc, sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch import forbidden')
sys.meta_path.insert(0, NoTorch())
sys.path[:0] = sys.argv[1:]
import evaluate_d92_branch_local_margin_probe
'''
        result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'tools'),
                                 str(ROOT / 'code')], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
