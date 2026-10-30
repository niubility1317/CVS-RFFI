"""Focused synthetic boundaries for source-only neighbor export and mining."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class ForbiddenP1IQ:
    def __getitem__(self, key):
        raise AssertionError('P1 IQ must be reused as teacher features, never encoded')


class FrozenReader(torch.nn.Module):
    def __init__(self, events, failure=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_buffer('offset', torch.zeros(1))
        self.events = events
        self.failure = failure
        self.eval().requires_grad_(False)

    def features(self, x):
        assert not torch.is_grad_enabled()
        is_smoke = bool((x[:, 0, 0] < 0).all())
        self.events.append(('smoke' if is_smoke else 'P0', len(x)))
        if is_smoke and self.failure == 'smoke':
            raise RuntimeError('synthetic checkpoint failure')
        if not is_smoke and self.failure == 'mutation':
            self.weight.add_(1.)
        z = torch.zeros((len(x), 160), dtype=torch.float32, device=x.device)
        z[:, 0] = self.weight
        z[:, 1] = x[:, 0, 0] + self.offset
        return z


class SourceNeighborIOTests(unittest.TestCase):
    def module(self, name):
        return importlib.import_module('experiments.pamf_phase3_open_world.star_source_neighbor_' + name)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def fixture(self, root):
        m = self.module('export')
        config = dict(run_id=m.RUN_ID, code_commit='a' * 40,
            reader_config=dict(base_release=str(root / 'release_F0'), source_output=str(root / 'original_F0'),
                frozen_sources={k: str(root / k) for k in ('F', 'AF', 'BASE', 'FOLD', 'G')}, model_seed=2026092701),
            source_contract=str(root / 'source_contract.json'), cache_root=str(root / 'cache'),
            teacher_root=str(root / 'teacher'), output_root=str(root / 'run/features'), batch_size=128)
        parent = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100, model_seed=2026092701)
        dura = dict(deployment_id='dura_v1', parameters=266951, epoch=20, updates=4940)
        counts = dict(L=6, OE=4)
        roles, units = {}, {}
        for index, (role, n) in enumerate(counts.items()):
            ids = np.asarray([role + '-opaque-' + str(i) for i in range(n)])
            metadata = dict(rx_i=np.asarray([1, 3, 4, 6, 8, 1][:n], np.int64),
                day_i=np.arange(n, dtype=np.int64) % 3 + 1, domain=np.arange(n, dtype=np.int64) % 15)
            if role == 'L':
                metadata['y'] = np.arange(n, dtype=np.int64)
            manifest = dict(role=role, count=n, parent_F0_identity=parent, dura_identity=dura,
                all_new_IQ_satellite=True, restored_input_path='P1')
            received = np.zeros((n, 2, 256), np.float32)
            received[:, 0, 0] = index + 2
            roles[role] = dict(ids=ids, metadata=metadata, manifest=manifest,
                               received=received, restored=ForbiddenP1IQ())
            units[role] = np.zeros((n, 160), np.float32)
            units[role][np.arange(n), np.arange(n) + 80] = 1.
        marker = dict(schema=m.teacher.TEACHER_SCHEMA, status=m.teacher.TEACHER_STATUS,
            code_commit='5aa041122396d48c49b8cb3d5912649bdc3217e1', parent_F0_identity=parent,
            dura_identity=dura, cache_handles={r: b['manifest'] for r, b in roles.items()},
            source_only=True, model_buffers_unchanged=True, feature='unit_fused160', input_path='cached_P1_only',
            target_reads=0, validation_reads=0, novel_validation_reads=0, original_U_s_reads=0,
            OE_truth_reads=0, encoder_updates=0, identity_head_updates=0, G_updates=0)
        teacher_root = Path(config['teacher_root'])
        self.write_json(teacher_root / 'complete.json', marker)
        for role in counts:
            np.savez(teacher_root / (role + '.npz'), ids=roles[role]['ids'], unit_features=units[role])
        return config, roles, units, parent, counts

    def execute_export(self, m, fixture, failure=None):
        config, roles, units, parent, counts = fixture
        events = []
        reader = FrozenReader(events, failure)
        before = {k: v.clone() for k, v in reader.state_dict().items()}
        smoke = torch.zeros((2, 2, 256), dtype=torch.float32)
        smoke[:, 0, 0] = -1.

        def load(root, role):
            self.assertEqual(events[0], ('smoke', 2))
            self.assertTrue((Path(config['output_root']) / 'smoke.json').is_file())
            self.assertEqual(Path(root), Path(config['cache_root']) / role)
            events.append(('load', role))
            return roles[role]

        with patch.dict(m.COUNTS, counts, clear=True), \
             patch.object(m.original, '_backend', return_value=dict(reader=reader, reader_identity=parent)) as backend, \
             patch.object(m.feature, '_synthetic', return_value=smoke), \
             patch.object(m, 'load_role', side_effect=load), patch.object(m, 'close_role') as close:
            done = m.export(config, 'cpu')
        backend.assert_called_once_with(config, 'cpu', with_dura=False)
        self.assertEqual(close.call_count, 2)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in reader.state_dict().items()))
        return done, events

    def test_export_rejects_private_fields_and_input_or_existing_output_collisions(self):
        m = self.module('export')
        for case in ('truth', 'target', 'overlap', 'existing'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                config, *_ = self.fixture(Path(td))
                if case == 'truth':
                    config['novel_truth'] = 'forbidden.npz'
                elif case == 'target':
                    config['target_root'] = 'forbidden-target'
                elif case == 'overlap':
                    config['output_root'] = str(Path(config['cache_root']) / 'new')
                else:
                    Path(config['output_root']).mkdir(parents=True)
                with patch.object(m.original, '_backend') as backend:
                    with self.assertRaises(FileExistsError if case == 'existing' else ValueError):
                        m.export(config, 'cpu')
                    backend.assert_not_called()

    def test_teacher_rejects_wrong_parent_cache_handle_or_extra_TX(self):
        m = self.module('export')
        for case in ('parent', 'cache', 'TX'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                config, roles, units, parent, counts = self.fixture(Path(td))
                root = Path(config['teacher_root'])
                marker = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
                if case == 'parent':
                    marker['parent_F0_identity']['model'] = 'candidate_not_original_F0'
                elif case == 'cache':
                    marker['cache_handles']['OE']['count'] += 1
                else:
                    np.savez(root / 'OE.npz', ids=roles['OE']['ids'], unit_features=units['OE'], tx=np.zeros(counts['OE']))
                self.write_json(root / 'complete.json', marker)
                with patch.dict(m.COUNTS, counts, clear=True), self.assertRaises(ValueError):
                    m._teacher(root, roles['OE'], 'OE', parent)

    def test_smoke_precedes_roles_only_P0_is_encoded_and_P1_teacher_is_exact(self):
        m = self.module('export')
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(Path(td))
            config, roles, units, _, counts = fixture
            done, events = self.execute_export(m, fixture)
            self.assertEqual(events, [('smoke', 2), ('load', 'L'), ('P0', 6), ('load', 'OE'), ('P0', 4)])
            self.assertEqual(done['real_F_forward_rows'], sum(counts.values()))
            self.assertEqual(done['P1_feature_rows_reused'], sum(counts.values()))
            self.assertEqual(done['restored_cache_values_read'], 0)
            self.assertEqual(done['model_updates'], 0)
            self.assertEqual(done['G_forward_rows'], 0)
            for role in counts:
                with np.load(Path(config['output_root']) / (role + '.npz')) as saved:
                    self.assertTrue(np.array_equal(saved['P1'], units[role]))
                    self.assertTrue(np.array_equal(saved['ids'], roles[role]['ids']))
                    self.assertEqual('y' in saved.files, role == 'L')
            self.assertEqual(done['status'], m.STATUS)

    def test_failed_smoke_or_mutating_backbone_cannot_complete(self):
        m = self.module('export')
        for failure in ('smoke', 'mutation'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                fixture = self.fixture(Path(td))
                with self.assertRaises(RuntimeError if failure == 'smoke' else ValueError):
                    self.execute_export(m, fixture, failure)
                self.assertFalse((Path(fixture[0]['output_root']) / 'complete.json').exists())
                if failure == 'smoke':
                    self.assertFalse(Path(fixture[0]['output_root']).exists())

    def test_miner_rejects_OE_y_or_TX_without_loading_algorithm_inputs(self):
        m = self.module('mine')
        for field in ('y', 'tx'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                n = 4
                unit = np.zeros((n, 160), np.float32)
                unit[:, 0] = 1.
                fields = dict(ids=np.asarray(['oe-' + str(i) for i in range(n)]),
                    rx_i=np.asarray([1, 3, 4, 6], np.int64), day_i=np.ones(n, np.int64), P0=unit, P1=unit.copy())
                fields[field] = np.zeros(n, np.int64)
                np.savez(root / 'OE.npz', **fields)
                with patch.dict(m.exported.COUNTS, dict(L=6, OE=n), clear=True), self.assertRaises(ValueError):
                    m._load(root, 'OE')

    def test_miner_rejects_incomplete_feature_marker_before_feature_arrays(self):
        m = self.module('mine')
        for case in ('status', 'frozen'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                config, _, _, _, counts = self.fixture(root)
                feature_root = Path(config['output_root'])
                marker = dict(config=config, schema=m.exported.SCHEMA, status=m.exported.STATUS,
                    run_id=m.exported.RUN_ID, code_commit=config['code_commit'], counts=counts, source_only=True,
                    model_states_unchanged=True, model_updates=0, optimizer_updates=0, target_reads=0,
                    private_truth_reads=0, source_V_reads=0, source_novel_reads=0, original_U_s_reads=0)
                marker['status' if case == 'status' else 'model_states_unchanged'] = 'RUNNING' if case == 'status' else False
                self.write_json(feature_root / 'complete.json', marker)
                with patch.dict(m.exported.COUNTS, counts, clear=True), patch.object(m, '_load') as load:
                    with self.assertRaises(ValueError):
                        m.mine(feature_root, root / 'mining', config['code_commit'])
                    load.assert_not_called()
                self.assertFalse((root / 'mining/complete.json').exists())

    def runtime(self, root):
        config, *_ = self.fixture(root)
        return dict(run_id=config['run_id'], code_commit=config['code_commit'], run_root=str(root / 'run'),
            release_root=str(root / 'release'), python='synthetic-python', launch_owner='test-owner',
            allowed_gpus=list(range(8)), minimum_free_mb=4096, max_added_workers_per_gpu=2, export_config=config)

    def test_launcher_export_failure_never_starts_miner(self):
        m = self.module('launch')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = self.runtime(root)
            Path(runtime['run_root']).mkdir()
            path = root / 'runtime.json'
            self.write_json(path, runtime)
            with patch.object(m, '_gpu_free', return_value=[2]), patch.object(m, '_job', return_value=False) as job:
                m.driver(path)
            self.assertEqual(job.call_count, 1)
            self.assertEqual(job.call_args.args[3], 'export')
            state = json.loads((Path(runtime['run_root']) / 'pipeline_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'FAILED_PRESERVED')
            self.assertIn('no mining', state['error'])
            self.assertFalse((Path(runtime['run_root']) / 'mining').exists())

    def test_launcher_accepts_exact_source_runtime_and_rejects_truth_fields(self):
        m = self.module('launch')
        with tempfile.TemporaryDirectory() as td:
            runtime = self.runtime(Path(td))
            self.assertEqual(m.validate(runtime), runtime)
            for field in ('novel_truth', 'target_truth'):
                with self.subTest(field=field):
                    bad = deepcopy(runtime)
                    bad[field] = 'forbidden.npz'
                    with self.assertRaises(ValueError):
                        m.validate(bad)


if __name__ == '__main__':
    unittest.main()
