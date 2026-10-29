from copy import deepcopy
from contextlib import contextmanager
import importlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.pamf_phase3_open_world import source_open_export as original
from experiments.pamf_phase3_open_world import source_open_readout as core


COUNTS = dict(source_L=6, source_V=6, exposure_train=6, novel_validation=6)
VIEWS = ['clean', 'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban']
ROLES = list(COUNTS)


class FakeReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25, dtype=torch.float32))
        self.register_buffer('fixed', torch.zeros(1, dtype=torch.float32))
        self.seen = []
        self.eval().requires_grad_(False)

    def features(self, x):
        self.seen.append(x.detach().cpu().clone())
        return x[:, 0, 0, None].repeat(1, 160)

    def classify_features(self, z):
        return z[:, :6] + torch.arange(6, dtype=torch.float32, device=z.device)[None]


class FakeDura(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('factor', torch.tensor(3., dtype=torch.float32))
        self.eval().requires_grad_(False)

    def forward(self, x):
        return x * self.factor


class SourcePracticalExportTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_practical_export.py'
        self.assertTrue(path.is_file(), 'Frozen source practical exporter is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_practical_export')

    def fixture(self, root):
        config = dict(run_id='20261029-phase3-source-practical-readout-s2026092701-r01', code_commit='a' * 40,
            source_row_root=str(root / '20261029-phase3-source-embedding-readout-s2026092701-r01/rows/uniform_oe'),
            risk_prediction_root=str(root / '20261029-phase3-source-baseline-risk-s2026092701-r01/rows/uniform_oe/predict'),
            oe_data_root=str(root / 'old-release/data/source_oe'), dura_checkpoint=str(root / 'R69/final.pt'),
            dura_release=str(root / 'frozen-G'), output_root=str(root / 'export'), batch_size=128)
        geometry = dict(schema=core.GEOMETRY_SCHEMA, class_count=6, branches={})
        for name, dimension in core.DIMENSIONS.items():
            geometry['branches'][name] = dict(mean=np.zeros(dimension), scale=np.ones(dimension),
                prototypes=np.eye(6, dimension), whitener=np.eye(dimension))
        handles = {role: dict(capsule_id='capsule-' + role, split_id='split-' + role, count=6)
                   for role in ('exposure_train', 'novel_validation')}
        marker = dict(encoder_state_identity={'fixed': 'same E51 uniform_oe'},
            parent_export_identity={'fixed': 'same original F0'}, source_OE_handles=handles)
        source = dict(marker=marker, geometry=geometry, state={}, thresholds=np.zeros(7, np.float64),
            train_config={'fixed': 'source config'})
        datasets = {}
        source_ids = {}
        n = np.arange(256)
        for role in ROLES:
            # Amplitudes deliberately differ, so an unauthorized second RMS
            # normalization cannot accidentally pass the fixture.
            iq = np.stack([np.stack(((1 + i / 5) * np.cos(.1 * n), (1 + i / 5) * np.sin(.1 * n)))
                           for i in range(6)]).astype(np.float32)
            ids = np.asarray([role + '-' + str(i) for i in range(6)])
            rx = np.asarray([1, 3, 4, 6, 8, 1], np.int64)
            day = np.asarray([1, 2, 3, 1, 2, 3], np.int64)
            meta = dict(ids=ids, rx_i=rx, day_i=day,
                domain=np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1)
            if role in ('source_L', 'source_V'):
                meta.update(y=np.arange(6, dtype=np.int64), eq_i=np.ones(6, np.int64))
                native = 'L_s' if role == 'source_L' else 'V'
                source_ids[native] = ids.tolist()
                datasets[native] = SimpleNamespace(x=torch.frombuffer(bytearray(iq.tobytes()), dtype=torch.float32).reshape(iq.shape), metadata=meta)
            else:
                public = Path(config['oe_data_root']) / role / 'public'
                public.mkdir(parents=True)
                np.save(public / 'iq.npy', iq, allow_pickle=False)
                np.save(public / 'ids.npy', ids, allow_pickle=False)
                np.savez(public / 'metadata.npz', **{key: meta[key] for key in ('rx_i', 'day_i', 'domain')})
        contract = dict(role_ids=source_ids)
        return config, source, datasets, contract, handles

    def encode(self, reader, x):
        fused = reader.features(x)
        data = {'features__' + name: np.ones((len(x), dimension), np.float32)
                for name, dimension in core.DIMENSIONS.items()}
        data['features__fused160'] = np.asarray(fused.tolist(), np.float32)
        data['features__reference29'][:, 27] = 0.
        data['logits'] = np.asarray(reader.classify_features(fused).tolist(), np.float32)
        return data

    @contextmanager
    def backend(self, module, source, datasets, contract, handles):
        reader, baseline, dura = FakeReader(), FakeReader(), FakeDura()
        with patch.object(module, 'COUNTS', COUNTS), patch.object(original, 'COUNTS', COUNTS), \
             patch.object(module.frozen, '_source_bundle', return_value=source), \
             patch.object(module.frozen, '_load_readers', return_value=(reader, baseline)), \
             patch.object(module.frozen, '_load_dura', return_value=(dura, lambda x: (dura(x),))), \
             patch.object(module.frozen, '_encode_new', side_effect=self.encode), \
             patch.object(module, '_source_contract', return_value=contract), \
             patch.object(module.runner, '_source_dataset', side_effect=lambda config, role, actual: datasets[role]) as source_loader, \
             patch.object(original, '_oe_handles', return_value=handles):
            yield reader, baseline, dura, source_loader

    def fake_channel(self, normalized, cfg, seed, receiver):
        self.assertEqual(normalized.dtype, np.float32)
        self.assertEqual(normalized.shape, (2, 256))
        return normalized * np.float32(2), {'processing_route': 'residual'}

    def test_full_fake_export_exact_four_roles_views_schema_and_last_marker(self):
        module = self.module()
        from experiments.phase1_external_dg import data as raw_source
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, source, datasets, contract, handles = self.fixture(root)
            source_before = deepcopy(source)
            writes = []
            actual_write = module._write
            def record(path, value):
                if Path(path).name == 'complete.json':
                    self.assertEqual(len(list((root / 'export').glob('*/*/data.npz'))), 16)
                writes.append(Path(path).name)
                return actual_write(path, value)
            with self.backend(module, source, datasets, contract, handles) as (reader, baseline, dura, loader), \
                 patch.object(module, '_apply_channel', side_effect=self.fake_channel) as channel, \
                 patch.object(raw_source, 'iq_record', side_effect=AssertionError('No second normalization')), \
                 patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('No NumPy ABI bridge')), \
                 patch.object(module, '_write', side_effect=record):
                result = module.export(config, 'cpu')
                self.assertEqual(result['status'], 'SOURCE_PRACTICAL_DESCRIPTORS_EXPORTED')
                self.assertEqual(result['physical_record_count'], 24)
                self.assertEqual(result['derived_view_record_count'], 96)
                self.assertEqual(result['view_order'], VIEWS)
                self.assertEqual(channel.call_count, 6 + 24 * 3)
                self.assertEqual([call.args[1] for call in loader.call_args_list], ['L_s', 'V'])
                self.assertEqual(sum(len(value) for value in reader.seen), 8 + 96)
                self.assertEqual(sum(len(value) for value in baseline.seen), 8 + 12 * 4)
                self.assertTrue(torch.equal(dura.factor, torch.tensor(3.)))
                for role in ROLES:
                    ids = None
                    for view in VIEWS:
                        path = root / 'export' / role / view / 'data.npz'
                        with np.load(path, allow_pickle=False) as data:
                            expected = {'ids', 'descriptor', 'static_scores', 'logits', 'baseline_closed', 'rx_i', 'day_i'}
                            if role in ('source_L', 'source_V'): expected.add('y')
                            self.assertEqual(set(data.files), expected)
                            self.assertEqual((data['descriptor'].dtype, data['descriptor'].shape), (np.dtype('float32'), (6, 100)))
                            self.assertEqual((data['static_scores'].dtype, data['static_scores'].shape), (np.dtype('float64'), (6, 5)))
                            self.assertEqual((data['logits'].dtype, data['logits'].shape), (np.dtype('float32'), (6, 6)))
                            self.assertEqual(data['baseline_closed'].dtype, np.int64)
                            if role in ('source_L', 'exposure_train'):
                                self.assertTrue(bool((data['baseline_closed'] == -2).all()))
                            else:
                                self.assertTrue(bool(((data['baseline_closed'] >= 0) & (data['baseline_closed'] < 6)).all()))
                            if ids is None: ids = data['ids'].copy()
                            else: self.assertTrue(np.array_equal(ids, data['ids']))
                        arrays = module.load_role_view(root / 'export', role, view, include_y=role in ('source_L', 'source_V'))
                        self.assertEqual('y' in arrays, role in ('source_L', 'source_V'))
                marker = module.export_marker(root / 'export')
                self.assertEqual(marker['geometry_source_fit_root'], str(Path(config['source_row_root']) / 'fit'))
                self.assertEqual(marker['descriptor_schema'], 'head_aligned_branch_summary_v1')
                self.assertEqual(marker['baseline_available'], dict(source_L=False, source_V=True, exposure_train=False, novel_validation=True))
                self.assertEqual(writes[-1], 'complete.json')
            self.assertTrue(np.array_equal(source['geometry']['branches']['fused160']['prototypes'], source_before['geometry']['branches']['fused160']['prototypes']))

    def test_role_blind_seeds_actual_RX_day_mapping_and_no_normalization(self):
        module = self.module()
        api = module._canonical_channel_api()
        channels = {scene: module._channel_config(scene) for scene in VIEWS[1:]}
        normalized = np.arange(512, dtype=np.float32).reshape(1, 2, 256) / 100
        meta = dict(ids=np.asarray(['same-opaque-id']), rx_i=np.asarray([3], np.int64), day_i=np.asarray([2], np.int64))
        received = []
        calls = []
        receiver_for_session = api[1]
        def record(cfg, seed, session):
            calls.append((cfg.scenario, seed, session))
            return receiver_for_session(cfg, seed, session)
        changed_api = (api[0], record, api[2], api[3])
        def capture(raw, cfg, seed, receiver):
            self.assertTrue(np.array_equal(raw, normalized[0]))
            self.assertEqual(seed, api[2](2026102931, cfg.scenario, 'same-opaque-id'))
            received.append(raw.copy())
            return raw * 2, {}
        sessions = {}
        with patch.object(module, '_apply_channel', side_effect=capture):
            for view in VIEWS[1:]:
                plain = module._synthesize(normalized, meta, view, channels, changed_api, sessions)
                labeled = module._synthesize(normalized, dict(meta, y=np.asarray([5])), view, channels, changed_api, sessions)
                self.assertTrue(np.array_equal(plain, normalized * 2))
                self.assertTrue(np.array_equal(plain, labeled))
        self.assertEqual(calls, [(VIEWS[1], 2026102912, '18-2/2021_03_15/1')])
        self.assertEqual(len(received), 6)
        self.assertNotIn('role', inspect.signature(module._synthesize).parameters)
        for bad in [dict(meta, rx_i=np.asarray([0], np.int64)), dict(meta, day_i=np.asarray([0], np.int64))]:
            with patch.object(module, '_apply_channel', side_effect=AssertionError('Validate source session first')), self.assertRaises(ValueError):
                module._synthesize(normalized, bad, VIEWS[1], channels, api, {})

    def test_canonical_channel_replaces_stale_import_before_frozen_code_activation(self):
        module = self.module()
        stale = ModuleType('leo_practical'); stale.__file__ = '/old-frozen/code/leo_practical/__init__.py'
        stale.__path__ = ['/old-frozen/code/leo_practical']
        child = ModuleType('leo_practical.channel'); child.__file__ = '/old-frozen/code/leo_practical/channel.py'
        with patch.dict(sys.modules, {'leo_practical': stale, 'leo_practical.channel': child}):
            api = module._canonical_channel_api()
            canonical = Path(module.__file__).resolve().parents[2] / 'code/leo_practical'
            self.assertTrue(Path(sys.modules['leo_practical.channel'].__file__).resolve().is_relative_to(canonical))
            self.assertEqual(api[0].__module__, 'leo_practical.channel')
        with tempfile.TemporaryDirectory() as tmp:
            config, source, datasets, contract, handles = self.fixture(Path(tmp))
            events = []
            original_api = module._canonical_channel_api
            with self.backend(module, source, datasets, contract, handles), \
                 patch.object(module, '_apply_channel', side_effect=self.fake_channel), \
                 patch.object(module, '_canonical_channel_api', side_effect=lambda: (events.append('canonical'), original_api())[1]), \
                 patch.object(module.frozen, '_load_readers', side_effect=lambda *args: (events.append('F'), (FakeReader(), FakeReader()))[1]):
                module.export(config, 'cpu')
            self.assertEqual(events[:2], ['canonical', 'F'])

    def test_real_ResidualChannel_three_scenes_small_synthetic_IQ(self):
        module = self.module()
        api = module._canonical_channel_api()
        channels = {scene: module._channel_config(scene) for scene in VIEWS[1:]}
        n = np.arange(256)
        iq = np.asarray([[np.cos(.13*n), np.sin(.13*n)]], np.float32)
        meta = dict(ids=np.asarray(['synthetic-only']), rx_i=np.asarray([1], np.int64), day_i=np.asarray([3], np.int64))
        for scene in VIEWS[1:]:
            cfg = channels[scene]
            self.assertEqual((cfg.fs_hz, cfg.fc_hz, cfg.processing_route, cfg.mode, cfg.equalization_enabled),
                             (25e6, 2462e6, 'residual', 'post_sync', False))
            self.assertEqual((cfg.reference_snr_db, cfg.boundary), (30., 'reflect'))
            first = module._synthesize(iq, meta, scene, channels, api, {})
            second = module._synthesize(iq, meta, scene, channels, api, {})
            self.assertEqual((first.dtype, first.shape), (np.dtype('float32'), (1, 2, 256)))
            self.assertTrue(np.isfinite(first).all())
            self.assertTrue(np.array_equal(first, second))

    def test_config_immutability_output_collision_and_no_forbidden_role_inputs(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, source, datasets, contract, handles = self.fixture(root)
            self.assertIs(module.validate_config(config), config)
            for key, value in [('target_root', 'forbidden'), ('truth', 'forbidden'), ('scenes', ['clean']),
                    ('receiver_seed', 1), ('batch_size', 64), ('oe_data_root', '')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: value}))
            Path(config['output_root']).mkdir()
            with patch.object(module, '_canonical_channel_api', side_effect=AssertionError('No IO after collision')), self.assertRaises(FileExistsError):
                module.export(config, 'cpu')

    def test_smoke_before_any_source_role_and_failures_never_complete(self):
        module = self.module()
        for failure in ('smoke', 'state', 'source_ID_overlap', 'OE_handle'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config, source, datasets, contract, handles = self.fixture(root)
                with self.backend(module, source, datasets, contract, handles) as (reader, baseline, dura, loader), \
                     patch.object(module, '_apply_channel', side_effect=self.fake_channel):
                    if failure == 'smoke':
                        change = patch.object(module.frozen, '_encode_new', side_effect=RuntimeError('synthetic smoke failure'))
                    elif failure == 'state':
                        def mutate(model, x):
                            model.fixed.add_(1)
                            return self.encode(model, x)
                        change = patch.object(module.frozen, '_encode_new', side_effect=mutate)
                    elif failure == 'source_ID_overlap':
                        public = Path(config['oe_data_root']) / 'exposure_train/public/ids.npy'
                        value = np.load(public, allow_pickle=False); value[0] = datasets['L_s'].metadata['ids'][0]
                        np.save(public, value, allow_pickle=False)
                        change = patch.object(module.runner, '_source_dataset', side_effect=lambda cfg, role, actual: datasets[role])
                    else:
                        change = patch.object(original, '_oe_handles', return_value={})
                    with change, self.assertRaises((ValueError, RuntimeError)):
                        module.export(config, 'cpu')
                    if failure in ('smoke', 'state', 'OE_handle'):
                        self.assertEqual(loader.call_count, 0)
                self.assertFalse((root / 'export/complete.json').exists())

    def test_loader_rejects_unknown_labels_and_partial_export_before_any_array_read(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, source, datasets, contract, handles = self.fixture(Path(tmp))
            with self.backend(module, source, datasets, contract, handles), \
                 patch.object(module, '_apply_channel', side_effect=self.fake_channel):
                module.export(config, 'cpu')
                with patch.object(np, 'load', side_effect=AssertionError('No unknown labels may be opened')):
                    for role in ('exposure_train', 'novel_validation'):
                        with self.subTest(role=role), self.assertRaises(ValueError):
                            module.load_role_view(config['output_root'], role, 'clean', include_y=True)
                marker = Path(config['output_root']) / 'complete.json'
                marker.unlink()
                with patch.object(np, 'load', side_effect=AssertionError('No partial arrays')), self.assertRaises(ValueError):
                    module.load_role_view(config['output_root'], 'source_L', 'clean', include_y=True)

    def test_known_source_labels_are_only_materialized_when_explicitly_requested(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, source, datasets, contract, handles = self.fixture(Path(tmp))
            with self.backend(module, source, datasets, contract, handles), \
                 patch.object(module, '_apply_channel', side_effect=self.fake_channel):
                module.export(config, 'cpu')
                actual_load = np.load
                @contextmanager
                def labels_closed(*args, **kwargs):
                    with actual_load(*args, **kwargs) as stored:
                        class PublicOnly:
                            files = stored.files
                            def __getitem__(self, key):
                                if key == 'y':
                                    raise AssertionError('include_y=False must not materialize labels')
                                return stored[key]
                        yield PublicOnly()
                with patch.object(np, 'load', side_effect=labels_closed):
                    for role in ('source_L', 'source_V'):
                        with self.subTest(role=role):
                            values = module.load_role_view(config['output_root'], role, 'clean')
                            self.assertEqual(set(values), module.DATA_FIELDS)


if __name__ == '__main__':
    unittest.main()
