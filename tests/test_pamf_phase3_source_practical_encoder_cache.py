from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.pamf_phase3_open_world import source_open_export as original
from experiments.pamf_phase3_open_world import source_open_readout as readout


COUNTS = {'source_L': 30, 'source_OE': 15}
VIEWS = ('clean', 'practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')


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

    def classify_features(self, features):
        return features[:, :6] + torch.arange(6, dtype=torch.float32, device=features.device)[None]


class FakeDura(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('factor', torch.tensor(3., dtype=torch.float32))
        self.eval().requires_grad_(False)

    def forward(self, x):
        return x * self.factor


class SourcePracticalEncoderCacheTests(unittest.TestCase):
    def module(self):
        name = 'experiments.pamf_phase3_open_world.source_practical_encoder_cache'
        self.assertTrue((Path(__file__).resolve().parents[1] / (name.replace('.', '/') + '.py')).is_file(),
                        'Source-only encoder IQ cache is not implemented')
        return importlib.import_module(name)

    def fixture(self, root):
        config = dict(run_id='20261029-phase3-source-practical-encoder-s2026092701-r01', code_commit='a' * 40,
            source_row_root=str(root / '20261029-phase3-source-embedding-readout-s2026092701-r01/rows/uniform_oe'),
            risk_prediction_root=str(root / '20261029-phase3-source-baseline-risk-s2026092701-r01/rows/uniform_oe/predict'),
            diverse_data_root=str(root / 'source21'), dura_checkpoint=str(root / 'R69/final.pt'),
            dura_release=str(root / 'frozen-G'), output_root=str(root / 'cache'), batch_size=128)
        geometry = dict(schema=readout.GEOMETRY_SCHEMA, class_count=6, branches={})
        for name, dimension in readout.DIMENSIONS.items():
            geometry['branches'][name] = dict(mean=np.zeros(dimension), scale=np.ones(dimension),
                prototypes=np.eye(6, dimension), whitener=np.eye(dimension))
        source = dict(marker=dict(encoder_state_identity={'fixed': 'E51 uniform_oe'},
            parent_export_identity={'fixed': 'original F0'}), geometry=geometry, state={},
            thresholds=np.zeros(7, np.float64), train_config={'source_contract': 'source contract'})
        native = None
        contract = dict(classes=['14-10', '14-7', '20-15', '20-19', '6-15', '8-20'],
            source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3], split_seed=392005,
            counts=dict(L_s=30, U_s=56700, V=27000), role_ids={})
        n = np.arange(256)
        inputs = {}
        for role, count in COUNTS.items():
            iq = np.asarray([[(1 + i / 10) * np.cos(.12 * n), (1 + i / 10) * np.sin(.12 * n)]
                             for i in range(count)], np.float32)
            rx = np.tile(np.repeat([1, 3, 4, 6, 8], 3), count // 15).astype(np.int64)
            day = np.tile([1, 2, 3], count // 3).astype(np.int64)
            meta = dict(ids=np.asarray([role + '-opaque-' + str(i) for i in range(count)]),
                rx_i=rx, day_i=day, domain=np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1)
            if role == 'source_L':
                meta.update(y=np.arange(count, dtype=np.int64) % 6, eq_i=np.ones(count, np.int64),
                    sig_i=np.arange(count, dtype=np.int64) % 6, base_index=np.arange(count, dtype=np.int64) + 100)
                contract['role_ids']['L_s'] = meta['ids'].tolist()
                x = torch.frombuffer(bytearray(iq.tobytes()), dtype=torch.float32).reshape(iq.shape)
                native = SimpleNamespace(x=x, metadata=meta, role='L_s')
            else:
                public = Path(config['diverse_data_root']) / 'exposure_train/public'
                public.mkdir(parents=True)
                np.save(public / 'iq.npy', iq, allow_pickle=False)
                np.save(public / 'ids.npy', meta['ids'], allow_pickle=False)
                np.savez(public / 'metadata.npz', **{key: meta[key] for key in ('rx_i', 'day_i', 'domain')})
            inputs[role] = iq
        handle = dict(capsule_id='source-oe-diverse-9dd73fa533b19a64fd4db8d9',
            split_id='5e0f620d5c7c72512e9e9edd', count=15)
        return config, source, native, contract, handle, inputs

    def encode(self, reader, x):
        fused = reader.features(x)
        result = {'features__' + key: np.ones((len(x), dimension), np.float32)
                  for key, dimension in readout.DIMENSIONS.items()}
        result['features__fused160'] = np.asarray(fused.tolist(), np.float32)
        result['features__reference29'][:, 27] = 0.
        result['logits'] = np.asarray(reader.classify_features(fused).tolist(), np.float32)
        return result

    @contextmanager
    def backend(self, module, source, native, contract, handle):
        reader, baseline, dura = FakeReader(), FakeReader(), FakeDura()
        actual_oe = module.diverse._public
        with patch.object(module, 'COUNTS', COUNTS), patch.object(original, 'COUNTS', {'source_L': 30}), \
             patch.object(module.diverse, 'COUNT', 15), \
             patch.object(module.frozen, '_source_bundle', return_value=source), \
             patch.object(module.frozen, '_load_readers', return_value=(reader, baseline)), \
             patch.object(module.frozen, '_load_dura', return_value=(dura, lambda x: (dura(x),))), \
             patch.object(module.frozen, '_encode_new', side_effect=self.encode), \
             patch.object(module.common, '_source_contract', return_value=contract), \
             patch.object(module.oldnative, '_load_labeled', return_value=native) as labeled, \
             patch.object(module.diverse, '_diverse_handle', return_value=handle), \
             patch.object(module.diverse, '_public', side_effect=actual_oe) as oe:
            yield reader, baseline, dura, labeled, oe

    def channel(self, raw, cfg, seed, receiver):
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(raw.shape, (2, 256))
        return raw * np.float32(2), {}

    def test_full_source_only_cache_exact_fields_order_and_frozen_models(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, source, native, contract, handle, inputs = self.fixture(Path(tmp))
            before = deepcopy(source)
            writes = []
            actual_write = module._write
            def record(path, value):
                writes.append(Path(path).name)
                if Path(path).name == 'complete.json':
                    self.assertEqual(len(list(Path(config['output_root']).glob('*/*.npy'))), 10)
                return actual_write(path, value)
            with self.backend(module, source, native, contract, handle) as (reader, baseline, dura, labeled, oe), \
                 patch.object(module.common, '_apply_channel', side_effect=self.channel) as channel, \
                 patch.object(module, '_write', side_effect=record), \
                 patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('No NumPy ABI bridge')):
                result = module.build(config, 'cpu')
                self.assertEqual(result['status'], 'SOURCE_PRACTICAL_ENCODER_CACHE_COMPLETE')
                self.assertEqual(result['physical_record_count'], 45)
                self.assertEqual(result['derived_view_record_count'], 180)
                self.assertEqual(channel.call_count, 6 + 45 * 3)
                self.assertEqual(sum(len(x) for x in reader.seen), 8)
                self.assertEqual(sum(len(x) for x in baseline.seen), 8)
                self.assertEqual(labeled.call_args.args, (source['train_config'], contract))
                self.assertEqual(oe.call_count, 1)
                self.assertEqual(writes[-1], 'complete.json')
                for role, count in COUNTS.items():
                    arrays, meta = module.load_role(config['output_root'], role)
                    try:
                        self.assertEqual(tuple(arrays), VIEWS)
                        self.assertEqual(set(meta), {'ids'} | module.METADATA_FIELDS[role])
                        self.assertEqual(len(set(meta['ids'].tolist())), count)
                        for view, values in arrays.items():
                            self.assertIsInstance(values, np.memmap)
                            self.assertFalse(values.flags.writeable)
                            self.assertEqual((values.dtype, values.shape), (np.dtype('float32'), (count, 2, 256)))
                            expected = inputs[role] if view == 'clean' else inputs[role] * 6
                            self.assertTrue(np.array_equal(values, expected))
                    finally:
                        for values in arrays.values(): values._mmap.close()
                self.assertTrue(np.array_equal(source['geometry']['branches']['fused160']['prototypes'],
                                               before['geometry']['branches']['fused160']['prototypes']))

    def test_bad_config_or_existing_output_never_activates_real_backend(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, *_ = self.fixture(Path(tmp))
            with patch.object(module.common, '_canonical_channel_api', side_effect=AssertionError('No backend')):
                for change in ({'target_root': 'no'}, {'truth': 'no'}, {'source_V': 'no'}, {'batch_size': 64},
                        {'run_id': 'other'}, {'channel_seed': 1}, {'diverse_data_root': None}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        module.build(dict(config, **change), 'cpu')
                Path(config['output_root']).mkdir()
                with self.assertRaises(FileExistsError): module.build(config, 'cpu')

    def test_synthetic_smoke_precedes_all_source_IQ_and_model_changes_never_complete(self):
        module = self.module()
        for failure in ('smoke', 'G_not_frozen', 'G_mutated', 'source_mutated'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                config, source, native, contract, handle, _ = self.fixture(Path(tmp))
                with self.backend(module, source, native, contract, handle) as (reader, baseline, dura, labeled, oe), \
                     patch.object(module.common, '_apply_channel', side_effect=self.channel):
                    if failure == 'smoke':
                        changed = patch.object(module.frozen, '_encode_new', side_effect=RuntimeError('smoke failed'))
                    elif failure == 'G_not_frozen':
                        dura.train()
                        changed = patch.object(module.frozen, '_encode_new', side_effect=self.encode)
                    elif failure == 'G_mutated':
                        def mutate(x):
                            dura.factor.add_(1.)
                            return (dura(x),)
                        changed = patch.object(module.frozen, '_load_dura', return_value=(dura, mutate))
                    else:
                        def mutate(reader, x):
                            source['thresholds'][0] += 1.
                            return self.encode(reader, x)
                        changed = patch.object(module.frozen, '_encode_new', side_effect=mutate)
                    with changed, self.assertRaises((ValueError, RuntimeError)):
                        module.build(config, 'cpu')
                    self.assertEqual(labeled.call_count, 0)
                    self.assertEqual(oe.call_count, 0)
                self.assertFalse((Path(config['output_root']) / 'complete.json').exists())

    def test_source_identity_metadata_and_role_overlap_fail_without_complete(self):
        module = self.module()
        for failure in ('L_order', 'L_private_field', 'OE_label', 'OE_overlap', 'OE_handle'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                config, source, native, contract, handle, _ = self.fixture(Path(tmp))
                public = Path(config['diverse_data_root']) / 'exposure_train/public'
                if failure == 'L_order': native.metadata['ids'] = native.metadata['ids'][::-1].copy()
                elif failure == 'L_private_field': native.metadata['truth'] = np.zeros(30, np.int64)
                elif failure == 'OE_label':
                    path = public / 'metadata.npz'
                    with np.load(path, allow_pickle=False) as data: meta = {k: data[k].copy() for k in data.files}
                    np.savez(path, **dict(meta, y=np.zeros(15, np.int64)))
                elif failure == 'OE_overlap':
                    ids = np.load(public / 'ids.npy', allow_pickle=False)
                    ids[0] = native.metadata['ids'][0]
                    np.save(public / 'ids.npy', ids, allow_pickle=False)
                else: handle['count'] = 6133
                with self.backend(module, source, native, contract, handle), \
                     patch.object(module.common, '_apply_channel', side_effect=self.channel), self.assertRaises(ValueError):
                    module.build(config, 'cpu')
                self.assertFalse((Path(config['output_root']) / 'complete.json').exists())

    def test_loader_rejects_partial_unknown_roles_labels_shape_and_domain_mutation(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, source, native, contract, handle, _ = self.fixture(Path(tmp))
            with self.backend(module, source, native, contract, handle), \
                 patch.object(module.common, '_apply_channel', side_effect=self.channel):
                module.build(config, 'cpu')
                root = Path(config['output_root'])
                with patch.object(np, 'load', side_effect=AssertionError('No forbidden role arrays')):
                    for role in ('source_V', 'novel_validation', 'U_s', 'target'):
                        with self.subTest(role=role), self.assertRaises(ValueError): module.load_role(root, role)
                for views in ((), ('unknown',), ('clean', 'clean')):
                    with self.subTest(views=views), self.assertRaises(ValueError): module.load_role(root, 'source_L', views)
                one, meta = module.load_role(root, 'source_OE', ('clean',))
                self.assertEqual(tuple(one), ('clean',)); one['clean']._mmap.close()
                path = root / 'source_OE/metadata.npz'
                with np.load(path, allow_pickle=False) as data: saved = {k: data[k].copy() for k in data.files}
                for broken in (dict(saved, y=np.zeros(15, np.int64)), dict(saved, rx_i=np.zeros(15, np.int64))):
                    np.savez(path, **broken)
                    with self.assertRaises(ValueError): module.load_role(root, 'source_OE')
                np.savez(path, **saved)
                np.save(root / 'source_OE/clean.npy', np.zeros((14, 2, 256), np.float32), allow_pickle=False)
                with self.assertRaises(ValueError): module.load_role(root, 'source_OE')
                (root / 'complete.json').unlink()
                with patch.object(np, 'load', side_effect=AssertionError('No partial arrays')), self.assertRaises(ValueError):
                    module.load_role(root, 'source_L')

    def test_real_residual_three_scenes_and_clean_copy_without_second_RMS(self):
        module = self.module()
        api = module.common._canonical_channel_api()
        channels = {view: module.common._channel_config(view) for view in VIEWS[1:]}
        n = np.arange(256)
        raw = np.asarray([[1.7 * np.cos(.13*n), 1.7 * np.sin(.13*n)]], np.float32)
        meta = dict(ids=np.asarray(['synthetic-source-only']), rx_i=np.asarray([3], np.int64), day_i=np.asarray([2], np.int64))
        with patch.object(module.common, '_synthesize', side_effect=AssertionError('Clean bypasses channel')):
            clean = module._view(raw, meta, 'clean', channels, api, {}, lambda x: (_ for _ in ()).throw(AssertionError('Clean bypasses G')), 'cpu')
        self.assertTrue(np.array_equal(raw, clean)); self.assertIsNot(raw, clean)
        dura = FakeDura()
        for scene in VIEWS[1:]:
            first = module._view(raw, meta, scene, channels, api, {}, lambda x: (dura(x),), 'cpu')
            second = module._view(raw, dict(meta, y=np.asarray([5])), scene, channels, api, {}, lambda x: (dura(x),), 'cpu')
            self.assertEqual((first.dtype, first.shape), (np.dtype('float32'), (1, 2, 256)))
            self.assertTrue(np.isfinite(first).all()); self.assertTrue(np.array_equal(first, second))

    def test_legal_source_OE_integer_metadata_is_preserved_in_int64_cache(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, source, native, contract, handle, _ = self.fixture(Path(tmp))
            path = Path(config['diverse_data_root']) / 'exposure_train/public/metadata.npz'
            with np.load(path, allow_pickle=False) as stored:
                narrow = {key: stored[key].astype(np.int32) for key in stored.files}
            np.savez(path, **narrow)
            with self.backend(module, source, native, contract, handle), \
                 patch.object(module.common, '_apply_channel', side_effect=self.channel):
                module.build(config, 'cpu')
                arrays, meta = module.load_role(config['output_root'], 'source_OE', ('clean',))
                try:
                    for key, value in narrow.items():
                        self.assertEqual(meta[key].dtype, np.int64)
                        self.assertTrue(np.array_equal(meta[key], value))
                finally:
                    for values in arrays.values(): values._mmap.close()


if __name__ == '__main__':
    unittest.main()
