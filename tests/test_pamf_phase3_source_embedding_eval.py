import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


class ToyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 160, bias=False)
        self.head = nn.Linear(160, 6, bias=False)
        self.bn = nn.BatchNorm1d(160)
        self.seen = []

    def features(self, x):
        self.seen.append(len(x))
        return self.bn(self.projection(x.mean(-1)))

    def classify_features(self, z):
        return self.head(z)


class EmbeddingEvalTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_eval.py'
        self.assertTrue(path.is_file(), 'New-encoder source evaluation is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_eval')

    def config(self, root):
        return dict(run_id='source-embedding-r01', row_id='uniform_oe', code_commit='a' * 40,
            train_config=str(root / 'train.json'), checkpoint=str(root / 'train/final.pt'),
            parent_run_root=str(root / '20261029-phase3-source-open-readout-s2026092701-r01'),
            novel_public_root=str(root / 'source_oe/novel_validation/public'), output_root=str(root / 'evaluation'))

    def fixtures(self):
        def source(role, repeats):
            y = np.tile(np.arange(6), 5 * repeats).astype(np.int64)
            rx = np.repeat([1, 3, 4, 6, 8], 6 * repeats).astype(np.int64)
            day = (np.arange(len(y)) % 3 + 1).astype(np.int64)
            meta = dict(ids=np.asarray([role + str(i) for i in range(len(y))]), y=y,
                rx_i=rx, day_i=day, domain=np.searchsorted([1, 3, 4, 6, 8], rx) * 3 + day - 1,
                eq_i=np.ones(len(y), dtype=np.int64))
            values = torch.randn(len(y), 2, 256, generator=torch.Generator().manual_seed(len(y)))
            return type('Source', (), dict(x=values, metadata=meta))()
        labeled, validation = source('L', 1), source('V', 2)
        novel_meta = dict(ids=np.asarray(['opaque' + str(i) for i in range(15)]),
            rx_i=np.repeat([1, 3, 4, 6, 8], 3).astype(np.int64), day_i=np.tile([1, 2, 3], 5).astype(np.int64),
            domain=np.arange(15, dtype=np.int64))
        novel = np.random.default_rng(1).normal(size=(15, 2, 256)).astype(np.float32)
        contract = dict(role_ids={'L_s': labeled.metadata['ids'].tolist(), 'V': validation.metadata['ids'].tolist()})
        return labeled, validation, novel, novel_meta, contract

    def test_exact_config_rejects_oe_training_truth_and_other_inputs(self):
        module = self.module()
        config = self.config(Path('toy'))
        module.validate_config(config)
        for key in ('OE_train', 'source_cache', 'U_s', 'truth', 'novel_labels', 'threshold', 'method'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.validate_config(dict(config, **{key: 'forbidden'}))

    def test_encode_uses_full_head_no_grad_and_preserves_buffers(self):
        module = self.module()
        reader = ToyReader().eval().requires_grad_(False)
        before = {k: v.clone() for k, v in reader.state_dict().items()}
        x = torch.randn(3, 2, 256)
        z, logits = module._encode(reader, x, 'cpu')
        self.assertEqual(z.shape, (3, 160))
        self.assertEqual(logits.shape, (3, 6))
        self.assertEqual(z.dtype, np.float32)
        with torch.no_grad():
            expected = reader.classify_features(reader.features(x))
        self.assertTrue(np.allclose(logits, np.asarray(expected.tolist()), atol=1e-6))
        for key, value in reader.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        self.assertTrue(all(p.grad is None for p in reader.parameters()))
        one_z, one_logits = module._encode(reader, x[:1], 'cpu')
        self.assertTrue(np.allclose(one_z, z[:1], atol=1e-6))
        self.assertTrue(np.allclose(one_logits, logits[:1], atol=1e-6))
        with self.assertRaises(ValueError):
            module._encode(reader, torch.full((2, 2, 256), float('nan')), 'cpu')

    def test_baseline_only_reads_ids_logits_and_aligns_by_exact_ids(self):
        module = self.module()
        ids = np.asarray(['b', 'a'])
        raw = np.asarray([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=np.float32)
        class Archive:
            files = list(module.parent.COMMON | {'y'})
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __getitem__(self, key):
                if key not in ('ids', 'logits'):
                    raise AssertionError('Old source labels/features must not be materialized')
                return {'ids': ids, 'logits': raw}[key]
        with patch('numpy.load', return_value=Archive()):
            value = module._baseline(Path('old.npz'), np.asarray(['a', 'b']), 'source_V')
        self.assertEqual(value.tolist(), [3, 1])
        with patch('numpy.load', return_value=Archive()), self.assertRaises(ValueError):
            module._baseline(Path('old.npz'), np.asarray(['a', 'missing']), 'source_V')

    def test_toy_end_to_end_l_smoke_then_v_risk_freeze_then_novel_and_marker_last(self):
        module = self.module()
        labeled, validation, novel, novel_meta, contract = self.fixtures()
        reader = ToyReader().eval().requires_grad_(False)
        model = type('Wrapper', (), {'id_backbone': type('Identity', (), {'encoder': reader})()})()
        counts = dict(source_L=30, source_V=60, novel_validation=15)
        parent_identity = {'model': {'source': 'original-full-six'}, 'source_OE_handles': {'novel_validation': {'capsule_id': 'cap', 'split_id': 'split', 'count': 15}}}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.config(root)
            traincfg = {'arm': 'uniform_oe', 'batch_size': 128, 'row_id': 'uniform_oe'}
            reads = []
            def load_source(cfg, role, c):
                reads.append(role)
                self.assertIs(c, contract)
                if role == 'V':
                    self.assertTrue((Path(config['output_root']) / 'smoke.json').is_file())
                self.assertIn(role, ('L_s', 'V'))
                return labeled if role == 'L_s' else validation
            def load_novel(path, identity):
                reads.append('novel_IQ')
                calibration = Path(config['output_root']) / 'calibration'
                self.assertTrue((calibration / 'thresholds.npy').is_file())
                frozen = json.loads((calibration / 'complete.json').read_text())
                self.assertEqual(frozen['unknown_calibration_scores_read'], 0)
                self.assertTrue(frozen['source_V_labels_read'])
                return novel, novel_meta, {'capsule_id': 'cap', 'split_id': 'split', 'count': 15}
            def baseline(path, ids, role):
                reads.append('baseline_' + role)
                if role == 'novel_validation':
                    self.assertTrue((Path(config['output_root']) / 'calibration/complete.json').is_file())
                return np.zeros(len(ids), dtype=np.int64)
            writes = []
            write = module.runner._write
            def record_write(path, value):
                writes.append(str(Path(path).relative_to(config['output_root'])))
                if Path(path) == Path(config['output_root']) / 'complete.json':
                    self.assertTrue((Path(config['output_root']) / 'predictions.npz').is_file())
                return write(path, value)
            with patch.object(module, '_training', return_value=(traincfg, {})), \
                    patch.object(module, '_parent', return_value=(root / 'parent/export', parent_identity)), \
                    patch.object(module.trainer, 'load_trained_model', return_value=model), \
                    patch.object(module.trainer, '_contract', return_value=contract), \
                    patch.object(module.source, '_validate_model'), patch.object(module, 'COUNTS', counts), \
                    patch.object(module.source, 'COUNTS', dict(counts, exposure_train=4200)), \
                    patch.object(module.runner, '_source_dataset', side_effect=load_source), \
                    patch.object(module, '_load_novel', side_effect=load_novel), \
                    patch.object(module, '_baseline', side_effect=baseline), \
                    patch.object(module.runner, '_write', side_effect=record_write), patch('builtins.print'):
                result = module.evaluate(config, 'cpu')
            self.assertEqual(reads, ['L_s', 'V', 'baseline_source_V', 'novel_IQ', 'baseline_novel_validation'])
            self.assertEqual(reader.seen, [2, 30, 60, 15])
            self.assertEqual(result['model_forward_samples'], 107)
            self.assertEqual(result['status'], 'SOURCE_EMBEDDING_PREDICTIONS_COMPLETE')
            self.assertEqual(result['method_names'], list(module.scores.METHODS))
            self.assertFalse(result['unknown_truth_read'])
            self.assertTrue(result['source_known_label_calibration'])
            self.assertEqual(writes[-1], 'complete.json')
            with np.load(Path(config['output_root']) / 'predictions.npz', allow_pickle=False) as stored:
                self.assertEqual(set(stored.files), {'ids', 'method_names', 'closed_prediction', 'knownness', 'accepted',
                    'prediction', 'rx_i', 'day_i', 'baseline_closed_prediction'})
                self.assertEqual(stored['knownness'].shape, (75, 5))
                self.assertEqual(stored['accepted'].dtype, np.bool_)
                self.assertTrue(np.array_equal(stored['ids'], np.sort(stored['ids'])))
                self.assertTrue(np.array_equal(stored['prediction'], np.where(stored['accepted'], stored['closed_prediction'][:, None], -1)))

    def test_bad_training_identity_and_output_collision_fail_before_iq(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.config(root)
            Path(config['output_root']).mkdir()
            with patch.object(module, '_training', side_effect=AssertionError('No parent or model IO')) as load:
                with self.assertRaises(FileExistsError):
                    module.evaluate(config, 'cpu')
                self.assertEqual(load.call_count, 0)
            config['output_root'] = str(root / 'failure')
            with patch.object(module, '_training', side_effect=ValueError('wrong own checkpoint')), \
                    patch.object(module.runner, '_source_dataset', side_effect=AssertionError('No IQ reads')) as iq:
                with self.assertRaises(ValueError):
                    module.evaluate(config, 'cpu')
                self.assertEqual(iq.call_count, 0)
            self.assertFalse((Path(config['output_root']) / 'complete.json').exists())

    def test_own_training_marker_binds_final_checkpoint_and_rejects_wrong_budget(self):
        module = self.module()
        from test_pamf_phase3_source_open_embedding_train import EmbeddingTrainTests
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); config = self.config(root)
            train_config = EmbeddingTrainTests().config(str(root / 'train'), 'uniform_oe')
            train_config.update(run_id=config['run_id'], row_id=config['row_id'])
            Path(config['train_config']).write_text(json.dumps(train_config), encoding='utf-8')
            marker = dict(status='SOURCE_OPEN_EMBEDDING_FROZEN', schema=module.trainer.SCHEMA, config=train_config,
                arm='uniform_oe', updates=222, parent_updates=11100, cumulative_updates=11322, optimizer_reset=True,
                source_only=True, default_head_joint=True, identity_head_replaced=False, model_classes=6, parameters=232993,
                U_reads=0, validation_reads=0, novel_validation_reads=0, target_reads=0, OE_truth_reads=0)
            (root / 'train').mkdir()
            path = root / 'train/complete.json'
            path.write_text(json.dumps(marker), encoding='utf-8')
            self.assertEqual(module._training(config)[0], train_config)
            for key, wrong in [('updates', 221), ('identity_head_replaced', True), ('U_reads', 1), ('target_reads', 1)]:
                path.write_text(json.dumps(dict(marker, **{key: wrong})), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module._training(config)
            path.write_text(json.dumps(marker), encoding='utf-8')
            with self.assertRaises(ValueError):
                module._training(dict(config, checkpoint=str(root / 'foreign/final.pt')))

    def test_novel_public_uses_relocated_root_without_opening_oe_train_or_truth(self):
        module = self.module()
        _, _, iq, meta, _ = self.fixtures()
        counts = dict(source_L=6300, source_V=27000, exposure_train=1, novel_validation=15)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); public = root / 'novel_validation/public'
            public.mkdir(parents=True)
            np.save(public / 'iq.npy', iq, allow_pickle=False)
            np.save(public / 'ids.npy', meta['ids'], allow_pickle=False)
            np.savez(public / 'metadata.npz', **{key: meta[key] for key in ('rx_i', 'day_i', 'domain')})
            marker = dict(status='SOURCE_OE_DATA_VALIDATED', data_status='VALIDATED_ONCE', source_only=True,
                target_reads=0, original_L_s_reads=0, original_U_s_reads=0, original_V_reads=0,
                model_reads=0, model_predictions=0, optimizer_updates=0, channel_observations_generated=0,
                counts={'exposure_train': 1, 'novel_validation': 15},
                source_domains=dict(receiver_names=module.source.RX_NAMES, receiver_indices=module.source.MAIN_RX,
                    dates=module.source.DATES, day_indices=[1, 2, 3], domain_count=15))
            for role, count in (('exposure_train', 1), ('novel_validation', 15)):
                marker[role] = dict(count=count, data_status='VALIDATED_ONCE', iq_dtype='float32', iq_shape=[count, 2, 256],
                    capsule_id=role + '-cap', split_id=role + '-split', public_root='E:/unreachable/builder/public',
                    truth_path='F:/unreachable/private/truth.npz')
            (root / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
            identity = {'source_OE_handles': {'novel_validation': {key: marker['novel_validation'][key] for key in ('capsule_id', 'split_id', 'count')}}}
            with patch.object(module.source, 'COUNTS', counts):
                values, returned, _ = module._load_novel(public, identity)
                self.assertTrue(np.array_equal(returned['ids'], meta['ids']))
                self.assertEqual(values.shape, (15, 2, 256))
                values._mmap.close()
                np.savez(public / 'metadata.npz', **meta, tx=np.asarray(['hidden'] * 15))
                with self.assertRaises(ValueError):
                    module._load_novel(public, identity)
            self.assertFalse((root / 'exposure_train').exists())
            self.assertFalse((root / 'novel_validation/truth').exists())

    def test_native_six_class_sinc_evaluation_cache_works_across_batches_without_updates(self):
        module = self.module()
        root = Path(__file__).resolve().parents[1]
        frozen = root / 'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
        config = dict(model_seed=2026092701, base_release=str(root),
            frozen_sources={key: str(frozen / key) for key in ('F', 'AF', 'BASE', 'FOLD', 'G')})
        module.runner._activate(config)
        native_config = dict(run_id='toy', row_id='toy', output_root='unused', model_seed=2026092701,
            method_id='pamf_t1_e50', source_rxs=[1, 3, 4, 6, 8], source_days=[1, 2, 3], num_classes=6,
            regime='matched_sat_supervised', ablation='F20', satellite_concat=False, step_offset=11100,
            segment_epochs=1, U_use=False, paper_method='synthetic test', view_comparison={'view_count': 3},
            repl_row={'group': 'main', 'structure': 'SHUFFLE_TIME', 'L_A': True, 'L_B': True, 'U': False})
        with patch('torch.load', side_effect=AssertionError('No weight read')):
            model = module.trainer._new_native(config, native_config, 'cpu').eval().requires_grad_(False)
        reader = model.id_backbone.encoder
        module.source._validate_model(reader)
        state = {key: value.clone() for key, value in reader.state_dict().items()}
        x = torch.randn(2, 2, 256, generator=torch.Generator().manual_seed(4))
        z, raw = module._encode(reader, x, 'cpu')
        one_z, one_raw = module._encode(reader, x[:1], 'cpu')
        self.assertTrue(np.allclose(z[:1], one_z, atol=2e-5))
        self.assertTrue(np.allclose(raw[:1], one_raw, atol=2e-5))
        for key, value in reader.state_dict().items():
            self.assertTrue(torch.equal(state[key], value), key)


if __name__ == '__main__':
    unittest.main()
