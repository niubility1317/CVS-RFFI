import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class FakeSourceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 160)
        self.norm = torch.nn.BatchNorm1d(160)
        self.head = torch.nn.Linear(160, 4, bias=False)
        self.validation_seen = 0
        self.training_seen = 0
        self.validation_bn_before = None

    def features(self, x):
        is_validation = bool((x[:, 0, 0] >= 10.).any())
        if is_validation:
            if self.training or torch.is_grad_enabled():
                raise AssertionError('Source V must never run with updates')
            self.validation_seen += len(x)
            if self.validation_bn_before is None:
                self.validation_bn_before = {key: value.clone() for key, value in self.norm.state_dict().items()}
        elif self.training:
            self.training_seen += len(x)
        return self.norm(self.projection(x[:, :, 0]))

    def classify_features(self, z):
        return self.head(z)

    def forward(self, x):
        return self.classify_features(self.features(x))


class SourceTrainTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_train.py'
        self.assertTrue(path.is_file(), 'Source-only fold trainer is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_train')

    def fixture(self, root):
        contract = root / 'source_contract.json'
        contract.write_text(json.dumps(dict(classes=['tx' + str(i) for i in range(6)],
            source_rxs=[1, 3, 4, 6, 8], counts=dict(L_s=6300, U_s=56700, V=27000))), encoding='utf-8')
        config = dict(run_id='source-dev', row_id='pair01-rx8', model_seed=2026092701,
            code_commit='test', code_runtime=dict(base_release='base', frozen_sources={key: key for key in ('F', 'AF', 'BASE', 'FOLD', 'G')}),
            source_cache='source-cache', source_contract=str(contract),
            fold=dict(unknown_source_labels=[0, 1], heldout_source_rx=8), output_root=str(root / 'output'),
            epochs=20, batch_size=128, shuffle_seed=2026102802,
            optimizer=dict(name='AdamW', lr=.0003, weight_decay=.0001, betas=[.9, .999], eps=1e-8),
            schedule='cosine_per_step', label_smoothing=0.)
        train_x = torch.zeros(3360, 2, 256)
        train_x[:, 0, 0] = torch.arange(3360) % 4
        train_x[:, 1, 0] = torch.arange(3360) % 3
        validation_x = torch.zeros(27000, 2, 256)
        validation_x[:, 0, 0] = 10.
        data = dict(train_x=train_x, train_y=np.arange(3360, dtype=np.int64) % 4,
            validation_x=validation_x, validation_y=np.arange(27000, dtype=np.int64) % 6,
            train_ids=np.asarray(['train-' + str(i) for i in range(3360)]),
            validation_ids=np.asarray(['validation-' + str(i) for i in range(27000)]),
            validation_receivers=np.asarray([1, 3, 4, 6, 8] * 5400, dtype=np.int64),
            manifest=dict(unknown_source_labels=[0, 1], heldout_source_rx=8),
            counts=dict(train=3360, validation=27000))
        return config, data

    def test_production_config_rejects_target_resume_and_budget_changes(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, _ = self.fixture(Path(tmp))
            module.validate_config(config)
            for key in ('target', 'truth', 'public_root', 'checkpoint', 'resume', 'teacher_checkpoint', 'U_s'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: 'forbidden'}))
            for key, value in (('epochs', 2), ('batch_size', 64), ('label_smoothing', .01), ('schedule', 'constant')):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: value}))
            for key, value in (('epochs', 20.), ('batch_size', 128.), ('shuffle_seed', 2026102802.),
                               ('fold', dict(unknown_source_labels=[0, 1], heldout_source_rx=8.))):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    module.validate_config(dict(config, **{key: value}))

    def test_short_internal_fit_keeps_tail_batches_and_cosine_endpoints(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeSourceModel()
            x = torch.zeros(6, 2, 256)
            x[:, 0, 0] = torch.arange(6) % 4
            result = module._fit(model, x, np.arange(6, dtype=np.int64) % 4,
                epochs=2, batch_size=4, shuffle_seed=2026102802,
                optimizer_options=dict(name='AdamW', lr=.0003, weight_decay=.0001, betas=[.9, .999], eps=1e-8),
                device='cpu', log_path=Path(tmp) / 'epochs.jsonl', row_id='tiny')
            self.assertEqual(result['updates'], 4)
            self.assertEqual(result['train_exposures'], 12)
            self.assertEqual(result['first_lr'], .0003)
            self.assertEqual(result['last_lr'], 0.)
            rows = [json.loads(line) for line in (Path(tmp) / 'epochs.jsonl').read_text().splitlines()]
            self.assertEqual([row['seen'] for row in rows], [6, 6])
            self.assertEqual([row['steps'] for row in rows], [2, 2])
            self.assertEqual([row['tail_batch_size'] for row in rows], [2, 2])

    def test_fixed_twenty_epochs_export_source_only_and_freeze_bn(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root)
            model = FakeSourceModel()
            original_savez = np.savez
            def guarded_savez(path, **values):
                self.assertFalse((root / 'output/complete.json').exists())
                return original_savez(path, **values)
            with patch.object(module, 'build_model', return_value=model), patch.object(module, '_load_fold', return_value=data), \
                 patch('torch.load', side_effect=AssertionError('No checkpoint initialization')), patch.object(module.np, 'savez', side_effect=guarded_savez):
                result = module.train(config, 'cpu')
            self.assertEqual(result['status'], 'SOURCE_DEVELOPMENT_FROZEN')
            self.assertEqual((result['epochs'], result['updates']), (20, 540))
            self.assertEqual((result['train_count'], result['validation_count']), (3360, 27000))
            self.assertEqual(result['checkpoint_initialization'], 'scratch')
            self.assertEqual(result['target_reads'], 0)
            self.assertEqual(result['U_reads'], 0)
            self.assertEqual(model.training_seen, 3360 * 20)
            self.assertEqual(model.validation_seen, 27000)
            for key, value in model.norm.state_dict().items():
                self.assertTrue(torch.equal(model.validation_bn_before[key], value), key)
            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            smoke = json.loads((root / 'output/smoke.json').read_text())
            self.assertEqual(smoke['optimizer_updates'], 0)
            self.assertEqual(smoke['validation_forwards'], 0)
            rows = [json.loads(line) for line in (root / 'output/epochs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(row['seen'] == 3360 and row['steps'] == 27 and row['tail_batch_size'] == 32 for row in rows))
            with np.load(root / 'output/source_features.npz', allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {'train_features', 'train_y', 'train_ids', 'validation_features',
                    'validation_logits', 'validation_y', 'validation_ids', 'validation_receivers'})
                self.assertEqual(archive['train_features'].shape, (3360, 160))
                self.assertEqual(archive['validation_logits'].shape, (27000, 4))
                self.assertEqual(archive['validation_features'].dtype, np.dtype('float32'))
                self.assertEqual(archive['train_y'].dtype, np.dtype('int64'))
                self.assertEqual(archive['train_ids'].dtype.kind, 'U')
            self.assertTrue((root / 'output/source_model.pt').is_file())

    def test_heldout_training_label_is_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, data = self.fixture(Path(tmp))
            data['train_y'][0] = 4
            with patch.object(module, 'build_model', return_value=FakeSourceModel()), patch.object(module, '_load_fold', return_value=data), \
                 patch.object(module, '_fit', side_effect=AssertionError('Cannot train leaked holdout')), self.assertRaises(ValueError):
                module.train(config, 'cpu')

    def test_train_validation_id_overlap_is_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            config, data = self.fixture(Path(tmp))
            data['validation_ids'][0] = data['train_ids'][0]
            with patch.object(module, 'build_model', return_value=FakeSourceModel()), patch.object(module, '_load_fold', return_value=data), \
                 patch.object(module, '_fit', side_effect=AssertionError('Cannot train overlap')), self.assertRaises(ValueError):
                module.train(config, 'cpu')

    def test_existing_output_fails_before_model_or_data(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _ = self.fixture(root)
            (root / 'output').mkdir()
            with patch.object(module, 'build_model', side_effect=AssertionError('Must preserve output')), self.assertRaises(FileExistsError):
                module.train(config, 'cpu')

    def test_export_failure_never_writes_complete_marker(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root)
            fit = dict(epochs=20, updates=540, train_exposures=67200, first_lr=.0003, last_lr=0.)
            with patch.object(module, 'build_model', return_value=FakeSourceModel()), patch.object(module, '_load_fold', return_value=data), \
                 patch.object(module, '_fit', return_value=fit), patch.object(module, '_export_features', side_effect=OSError('Incomplete export')):
                with self.assertRaises(OSError):
                    module.train(config, 'cpu')
            self.assertFalse((root / 'output/complete.json').exists())

    def test_smoke_is_written_before_optimizer_or_validation_forward(self):
        module = self.module()
        class ReachedTraining(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, data = self.fixture(root)
            model = FakeSourceModel()
            def after_smoke(*args, **kwargs):
                marker = json.loads((root / 'output/smoke.json').read_text())
                self.assertEqual(marker['optimizer_updates'], 0)
                self.assertEqual(model.validation_seen, 0)
                self.assertEqual(model.training_seen, 0)
                raise ReachedTraining()
            with patch.object(module, 'build_model', return_value=model), patch.object(module, '_load_fold', return_value=data), \
                 patch.object(module, '_fit', side_effect=after_smoke), patch('torch.optim.AdamW', side_effect=AssertionError('Optimizer before smoke')):
                with self.assertRaises(ReachedTraining):
                    module.train(config, 'cpu')
            self.assertFalse((root / 'output/complete.json').exists())

    def test_cli_has_no_resume_or_target_option(self):
        self.module()
        command = [sys.executable, '-B', '-m', 'experiments.pamf_phase3_open_world.source_train', '--help']
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--config', result.stdout)
        self.assertNotIn('--resume', result.stdout)
        self.assertNotIn('--target', result.stdout)


if __name__ == '__main__':
    unittest.main()
