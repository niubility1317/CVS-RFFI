import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class FixtureReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(160)

    def features(self, x):
        return self.norm(x[:, 0, :160])

    def classify_features(self, z):
        return 10. * z[:, :6]


class Phase3RunnerTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/runner.py'
        self.assertTrue(path.is_file(), 'Phase3 frozen source/prediction runner is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.runner')

    def fixture(self, root):
        classes = ['tx' + str(i) for i in range(6)]
        contract = dict(classes=classes, counts=dict(L_s=6, V=12, U_s=24))
        contract_path = root / 'contract.json'
        contract_path.write_text(json.dumps(contract), encoding='utf-8')
        common = dict(run_id='phase3-test', row_id='seed01', model_seed=2026092701,
            base_release=str(root / 'base'), frozen_sources={key: str(root / key) for key in ('F', 'AF', 'BASE', 'FOLD', 'G')},
            source_output=str(root / 'source'), source_contract=str(contract_path), registered_classes=classes,
            code_commit='test-commit', batch_size=128, output_root=str(root / 'calibration'))
        labeled = torch.zeros(6, 2, 256)
        labeled[torch.arange(6), 0, torch.arange(6)] = 1.
        data = dict(L_s=SimpleNamespace(x=labeled, metadata=dict(y=np.arange(6))),
                    V=SimpleNamespace(x=labeled.repeat(2, 1, 1), metadata=dict(y=np.tile(np.arange(6), 2))))
        source = dict(common, source_cache=str(root / 'cache'), retention=.99)
        public = (dict(registered_classes=classes), np.asarray(['id0', 'id1', 'id2', 'id3']),
                  labeled[:4].tolist())
        public = public[:2] + (np.asarray(public[2], dtype=np.float32),)
        return common, source, data, public

    def calibrated(self, module, root):
        common, config, data, public = self.fixture(root)
        reader = FixtureReader()
        with patch.object(module, '_load_model', return_value=reader), patch.object(module, '_source_dataset', side_effect=lambda c, role, ct: data[role]):
            module.calibrate(config, 'cpu')
        return common, config, data, public, reader

    def test_mode_whitelists_reject_cross_role_inputs(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            common, source, _, _ = self.fixture(Path(tmp))
            predict = dict(common, public_root='public', calibration_root='calibration')
            module.validate_config('predict', predict)
            module.validate_config('calibrate', source)
            module.validate_config('smoke', source)
            for key in ('source_cache', 'truth', 'target_truth', 'roles', 'unknown_classes', 'unknown_tx_ids'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config('predict', dict(predict, **{key: 'forbidden'}))
            for key in ('public_root', 'calibration_root', 'truth'):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_config('calibrate', dict(source, **{key: 'forbidden'}))

    def test_source_class_order_mismatch_fails_before_model_load(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            _, source, _, _ = self.fixture(Path(tmp))
            source['registered_classes'] = source['registered_classes'][::-1]
            with patch.object(module, '_load_model', side_effect=AssertionError('must stay closed')), self.assertRaises(ValueError):
                module.calibrate(source, 'cpu')

    def test_calibration_uses_l_for_centers_and_v_only_for_thresholds_without_updates(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source, data, _ = self.fixture(root)
            reader = FixtureReader()
            before = {key: value.clone() for key, value in reader.state_dict().items()}
            roles = []
            def source_loader(c, role, ct):
                roles.append(role)
                return data[role]
            with patch.object(module, '_load_model', return_value=reader), patch.object(module, '_source_dataset', side_effect=source_loader):
                result = module.calibrate(source, 'cpu')
            self.assertEqual(roles, ['L_s', 'V'])
            self.assertEqual(result['status'], 'SOURCE_CALIBRATION_FROZEN')
            self.assertEqual(result['source_counts'], dict(L_s=6, V=12))
            for key, value in reader.state_dict().items():
                self.assertTrue(torch.equal(before[key], value), key)
            self.assertFalse(reader.training)
            self.assertFalse(any(p.requires_grad for p in reader.parameters()))
            with np.load(root / 'calibration/bundle.npz', allow_pickle=False) as bundle:
                self.assertEqual(bundle['prototypes'].shape, (6, 160))
                self.assertEqual(bundle['thresholds'].shape, (5,))
                self.assertEqual(bundle['method_names'].tolist(), ['msp', 'maxlogit', 'margin', 'energy', 'cosine'])

    def test_smoke_uses_only_two_source_l_samples(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source, data, _ = self.fixture(root)
            source['output_root'] = str(root / 'smoke')
            roles = []
            def source_loader(c, role, ct):
                roles.append(role)
                return data[role]
            with patch.object(module, '_load_model', return_value=FixtureReader()), patch.object(module, '_source_dataset', side_effect=source_loader):
                result = module.smoke(source, 'cpu')
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['source_samples'], 2)
            self.assertEqual(result['target_reads'], 0)
            self.assertEqual(roles, ['L_s'])

    def test_calibration_identity_mismatch_keeps_public_data_closed(self):
        module = self.module()
        for key, value in [('model_seed', 2026092702), ('source_output', 'other-source'), ('contract_classes', ['wrong'] * 6)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                common, _, _, _, reader = self.calibrated(module, root)
                marker = root / 'calibration/complete.json'
                metadata = json.loads(marker.read_text())
                metadata[key] = value
                marker.write_text(json.dumps(metadata), encoding='utf-8')
                config = dict(common, public_root='public', calibration_root=str(root / 'calibration'), output_root=str(root / 'predictions'))
                with patch.object(module, '_load_model', return_value=reader), patch.object(module, '_load_public', side_effect=AssertionError('must stay closed')), self.assertRaises(ValueError):
                    module.predict(config, 'cpu')

    def test_public_class_mismatch_fails(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common, _, _, public, reader = self.calibrated(module, root)
            config = dict(common, public_root='public', calibration_root=str(root / 'calibration'), output_root=str(root / 'predictions'))
            public[0]['registered_classes'] = public[0]['registered_classes'][::-1]
            with patch.object(module, '_load_model', return_value=reader), patch.object(module, '_load_public', return_value=public), self.assertRaises(ValueError):
                module.predict(config, 'cpu')

    def test_predict_has_no_source_access_and_writes_marker_after_predictions(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common, _, _, public, reader = self.calibrated(module, root)
            config = dict(common, public_root='public', calibration_root=str(root / 'calibration'), output_root=str(root / 'predictions'))
            original_save = np.savez
            def guarded_save(path, **values):
                self.assertFalse((root / 'predictions/complete.json').exists())
                return original_save(path, **values)
            with patch.object(module, '_load_model', return_value=reader), patch.object(module, '_load_public', return_value=public), \
                 patch.object(module, '_source_dataset', side_effect=AssertionError('predict must not read source IQ')), \
                 patch.object(module.np, 'savez', side_effect=guarded_save):
                result = module.predict(config, 'cpu')
            self.assertEqual(result['status'], 'ALL_PREDICTIONS_COMPLETE')
            self.assertFalse(result['truth_read'])
            self.assertFalse(result['query_fit'])
            with np.load(root / 'predictions/predictions.npz', allow_pickle=False) as payload:
                self.assertEqual(payload['ids'].tolist(), public[1].tolist())
                self.assertEqual(payload['knownness'].shape, (4, 5))
                self.assertEqual(payload['accepted'].dtype, np.dtype(bool))
                self.assertTrue(np.array_equal(payload['prediction'], np.where(payload['accepted'], payload['closed_prediction'][:, None], -1)))

    def test_failed_prediction_never_writes_completion(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common, _, _, public, reader = self.calibrated(module, root)
            config = dict(common, public_root='public', calibration_root=str(root / 'calibration'), output_root=str(root / 'predictions'))
            with patch.object(module, '_load_model', return_value=reader), patch.object(module, '_load_public', return_value=public), \
                 patch.object(module.np, 'savez', side_effect=OSError('simulated incomplete write')):
                with self.assertRaises(OSError):
                    module.predict(config, 'cpu')
            self.assertFalse((root / 'predictions/complete.json').exists())

    def test_existing_output_is_preserved_without_loading_model(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source, _, _ = self.fixture(root)
            (root / 'calibration').mkdir()
            with patch.object(module, '_load_model', side_effect=AssertionError('must not load')), self.assertRaises(FileExistsError):
                module.calibrate(source, 'cpu')

    def test_model_loader_rejects_seed_source_and_structure_mismatch(self):
        module = self.module()
        for difference in ('seed', 'source', 'views', 'order', 'parameters'):
            with self.subTest(difference=difference), tempfile.TemporaryDirectory() as tmp:
                common, _, _, _ = self.fixture(Path(tmp))
                reader = torch.nn.Module()
                count = 232992 if difference == 'parameters' else 232993
                reader.register_parameter('identity_parameters', torch.nn.Parameter(torch.zeros(count)))
                order = list(range(8)) if difference == 'order' else [0, 5, 3, 6, 2, 1, 4, 7]
                reader.sequence_pool = SimpleNamespace(views=[SimpleNamespace(window_order=torch.tensor(order)) for _ in range(3)],
                    output_projection=SimpleNamespace(weight=torch.zeros(160, 336)))
                actual = dict(model_seed=2026092702 if difference == 'seed' else common['model_seed'], num_classes=6,
                              output_root='other-source' if difference == 'source' else common['source_output'],
                              view_comparison=dict(view_count=1 if difference == 'views' else 3, model_seed=common['model_seed']))
                source_module = SimpleNamespace(load_reader=lambda *args: (reader, actual))
                with patch.object(module, '_activate'), patch.dict(sys.modules, {'experiments.pamf_repl_view_shuffle.source': source_module}), self.assertRaises(ValueError):
                    module._load_model(common, {}, 'cpu')

    def test_prediction_artifacts_are_consumable_by_independent_scorer(self):
        module = self.module()
        from experiments.pamf_phase3_open_world.scoring import score
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common, _, _, public, reader = self.calibrated(module, root)
            received = public[2].copy()
            received[2:] = 0.
            received[2, 0, 6] = received[3, 0, 7] = 1.
            public = (public[0], public[1], received)
            config = dict(common, public_root='public', calibration_root=str(root / 'calibration'), output_root=str(root / 'predictions'))
            with patch.object(module, '_load_model', return_value=reader), patch.object(module, '_load_public', return_value=public):
                module.predict(config, 'cpu')
            order = np.asarray([3, 0, 2, 1])
            truth = root / 'truth.npz'
            np.savez(truth, ids=public[1][order], y=np.asarray([0, 1, 6, 7])[order],
                     receiver=np.asarray([2, 3, 2, 3])[order], scenario=np.asarray(['low', 'urban', 'low', 'urban'])[order])
            result = score(root / 'predictions/predictions.npz', truth, root / 'scores')
            self.assertTrue(all(row['goals']['pass'] for row in result['methods'].values()))

    def test_cli_exposes_only_declared_actions(self):
        self.module()
        command = [sys.executable, '-B', '-m', 'experiments.pamf_phase3_open_world.runner', '--help']
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('{smoke,calibrate,predict}', result.stdout)


if __name__ == '__main__':
    unittest.main()
