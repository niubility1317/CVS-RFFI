import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


SMALL = dict(source_L=12, source_V=18, exposure_train=12, novel_validation=12)


class SourceOpenRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_open_run')

    def fixture(self, root):
        from experiments.pamf_phase3_open_world.source_open_export import FEATURE_REFS, DIMENSIONS
        export = root / 'export'; export.mkdir()
        rng = np.random.default_rng(35)
        for role, count in SMALL.items():
            labels = np.arange(count) % 6
            logits = rng.normal(0., .05, (count, 6)).astype(np.float32)
            if role.startswith('source_'): logits[np.arange(count), labels] += 3.
            arrays = dict(ids=np.asarray([role + '-id-%03d' % i for i in range(count)]), logits=logits,
                rx_i=np.resize(np.asarray([1, 3, 4, 6, 8], dtype=np.int64), count),
                day_i=np.resize(np.asarray([1, 2, 3], dtype=np.int64), count))
            for name, dimension in DIMENSIONS.items():
                raw = rng.normal(0., .03, (count, dimension)).astype(np.float32)
                if role.startswith('source_'): raw[np.arange(count), labels] += 1.
                else: raw[:, 10] += 1.
                if name == 'reference29': raw[:, 27] = 0.
                arrays['features__' + name] = raw
            if role == 'source_L': arrays['y'] = labels.astype(np.int64)
            if role == 'source_V': arrays['y'] = np.asarray([object()] * count)  # Must never be read by predict.
            path = export / FEATURE_REFS[role]; path.parent.mkdir(parents=True)
            np.savez(path, **arrays)
        marker = dict(status='SOURCE_OPEN_FEATURES_EXPORTED', source_only=True, target_reads=0,
            optimizer_updates=0, head_changed=False, model_classes=6, parameters=232993,
            original_U_s_reads=0, OE_truth_reads=0, counts=SMALL,
            config=dict(model_seed=2026092701, source_output='full6', source_contract='contract',
                base_release='code', frozen_sources={name: name for name in ['F', 'AF', 'BASE', 'FOLD', 'G']},
                registered_classes=['old%d' % i for i in range(6)], code_commit='commit'),
            source_OE_handles={role: dict(capsule_id=role, split_id='split-' + role, count=SMALL[role])
                for role in ['exposure_train', 'novel_validation']}, feature_refs=FEATURE_REFS)
        (export / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
        return export

    def test_real_core_full_synthetic_chain_roles_outputs_and_fixed_head(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export = self.fixture(root)
            original_load = module._load_features
            fit_reads, prediction_reads = [], []
            def fit_only(root_path, role, include_y=False):
                self.assertIn(role, ['source_L', 'exposure_train']); fit_reads.append(role)
                return original_load(root_path, role, include_y)
            with patch.object(module, 'COUNTS', SMALL), patch.object(module, '_load_features', side_effect=fit_only):
                fitted = module.fit(export, 2026092701, root / 'fit')
            self.assertEqual(fit_reads, ['source_L', 'exposure_train'])
            self.assertEqual(fitted['readout_steps_per_model'], 200)
            self.assertEqual(fitted['source_validation_reads'], 0)
            training = json.loads((root / 'fit/training.json').read_text())
            self.assertEqual(training['training']['parameter_counts'], {'linear_oe': 101, 'mlp16_oe': 1633})
            self.assertEqual({len(trace) for trace in training['training']['losses'].values()}, {200})
            with np.load(root / 'fit/geometry.npz', allow_pickle=False) as stored:
                before_geometry = {key: stored[key].copy() for key in stored.files}
            with np.load(root / 'fit/readouts.npz', allow_pickle=False) as stored:
                before_readouts = {key: stored[key].copy() for key in stored.files}
            def validation_only(root_path, role, include_y=False):
                self.assertIn(role, ['source_V', 'novel_validation']); self.assertFalse(include_y)
                if role == 'novel_validation':
                    self.assertTrue((root / 'predict/calibration/complete.json').is_file())
                prediction_reads.append(role); return original_load(root_path, role, include_y)
            with patch.object(module, 'COUNTS', SMALL), patch.object(module, '_load_features', side_effect=validation_only), \
                 patch.object(module.core, 'fit_geometry', side_effect=AssertionError('No validation fit')), \
                 patch.object(module.core, 'fit_readouts', side_effect=AssertionError('No validation updates')):
                result = module.predict(export, root / 'fit', root / 'predict')
            self.assertEqual(prediction_reads, ['source_V', 'novel_validation'])
            self.assertEqual(result['status'], 'ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE')
            self.assertIs(result['truth_read'], False); self.assertIs(result['query_fit'], False)
            with np.load(root / 'predict/predictions.npz', allow_pickle=False) as predictions:
                self.assertEqual(set(predictions.files), {'ids', 'method_names', 'closed_prediction', 'knownness', 'accepted', 'prediction', 'rx_i', 'day_i'})
                self.assertEqual(predictions['knownness'].shape, (30, 7)); self.assertTrue(np.isfinite(predictions['knownness']).all())
                self.assertEqual(predictions['accepted'].dtype.kind, 'b')
                self.assertEqual(predictions['method_names'].tolist(), list(module.METHODS))
                np.testing.assert_array_equal(predictions['ids'], np.sort(predictions['ids']))
                np.testing.assert_array_equal(predictions['prediction'], np.where(predictions['accepted'], predictions['closed_prediction'][:, None], -1))
                ids = predictions['ids'].tolist()
                for role in ['source_V', 'novel_validation']:
                    with patch.object(module, 'COUNTS', SMALL):
                        data = original_load(export, role, False)
                    rows = [ids.index(identifier) for identifier in data['ids']]
                    np.testing.assert_array_equal(predictions['closed_prediction'][rows], data['logits'].argmax(axis=1))
            for filename, before in [('geometry.npz', before_geometry), ('readouts.npz', before_readouts)]:
                with np.load(root / 'fit' / filename, allow_pickle=False) as stored:
                    for key in before: np.testing.assert_array_equal(stored[key], before[key])

    def test_future_scores_are_row_local_and_novel_does_not_set_thresholds(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export = self.fixture(root)
            with patch.object(module, 'COUNTS', SMALL):
                module.fit(export, 2026092701, root / 'fit')
                geometry, readouts = module._load_aggregates(root / 'fit')
                data = module._load_features(export, 'novel_validation', False)
                scores = module._score(data, geometry, readouts)
                for i in range(len(scores)):
                    alone = {key: value[i:i+1] for key, value in data.items()}
                    np.testing.assert_allclose(module._score(alone, geometry, readouts)[0], scores[i], atol=2e-5, rtol=2e-5)
                module.predict(export, root / 'fit', root / 'predict-a')
                path = export / module.FEATURE_REFS['novel_validation']
                with np.load(path, allow_pickle=False) as stored: arrays = {key: stored[key].copy() for key in stored.files}
                arrays['logits'] += 100.
                np.savez(path, **arrays)
                module.predict(export, root / 'fit', root / 'predict-b')
            np.testing.assert_array_equal(np.load(root / 'predict-a/calibration/thresholds.npy'), np.load(root / 'predict-b/calibration/thresholds.npy'))

    def test_markers_last_output_protection_and_no_novel_before_frozen_fit(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); export = self.fixture(root)
            original, writes = module._write, []
            def writing(path, value):
                if Path(path) == root / 'fit/complete.json':
                    self.assertTrue((root / 'fit/geometry.npz').is_file()); self.assertTrue((root / 'fit/readouts.npz').is_file())
                    self.assertTrue((root / 'fit/training.json').is_file())
                if Path(path) == root / 'predict/complete.json':
                    self.assertTrue((root / 'predict/predictions.npz').is_file())
                writes.append(str(path)); return original(path, value)
            with patch.object(module, 'COUNTS', SMALL), patch.object(module, '_write', side_effect=writing):
                module.fit(export, 2026092701, root / 'fit')
                module.predict(export, root / 'fit', root / 'predict')
                with patch.object(module, '_export_marker', side_effect=AssertionError('No reads after collision')), self.assertRaises(FileExistsError):
                    module.fit(export, 2026092701, root / 'fit')
                with patch.object(module, '_export_marker', side_effect=AssertionError('No reads after collision')), self.assertRaises(FileExistsError):
                    module.predict(export, root / 'fit', root / 'predict')
            self.assertTrue(writes[-1].endswith('predict\\complete.json') or writes[-1].endswith('predict/complete.json'))
            marker = json.loads((root / 'fit/complete.json').read_text()); marker['status'] = 'INCOMPLETE'
            (root / 'fit/complete.json').write_text(json.dumps(marker), encoding='utf-8')
            with patch.object(module, 'COUNTS', SMALL), patch.object(module, '_load_features', side_effect=AssertionError('No feature reads before fit is frozen')), self.assertRaises(ValueError):
                module.predict(export, root / 'fit', root / 'predict-invalid')
            self.assertFalse((root / 'predict-invalid/complete.json').exists())

    def test_bad_export_labels_oe_label_or_fit_failure_cannot_complete(self):
        module = self.module()
        for failure in ['marker', 'source_labels', 'oe_label', 'training']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); export = self.fixture(root)
                if failure == 'marker':
                    path = export / 'complete.json'; marker = json.loads(path.read_text()); marker['model_classes'] = 4
                    path.write_text(json.dumps(marker), encoding='utf-8')
                if failure in ['source_labels', 'oe_label']:
                    role = 'source_L' if failure == 'source_labels' else 'exposure_train'
                    path = export / module.FEATURE_REFS[role]
                    with np.load(path, allow_pickle=False) as stored: arrays = {key: stored[key].copy() for key in stored.files}
                    arrays['y'] = np.zeros(SMALL[role], dtype=np.int64)
                    np.savez(path, **arrays)
                training_error = ValueError('synthetic training failure') if failure == 'training' else AssertionError('Invalid input must fail before training')
                with patch.object(module, 'COUNTS', SMALL), patch.object(module.core, 'fit_readouts', side_effect=training_error), self.assertRaises(ValueError):
                    module.fit(export, 2026092701, root / 'fit')
                self.assertFalse((root / 'fit/complete.json').exists())


if __name__ == '__main__':
    unittest.main()
