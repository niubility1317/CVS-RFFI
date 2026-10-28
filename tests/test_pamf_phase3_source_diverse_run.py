import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_pamf_phase3_source_risk_run as fixture_module


COUNTS = dict(source_L=30, source_V=300, exposure_train=30, novel_validation=12)
OE_COUNT = 45


class SourceDiverseRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_diverse_run')

    def fixture(self, root, module):
        with patch.object(fixture_module, 'SMALL', COUNTS):
            base = fixture_module.SourceRiskRunTests().fixture(root, module)
        export = root / 'diverse-export'; public = export / 'exposure_train/public'; public.mkdir(parents=True)
        with np.load(base / 'export/exposure_train/public/features.npz', allow_pickle=False) as stored:
            arrays = {key: stored[key][np.arange(OE_COUNT) % 30].copy() for key in stored.files}
        arrays['ids'] = np.asarray(['new-oe-%03d' % i for i in range(OE_COUNT)])
        np.savez(public / 'features.npz', **arrays)
        with patch.object(module.parent, 'COUNTS', COUNTS):
            identity = module.parent._identity(module.parent._export_marker(base / 'export'))
        marker = dict(status='SOURCE_DIVERSE_OE_FEATURES_EXPORTED', source_only=True, target_reads=0,
            optimizer_updates=0, original_U_s_reads=0, OE_truth_reads=0, head_changed=False,
            model_classes=6, parameters=232993, count=OE_COUNT, feature_ref='exposure_train/public/features.npz',
            parent_export_identity=identity, source_OE_handle=dict(capsule_id='diverse-capsule', split_id='diverse-split', count=OE_COUNT))
        (export / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
        return base, export

    def test_fit_only_parent_l_and_new_oe_uses_fixed_domain_weighting_and_artifacts(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, export = self.fixture(root, module)
            original, roles = module.parent._load_features, []
            def loading(root, role, include_y=False):
                self.assertEqual(role, 'source_L'); roles.append(role); return original(root, role, include_y)
            with patch.object(module.parent, 'COUNTS', COUNTS), patch.object(module, 'OE_COUNT', OE_COUNT), \
                 patch.object(module.parent, '_load_features', side_effect=loading), \
                 patch.object(module.parent.core, 'fit_geometry', side_effect=AssertionError('Reuse L geometry')):
                result = module.fit(base, export, 2026092701, root / 'fit')
            self.assertEqual(roles, ['source_L']); self.assertEqual(result['readout_steps_per_model'], 200)
            self.assertEqual(result['readout_training_regime'], 'source_role_domain_balanced_v1')
            with np.load(root / 'fit/readouts.npz', allow_pickle=False) as stored:
                self.assertEqual(stored['descriptor_schema'].item(), 'head_aligned_branch_summary_v1')
                self.assertEqual(stored['readout_training_regime'].item(), result['readout_training_regime'])
            training = json.loads((root / 'fit/training.json').read_text())['training']
            self.assertEqual(training['normalizer_role_prior'], {'known': .6, 'oe': .4})
            self.assertEqual(training['loss_role_prior'], {'known': .5, 'oe': .5})
            self.assertEqual(training['parameter_counts'], {'linear_oe': 101, 'mlp16_oe': 1633})
            self.assertEqual(sum(training['domain_counts']['oe']), OE_COUNT)
            self.assertEqual((root / 'fit/geometry.npz').read_bytes(), (base / 'fit/geometry.npz').read_bytes())

    def test_predict_freezes_risk_before_novel_and_does_not_read_new_training_rows(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, export = self.fixture(root, module)
            with patch.object(module.parent, 'COUNTS', COUNTS), patch.object(module, 'OE_COUNT', OE_COUNT):
                module.fit(base, export, 2026092701, root / 'fit')
            original, roles = module.parent._load_features, []
            def loading(root_path, role, include_y=False):
                self.assertIn(role, ['source_V', 'novel_validation']); self.assertFalse(include_y); roles.append(role)
                if role == 'novel_validation': self.assertTrue((root / 'predict/calibration/complete.json').is_file())
                return original(root_path, role, include_y)
            with patch.object(module.parent, 'COUNTS', COUNTS), patch.object(module, 'OE_COUNT', OE_COUNT), \
                 patch.object(module.parent, '_load_features', side_effect=loading), \
                 patch.object(module, '_load_diverse_features', side_effect=AssertionError('No deployment training-role reads')):
                result = module.predict(base, export, root / 'fit', root / 'predict')
            self.assertEqual(roles, ['source_V', 'novel_validation']); self.assertFalse(result['truth_read'])
            self.assertEqual(result['readout_training_regime'], 'source_role_domain_balanced_v1')
            with np.load(root / 'predict/predictions.npz', allow_pickle=False) as pred:
                np.testing.assert_array_equal(pred['prediction'], np.where(pred['accepted'], pred['closed_prediction'][:, None], -1))
            geometry, state = module._load_new(root / 'fit')
            with patch.object(module.parent, 'COUNTS', COUNTS): data = original(base / 'export', 'novel_validation', False)
            before = module._score(data, geometry, state)
            data['rx_i'][:] = 8; data['day_i'][:] = 3
            np.testing.assert_array_equal(before, module._score(data, geometry, state))

    def test_old_regime_mismatched_handle_and_existing_outputs_are_rejected(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base, export = self.fixture(root, module)
            with patch.object(module.parent, 'COUNTS', COUNTS), patch.object(module, 'OE_COUNT', OE_COUNT):
                module.fit(base, export, 2026092701, root / 'fit')
                with self.assertRaises(FileExistsError): module.fit(base, export, 2026092701, root / 'fit')
                path = root / 'fit/readouts.npz'
                with np.load(path, allow_pickle=False) as stored: arrays = {key: stored[key].copy() for key in stored.files}
                arrays['readout_training_regime'] = np.asarray('old_unweighted')
                np.savez(path, **arrays)
                with patch.object(module.parent, '_load_features', side_effect=AssertionError('No V before regime check')), self.assertRaises(ValueError):
                    module.predict(base, export, root / 'fit', root / 'bad-regime')
                marker = json.loads((export / 'complete.json').read_text()); marker['source_OE_handle']['capsule_id'] = 'other-data'
                (export / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
                with self.assertRaises(ValueError): module.predict(base, export, root / 'fit', root / 'bad-handle')
            self.assertFalse((root / 'bad-regime/complete.json').exists())


if __name__ == '__main__': unittest.main()
