import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_pamf_phase3_source_open_run as source_fixture


SMALL = dict(source_L=12, source_V=300, exposure_train=12, novel_validation=12)
PARENT = '20261029-phase3-source-open-readout-s2026092701-r01'


class SourceRiskRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_risk_run')

    def fixture(self, root, module):
        parent = root / PARENT; parent.mkdir()
        with patch.object(source_fixture, 'SMALL', SMALL):
            export = source_fixture.SourceOpenRunTests().fixture(parent)
        path = export / module.parent.FEATURE_REFS['source_V']
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files if key != 'y'}
        arrays['y'] = (np.arange(300) % 6).astype(np.int64)
        np.savez(path, **arrays)
        with patch.object(module.parent, 'COUNTS', SMALL):
            module.parent.fit(export, 2026092701, parent / 'fit')
        return parent

    def test_smoke_then_v_only_calibration_then_novel_and_fixed_head(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = self.fixture(root, module); output = root / 'predict'
            original_load, original_score = module.parent._load_features, module.parent._score
            roles, scored = [], []
            def loading(export, role, include_y=False):
                self.assertFalse(include_y); roles.append(role)
                if role == 'source_V': self.assertTrue((output / 'smoke.json').is_file())
                if role == 'novel_validation': self.assertTrue((output / 'calibration/complete.json').is_file())
                return original_load(export, role, include_y)
            def scoring(data, geometry, readouts):
                scored.append(len(data['ids'])); return original_score(data, geometry, readouts)
            with patch.object(module.parent, 'COUNTS', SMALL), patch.object(module.parent, '_load_features', side_effect=loading), \
                 patch.object(module.parent, '_score', side_effect=scoring), \
                 patch.object(module.parent.core, 'fit_readouts', side_effect=AssertionError('No retraining')):
                result = module.predict(parent, output)
            self.assertEqual(roles, ['source_L', 'source_V', 'novel_validation']); self.assertEqual(scored, [2, 300, 12])
            self.assertEqual(result['status'], 'ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE')
            self.assertTrue(result['source_V_labels_read']); self.assertTrue(result['source_known_label_calibration'])
            self.assertFalse(result['truth_read']); self.assertFalse(result['query_fit'])
            self.assertFalse(result['independent_confirmation'])
            with np.load(output / 'predictions.npz', allow_pickle=False) as pred:
                self.assertEqual(set(pred.files), {'ids', 'method_names', 'closed_prediction', 'knownness', 'accepted', 'prediction', 'rx_i', 'day_i'})
                np.testing.assert_array_equal(pred['prediction'], np.where(pred['accepted'], pred['closed_prediction'][:, None], -1))
                with patch.object(module.parent, 'COUNTS', SMALL):
                    known = original_load(parent / 'export', 'source_V', False)
                    novel = original_load(parent / 'export', 'novel_validation', False)
                ids = pred['ids'].tolist()
                for data in [known, novel]:
                    positions = [ids.index(identifier) for identifier in data['ids']]
                    np.testing.assert_array_equal(pred['closed_prediction'][positions], data['logits'].argmax(axis=1))
            calibration = json.loads((output / 'calibration/complete.json').read_text())
            self.assertEqual(calibration['unknown_calibration_scores_read'], 0)
            self.assertEqual(calibration['limits'], {'known_macro_frr': .04, 'known_macro_correct_rejection_loss': .015})
            self.assertTrue(calibration['constraints_passed'])

    def test_novel_score_changes_do_not_affect_thresholds_or_parent_state(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = self.fixture(root, module)
            snapshots = {name: (parent / 'fit' / name).read_bytes() for name in ['geometry.npz', 'readouts.npz']}
            with patch.object(module.parent, 'COUNTS', SMALL):
                module.predict(parent, root / 'a')
                path = parent / 'export' / module.parent.FEATURE_REFS['novel_validation']
                with np.load(path, allow_pickle=False) as stored: arrays = {key: stored[key].copy() for key in stored.files}
                arrays['logits'] += 1000.; np.savez(path, **arrays)
                module.predict(parent, root / 'b')
            np.testing.assert_array_equal(np.load(root / 'a/calibration/thresholds.npy'), np.load(root / 'b/calibration/thresholds.npy'))
            for name, before in snapshots.items(): self.assertEqual(before, (parent / 'fit' / name).read_bytes())

    def test_failures_before_novel_have_no_complete_and_output_cannot_overwrite(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent = self.fixture(root, module); output = root / 'predict'
            with patch.object(module.parent, 'COUNTS', SMALL), \
                 patch.object(module, '_known_labels', side_effect=ValueError('Invalid known label association')), self.assertRaises(ValueError):
                module.predict(parent, output)
            self.assertFalse((output / 'calibration/complete.json').exists()); self.assertFalse((output / 'complete.json').exists())
            with patch.object(module.parent, '_export_marker', side_effect=AssertionError('No reads after collision')), self.assertRaises(FileExistsError):
                module.predict(parent, output)


if __name__ == '__main__':
    unittest.main()
