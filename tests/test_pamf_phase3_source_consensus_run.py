import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_pamf_phase3_source_risk_run as fixture_module


SMALL = fixture_module.SMALL


class SourceConsensusRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_consensus_run')

    def fixture(self, root, module):
        return fixture_module.SourceRiskRunTests().fixture(root, module)

    def test_fit_smokes_real_parent_and_new_descriptor_then_only_l_and_oe(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = self.fixture(root, module)
            original_load, original_score = module.parent._load_features, module.parent._score
            roles, smoke_counts = [], []
            def loading(export, role, include_y=False):
                self.assertIn(role, ['source_L', 'exposure_train']); roles.append(role)
                if role == 'exposure_train': self.assertTrue((root / 'fit/smoke.json').is_file())
                return original_load(export, role, include_y)
            def scoring(data, geometry, state):
                smoke_counts.append(len(data['ids'])); return original_score(data, geometry, state)
            with patch.object(module.parent, 'COUNTS', SMALL), patch.object(module.parent, '_load_features', side_effect=loading), \
                 patch.object(module.parent, '_score', side_effect=scoring), \
                 patch.object(module.parent.core, 'fit_geometry', side_effect=AssertionError('Geometry is inherited, not refitted')):
                result = module.fit(base, 2026092701, root / 'fit')
            self.assertEqual(roles, ['source_L', 'exposure_train']); self.assertEqual(smoke_counts, [2])
            self.assertEqual(result['descriptor_schema'], 'head_aligned_branch_summary_v1')
            self.assertEqual(result['readout_steps_per_model'], 200)
            self.assertEqual((root / 'fit/geometry.npz').read_bytes(), (base / 'fit/geometry.npz').read_bytes())
            with np.load(root / 'fit/readouts.npz', allow_pickle=False) as data:
                self.assertEqual(data['descriptor_schema'].item(), result['descriptor_schema'])
            training = json.loads((root / 'fit/training.json').read_text())
            self.assertEqual({len(values) for values in training['training']['losses'].values()}, {200})
            with patch.object(module.parent, 'COUNTS', SMALL), self.assertRaises(FileExistsError):
                module.fit(base, 2026092701, root / 'fit')

    def test_predict_new_state_risk_freeze_before_novel_same_original_head(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = self.fixture(root, module)
            with patch.object(module.parent, 'COUNTS', SMALL): module.fit(base, 2026092701, root / 'fit')
            original_load, roles = module.parent._load_features, []
            def loading(export, role, include_y=False):
                self.assertIn(role, ['source_V', 'novel_validation']); self.assertFalse(include_y); roles.append(role)
                if role == 'novel_validation': self.assertTrue((root / 'predict/calibration/complete.json').is_file())
                return original_load(export, role, include_y)
            with patch.object(module.parent, 'COUNTS', SMALL), patch.object(module.parent, '_load_features', side_effect=loading), \
                 patch.object(module.parent.core, 'fit_readouts', side_effect=AssertionError('No validation fit')):
                result = module.predict(base, root / 'fit', root / 'predict')
            self.assertEqual(roles, ['source_V', 'novel_validation'])
            self.assertTrue(result['source_V_labels_read']); self.assertFalse(result['truth_read'])
            self.assertEqual(result['descriptor_schema'], module.consensus.DESCRIPTOR_SCHEMA)
            with np.load(root / 'predict/predictions.npz', allow_pickle=False) as pred:
                np.testing.assert_array_equal(pred['prediction'], np.where(pred['accepted'], pred['closed_prediction'][:, None], -1))
                ids = pred['ids'].tolist()
                with patch.object(module.parent, 'COUNTS', SMALL):
                    for role in ['source_V', 'novel_validation']:
                        data = original_load(base / 'export', role, False)
                        np.testing.assert_array_equal(pred['closed_prediction'][[ids.index(value) for value in data['ids']]], data['logits'].argmax(axis=1))
            with patch.object(module.parent, 'COUNTS', SMALL), self.assertRaises(FileExistsError):
                module.predict(base, root / 'fit', root / 'predict')

    def test_old_descriptor_state_and_wrong_marker_fail_before_v(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); base = self.fixture(root, module)
            with patch.object(module.parent, 'COUNTS', SMALL), patch.object(module.parent, '_load_features', side_effect=AssertionError('No V before new fit identity')), self.assertRaises(ValueError):
                module.predict(base, base / 'fit', root / 'old-state')
            with patch.object(module.parent, 'COUNTS', SMALL): module.fit(base, 2026092701, root / 'fit')
            path = root / 'fit/readouts.npz'
            with np.load(path, allow_pickle=False) as stored: arrays = {key: stored[key].copy() for key in stored.files}
            arrays['descriptor_schema'] = np.asarray('old_sorted_descriptor')
            np.savez(path, **arrays)
            with patch.object(module.parent, 'COUNTS', SMALL), patch.object(module.parent, '_load_features', side_effect=AssertionError('No V on descriptor mismatch')), self.assertRaises(ValueError):
                module.predict(base, root / 'fit', root / 'bad-state')
            self.assertFalse((root / 'bad-state/complete.json').exists())


if __name__ == '__main__': unittest.main()
