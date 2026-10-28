import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import test_pamf_phase3_source_risk_run as old


class EmbeddingReadoutRunTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_readout_run.py'
        self.assertTrue(path.is_file(), 'Frozen E51 feature readouts are not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_readout_run')

    def fixture(self, root, module):
        base = old.SourceRiskRunTests().fixture(root, module)
        export = root / 'new-export'; shutil.copytree(base / 'export', export)
        marker = dict(arm='supervised_only', config={'parent_run_root': str(base)},
            encoder_state_identity={'arm': 'supervised_only', 'row_id': 'supervised_only-s2026092701'},
            parent_export_identity={'original': 'full-F6'}, source_OE_handles={}, counts=old.SMALL)
        return base, export, marker

    def test_new_l_geometry_fit_only_l_oe_then_risk_before_novel_and_same_head(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); base, export, marker = self.fixture(root, module)
            load = module.parent._load_features; fit_geometry = module.parent.core.fit_geometry
            roles = []
            def training(e, role, include_y=False):
                self.assertIn(role, ['source_L', 'exposure_train']); roles.append(role)
                return load(e, role, include_y)
            with patch.object(module, '_export_marker', return_value=marker), patch.object(module.parent, 'COUNTS', old.SMALL), \
                    patch.object(module.parent, '_load_features', side_effect=training), \
                    patch.object(module.parent.core, 'fit_geometry', wraps=fit_geometry) as geometry, patch('builtins.print'):
                fitted = module.fit(export, 2026092701, root / 'fit')
            self.assertEqual(roles, ['source_L', 'exposure_train']); self.assertEqual(geometry.call_count, 1)
            self.assertTrue(fitted['geometry_fitted']); self.assertEqual(fitted['readout_steps_per_model'], 200)
            frozen = {name: (root / 'fit' / name).read_bytes() for name in ('geometry.npz', 'readouts.npz')}
            roles.clear()
            def validation(e, role, include_y=False):
                self.assertIn(role, ['source_V', 'novel_validation']); roles.append(role)
                if role == 'novel_validation': self.assertTrue((root / 'predict/calibration/complete.json').is_file())
                return load(e, role, include_y)
            with patch.object(module, '_export_marker', return_value=marker), patch.object(module.parent, 'COUNTS', old.SMALL), \
                    patch.object(module.parent, '_load_features', side_effect=validation), \
                    patch.object(module.parent.core, 'fit_readouts', side_effect=AssertionError('No validation fitting')), patch('builtins.print'):
                result = module.predict(export, root / 'fit', root / 'predict')
            self.assertEqual(roles, ['source_V', 'novel_validation'])
            self.assertEqual(result['method_names'], list(module.METHODS)); self.assertEqual(result['new_encoder_updates'], 0)
            self.assertEqual(result['status'], 'SOURCE_EMBEDDING_READOUT_PREDICTIONS_COMPLETE')
            for name, content in frozen.items(): self.assertEqual(content, (root / 'fit' / name).read_bytes())
            with np.load(root / 'predict/predictions.npz', allow_pickle=False) as pred:
                self.assertEqual(pred['knownness'].shape, (312, 7))
                self.assertIn('baseline_closed_prediction', pred.files)
                np.testing.assert_array_equal(pred['prediction'], np.where(pred['accepted'], pred['closed_prediction'][:, None], -1))
            with self.assertRaises(FileExistsError): module.fit(export, 2026092701, root / 'fit')

    def test_wrong_encoder_or_regime_stops_before_validation(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, export, marker = self.fixture(root, module)
            with patch.object(module, '_export_marker', return_value=marker), patch.object(module.parent, 'COUNTS', old.SMALL), patch('builtins.print'):
                module.fit(export, 2026092701, root / 'fit')
            path = root / 'fit/complete.json'; value = json.loads(path.read_text())
            value['encoder_state_identity'] = {'arm': 'uniform_oe'}; path.write_text(json.dumps(value))
            with patch.object(module, '_export_marker', return_value=marker), \
                    patch.object(module.parent, '_load_features', side_effect=AssertionError('No V on wrong state')), self.assertRaises(ValueError):
                module.predict(export, root / 'fit', root / 'wrong')
            self.assertFalse((root / 'wrong/complete.json').exists())


if __name__ == '__main__': unittest.main()
