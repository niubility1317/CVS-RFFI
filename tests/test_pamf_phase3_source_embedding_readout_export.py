import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import test_pamf_phase3_source_open_export as old


class EmbeddingReadoutExportTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_readout_export.py'
        self.assertTrue(path.is_file(), 'Frozen E51 branch exporter is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_readout_export')

    def test_export_own_smoke_then_all_roles_no_truth_and_frozen_marker_last(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); base, contract, loader, reader = old.SourceOpenExportTests().fixture(root)
            config = dict(run_id='new-readout', row_id='supervised_only-s2026092701', code_commit='a' * 40,
                train_config=str(root / 'train.json'), checkpoint=str(root / 'final.pt'),
                parent_run_root=str(root / 'parent'), oe_data_root=base['oe_data_root'], output_root=str(root / 'export'))
            train = dict(base, arm='supervised_only')
            identity = {'arm': 'supervised_only', 'row_id': config['row_id']}
            original = {'original': 'full-F6', 'source_OE_handles': {role: dict(capsule_id='capsule-' + role,
                split_id='split-' + role, count=6) for role in ('exposure_train', 'novel_validation')}}
            roles, writes = [], []
            def source(cfg, role, c):
                if role == 'V': self.assertTrue((root / 'export/smoke.json').is_file())
                roles.append(role); return loader(cfg, role, c)
            write = module.runner._write
            def record(path, value):
                writes.append(Path(path).name)
                if Path(path).name == 'complete.json': self.assertTrue((root / 'export/novel_validation/public/features.npz').is_file())
                return write(path, value)
            with patch.object(module, '_encoder', return_value=(reader, train, identity)), \
                    patch.object(module, '_parent', return_value=original), \
                    patch.object(module.trainer, '_contract', return_value=contract), \
                    patch.object(module.runner, '_source_dataset', side_effect=source), \
                    patch.object(module.source, 'COUNTS', old.SMALL), patch.object(module, 'COUNTS', old.SMALL), \
                    patch.object(module.runner, '_write', side_effect=record), patch('builtins.print'):
                result = module.export(config, 'cpu')
            self.assertEqual(roles, ['L_s', 'V']); self.assertEqual(reader.seen, 26)
            self.assertEqual(writes[0], 'smoke.json'); self.assertEqual(writes[-1], 'complete.json')
            self.assertEqual(result['status'], 'SOURCE_EMBEDDING_BRANCH_FEATURES_EXPORTED')
            self.assertEqual(result['encoder_state_identity'], identity)
            self.assertEqual(result['new_encoder_updates'], 0)
            for role, ref in result['feature_refs'].items():
                with np.load(root / 'export' / ref, allow_pickle=False) as stored:
                    expected = module.parent.COMMON | ({'y'} if role in ('source_L', 'source_V') else set())
                    self.assertEqual(set(stored.files), expected)
                    self.assertTrue((stored['features__reference29'][:, 27] == 0).all())
            with self.assertRaises(FileExistsError): module.export(config, 'cpu')

    def test_identity_and_config_reject_wrong_encoder_before_iq(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            identity = dict(run_id=module.EMBEDDING_RUN, row_id='uniform_oe-s2026092701', arm='uniform_oe',
                code_commit=module.EMBEDDING_COMMIT, checkpoint=str(root / module.EMBEDDING_RUN / 'rows/uniform_oe/train/final.pt'))
            module.validate_encoder_identity(identity, 'uniform_oe')
            with self.assertRaises(ValueError): module.validate_encoder_identity(identity, 'supervised_only')
            with self.assertRaises(ValueError): module.validate_encoder_identity(dict(identity, code_commit='0' * 40), 'uniform_oe')
            with self.assertRaises(ValueError): module.validate_config({'target': 'forbidden'})


if __name__ == '__main__': unittest.main()
