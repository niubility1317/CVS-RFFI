import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import test_pamf_phase3_source_open_export as old


class BudgetExportTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_budget_export.py'
        self.assertTrue(path.is_file(), 'E60 frozen branch export is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_budget_export')

    def test_budget_identity_requires_own_e60_not_old_e51(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); arm = 'uniform_oe'
            value = dict(run_id=module.RUN_ID, row_id=arm + '-s2026092701', arm=arm, code_commit='a' * 40,
                checkpoint=str(root / module.RUN_ID / 'rows' / arm / 'train/final.pt'), updates=2220, cumulative_updates=13320)
            module.validate_encoder_identity(value, arm)
            for change in ({'updates': 222}, {'cumulative_updates': 11322}, {'run_id': 'old-E51'}, {'arm': 'supervised_only'}):
                with self.subTest(change=change), self.assertRaises(ValueError): module.validate_encoder_identity(dict(value, **change), arm)

    def test_full_export_frozen_smoke_then_all_roles_40652_contract(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); base, contract, loader, reader = old.SourceOpenExportTests().fixture(root)
            config = dict(run_id=module.RUN_ID, row_id='supervised_only-s2026092701', code_commit='a' * 40,
                train_config=str(root / 'train.json'), checkpoint=str(root / 'final.pt'), parent_run_root=str(root / 'parent'),
                oe_data_root=base['oe_data_root'], output_root=str(root / 'export'))
            train = dict(base, arm='supervised_only', steps=2220)
            identity = dict(arm='supervised_only', updates=2220, cumulative_updates=13320)
            original = dict(source_OE_handles={role: dict(capsule_id='capsule-' + role, split_id='split-' + role, count=6)
                for role in ('exposure_train', 'novel_validation')})
            roles = []
            def source(cfg, role, c):
                roles.append(role)
                if role == 'V': self.assertTrue((root / 'export/smoke.json').is_file())
                return loader(cfg, role, c)
            with patch.object(module, '_encoder', return_value=(reader, train, identity)), \
                    patch.object(module, '_parent', return_value=original), patch.object(module.trainer, '_contract', return_value=contract), \
                    patch.object(module.runner, '_source_dataset', side_effect=source), \
                    patch.object(module.source, 'COUNTS', old.SMALL), patch.object(module, 'COUNTS', old.SMALL), patch('builtins.print'):
                result = module.export(config, 'cpu')
            self.assertEqual(result['status'], 'SOURCE_BUDGET_BRANCH_FEATURES_EXPORTED')
            self.assertEqual(result['historical_encoder_updates'], 2220)
            self.assertEqual(result['new_encoder_updates'], 0)
            self.assertEqual(result['model_forward_samples'], 26)
            self.assertEqual(reader.seen, 26); self.assertEqual(roles, ['L_s', 'V'])
            for role, ref in result['feature_refs'].items():
                with np.load(root / 'export' / ref, allow_pickle=False) as values:
                    self.assertEqual(set(values.files), module.parent.COMMON | ({'y'} if role in ('source_L', 'source_V') else set()))
            with self.assertRaises(FileExistsError): module.export(config, 'cpu')

    def test_complete_and_payload_loader_budget_are_checked_before_source_iq(self):
        module = self.module()
        from test_pamf_phase3_source_open_embedding_train import EmbeddingTrainTests
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); arm = 'supervised_only'; train_root = root / module.RUN_ID / 'rows' / arm / 'train'
            train_root.mkdir(parents=True)
            train = EmbeddingTrainTests().config(str(train_root))
            train.update(run_id=module.RUN_ID, row_id=arm + '-s2026092701', steps=2220)
            train_path = root / 'train.json'; train_path.write_text(json.dumps(train), encoding='utf-8')
            config = dict(run_id=module.RUN_ID, row_id=train['row_id'], code_commit=train['code_commit'],
                train_config=str(train_path), checkpoint=str(train_root / 'final.pt'), parent_run_root='parent',
                oe_data_root='oe', output_root=str(root / 'export'))
            marker = dict(status='SOURCE_OPEN_EMBEDDING_FROZEN', schema=module.trainer.SCHEMA, config=train,
                arm=arm, updates=2220, parent_updates=11100, cumulative_updates=13320, increment_epoch=60,
                optimizer_reset=True, source_only=True, default_head_joint=True, identity_head_replaced=False,
                model_classes=6, parameters=232993, U_reads=0, validation_reads=0, novel_validation_reads=0,
                target_reads=0, OE_truth_reads=0)
            path = train_root / 'complete.json'; path.write_text(json.dumps(marker), encoding='utf-8')
            reader = old.Reader().eval().requires_grad_(False)
            wrapper = SimpleNamespace(id_backbone=SimpleNamespace(encoder=reader))
            with patch.object(module.trainer, 'load_trained_model', return_value=wrapper) as load:
                _, actual, identity = module._encoder(config, 'cpu')
            self.assertEqual(actual['steps'], 2220); self.assertEqual(identity['cumulative_updates'], 13320)
            load.assert_called_once_with(train, config['checkpoint'], 'cpu')
            for changed in ({'updates': 222}, {'cumulative_updates': 11322}, {'increment_epoch': 51}, {'U_reads': 1}):
                path.write_text(json.dumps(dict(marker, **changed)), encoding='utf-8')
                with self.subTest(changed=changed), patch.object(module.trainer, 'load_trained_model', side_effect=AssertionError('No payload after invalid marker')) as load:
                    with self.assertRaises(ValueError): module._encoder(config, 'cpu')
                    self.assertEqual(load.call_count, 0)


if __name__ == '__main__': unittest.main()
