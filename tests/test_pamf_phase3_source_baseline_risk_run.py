import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import test_pamf_phase3_source_embedding_readout_run as old
import test_pamf_phase3_source_embedding_readout_launch as launch_fixture


class BaselineRiskRunTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_baseline_risk_run.py'
        self.assertTrue(path.is_file(), 'Frozen baseline risk predictor is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_baseline_risk_run')

    def test_smoke_then_known_baseline_risk_freeze_then_novel_no_refit_and_shared_head(self):
        module = self.module(); readout = module.readout
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _, export, marker = old.EmbeddingReadoutRunTests().fixture(root, readout)
            marker['encoder_state_identity'] = dict(arm='supervised_only', row_id='supervised_only-s2026092701',
                run_id=readout.exporter.EMBEDDING_RUN, code_commit=readout.exporter.EMBEDDING_COMMIT,
                checkpoint=str(root / readout.exporter.EMBEDDING_RUN / 'rows/supervised_only/train/final.pt'))
            with patch.object(readout, '_export_marker', return_value=marker), patch.object(module.parent, 'COUNTS', old.old.SMALL), patch('builtins.print'):
                readout.fit(export, 2026092701, root / 'fit')
            identity = dict(marker=marker, export_root=export, fit_root=root / 'fit', arm='supervised_only')
            loading = module.parent._load_features; roles = []
            def load(e, role, include_y=False):
                roles.append(role); self.assertFalse(include_y)
                if role == 'source_V': self.assertTrue((root / 'predict/smoke.json').is_file())
                if role == 'novel_validation': self.assertTrue((root / 'predict/calibration/complete.json').is_file())
                return loading(e, role, include_y)
            before = {name: (root / 'fit' / name).read_bytes() for name in ('geometry.npz', 'readouts.npz')}
            with patch.object(module, '_parent_row', return_value=identity), patch.object(module.parent, 'COUNTS', old.old.SMALL), \
                    patch.object(module.parent, '_load_features', side_effect=load), \
                    patch.object(module.parent.core, 'fit_readouts', side_effect=AssertionError('No new fit')), patch('builtins.print'):
                result = module.predict(root / 'parent-row', root / 'predict')
            self.assertEqual(roles, ['source_L', 'source_V', 'novel_validation'])
            self.assertEqual(result['calibration_schema'], module.calibration.SCHEMA)
            self.assertEqual(result['new_readout_updates'], 0)
            self.assertEqual(result['new_encoder_updates'], 0)
            with np.load(root / 'predict/predictions.npz', allow_pickle=False) as pred:
                self.assertEqual(len(pred.files), 9); self.assertEqual(pred['knownness'].shape, (312, 7))
                np.testing.assert_array_equal(pred['prediction'], np.where(pred['accepted'], pred['closed_prediction'][:, None], -1))
            from experiments.pamf_phase3_open_world import source_embedding_readout_scoring as independent
            predictions, checked = independent._predictions(root / 'predict/predictions.npz')
            self.assertEqual(predictions['knownness'].shape, (312, 7))
            self.assertEqual(checked['calibration_schema'], module.calibration.SCHEMA)
            for name, value in before.items(): self.assertEqual(value, (root / 'fit' / name).read_bytes())
            with self.assertRaises(FileExistsError): module.predict(root / 'parent-row', root / 'predict')

    def test_parent_must_be_fixed_completed_combo_and_failure_keeps_no_marker(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError): module._parent_row(root / 'foreign/rows/uniform_oe')
            with patch.object(module, '_parent_row', side_effect=ValueError('bad frozen state')), self.assertRaises(ValueError):
                module.predict(root / 'bad', root / 'out')
            self.assertFalse((root / 'out/complete.json').exists())

    def test_cpu_two_rows_complete_or_partial_without_any_gpu_or_foreign_stop(self):
        module = self.module()
        for failure in (None, 'supervised_only'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                root = Path(td); rt = dict(run_id=module.RUN_ID, run_root=str(root / module.RUN_ID), release_root=str(root / 'release'),
                    python=str(root / 'python'), code_commit='a' * 40, launch_owner='baseline-risk',
                    parent_run_root=str(root / module.PARENT_RUN), max_concurrent_rows=2)
                seen = []; barrier = threading.Barrier(2)
                class Child:
                    pid = 92
                    def __init__(child, command, **kw):
                        child.kind = 'predict' if command[4].endswith('source_baseline_risk_run') else 'score'
                        child.output = Path(command[command.index('--output') + 1]); child.arm = child.output.parent.name
                        seen.append((child.arm, child.kind, kw['env']['CUDA_VISIBLE_DEVICES']))
                    def wait(child):
                        if child.kind == 'predict': barrier.wait(timeout=5)
                        child.output.mkdir()
                        if child.kind == 'predict' and child.arm == failure: return 3
                        (child.output / 'complete.json').write_text(json.dumps({'arm': child.arm, 'kind': child.kind,
                            'eligible_methods': ['mlp16_oe'], 'selected_method': 'mlp16_oe', 'next_step': module.PASS_NEXT}))
                        return 0
                with patch.object(module.subprocess, 'Popen', Child), patch.object(module, '_validate_marker'), \
                        patch.object(module, '_score_command', return_value=['python', '-B', '-u', '-m', 'scorer', '--output', 'unused']) as score, \
                        patch.object(module.os, 'kill', side_effect=AssertionError('No process stop')), patch('builtins.print'):
                    score.side_effect = lambda rt, root, arm: ['python', '-B', '-u', '-m', 'scorer', '--output', str(root / 'rows' / arm / 'score')]
                    result = module.dispatch(rt)
                self.assertTrue(all(gpu == '' for _, _, gpu in seen))
                self.assertEqual(result['foreign_processes_stopped'], 0)
                self.assertEqual(result['status'], 'PARTIAL_SOURCE_ONLY' if failure else 'ANALYZED_SOURCE_ONLY')
                self.assertEqual(result['selected_candidate'], None if failure else 'supervised_only::mlp16_oe')

    def test_score_paths_use_original_f6_baseline_and_only_score_gets_private_truth(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rt = dict(python='python', parent_run_root=str(root / module.PARENT_RUN))
            ctx = dict(marker={'config': {'parent_run_root': str(root / 'original-F6-source-open'), 'oe_data_root': str(root / 'oe')}})
            with patch.object(module, '_parent_row', return_value=ctx):
                command = module._score_command(rt, root / 'new', 'uniform_oe')
            self.assertEqual(command[command.index('--known-features') + 1], str(root / 'original-F6-source-open/export/source_V/features.npz'))
            self.assertEqual(command[command.index('--novel-truth') + 1], str(root / 'oe/novel_validation/truth/truth.npz'))


if __name__ == '__main__': unittest.main()
