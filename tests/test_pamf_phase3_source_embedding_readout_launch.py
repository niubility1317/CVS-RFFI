import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import test_pamf_phase3_source_embedding_launch as old


class EmbeddingReadoutLaunchTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_readout_launch.py'
        self.assertTrue(path.is_file(), 'E51 readout four-stage launcher is not implemented')
        return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_readout_launch')

    def runtime(self, root, module):
        rt = old.EmbeddingLaunchTests().runtime(root)
        for key in ('model', 'parent_oe_data_root', 'diverse_data_root'): rt.pop(key)
        rt.update(embedding_run_root=str(root / module.exporter.EMBEDDING_RUN), oe_data_root=str(root / 'data/source_oe'))
        return rt

    def test_runtime_has_no_model_training_inputs_and_commands_isolate_truth(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); rt = self.runtime(root, module)
            module.validate_runtime(rt)
            for arm in module.ARMS:
                config = module.configuration(rt, Path(rt['run_root']), arm)
                self.assertEqual(set(config), module.exporter.KEYS)
                cmd = module.commands(rt, Path(rt['run_root']), arm)
                self.assertEqual(set(cmd), {'export', 'fit', 'predict', 'score'})
                for stage in ('export', 'fit', 'predict'): self.assertNotIn('--novel-truth', cmd[stage])
                self.assertIn('--novel-truth', cmd['score'])
            with self.assertRaises(ValueError): module.validate_runtime(dict(rt, model={}))

    def test_only_mlp_can_advance_and_supervised_preferred_after_both_complete(self):
        module = self.module()
        score = dict(status='SCORED_SOURCE_EMBEDDING_READOUT_COMPLETE', source_only=True, target_reads=0,
            arm='supervised_only', truth_last=True, final_phase3_complete=False, eligible_methods=['mlp16_oe'],
            selected_method='mlp16_oe', next_step=module.PASS_NEXT)
        module._validate_marker('score', score, 'supervised_only')
        with self.assertRaises(ValueError): module._validate_marker('score', dict(score, eligible_methods=['msp'], selected_method='msp'), 'supervised_only')
        outcomes = {arm: {'complete': True, 'marker': dict(score, arm=arm)} for arm in module.ARMS}
        self.assertEqual(module.selection(outcomes), 'supervised_only::mlp16_oe')
        outcomes['supervised_only']['complete'] = False
        self.assertIsNone(module.selection(outcomes))

    def test_two_rows_four_stages_gpu_export_only_and_failed_row_preserves_healthy(self):
        module = self.module()
        for failed in (None, 'supervised_only'):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as td:
                rt = self.runtime(Path(td), module); seen = []; barrier = threading.Barrier(2)
                lock = threading.Lock(); active = [0, 0]
                class Child:
                    pid = 9012
                    def __init__(child, command, **kw):
                        if command[4].endswith('source_embedding_readout_export'):
                            stage = 'export'; config = json.loads(Path(command[command.index('--config') + 1]).read_text()); output = Path(config['output_root'])
                        else:
                            stage = command[5] if command[4].endswith('source_embedding_readout_run') else 'score'
                            output = Path(command[command.index('--output') + 1])
                        child.stage, child.arm, child.output = stage, output.parent.name, output
                        with lock:
                            seen.append((child.arm, stage, kw['env']['CUDA_VISIBLE_DEVICES']))
                            active[0] += 1; active[1] = max(active)
                    def wait(child):
                        if child.stage == 'export': barrier.wait(timeout=5)
                        child.output.mkdir()
                        if child.arm == failed and child.stage == 'export':
                            (child.output / 'partial').write_text('preserve')
                            with lock: active[0] -= 1
                            return 8
                        state = dict(run_id=module.exporter.EMBEDDING_RUN, code_commit=module.exporter.EMBEDDING_COMMIT,
                            arm=child.arm, row_id=child.arm + '-s2026092701',
                            checkpoint=str(Path(rt['embedding_run_root']) / 'rows' / child.arm / 'train/final.pt'))
                        marker = dict(source_only=True, target_reads=0, arm=child.arm, encoder_state_identity=state,
                            new_encoder_updates=0, new_identity_head_updates=0)
                        if child.stage == 'export': marker.update(status='SOURCE_EMBEDDING_BRANCH_FEATURES_EXPORTED',
                            head_changed=False, counts=module.exporter.COUNTS, model_classes=6, parameters=232993,
                            optimizer_updates=0, original_U_s_reads=0, OE_truth_reads=0)
                        elif child.stage == 'fit': marker.update(status='SOURCE_EMBEDDING_READOUTS_FROZEN',
                            geometry_fitted=True, descriptor_schema=module.run.consensus.DESCRIPTOR_SCHEMA,
                            readout_training_regime=module.run.REGIME, parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633},
                            readout_steps_per_model=200, source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0)
                        elif child.stage == 'predict': marker.update(status='SOURCE_EMBEDDING_READOUT_PREDICTIONS_COMPLETE',
                            unknown_truth_read=False, query_fit=False, default_head_replaced=False, method_names=list(module.METHODS),
                            source_V_labels_read=True, source_known_label_calibration=True, encoder_updates=222,
                            identity_head_updates=222, readout_steps_per_model=200, known_count=27000, novel_count=3150, count=30150)
                        else: marker.update(status='SCORED_SOURCE_EMBEDDING_READOUT_COMPLETE', truth_last=True,
                            final_phase3_complete=False, eligible_methods=['mlp16_oe'], selected_method='mlp16_oe', next_step=module.PASS_NEXT)
                        (child.output / 'complete.json').write_text(json.dumps(marker))
                        with lock: active[0] -= 1
                        return 0
                with patch.object(module, 'select_gpus', return_value=[0, 0]), patch.object(module.subprocess, 'Popen', Child), \
                        patch.object(module.os, 'kill', side_effect=AssertionError('No foreign stop')), patch('builtins.print'):
                    result = module.dispatch(rt)
                self.assertEqual(active[1], 2)
                for arm in module.ARMS:
                    jobs = [(stage, gpu) for row, stage, gpu in seen if row == arm]
                    expected = [('export', '0')] if arm == failed else [('export', '0'), ('fit', ''), ('predict', ''), ('score', '')]
                    self.assertEqual(jobs, expected)
                self.assertEqual(result['foreign_processes_stopped'], 0)
                self.assertFalse(result['final_phase3_complete'])
                if failed:
                    self.assertEqual(result['status'], 'PARTIAL_SOURCE_ONLY'); self.assertIsNone(result['selected_candidate'])
                    self.assertTrue((Path(rt['run_root']) / 'rows/supervised_only/export/partial').is_file())
                else:
                    self.assertEqual(result['status'], 'ANALYZED_SOURCE_ONLY')
                    self.assertEqual(result['selected_candidate'], 'supervised_only::mlp16_oe')
                with self.assertRaises(FileExistsError): module.dispatch(rt)


if __name__ == '__main__': unittest.main()
