import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PARENT = '20261029-phase3-source-open-readout-s2026092701-r01'


class SourceRiskLaunchTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_risk_launch')

    def runtime(self, root):
        return dict(run_id='source-risk', run_root=str(root / 'source-risk'), release_root=str(root),
            python='python', code_commit='a' * 40, launch_owner='risk-owner', parent_run_root=str(root / PARENT),
            parent_oe_data_root=str(root / 'data/source_oe'), max_added_workers_per_gpu=2)

    def markers(self):
        common = dict(source_only=True, target_reads=0)
        return dict(predict=dict(common, status='ALL_SOURCE_OPEN_PREDICTIONS_COMPLETE', truth_read=False,
            query_fit=False, source_V_labels_read=True, source_known_label_calibration=True,
            unknown_calibration_scores_read=0, unknown_calibration_labels_read=0,
            known_count=27000, novel_count=3150, count=30150,
            method_names=['msp', 'maxlogit', 'margin', 'energy', 'cosine', 'linear_oe', 'mlp16_oe'],
            encoder_updates=0, identity_head_updates=0, readout_updates=0,
            independent_confirmation=False, constraints_passed=True,
            limits={'known_macro_frr': .04, 'known_macro_correct_rejection_loss': .015}),
            score=dict(common, status='SCORED_SOURCE_OPEN_COMPLETE', truth_last=True, eligible_methods=[],
                selected_method=None, next_step='NO_PROMOTION', final_phase3_complete=False))

    def child(self, rt, markers):
        root = Path(rt['run_root']); seen = []
        class Child:
            pid = 1234
            def __init__(child, command, **kwargs):
                kind = 'predict' if command[4].endswith('.source_risk_run') else 'score'
                seen.append((kind, command, kwargs)); child.kind = kind
                folder = root / kind; folder.mkdir()
                (folder / 'complete.json').write_text(json.dumps(markers[kind]), encoding='utf-8')
                kwargs['stdout'].write(kind + '\n')
            def wait(child): return 0
        return Child, seen

    def test_two_cpu_stages_finish_without_claiming_independent_confirmation(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(Path(tmp)); child, seen = self.child(rt, self.markers())
            with patch.object(module.subprocess, 'Popen', child), patch.object(module.os, 'kill', side_effect=AssertionError('No process stops')):
                result = module.dispatch(rt)
            self.assertEqual([kind for kind, _, _ in seen], ['predict', 'score'])
            self.assertTrue(all(kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' for _, _, kwargs in seen))
            self.assertEqual(result['status'], 'ANALYZED_SOURCE_ONLY'); self.assertEqual(result['added_GPU_workers'], 0)
            self.assertFalse(result['final_phase3_complete']); self.assertFalse(result['independent_confirmation'])
            self.assertIn(str(Path(rt['parent_run_root']) / 'export/source_V/features.npz'), seen[1][1])
            self.assertIn(str(Path(rt['parent_oe_data_root']) / 'novel_validation/truth/truth.npz'), seen[1][1])
            self.assertEqual(result['foreign_processes_stopped'], 0)
            with self.assertRaises(FileExistsError): module.dispatch(rt)

    def test_bad_predict_boundary_stops_before_truth_scorer(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(Path(tmp)); markers = self.markers(); markers['predict']['source_known_label_calibration'] = False
            child, seen = self.child(rt, markers)
            with patch.object(module.subprocess, 'Popen', child), self.assertRaises(ValueError): module.dispatch(rt)
            self.assertEqual(len(seen), 1)
            state = json.loads((Path(rt['run_root']) / 'pipeline_state.json').read_text())
            self.assertEqual(state['status'], 'STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE')

    def test_strict_runtime_and_no_static_control_promotion(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(Path(tmp))
            for change in [{'target_root': 'forbidden'}, {'max_added_workers_per_gpu': 3}, {'parent_run_root': str(Path(tmp) / 'other-parent')}]:
                with self.subTest(change=change), self.assertRaises(ValueError): module.validate_runtime(dict(rt, **change))
            marker = self.markers()['score']; marker['eligible_methods'] = ['cosine']
            with self.assertRaises(ValueError): module._validate_marker('score', marker)


if __name__ == '__main__':
    unittest.main()
