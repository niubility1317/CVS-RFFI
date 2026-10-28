import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_pamf_phase3_source_risk_launch as risk_fixture


class SourceDiverseLaunchTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_diverse_launch')

    def runtime(self, root):
        rt = risk_fixture.SourceRiskLaunchTests().runtime(root)
        rt.update(run_id='source-diverse', run_root=str(root / 'source-diverse'),
            diverse_data_root=str(root / 'data/source_oe_diverse'), allowed_gpus=[2, 3], minimum_free_mb=4096)
        return rt

    def markers(self):
        markers = risk_fixture.SourceRiskLaunchTests().markers()
        schema, regime = 'head_aligned_branch_summary_v1', 'source_role_domain_balanced_v1'
        markers['export'] = dict(status='SOURCE_DIVERSE_OE_FEATURES_EXPORTED', source_only=True, target_reads=0,
            optimizer_updates=0, original_U_s_reads=0, OE_truth_reads=0, head_changed=False, model_classes=6,
            parameters=232993, count=6134, feature_ref='exposure_train/public/features.npz',
            parent_export_identity={'source': 'fixture'}, source_OE_handle=dict(capsule_id='diverse', split_id='diverse-split', count=6134))
        markers['fit'] = dict(status='SOURCE_DIVERSE_READOUTS_FROZEN', source_only=True, target_reads=0,
            descriptor_schema=schema, readout_training_regime=regime, geometry_fitted=False, model_seed=2026092701,
            encoder_updates=0, identity_head_updates=0, readout_steps_per_model=200, source_validation_reads=0,
            novel_validation_reads=0, OE_truth_reads=0, method_names=markers['predict']['method_names'],
            parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633}, source_L_count=6300, source_OE_count=6134)
        markers['predict'].update(descriptor_schema=schema, readout_training_regime=regime, domain_metadata_used_for_deployed_scores=False)
        return markers

    def child(self, rt, markers):
        root = Path(rt['run_root']); seen, active = [], [0, 0]
        class Child:
            pid = 3456
            def __init__(child, command, **kwargs):
                if command[4].endswith('.source_diverse_export'): kind = 'export'
                elif command[4].endswith('.source_diverse_run'): kind = command[5]
                else: kind = 'score'
                self.assertEqual(len(seen), ['export', 'fit', 'predict', 'score'].index(kind))
                active[0] += 1; active[1] = max(active); seen.append((kind, command, kwargs)); child.kind = kind
                folder = root / kind; folder.mkdir(); (folder / 'complete.json').write_text(json.dumps(markers[kind]), encoding='utf-8')
                kwargs['stdout'].write(kind + '\n')
            def wait(child): active[0] -= 1; return 0
        return Child, seen, active

    def test_export_gpu_then_three_cpu_stages_and_minimal_export_config(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(Path(tmp)); child, seen, active = self.child(rt, self.markers())
            with patch.object(module.subprocess, 'Popen', child), patch.object(module, 'choose_gpu', return_value=3) as gpu, \
                 patch.object(module.os, 'kill', side_effect=AssertionError('No foreign process stops')):
                state = module.dispatch(rt)
            self.assertEqual([kind for kind, _, _ in seen], ['export', 'fit', 'predict', 'score'])
            self.assertEqual(gpu.call_count, 1); self.assertEqual(active[1], 1)
            self.assertEqual([kwargs['env']['CUDA_VISIBLE_DEVICES'] for _, _, kwargs in seen], ['3', '', '', ''])
            config = json.loads((Path(rt['run_root']) / 'configs/export.json').read_text())
            self.assertEqual(set(config), {'run_id', 'row_id', 'code_commit', 'parent_run_root', 'diverse_data_root', 'output_root'})
            self.assertEqual(config['diverse_data_root'], rt['diverse_data_root'])
            self.assertIn(str(Path(rt['run_root']) / 'export'), seen[1][1]); self.assertIn(str(Path(rt['run_root']) / 'fit'), seen[2][1])
            self.assertEqual(state['status'], 'ANALYZED_SOURCE_ONLY'); self.assertEqual(state['next_step'], 'NO_PROMOTION')
            self.assertFalse(state['final_phase3_complete']); self.assertFalse(state['independent_confirmation'])
            self.assertEqual(state['foreign_processes_stopped'], 0)
            with self.assertRaises(FileExistsError): module.dispatch(rt)

    def test_wrong_count_or_old_regime_stops_without_later_stage(self):
        module = self.module()
        for failure in ['export', 'fit', 'predict']:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                rt = self.runtime(Path(tmp)); markers = self.markers()
                if failure == 'export': markers['export']['count'] = 4200
                else: markers[failure]['readout_training_regime'] = 'old_unweighted'
                child, seen, _ = self.child(rt, markers)
                with patch.object(module.subprocess, 'Popen', child), patch.object(module, 'choose_gpu', return_value=2), self.assertRaises(ValueError):
                    module.dispatch(rt)
                self.assertEqual(len(seen), ['export', 'fit', 'predict', 'score'].index(failure) + 1)
                self.assertTrue((Path(rt['run_root']) / 'logs/failure.log').is_file())

    def test_capacity_foreign_preservation_and_static_control_boundaries(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(Path(tmp)); module.validate_runtime(rt)
            with self.assertRaises(ValueError): module.validate_runtime(dict(rt, max_added_workers_per_gpu=3))
            with self.assertRaises(ValueError): module.validate_runtime(dict(rt, allowed_gpus=[True]))
            marker = self.markers()['score']; marker['eligible_methods'] = ['cosine']
            with self.assertRaises(ValueError): module._validate_marker('score', marker)


if __name__ == '__main__': unittest.main()
