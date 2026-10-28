import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_pamf_phase3_source_risk_launch as risk_fixture


class SourceConsensusLaunchTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_consensus_launch')

    def runtime(self, root):
        result = risk_fixture.SourceRiskLaunchTests().runtime(root)
        result.update(run_id='source-consensus', run_root=str(root / 'source-consensus'))
        return result

    def markers(self):
        markers = risk_fixture.SourceRiskLaunchTests().markers()
        schema = 'head_aligned_branch_summary_v1'
        markers['predict']['descriptor_schema'] = schema
        markers['fit'] = dict(status='SOURCE_CONSENSUS_READOUTS_FROZEN', source_only=True, target_reads=0,
            descriptor_schema=schema, geometry_fitted=False, model_seed=2026092701,
            encoder_updates=0, identity_head_updates=0, readout_steps_per_model=200,
            source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0,
            method_names=markers['predict']['method_names'], parameter_counts={'linear_oe': 101, 'mlp16_oe': 1633})
        return markers

    def child(self, rt, markers):
        root = Path(rt['run_root']); seen, active = [], [0, 0]
        class Child:
            pid = 2345
            def __init__(child, command, **kwargs):
                kind = command[5] if command[4].endswith('.source_consensus_run') else 'score'
                self.assertEqual(len(seen), ['fit', 'predict', 'score'].index(kind))
                seen.append((kind, command, kwargs)); child.kind = kind
                active[0] += 1; active[1] = max(active)
                folder = root / kind; folder.mkdir()
                (folder / 'complete.json').write_text(json.dumps(markers[kind]), encoding='utf-8')
                kwargs['stdout'].write(kind + '\n')
            def wait(child): active[0] -= 1; return 0
        return Child, seen, active

    def test_three_cpu_stages_and_no_promotion_finish_without_global_goal_claim(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.runtime(Path(tmp)); child, seen, active = self.child(rt, self.markers())
            with patch.object(module.subprocess, 'Popen', child), patch.object(module.os, 'kill', side_effect=AssertionError('No foreign stops')):
                state = module.dispatch(rt)
            self.assertEqual([kind for kind, _, _ in seen], ['fit', 'predict', 'score']); self.assertEqual(active[1], 1)
            self.assertTrue(all(kwargs['env']['CUDA_VISIBLE_DEVICES'] == '' for _, _, kwargs in seen))
            self.assertIn(str(Path(rt['run_root']) / 'fit'), seen[1][1])
            self.assertEqual(state['status'], 'ANALYZED_SOURCE_ONLY'); self.assertEqual(state['next_step'], 'NO_PROMOTION')
            self.assertEqual(state['descriptor_schema'], 'head_aligned_branch_summary_v1')
            self.assertEqual(state['added_GPU_workers'], 0); self.assertFalse(state['independent_confirmation'])
            self.assertFalse(state['final_phase3_complete']); self.assertEqual(state['foreign_processes_stopped'], 0)
            with self.assertRaises(FileExistsError): module.dispatch(rt)

    def test_descriptor_failure_stops_before_later_stages(self):
        module = self.module()
        for bad in ['fit', 'predict']:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                rt = self.runtime(Path(tmp)); markers = self.markers(); markers[bad]['descriptor_schema'] = 'old_sorted_descriptor'
                child, seen, _ = self.child(rt, markers)
                with patch.object(module.subprocess, 'Popen', child), self.assertRaises(ValueError): module.dispatch(rt)
                self.assertEqual(len(seen), ['fit', 'predict', 'score'].index(bad) + 1)
                self.assertTrue((Path(rt['run_root']) / 'logs/failure.log').is_file())

    def test_budget_and_static_control_promotion_are_rejected(self):
        module = self.module(); marker = self.markers()['fit']; marker['readout_steps_per_model'] = 201
        with self.assertRaises(ValueError): module._validate_marker('fit', marker)
        marker = self.markers()['score']; marker['eligible_methods'] = ['cosine']
        with self.assertRaises(ValueError): module._validate_marker('score', marker)


if __name__ == '__main__': unittest.main()
