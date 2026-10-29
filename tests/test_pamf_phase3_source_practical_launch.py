import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class PracticalLaunchTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.source_practical_launch')

    def runtime(self, base):
        run='20261029-phase3-source-practical-readout-s2026092701-r01'
        root=base/run
        return dict(run_id=run,run_root=str(root),release_root=str(base/'release'),python='python',
            code_commit='a'*40,launch_owner='owner',allowed_gpus=[0],max_added_workers_per_gpu=2,
            minimum_free_mb=4096,targets_path=str(base/'targets.json'),
            export_config=dict(output_root=str(root/'source/export')),
            source_fit_root=str(base/'oldsource/fit'),novel_truth=str(base/'novel.truth.npz'),
            rows={c:dict(predict_config=dict(output_root=str(root/'rows'/c/'predict')),truth_path=str(base/(c+'.truth.npz'))) for c in ('C0','P1')})

    def test_source_predictions_all_frozen_before_truth_and_target(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));jobs=module.commands(rt)
            self.assertEqual([j['name'] for j in jobs],['export','fit','calibrate','predict_source','score_source','C0_predict','C0_score','P1_predict','P1_score'])
            for job in jobs:
                if job['name'] not in ('score_source','C0_score','P1_score'):
                    self.assertNotIn('--truth',job['argv']);self.assertNotIn('--novel-truth',job['argv'])
                self.assertEqual(job['gpu'],job['name'] in ('export','C0_predict','P1_predict'))
            self.assertIn(rt['novel_truth'],jobs[4]['argv'])

    def test_existing_output_never_launches(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));root=Path(rt['run_root']);root.mkdir();(root/'keep').write_text('keep')
            with patch.object(module,'validate_runtime',return_value=rt),patch.object(module.subprocess,'Popen') as child:
                with self.assertRaises(FileExistsError):module.launch(rt,'runtime.json')
            child.assert_not_called();self.assertEqual((root/'keep').read_text(),'keep')

    def test_stage_failure_stops_before_truth_without_modifying_other_processes(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));root=Path(rt['run_root']);root.mkdir();(root/'configs').mkdir();(root/'logs').mkdir()
            calls=[]
            class Failed:
                pid=123
                def wait(self):return 1
            def child(argv,**kwargs):calls.append((argv,kwargs));return Failed()
            with patch.object(module,'validate_runtime',return_value=rt),patch.object(module,'choose_gpu',return_value=0),patch.object(module.subprocess,'Popen',side_effect=child):
                state=module.dispatch(rt)
            self.assertEqual(state['status'],'FAILED');self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][1]['env']['CUDA_VISIBLE_DEVICES'],'0')
            self.assertEqual(state['foreign_processes_stopped'],0)
            self.assertTrue((root/'pipeline_state.json').is_file())


if __name__=='__main__':unittest.main()
