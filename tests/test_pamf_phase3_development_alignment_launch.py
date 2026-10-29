import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class AlignmentLaunchTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('experiments.pamf_phase3_open_world.development_alignment_launch')

    def runtime(self, base):
        run='20261029-phase3-development-alignment-s2026092701-r01'
        return dict(run_id=run,run_root=str(base/run),release_root=str(base/'release'),
            python='python',code_commit='a'*40,launch_owner='phase3-owner',allowed_gpus=[0,1],
            max_added_workers_per_gpu=2,minimum_free_mb=4096,
            targets_path=str(base/'release/configs/targets.json'),
            rows={condition:dict(predict_config={'condition':condition},truth_path=str(base/(condition+'.truth.npz')))
                  for condition in ('C0','P1')})

    def test_existing_run_never_launches_or_overwrites(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));root=Path(rt['run_root']);root.mkdir()
            marker=root/'existing';marker.write_text('preserve')
            with patch.object(module,'validate_runtime',return_value=rt),patch.object(module.subprocess,'Popen') as popen:
                with self.assertRaises(FileExistsError):module.launch(rt,'runtime.json')
            popen.assert_not_called();self.assertEqual(marker.read_text(),'preserve')

    def test_commands_keep_private_truth_out_of_predict_config_and_cli(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));root=Path(rt['run_root'])
            commands=module.commands(rt,'C0',root/'configs/C0.predict.json')
            self.assertNotIn('--truth',commands['predict'])
            self.assertIn('--truth',commands['score'])
            self.assertIn(rt['rows']['C0']['truth_path'],commands['score'])
            self.assertIn('--targets',commands['score'])

    def test_real_child_failure_preserves_output_and_does_not_score(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temp:
            rt=self.runtime(Path(temp));root=Path(rt['run_root']);root.mkdir();(root/'configs').mkdir();(root/'rows').mkdir()
            calls=[]
            class Failed:
                pid=12345
                def wait(self):return 2
            def popen(argv,**kwargs):
                calls.append((argv,kwargs));return Failed()
            with patch.object(module,'validate_runtime',return_value=rt),patch.object(module,'choose_gpu',return_value=1),patch.object(module.subprocess,'Popen',side_effect=popen):
                state=module.dispatch(rt)
            self.assertEqual(state['status'],'FAILED')
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][1]['env']['CUDA_VISIBLE_DEVICES'],'1')
            self.assertEqual(state['foreign_processes_stopped'],0)
            self.assertTrue((root/'pipeline_state.json').is_file())
            self.assertFalse((root/'rows/P1').exists())


if __name__=='__main__':unittest.main()
