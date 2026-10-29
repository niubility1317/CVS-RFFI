import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pamf_phase3_open_world import star_representation_launch as launch


class RepresentationLaunchSafety(unittest.TestCase):
    def test_existing_output_preserved_without_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'owned';root.mkdir();(root/'old').write_text('preserved')
            runtime=Path(directory)/'runtime.json';runtime.write_text(json.dumps({'run_root':str(root)}))
            with patch.object(launch,'validate',side_effect=lambda x:x),patch.object(launch.subprocess,'Popen') as spawn:
                with self.assertRaises(FileExistsError):launch.start(runtime)
                spawn.assert_not_called()
            self.assertEqual((root/'old').read_text(),'preserved')

    def test_cache_failure_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);r=dict(run_root=str(root),release_root=str(root),python='python',code_commit='a'*40)
            runtime=root/'runtime.json';runtime.write_text(json.dumps(r))
            with patch.object(launch,'validate',side_effect=lambda x:x),patch.object(launch,'_gpu_free',return_value=[0]),patch.object(launch,'_job',return_value=False),patch.object(launch.subprocess,'Popen') as spawn:
                launch.driver(runtime);spawn.assert_not_called()
            state=json.loads((root/'pipeline_state.json').read_text())
            self.assertEqual(state['status'],'FAILED_PRESERVED')
            self.assertFalse((root/'selection.json').exists())

    def test_control_success_cannot_replace_failed_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir();(root/'rows').mkdir()
            for arm in launch.ARMS:(root/'rows'/arm).mkdir()
            r=dict(run_root=str(root),release_root=str(root),python='python',code_commit='a'*40,max_concurrent_rows=3)
            runtime=root/'runtime.json';runtime.write_text(json.dumps(r))
            class Finished:
                pid=12345
                def __init__(self,command,**kwargs):
                    arm=command[command.index('--arm')+1];folder=root/'rows'/arm
                    (folder/'state.json').write_text(json.dumps({'status':'ANALYZED_SOURCE_ONLY'}))
                    (folder/'score').mkdir()
                    (folder/'score/complete.json').write_text(json.dumps({'arm_source_gate_passed':arm!='oe_vic'}))
                def poll(self):return 0
            with patch.object(launch,'validate',side_effect=lambda x:x),patch.object(launch,'_gpu_free',return_value=[0,1,2]),patch.object(launch,'_job',return_value=True),patch.object(launch.subprocess,'Popen',side_effect=Finished):
                launch.driver(runtime)
            selection=json.loads((root/'selection.json').read_text())
            self.assertTrue(selection['all_rows_complete'])
            self.assertFalse(selection['source_gate_passed'])
            self.assertEqual(selection['eligible_candidates'],[])

    def test_completed_primary_success_is_not_lost_by_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir();(root/'rows').mkdir()
            for arm in launch.ARMS:(root/'rows'/arm).mkdir()
            runtime=root/'runtime.json';runtime.write_text(json.dumps(dict(run_root=str(root),release_root=str(root),python='python',code_commit='a'*40,max_concurrent_rows=3)))
            class Finished:
                pid=12346
                def __init__(self,command,**kwargs):
                    arm=command[command.index('--arm')+1];folder=root/'rows'/arm
                    (folder/'state.json').write_text(json.dumps({'status':'ANALYZED_SOURCE_ONLY'}))
                    (folder/'score').mkdir()
                    (folder/'score/complete.json').write_text(json.dumps({'arm_source_gate_passed':arm=='oe_vic'}))
                def poll(self):return 0
            with patch.object(launch,'validate',side_effect=lambda x:x),patch.object(launch,'_gpu_free',return_value=[0,1,2]),patch.object(launch,'_job',return_value=True),patch.object(launch.subprocess,'Popen',side_effect=Finished):
                launch.driver(runtime)
            selection=json.loads((root/'selection.json').read_text())
            self.assertTrue(selection['source_gate_passed'])
            self.assertEqual(selection['eligible_candidates'],['oe_vic'])


if __name__=='__main__':unittest.main()
