import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pamf_phase3_open_world import branch_launch


class BranchLaunchTests(unittest.TestCase):
    def test_runtime_has_no_target_or_training_inputs(self):
        with self.assertRaises(ValueError):branch_launch.validate_runtime({'target_root':'bad'})

    def test_export_completes_before_rules_and_new_root_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'branch-pilot'
            rt=dict(run_id='branch-pilot',run_root=str(root),release_root=folder,python='python',code_commit='a'*40,
                launch_owner='owner',allowed_gpus=[2],max_added_workers_per_gpu=2,minimum_free_mb=4096,
                export=dict(fold=dict(heldout_source_rx=8)))
            seen=[]
            class Child:
                pid=123
                def __init__(self,command,**kwargs):
                    if command[4].endswith('.branch_export'):
                        seen.append('export');output=root/'export';output.mkdir()
                        done=dict(status='SOURCE_BRANCHES_EXPORTED',source_only=True,target_reads=0,optimizer_updates=0,
                            head_changed=False,train_count=3360,validation_count=27000,parent_epochs=20,parent_updates=540)
                    else:
                        if not (root/'export/complete.json').is_file():raise AssertionError('early rules')
                        seen.append('rules');output=root/'rules';output.mkdir()
                        done=dict(status='SOURCE_BRANCH_RULES_EVALUATED',source_only=True,target_reads=0,
                            next_step='NO_PROMOTION',eligible_methods=[])
                    (output/'complete.json').write_text(json.dumps(done),encoding='utf-8')
                def wait(self):return 0
            with patch.object(branch_launch,'choose_gpu',return_value=2),patch.object(branch_launch.subprocess,'Popen',Child):
                branch_launch.dispatch(rt)
            self.assertEqual(seen,['export','rules'])
            self.assertEqual(json.loads((root/'pipeline_state.json').read_text())['status'],'ANALYZED_SOURCE_ONLY')
            with self.assertRaises(FileExistsError):branch_launch.dispatch(rt)


if __name__=='__main__':unittest.main()
