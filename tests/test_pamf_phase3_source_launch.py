import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pamf_phase3_open_world import source_launch


class SourceLaunchTests(unittest.TestCase):
    def test_runtime_rejects_target_or_checkpoint_inputs(self):
        with self.assertRaises(ValueError):source_launch.validate_runtime({'target_root':'query'})

    def test_training_freezes_before_source_rules_and_root_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'source-pilot'
            runtime=dict(run_id='source-pilot',run_root=str(root),release_root=folder,python='python',
                code_commit='a'*40,launch_owner='source-owner',allowed_gpus=[2],max_added_workers_per_gpu=2,
                minimum_free_mb=4096,train=dict(fold=dict(heldout_source_rx=8)))
            observed=[]
            class Child:
                pid=123
                def __init__(self,command,**kwargs):
                    if command[4].endswith('.source_train'):
                        observed.append('train');output=root/'train';output.mkdir()
                        (output/'complete.json').write_text(json.dumps(dict(status='SOURCE_DEVELOPMENT_FROZEN',updates=540,
                            train_count=3360,validation_count=27000,source_only=True,target_reads=0)),encoding='utf-8')
                    else:
                        if not (root/'train/complete.json').is_file():raise AssertionError('rules before model freeze')
                        observed.append('rules');output=root/'rules';output.mkdir()
                        (output/'complete.json').write_text(json.dumps(dict(status='SOURCE_RULES_EVALUATED',source_only=True,
                            target_reads=0,next_step='NO_PROMOTION',eligible_methods=[])),encoding='utf-8')
                def wait(self):return 0
            with patch.object(source_launch,'choose_gpu',return_value=2),patch.object(source_launch.subprocess,'Popen',Child):
                source_launch.dispatch(runtime)
            self.assertEqual(observed,['train','rules'])
            state=json.loads((root/'pipeline_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'],'ANALYZED_SOURCE_ONLY')
            self.assertEqual(state['next_step'],'NO_PROMOTION')
            with self.assertRaises(FileExistsError):source_launch.dispatch(runtime)


if __name__=='__main__':unittest.main()
