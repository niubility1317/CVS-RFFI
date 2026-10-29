import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pamf_phase3_open_world import star_neighborhood_launch as launch


class OwnedDevelopmentChain(unittest.TestCase):
    def runtime(self, root):
        return dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root='/release',
            python='/python',launch_owner='owned',allowed_gpus=list(range(8)),minimum_free_mb=4096,
            max_added_workers_per_gpu=2,source_config={},target_config={},U_truth='/truth/U.npz',
            Q_truth='/truth/Q.npz',targets_path='/release/targets.json')

    def invoke(self, root, job, gpus=(2,)):
        root.mkdir();(root/'logs').mkdir();config=root/'runtime.json'
        config.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
        with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch,'_gpu_free',return_value=list(gpus)),patch.object(launch,'_job',side_effect=job):
            launch.driver(config)
        return json.loads((root/'pipeline_state.json').read_text())

    def test_source_failure_never_opens_target_or_truth(self):
        calls=[]
        def job(r,state,path,name,command,gpu): calls.append(name);return False
        with tempfile.TemporaryDirectory() as d:state=self.invoke(Path(d)/'run',job)
        self.assertEqual(calls,['source']);self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_target_failure_never_scores_truth(self):
        calls=[]
        def job(r,state,path,name,command,gpu): calls.append(name);return name=='source'
        with tempfile.TemporaryDirectory() as d:state=self.invoke(Path(d)/'run',job)
        self.assertEqual(calls,['source','target']);self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_no_memory_preserves_without_starting_job(self):
        with tempfile.TemporaryDirectory() as d:
            state=self.invoke(Path(d)/'run',lambda *a:self.fail('No GPU child allowed'),gpus=())
        self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_existing_root_never_launches_another_driver(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=root/'runtime.json';config.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
            with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch.subprocess,'Popen') as popen:
                with self.assertRaises(FileExistsError):launch.start(config)
            popen.assert_not_called()


if __name__=='__main__':unittest.main()
