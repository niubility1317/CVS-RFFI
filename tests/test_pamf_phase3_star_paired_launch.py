import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pamf_phase3_open_world import star_paired_launch as launch


class OwnedSourceChain(unittest.TestCase):
    def runtime(self, root):
        return dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root='/release',
                    python='/python',launch_owner='owned',allowed_gpus=list(range(8)),minimum_free_mb=4096,
                    max_added_workers_per_gpu=2,source_config={},novel_truth='/separate/source_truth.npz')

    def invoke(self, root, job, gpus=(2,)):
        root.mkdir();(root/'logs').mkdir();config=root/'runtime.json'
        config.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
        with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch,'_gpu_free',return_value=list(gpus)),patch.object(launch,'_job',side_effect=job):
            launch.driver(config)
        return json.loads((root/'pipeline_state.json').read_text())

    def test_prediction_failure_never_runs_truth_score(self):
        calls=[]
        def job(r,state,path,name,command,gpu):
            calls.append(name);return False
        with tempfile.TemporaryDirectory() as directory:
            state=self.invoke(Path(directory)/'run',job)
        self.assertEqual(calls,['source_predict'])
        self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_no_memory_preserves_without_starting_job(self):
        with tempfile.TemporaryDirectory() as directory:
            state=self.invoke(Path(directory)/'run',lambda *args:self.fail('No GPU child allowed'),gpus=())
        self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_existing_root_never_launches_another_driver(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'runtime.json'
            config.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
            with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch.subprocess,'Popen') as popen:
                with self.assertRaises(FileExistsError):launch.start(config)
            popen.assert_not_called()

    def test_wrong_terminal_identity_cannot_become_analyzed(self):
        from experiments.pamf_phase3_open_world import star_paired_run as predictor,star_paired_score as scorer
        calls=[]
        def job(r,state,path,name,command,gpu):
            calls.append((name,gpu))
            output=Path(r['run_root'])/('prediction' if name=='source_predict' else 'score')
            output.mkdir()
            marker=dict(run_id=launch.RUN,code_commit='b'*40,status=predictor.STATUS if name=='source_predict' else scorer.STATUS,
                        source_gate_passed=False,next_step='NO_PROMOTION')
            (output/'complete.json').write_text(json.dumps(marker),encoding='utf-8');return True
        with tempfile.TemporaryDirectory() as directory:
            state=self.invoke(Path(directory)/'run',job)
        self.assertEqual(calls,[('source_predict',2),('source_score',None)])
        self.assertEqual(state['status'],'FAILED_PRESERVED')


if __name__=='__main__':unittest.main()
