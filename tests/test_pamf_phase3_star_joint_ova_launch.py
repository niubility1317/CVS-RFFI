import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pamf_phase3_open_world import star_joint_ova_launch as launch


class OwnedSourceRows(unittest.TestCase):
    def runtime(self,root):
        return dict(run_id=launch.RUN,code_commit='a'*40,run_root=str(root),release_root='/release',python='/python',
            launch_owner='owned',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
            max_concurrent_rows=2,teacher_config={},rows={a:{} for a in launch.ARMS},novel_truth='/separate/truth.npz')

    def invoke(self,root,job,mode='row',gpus=(2,)):
        root.mkdir();(root/'logs').mkdir();(root/'rows').mkdir()
        for arm in launch.ARMS:(root/'rows'/arm).mkdir()
        path=root/'runtime.json';path.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
        with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch,'_gpu_free',return_value=list(gpus)),patch.object(launch,'_job',side_effect=job):
            if mode=='row':launch.row_driver(path,'joint',2)
            else:launch.driver(path)
        target=root/'rows/joint/state.json' if mode=='row' else root/'pipeline_state.json'
        return json.loads(target.read_text())

    def test_teacher_failure_never_starts_training_rows(self):
        calls=[]
        def job(r,state,path,name,command,gpu):calls.append(name);return False
        with tempfile.TemporaryDirectory() as d,patch.object(launch.subprocess,'Popen') as spawn:
            state=self.invoke(Path(d)/'run',job,mode='driver')
        self.assertEqual(calls,['teacher_export']);spawn.assert_not_called();self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_train_failure_never_opens_validation_or_truth(self):
        calls=[]
        def job(r,state,path,name,command,gpu):calls.append(name);return False
        with tempfile.TemporaryDirectory() as d:state=self.invoke(Path(d)/'run',job)
        self.assertEqual(calls,['joint_train']);self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_eval_failure_never_scores_private_truth(self):
        calls=[]
        def job(r,state,path,name,command,gpu):calls.append(name);return name.endswith('_train')
        with tempfile.TemporaryDirectory() as d:state=self.invoke(Path(d)/'run',job)
        self.assertEqual(calls,['joint_train','joint_eval']);self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_no_memory_before_teacher_never_launches_job(self):
        with tempfile.TemporaryDirectory() as d:
            state=self.invoke(Path(d)/'run',lambda *a:self.fail('No GPU launch without actual memory'),mode='driver',gpus=())
        self.assertEqual(state['status'],'FAILED_PRESERVED')

    def test_successful_scorer_arm_fields_finish_row(self):
        from experiments.pamf_phase3_open_world.star_joint_ova_score import STATUS
        calls=[]
        def job(r,state,path,name,command,gpu):
            calls.append(name)
            if name.endswith('_score'):
                output=Path(r['run_root'])/'rows/joint/score';output.mkdir()
                (output/'complete.json').write_text(json.dumps(dict(status=STATUS,run_id=launch.RUN,arm='joint',
                    arm_source_gate_passed=True,eligible_primary=True,next_step='SOURCE_JOINT_OVA_READY_FOR_TARGET_FALSIFICATION')),encoding='utf-8')
            return True
        with tempfile.TemporaryDirectory() as d:state=self.invoke(Path(d)/'run',job)
        self.assertEqual(calls,['joint_train','joint_eval','joint_score'])
        self.assertEqual(state['status'],'ANALYZED_SOURCE_ONLY');self.assertTrue(state['source_gate_passed'])

    def test_existing_root_is_never_relaunched(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime=root/'runtime.json';runtime.write_text(json.dumps(self.runtime(root)),encoding='utf-8')
            with patch.object(launch,'validate',side_effect=lambda r:r),patch.object(launch.subprocess,'Popen') as spawn:
                with self.assertRaises(FileExistsError):launch.start(runtime)
            spawn.assert_not_called()


if __name__=='__main__':unittest.main()
