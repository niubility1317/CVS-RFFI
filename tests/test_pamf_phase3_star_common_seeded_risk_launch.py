import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
from tests import test_pamf_phase3_star_oe_risk_run as fixtures


class CommonSeededRiskLaunchTests(unittest.TestCase):
    def fixture(self,root):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_common_seeded_risk_launch')
        c,*_=fixtures.OERiskRunTests().fixture(root);run=root/'run'
        c['run_id']=m.RUN;c['OE_prediction_root']=str(root/'oe_predictions');del c['OE_root']
        r=dict(run_id=m.RUN,code_commit='b'*40,run_root=str(run),release_root=str(root/'release'),python='python',
            launch_owner='test-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
            max_concurrent_rows=2,rows={},novel_truth='private.npz')
        for arm in m.ARMS:
            cfg=deepcopy(c);cfg['arm']=arm;cfg['parent_eval_config']['arm']=arm;cfg['parent_eval_config']['train_config']['arm']=arm
            cfg['output_root']=str(run/'rows'/arm/'predict');r['rows'][arm]={'source_config':cfg}
        path=root/'runtime.json';path.write_text(json.dumps(r));return m,r,path

    def test_runtime_rejects_extra_target_wrong_root_and_capacity(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,_=self.fixture(Path(td));self.assertIs(m.validate(r),r)
            for defect in ('target','root','capacity'):
                with self.subTest(defect=defect):
                    bad=deepcopy(r)
                    if defect=='target':bad['rows']['joint']['source_config']['target_query']='forbidden'
                    elif defect=='root':bad['rows']['joint']['source_config']['output_root']='other-owner'
                    else:bad['max_added_workers_per_gpu']=3
                    with self.assertRaises(ValueError):m.validate(bad)

    def test_source_immediately_scores_on_cpu_with_correct_terminal_keys(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'rows/joint').mkdir(parents=True);commands=[]
            def job(rt,state,sp,name,cmd,device):
                commands.append((name,cmd,device))
                if name.endswith('_score'):
                    out=root/'rows/joint/score';out.mkdir();(out/'complete.json').write_text(json.dumps(dict(
                        status='STAR_COMMON_SEEDED_RISK_SCORES_COMPLETE',run_id=m.RUN,arm='joint',arm_source_gate_passed=False,
                        eligible_primary=False,next_step='NO_PROMOTION')))
                return True
            with patch.object(m,'_job',side_effect=job),patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,'joint',2)
            self.assertEqual([v[0] for v in commands],['joint_source','joint_score']);self.assertIsNone(commands[1][2])
            self.assertIn('experiments.pamf_phase3_open_world.star_common_seeded_risk_run',commands[0][1]);self.assertNotIn('run',commands[0][1])
            row=json.loads((root/'rows/joint/state.json').read_text());self.assertEqual(row['status'],'ANALYZED_SOURCE_ONLY')
            self.assertFalse(row['source_gate_passed'])

    def test_source_failure_preserves_partial_without_score_or_restart(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'rows/joint').mkdir(parents=True)
            with patch.object(m,'_job',return_value=False) as job,patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,'joint',2)
            self.assertEqual(job.call_count,1);self.assertEqual(json.loads((root/'rows/joint/state.json').read_text())['status'],'FAILED_PRESERVED')
            with self.assertRaises(FileExistsError):m.start(path)


if __name__=='__main__':unittest.main()
