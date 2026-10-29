import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
from tests import test_pamf_phase3_star_oe_risk_run as fixtures


class OERiskLaunchTests(unittest.TestCase):
    def fixture(self,root):
        m=importlib.import_module('experiments.pamf_phase3_open_world.star_oe_risk_launch')
        c,*_=fixtures.OERiskRunTests().fixture(root);run=root/'run'
        r=dict(run_id=m.RUN,code_commit='b'*40,run_root=str(run),release_root=str(root/'release'),python='python',
            launch_owner='test-owner',allowed_gpus=list(range(8)),minimum_free_mb=4096,max_added_workers_per_gpu=2,
            max_concurrent_rows=2,rows={},novel_truth='private.npz')
        for arm in m.ARMS:
            cfg=deepcopy(c);cfg['arm']=arm;cfg['parent_eval_config']['arm']=arm;cfg['parent_eval_config']['train_config']['arm']=arm
            cfg['output_root']=str(run/'rows'/arm/'predict');r['rows'][arm]={'source_config':cfg}
        path=root/'runtime.json';path.write_text(json.dumps(r));return m,r,path

    def test_runtime_cannot_add_target_or_change_owned_root_or_foreign_policy(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,_=self.fixture(Path(td));self.assertIs(m.validate(r),r)
            for defect in ('truth','root','capacity'):
                with self.subTest(defect=defect):
                    bad=deepcopy(r)
                    if defect=='truth':bad['rows']['joint']['source_config']['target_query']='forbidden'
                    elif defect=='root':bad['rows']['joint']['source_config']['output_root']='other-owner'
                    else:bad['max_added_workers_per_gpu']=3
                    with self.assertRaises(ValueError):m.validate(bad)

    def test_source_then_cpu_score_uses_real_terminal_fields_and_no_extra_cli_mode(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'rows/joint').mkdir(parents=True)
            commands=[]
            def job(rt,state,sp,name,cmd,device):
                commands.append((name,cmd,device))
                if name.endswith('_score'):
                    out=root/'rows/joint/score';out.mkdir();(out/'complete.json').write_text(json.dumps(dict(
                        status='STAR_OE_RISK_SCORES_COMPLETE',run_id=m.RUN,arm='joint',arm_source_gate_passed=False,
                        eligible_primary=False,next_step='NO_PROMOTION')))
                return True
            with patch.object(m,'_job',side_effect=job),patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,'joint',2)
            self.assertEqual([v[0] for v in commands],['joint_source','joint_score']);self.assertEqual(commands[1][2],None)
            self.assertIn('experiments.pamf_phase3_open_world.star_oe_risk_run',commands[0][1])
            self.assertNotIn('run',commands[0][1]);state=json.loads((root/'rows/joint/state.json').read_text())
            self.assertEqual(state['status'],'ANALYZED_SOURCE_ONLY');self.assertFalse(state['source_gate_passed'])

    def test_failed_source_preserves_row_and_does_not_score_or_restart(self):
        with tempfile.TemporaryDirectory() as td:
            m,r,path=self.fixture(Path(td));root=Path(r['run_root']);(root/'rows/joint').mkdir(parents=True)
            with patch.object(m,'_job',return_value=False) as job,patch.object(m,'_gpu_free',return_value=[2]):m.row_driver(path,'joint',2)
            self.assertEqual(job.call_count,1);self.assertEqual(json.loads((root/'rows/joint/state.json').read_text())['status'],'FAILED_PRESERVED')
            with self.assertRaises(FileExistsError):m.start(path)


if __name__=='__main__':unittest.main()
