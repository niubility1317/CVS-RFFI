import importlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

class LaunchTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_relative_launch')
    def test_source_then_each_four_state_and_truth_only_scorer(self):
        m=self.module();root=Path.cwd()/m.RUN
        rt=dict(run_root=str(root),python='python',targets_path='/targets.json',rows={p:dict(U_truth='/private/'+p+'/U.npz',Q_truth='/private/'+p+'/Q.npz') for p in ('P0','P1')})
        jobs=m.commands(rt);self.assertEqual([x['name'] for x in jobs],['source','P0_predict','P0_score','P1_predict','P1_score'])
        self.assertEqual([x['gpu'] for x in jobs],[True,True,False,True,False]);self.assertIn('run',jobs[0]['argv'])
        for x in jobs:
            with self.subTest(job=x['name']):
                if x['name'].endswith('score'):self.assertIn('--U-truth',x['argv']);self.assertIn('--Q-truth',x['argv'])
                else:self.assertFalse(any('private' in a or 'truth' in a.lower() for a in x['argv']))
    def test_technical_failure_does_not_launch_next_job(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'configs').mkdir();(r/'logs').mkdir()
            rt=dict(run_root=d,run_id=m.RUN,code_commit='a'*40,launch_owner='test',source_config={},rows={p:dict(predict_config={}) for p in ('P0','P1')})
            with patch.object(m,'validate_runtime'),patch.object(m,'commands',return_value=[dict(name='source',gpu=True),dict(name='P0_predict',gpu=True)]),\
                 patch.object(m,'choose_gpu',return_value=0),patch.object(m,'_run_job',side_effect=RuntimeError('technical')) as job:
                state=m.dispatch(rt)
            self.assertEqual(state['status'],'FAILED');self.assertEqual(job.call_count,1);self.assertEqual(state['foreign_processes_stopped'],0)

if __name__=='__main__':unittest.main()
