import importlib
import unittest
from pathlib import Path
from unittest.mock import patch


class LaunchTests(unittest.TestCase):
    def test_sequential_five_jobs_truth_only_scorer(self):
        m=importlib.import_module('experiments.pamf_phase3_open_world.unlabeled_registration_launch')
        root=Path.cwd()/'registration-test'/m.RUN
        rt=dict(run_root=str(root),python='python',targets_path='/targets.json',rows={c:dict(U_truth='/private/'+c+'/U.npz',Q_truth='/private/'+c+'/Q.npz') for c in ('C0','P1')})
        jobs=m.commands(rt)
        self.assertEqual([j['name'] for j in jobs],['source','C0_predict','C0_score','P1_predict','P1_score'])
        self.assertEqual([j['gpu'] for j in jobs],[True,True,False,True,False])
        for j in jobs:
            with self.subTest(job=j['name']):
                if not j['name'].endswith('_score'):
                    self.assertFalse(any('private' in a or 'truth' in a.lower() for a in j['argv']))
                else:
                    self.assertIn('--U-truth',j['argv']);self.assertIn('--Q-truth',j['argv'])

    def test_dispatch_stops_on_technical_failure(self):
        m=importlib.import_module('experiments.pamf_phase3_open_world.unlabeled_registration_launch')
        import tempfile,json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'configs').mkdir();(root/'logs').mkdir()
            rt=dict(run_root=d,run_id=m.RUN,code_commit='a'*40,release_root=d,launch_owner='test',source_config={},rows={c:dict(predict_config={}) for c in ('C0','P1')})
            jobs=[dict(name='source',gpu=True),dict(name='C0_predict',gpu=True)]
            with patch.object(m,'validate_runtime'),patch.object(m,'commands',return_value=jobs),patch.object(m,'choose_gpu',return_value=0),patch.object(m,'_run_job',side_effect=RuntimeError('source failed')) as run:
                result=m.dispatch(rt)
            self.assertEqual(result['status'],'FAILED');self.assertEqual(run.call_count,1)
            self.assertEqual(json.loads((root/'pipeline_state.json').read_text())['foreign_processes_stopped'],0)


if __name__=='__main__':unittest.main()
