from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.pamf_ssl_research.plan import SEEDS
from experiments.pamf_ssl_research.u_mixstyle import RUN,PAIR_RUN,UCRIC_RUN,second_slot_ready
from experiments.pamf_ssl_research.u_mixstyle_publish import prepare
from tests.test_pamf_ssl_paired_u_release import fixture
from tests.test_pamf_ssl_goal_counts import save


class UnlabeledMixStyleReleaseTest(unittest.TestCase):
    def test_three_source_only_rows_match_both_readouts_and_dura_is_eval_only(self):
        c=fixture();before=deepcopy(c);rt,blobs=prepare(c,'/p','/p/releases/'+RUN,'new')
        self.assertEqual(c,before);self.assertEqual(rt['methods'],['UMIX'])
        self.assertEqual(rt['max_compute_processes_per_gpu'],3)
        self.assertEqual(rt['paired_GPU_owner_state'],'/p/runs/'+PAIR_RUN+'/state.json')
        self.assertEqual(rt['gpu_predecessor_state'],'/p/runs/'+UCRIC_RUN+'/state.json')
        self.assertEqual(rt['prior_score_owner_state'],rt['gpu_predecessor_state'])
        for row in rt['rows']:
            stem='configs/'+row['row_id'];t,p,s=(blobs[stem+'-'+kind+'.json'] for kind in ('train','predict','score'))
            self.assertEqual(t['algorithm_id'],'pamf_u_mixstyle_v1')
            self.assertNotIn('u_cric',t);self.assertNotIn('paired_u',t)
            self.assertNotIn('tests',t);self.assertNotIn('source_restorer',t)
            self.assertNotIn('truth',p['tests']['main']);self.assertEqual(list(p['readouts']),['raw','DURA-LU'])
            self.assertEqual(s['readouts'],['raw','DURA-LU']);self.assertEqual(t['output_root'],p['source_output'])

    def test_second_slot_waits_for_complete_paired_gpu_and_ucric_dispatch(self):
        a=dict(jobs={'PAIREDU-s'+str(SEEDS[0]):dict(status='RUNNING')})
        b=dict(jobs={'UCRIC-s'+str(SEEDS[0]):dict(status='RUNNING')})
        rt=dict(paired_GPU_owner_state='a',gpu_predecessor_state='b')
        with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda path:a if path=='a' else b):
            self.assertFalse(second_slot_ready(rt,SEEDS[0]))
            a['jobs']['PAIREDU-s'+str(SEEDS[0])]['status']='PREDICTIONS_COMPLETE'
            self.assertTrue(second_slot_ready(rt,SEEDS[0]))
            b['jobs']['UCRIC-s'+str(SEEDS[0])]['status']='QUEUED'
            self.assertFalse(second_slot_ready(rt,SEEDS[0]))
            self.assertFalse(second_slot_ready(rt,SEEDS[1]))

    def test_smoke_cannot_dispatch_into_the_still_occupied_paired_slot(self):
        from experiments.pamf_ssl_research.control import run
        class ReachedSmoke(Exception):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rt,_=prepare(fixture(),'/p',str(root),'new')
            rt.update(run_root=str(root/'new'),runtime_file=str(root/'runtime.json'))
            def smoke(command,**kwargs):
                self.assertIn('experiments.pamf_ssl_research.u_mixstyle_worker',command)
                raise ReachedSmoke
            with patch('experiments.pamf_ssl_research.u_mixstyle.second_slot_ready',side_effect=[False,True]), \
                 patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
                 patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
                 patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=smoke), \
                 patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
                with self.assertRaises(ReachedSmoke):run(rt)
                sleep.assert_called_once_with(20);launch.assert_not_called()

    def test_complete_dual_report_cannot_silently_mislabel_umix(self):
        from experiments.pamf_ssl_research.dura_reference_report import report
        from tests.test_pamf_ssl_dura_reference import report_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=['UMIX'])
            rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
            for row in rt['rows']:
                row.update(method='UMIX',row_id='UMIX-s'+str(row['model_seed']))
                c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id'])
                save(Path(row['score_config']),c);p=Path(c['output_root'])/'scores.json'
                value=json.loads(p.read_text());value.update(run_id=RUN,row_id=row['row_id']);save(p,value)
            report(rt,root/'report');a=json.loads((root/'report/analysis.json').read_text())
            self.assertEqual({v['method'] for v in a['verdicts']},{'UMIX'});self.assertFalse(a['goal_met'])
            self.assertIn('| UMIX |',(root/'report/report.md').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
