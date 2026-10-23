from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pamf_ssl_research.plan import configuration,SEEDS,VIEWS
from experiments.pamf_ssl_research.paired_u import RUN,REFERENCE_RUN,reference_gpu_work_finished
from experiments.pamf_ssl_research.paired_u_publish import prepare
from experiments.pamf_ssl_research.dura_reference import DURA_RELATIVE
from tests.test_pamf_ssl_research import template
from tests.test_pamf_ssl_goal_counts import save


def fixture():
    c={'_rxpsm_parents':{},'_parent_timing':{}}
    g=dict(kind='original_F0_LU',roles='LU',checkpoint='/p/runs/'+DURA_RELATIVE,source_contract='/p/source_contract.json')
    for seed in SEEDS:
        base=dict(template(),source_contract='/p/source_contract.json',frozen_sources={'F':'/f'},
                  output_root='/p/baseline/'+str(seed)+'/source')
        score=dict(output_root='/p/baseline/'+str(seed),tests={'main':{'truth':'/private/truth'}})
        prediction=dict(tests={'main':{'truth':'/private/truth','iq':'/target/iq'}},views=list(VIEWS),
                        readouts={'raw':{},'DURA-LU':deepcopy(g)})
        c[str(seed)]={arm:dict(train=base,score=score,predict=prediction) for arm in ('F20','F21')}
        c['_rxpsm_parents'][str(seed)]=configuration(base,'RXPSM',seed,'/p','old')
        c['_parent_timing'][str(seed)]=5000.
    return c


class PairedUReleaseTest(unittest.TestCase):
    def test_actual_three_seed_release_has_both_readouts_and_no_raw_reuse(self):
        original=fixture()
        before=deepcopy(original)
        rt,blobs=prepare(original,'/p','/p/releases/'+RUN,'new')
        self.assertEqual(original,before)
        self.assertEqual(rt['methods'],['PAIREDU'])
        self.assertEqual(rt['allowed_gpus'],[5,6,7])
        self.assertEqual(len(rt['rows']),3)
        self.assertEqual(rt['gpu_predecessor_state'],'/p/runs/'+REFERENCE_RUN+'/state.json')
        self.assertNotIn('replaces_queued_runtime',rt)
        for row in rt['rows']:
            stem='configs/'+row['row_id']
            t,p,s=(blobs[stem+'-'+kind+'.json'] for kind in ('train','predict','score'))
            self.assertEqual(t['algorithm_id'],'pamf_pairedu_v1')
            self.assertEqual(t['optimizer_updates'],11100)
            self.assertNotIn('view_anchor',t)
            self.assertNotIn('tests',t)
            self.assertNotIn('truth',p['tests']['main'])
            self.assertNotIn('raw_predictions_from',p)
            self.assertEqual(list(p['readouts']),['raw','DURA-LU'])
            self.assertEqual(s['readouts'],['raw','DURA-LU'])
            self.assertEqual(p['source_output'],t['output_root'])
            self.assertEqual(s['prediction_output'],p['output_root'])

    def test_card_handoff_waits_until_all_three_previous_gpu_rows_finish(self):
        state=dict(jobs={m+'-s'+str(SEEDS[0]):dict(status='PREDICTIONS_COMPLETE')
                         for m in ('PSM','RXPSM','VIEWANCHOR')})
        with patch('experiments.cric_dura_comparison.io.read',return_value=state):
            self.assertTrue(reference_gpu_work_finished(dict(gpu_predecessor_state='state'),SEEDS[0]))
            state['jobs']['VIEWANCHOR-s'+str(SEEDS[0])]['status']='RUNNING'
            self.assertFalse(reference_gpu_work_finished(dict(gpu_predecessor_state='state'),SEEDS[0]))
            state['jobs'].pop('VIEWANCHOR-s'+str(SEEDS[0]))
            self.assertFalse(reference_gpu_work_finished(dict(gpu_predecessor_state='state'),SEEDS[0]))

    def test_controller_waits_for_predecessor_before_smoke_or_training(self):
        from experiments.pamf_ssl_research.control import run
        class ReachedSmoke(Exception):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rt,_=prepare(fixture(),'/p',str(root),'new')
            rt.update(run_root=str(root/'new'),runtime_file=str(root/'runtime.json'))
            def smoke(command,**kwargs):
                self.assertIn('experiments.pamf_ssl_research.paired_u_worker',command)
                raise ReachedSmoke
            with patch('experiments.pamf_ssl_research.paired_u.reference_gpu_work_finished',side_effect=[False,True]), \
                 patch('experiments.pamf_ssl_research.control.capacity',return_value=(0,20000)), \
                 patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
                 patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=smoke), \
                 patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
                with self.assertRaises(ReachedSmoke):run(rt)
                sleep.assert_called_once_with(20)
                launch.assert_not_called()

    def test_dual_report_preserves_real_paired_method_name(self):
        from experiments.pamf_ssl_research.dura_reference_report import report
        from tests.test_pamf_ssl_dura_reference import report_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rt=report_fixture(root)
            rt['run_id']=RUN
            rt['methods']=['PAIREDU']
            rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
            for row in rt['rows']:
                row['method']='PAIREDU'
                row['row_id']='PAIREDU-s'+str(row['model_seed'])
                config=json.loads(Path(row['score_config']).read_text())
                config.update(run_id=RUN,row_id=row['row_id'])
                save(Path(row['score_config']),config)
                p=Path(config['output_root'])/'scores.json'
                value=json.loads(p.read_text())
                value.update(run_id=RUN,row_id=row['row_id'])
                save(p,value)
            report(rt,root/'report')
            a=json.loads((root/'report/analysis.json').read_text())
            self.assertEqual({v['method'] for v in a['verdicts']},{'PAIREDU'})
            self.assertEqual({v['readout'] for v in a['summary']},{'raw','DURA-LU'})
            self.assertFalse(a['goal_met'])
            self.assertIn('| PAIREDU |',(root/'report/report.md').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
