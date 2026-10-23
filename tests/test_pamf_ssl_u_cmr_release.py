from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from experiments.pamf_ssl_research.plan import SEEDS
from experiments.pamf_ssl_research.u_cmr import RUN,UMIX_RUN,UGP_RUN,second_slot_ready
from experiments.pamf_ssl_research.u_cmr_publish import prepare
from tests.test_pamf_ssl_paired_u_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_source_only_three_seed_rows_both_readouts_and_separate_scorer_owner():
    c=fixture();before=deepcopy(c);rt,blobs=prepare(c,'/p','/p/releases/'+RUN,'new')
    assert c==before and rt['methods']==['UCMR'] and len(rt['rows'])==3
    assert rt['max_compute_processes_per_gpu']==3
    assert rt['umix_GPU_owner_state']=='/p/runs/'+UMIX_RUN+'/state.json'
    assert rt['gpu_predecessor_state']==rt['prior_score_owner_state']=='/p/runs/'+UGP_RUN+'/state.json'
    for row in rt['rows']:
        stem='configs/'+row['row_id'];t,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert t['algorithm_id']=='pamf_u_cmr_v1'
        assert not any(k in t for k in ('u_gradprotect','u_mixstyle','u_cric','paired_u','tests','truth','source_restorer'))
        assert 'truth' not in p['tests']['main'] and list(p['readouts'])==['raw','DURA-LU']
        assert s['readouts']==['raw','DURA-LU'] and p['source_output']==t['output_root']


def test_second_slot_requires_umix_gpu_complete_and_ugp_dispatched():
    a=dict(jobs={'UMIX-s'+str(SEEDS[0]):dict(status='RUNNING')})
    b=dict(jobs={'UGP-s'+str(SEEDS[0]):dict(status='RUNNING')})
    rt=dict(umix_GPU_owner_state='a',gpu_predecessor_state='b')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:a if p=='a' else b):
        assert not second_slot_ready(rt,SEEDS[0])
        a['jobs']['UMIX-s'+str(SEEDS[0])]['status']='PREDICTIONS_COMPLETE'
        assert second_slot_ready(rt,SEEDS[0])
        b['jobs']['UGP-s'+str(SEEDS[0])]['status']='QUEUED'
        assert not second_slot_ready(rt,SEEDS[0])
        assert not second_slot_ready(rt,SEEDS[1])


def test_source_smoke_waits_for_the_vacated_umix_slot():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt,_=prepare(fixture(),'/p',str(root),'new')
        rt.update(run_root=str(root/'new'),runtime_file=str(root/'runtime.json'))
        def smoke(command,**kwargs):
            assert 'experiments.pamf_ssl_research.u_cmr_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.u_cmr.second_slot_ready',side_effect=[False,True]), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=smoke), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            sleep.assert_called_once_with(20);launch.assert_not_called()


def test_complete_dual_report_uses_actual_ucmr_identity():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=['UCMR'])
        rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method='UCMR',row_id='UCMR-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id'])
            save(Path(row['score_config']),c);p=Path(c['output_root'])/'scores.json'
            v=json.loads(p.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(p,v)
        report(rt,root/'report');a=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in a['verdicts']}=={'UCMR'} and not a['goal_met']
        assert '| UCMR |' in (root/'report/report.md').read_text(encoding='utf-8')
