from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from experiments.pamf_ssl_research.plan import SEEDS
from experiments.pamf_ssl_research.u_dualanchor import RUN,UGP_RUN,UCMR_RUN,second_slot_ready
from experiments.pamf_ssl_research.u_dualanchor_publish import prepare
from tests.test_pamf_ssl_paired_u_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_original_e30_three_seed_parents_both_readouts_and_separate_scorer():
    context=fixture();before=deepcopy(context);rt,blobs=prepare(context,'/p','/p/releases/'+RUN,'new')
    assert context==before and rt['methods']==['UDAP'] and len(rt['rows'])==3
    assert rt['max_compute_processes_per_gpu']==3
    assert rt['ugp_GPU_owner_state']=='/p/runs/'+UGP_RUN+'/state.json'
    assert rt['gpu_predecessor_state']==rt['prior_score_owner_state']=='/p/runs/'+UCMR_RUN+'/state.json'
    assert {row['gpu'] for row in rt['rows']}=={5,6,7}
    for row in rt['rows']:
        stem='configs/'+row['row_id'];t,p,s=(blobs[stem+'-'+key+'.json'] for key in ('train','predict','score'))
        assert t['algorithm_id']=='pamf_u_dualanchor_v1' and t['epochs']==50 and t['optimizer_updates']==11100
        assert '/'+UGP_RUN+'/' not in t['fork_parent_source'] and '/'+UCMR_RUN+'/' not in t['fork_parent_source']
        assert not any(t.get(k) for k in ('u_gradprotect','u_cmr','u_mixstyle','u_cric','paired_u','tests','truth','source_restorer'))
        assert 'truth' not in p['tests']['main'] and list(p['readouts'])==s['readouts']==['raw','DURA-LU']
        assert p['source_output']==t['output_root']


def test_second_slot_requires_ugp_gpu_complete_and_cmr_dispatched():
    a=dict(jobs={'UGP-s'+str(SEEDS[0]):dict(status='RUNNING')})
    b=dict(jobs={'UCMR-s'+str(SEEDS[0]):dict(status='RUNNING')})
    rt=dict(ugp_GPU_owner_state='a',gpu_predecessor_state='b')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda path:a if path=='a' else b):
        assert not second_slot_ready(rt,SEEDS[0])
        a['jobs']['UGP-s'+str(SEEDS[0])]['status']='PREDICTIONS_COMPLETE'
        assert second_slot_ready(rt,SEEDS[0])
        b['jobs']['UCMR-s'+str(SEEDS[0])]['status']='QUEUED'
        assert not second_slot_ready(rt,SEEDS[0]) and not second_slot_ready(rt,SEEDS[1])


def test_smoke_waits_for_vacated_gpu_slot_without_launching_queued_trainers():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt,_=prepare(fixture(),'/p',str(root),'new')
        rt.update(run_root=str(root/'new'),runtime_file=str(root/'runtime.json'))
        def smoke(command,**kwargs):
            assert 'experiments.pamf_ssl_research.u_dualanchor_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.u_dualanchor.second_slot_ready',side_effect=[False,True]), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=smoke), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            sleep.assert_called_once_with(20);launch.assert_not_called()


def test_full_dual_report_uses_udap_identity_and_raw_only_goal():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=['UDAP'])
        rt['rows']=[row for row in rt['rows'] if row['method']=='PSM']
        for row in rt['rows']:
            row.update(method='UDAP',row_id='UDAP-s'+str(row['model_seed']))
            config=json.loads(Path(row['score_config']).read_text());config.update(run_id=RUN,row_id=row['row_id'])
            save(Path(row['score_config']),config);path=Path(config['output_root'])/'scores.json'
            values=json.loads(path.read_text());values.update(run_id=RUN,row_id=row['row_id']);save(path,values)
        report(rt,root/'report');analysis=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in analysis['verdicts']}=={'UDAP'} and not analysis['goal_met']
        assert '| UDAP |' in (root/'report/report.md').read_text(encoding='utf-8')
