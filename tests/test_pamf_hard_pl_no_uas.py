from copy import deepcopy
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import pytest
from experiments.pamf_ssl_research.hard_pl_no_uas import RUN,configuration,validate,slot_ready,RESOURCE_GPU_BY_SEED
from experiments.pamf_ssl_research.hard_pl_no_uas_publish import prepare
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_one_variable_only_no_U_channel_same_parent_gate_weight_and_dose():
    from experiments.pamf_ssl_research.hard_pl import configuration as original,validate as validate_original
    context=fixture();base=context['2026092701']['F21']['train']
    a=original(base,'HPLT1U','/p','new');b=configuration(base,'/p','new')
    assert b['hard_pl']==a['hard_pl'] and b['hard_pl_no_uas'] is True
    assert b['fork_parent_source']==a['fork_parent_source'] and b['hard_pl']['weight']==2. and b['hard_pl']['threshold']==.95
    with pytest.raises(ValueError):validate_original(b)
    bad=deepcopy(b);bad['hard_pl_no_uas']=False
    with pytest.raises(ValueError):validate(bad)
    bad=deepcopy(b);bad['hard_pl']['As_KL']=1.
    with pytest.raises(ValueError):validate(bad)


def test_three_seed_matrix_auto_both_readouts_and_chained_scorer():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==['HPL_NO_UAS'] and len(rt['rows'])==3 and rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['gpu_predecessor_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert c['hard_pl_no_uas'] and row['gpu']==RESOURCE_GPU_BY_SEED[row['model_seed']]
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']
        assert c['output_root']==p['source_output'] and c['row_id']==p['source_row_id']


def test_waits_until_HPL2_and_UVB_GPU_done_and_HPLT1U_dispatched():
    seed=2026092701;u=dict(jobs={'UVB-s'+str(seed):dict(status='RUNNING')})
    h=dict(jobs={'HPL2-s'+str(seed):dict(status='RUNNING'),'HPLT1U-s'+str(seed):dict(status='QUEUED')})
    rt=dict(uvb_GPU_owner_state='u',gpu_predecessor_state='h')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:u if p=='u' else h):
        assert not slot_ready(rt,seed)
        h['jobs']['HPL2-s'+str(seed)]['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,seed)
        u['jobs']['UVB-s'+str(seed)]['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,seed)
        h['jobs']['HPLT1U-s'+str(seed)]['status']='RUNNING'
        assert slot_ready(rt,seed) and not slot_ready(rt,2026092702)


def test_source_smoke_waits_for_released_slot_and_dedicated_profile():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_no_uas_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_no_uas.slot_ready',side_effect=[False,True]), \
             patch('experiments.pamf_ssl_research.hard_pl_no_uas.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            sleep.assert_called_once_with(20);launch.assert_not_called()


def test_dispatched_predecessor_reserves_same_seed_slot_before_CUDA():
    from types import SimpleNamespace
    from experiments.pamf_ssl_research.control import pending_gpu_pids,capacity
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6},gpu_predecessor_state='h')
    jobs={'HPLT1U-s2026092701':dict(status='RUNNING',pid=99),
          'HPLT1U-s2026092702':dict(status='RUNNING',pid=100)}
    own=[dict(gpu=5,child=SimpleNamespace(pid=101)),dict(gpu=6,child=SimpleNamespace(pid=102))]
    with patch('experiments.cric_dura_comparison.io.read',return_value=dict(jobs=jobs)):
        assert pending_gpu_pids(rt,('HPL_NO_UAS',),2026092701)=={99}
        assert pending_gpu_pids(rt,('HPL_NO_UAS',),2026092701,own)=={99,101}
        # Two external CUDA processes plus the not-yet-CUDA predecessor fill the limit.
        with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['12\n13\n','19000']):
            assert capacity(5,pending_gpu_pids(rt,('HPL_NO_UAS',),2026092701))==(3,19000)
        jobs['HPLT1U-s2026092701']['status']='PREDICTIONS_COMPLETE'
        assert pending_gpu_pids(rt,('HPL_NO_UAS',),2026092701,own)=={101}


def test_full_report_has_original_primary_goal_and_profile_identity():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=['HPL_NO_UAS'])
        rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method='HPL_NO_UAS',row_id='HPL_NO_UAS-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';v=json.loads(path.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(path,v)
        report(rt,root/'report');a=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in a['verdicts']}=={'HPL_NO_UAS'} and not a['goal_met']
