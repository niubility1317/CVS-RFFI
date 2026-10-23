from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from experiments.pamf_ssl_research.hard_pl import RUN,PARENT_RUN,METHODS,SEEDS,RESOURCE_GPU_BY_SEED,slot_ready,validate
from experiments.pamf_ssl_research.hard_pl_publish import prepare
from tests.test_pamf_ssl_paired_u_release import fixture as scaffold_fixture
from tests.test_pamf_ssl_goal_counts import save


def fixture(project='/p'):
    from experiments.pamf_ssl_research.dura_reference import DURA_RELATIVE
    context=scaffold_fixture()
    for seed in SEEDS:
        t=context[str(seed)]['F21']['train']
        t.update(run_id=PARENT_RUN,row_id='F21-s'+str(seed),model_seed=seed,ablation='F21',
            regime='native_cric',satellite_concat=False,trajectory=[20,10,20],optimizer_updates=11100,
            output_root=project+'/runs/'+PARENT_RUN+'/F21-s'+str(seed)+'/source')
        for arm in ('F20','F21'):
            context[str(seed)][arm]['predict']['readouts']['DURA-LU']['checkpoint']=project+'/runs/'+DURA_RELATIVE
    return context


def test_two_methods_three_seeds_f21_parents_dual_readouts_and_fixed_G():
    c=fixture();before=deepcopy(c);rt,blobs=prepare(c,'/p','/p/releases/'+RUN,'new')
    assert c==before and rt['methods']==list(METHODS) and len(rt['rows'])==6
    assert rt['max_per_gpu']==2 and rt['max_total_workers']==6 and rt['max_compute_processes_per_gpu']==3
    assert rt['gpu_predecessor_state']==rt['prior_score_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];t,p,s=(blobs[stem+'-'+key+'.json'] for key in ('train','predict','score'))
        assert t['hard_pl']['threshold']==.95 and t['hard_pl']['weight']==2. and not t['own_EMA']
        assert t['epochs']==50 and t['optimizer_updates']==11100 and t['posthoc_head_fit'] is False
        assert '/'+PARENT_RUN+'/F21-s'+str(row['model_seed'])+'/source' in t['fork_parent_source']
        assert not any(t.get(k) for k in ('ssl','tests','truth','u_viewbalance','u_dualanchor'))
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']
        assert p['source_output']==t['output_root'] and row['gpu']==RESOURCE_GPU_BY_SEED[row['model_seed']]
        for key,value in (('target_truth','x'),('plain_PL_threshold',.98),('own_EMA',True)):
            bad=deepcopy(t);bad[key]=value
            if key=='plain_PL_threshold':bad['hard_pl']['threshold']=.98
            with pytest.raises(ValueError):validate(bad)


def test_successor_cannot_create_a_third_own_worker_even_on_card_without_foreign_job():
    seed=SEEDS[0];u=dict(jobs={'UDAP-s'+str(seed):dict(status='RUNNING')})
    v=dict(jobs={'UVB-s'+str(seed):dict(status='RUNNING')});jobs={'HPL2-s'+str(seed):dict(status='QUEUED')}
    rt=dict(udap_GPU_owner_state='u',gpu_predecessor_state='v')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:u if p=='u' else v):
        assert not slot_ready(rt,seed,'HPL2',jobs) and not slot_ready(rt,seed,'HPLT1U',jobs)
        u['jobs']['UDAP-s'+str(seed)]['status']='PREDICTIONS_COMPLETE'
        assert slot_ready(rt,seed,'HPL2',jobs) and not slot_ready(rt,seed,'HPLT1U',jobs)
        jobs['HPL2-s'+str(seed)]['status']='RUNNING'
        assert not slot_ready(rt,seed,'HPLT1U',jobs)
        v['jobs']['UVB-s'+str(seed)]['status']='PREDICTIONS_COMPLETE'
        assert slot_ready(rt,seed,'HPLT1U',jobs)
        assert not slot_ready(rt,SEEDS[1],'HPL2',jobs)


def test_capacity_reserves_dispatched_children_before_CUDA_without_double_counting():
    from experiments.pamf_ssl_research.control import capacity
    with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['101\n202\n','19000\n']):
        assert capacity(5,[303])==(3,19000)
    with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['101\n202\n303\n','19000\n']):
        assert capacity(5,[303])==(3,19000)


def test_actual_smoke_waits_for_udap_slot_and_uses_dedicated_worker():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def smoke(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl.slot_ready',side_effect=[False,True]), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.time.sleep') as sleep, \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=smoke), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            sleep.assert_called_once_with(20);launch.assert_not_called()


def test_report_requires_full_six_rows_and_reports_both_methods():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS))
        rt['rows']=[r for r in rt['rows'] if r['method'] in ('PSM','RXPSM')]
        for row in rt['rows']:
            method='HPL2' if row['method']=='PSM' else 'HPLT1U';row.update(method=method,row_id=method+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';s=json.loads(path.read_text());s.update(run_id=RUN,row_id=row['row_id']);save(path,s)
        assert len(rt['rows'])==6
        report(rt,root/'report');a=json.loads((root/'report/analysis.json').read_text())
        assert {v['method'] for v in a['verdicts']}==set(METHODS) and len(a['verdicts'])==4 and not a['goal_met']
        rt['rows'].pop()
        with pytest.raises(ValueError):report(rt,root/'bad-report')


def test_parent_source_rejects_target_contamination_and_actual_external_teacher():
    from experiments.pamf_ssl_research.hard_pl_train import source_parents
    with tempfile.TemporaryDirectory() as tmp:
        context=fixture(tmp);rt,blobs=prepare(context,tmp,tmp+'/release','new')
        c=blobs['configs/'+rt['rows'][0]['row_id']+'-train.json'];c['source_contract']=tmp+'/contract.json'
        contract=dict(role_ids={'L_s':['a'],'U_s':['b'],'V':['c']});save(Path(c['source_contract']),contract)
        previous=None;payloads={};paths=[]
        for end,epochs,offset in ((20,20,0),(30,10,20)):
            p=Path(c['fork_parent_source'])/('segment%d'%end);paths.append(p)
            pc=dict(context[str(SEEDS[0])]['F21']['train'],source_contract=c['source_contract'],
                parent_source=previous,step_offset=offset*222,epoch_offset=offset,regime='native_cric',satellite_concat=False)
            done=dict(status='SOURCE_FROZEN',epoch=epochs,updates=epochs*222,config=pc,
                target_access=False,U_truth_read=False,head_fit_updates=0)
            save(p/'complete.json',done);save(p/'progress.json',dict(PL_seen=396900 if end==20 else 567000))
            payloads[str(p/'final_ssdg.pth')]=dict(run_id=PARENT_RUN,candidate_id=pc['row_id'],epoch=epochs,
                a1_ema_successful_updates=epochs*222,args=dict(teacher_ckpt='',use_ema_teacher=False,
                    baseline_ckpt='' if previous is None else previous+'/final_ssdg.pth'))
            previous=str(p)
        with patch('torch.load',side_effect=lambda p,**kw:payloads[str(p)]):
            assert len(source_parents(c)[0])==2
            payloads[str(paths[1]/'final_ssdg.pth')]['args']['teacher_ckpt']='external'
            with pytest.raises(ValueError,match='PROVENANCE'):source_parents(c)
            payloads[str(paths[1]/'final_ssdg.pth')]['args']['teacher_ckpt']=''
            d=json.loads((paths[0]/'complete.json').read_text());d['target_access']=True;save(paths[0]/'complete.json',d)
            with pytest.raises(ValueError,match='CONTRACT'):source_parents(c)
