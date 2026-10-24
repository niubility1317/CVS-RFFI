from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from experiments.pamf_ssl_research.hard_pl_step_no_cap import RUN,METHODS,NO_CAP,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_step_no_cap_publish import prepare
from experiments.pamf_ssl_research.hard_pl_branch_guard import METHODS as BG
from experiments.pamf_ssl_research.hard_pl_step_guard import METHODS as SG,STEP_GUARD
from experiments.pamf_ssl_research.hard_pl_branch_guard_objective import protected_branch_loss
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_step_guard import states
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def config():return configuration(fixture()['2026092701']['F21']['train'],'/p','new')


def test_only_raw_norm_cap_changes_from_matched_step_guard():
    c=config();assert validate(c) is c and c['hard_pl_step_no_cap']==NO_CAP
    assert c['hard_pl_step_guard']==STEP_GUARD and c['hard_pl_gradient_guard']['U_to_L_norm_cap'] is None
    assert c['hard_pl_branch_guard']['U_to_L_norm_cap'] is None and 'hard_pl_native_cap2' not in c
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95
    assert c['hard_pl_parent_epoch']==20 and c['optimizer_updates']==11100 and not c['posthoc_head_fit']
    from experiments.pamf_ssl_research.hard_pl_step_guard import validate as old_validate
    with pytest.raises(ValueError):old_validate(c)


@pytest.mark.parametrize('change',[{'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},
    {'hard_pl_step_no_cap':dict(NO_CAP,raw_norm_nonexpansion=False)},
    {'hard_pl_native_cap2':{'U_to_L_norm_cap':2.}}])
def test_invalid_data_or_cap_recipe_rejected(change):
    c=config();c.update(change)
    with pytest.raises(ValueError):validate(c)


def test_real_explicit_bridge_accepts_uncapped_U_and_keeps_complete_L_once():
    p=nn.Parameter(torch.tensor([.2,.3]));named=[('encoder.t2.weight',p)]
    clean=p[0];physical=p[1];labeled=clean+physical;unlabeled=10*p[0]+10*p[1]
    value,details,gradients=protected_branch_loss(labeled,unlabeled,named,clean,physical,max_u_ratio=None)
    value.backward();torch.testing.assert_close(p.grad,torch.tensor([11.,11.]))
    assert torch.equal(p.grad,gradients[0]) and not details['t2']['capped']
    assert details['t2']['applied_U_norm']>2*details['t2']['L_norm']


def test_complete_native_U_objective_reports_uncapped_without_extra_gradients():
    from experiments.pamf_ssl_research.hard_pl_step_guard_objective import StepGuardedNativeAs
    from tests.test_pamf_hard_pl_native_as import batch
    from experiments.cric_dura_comparison import t1_model
    model,_,x=batch(smoke=True);c=config();c.update(hard_pl_smoke=True,source_rxs=[1,3],step_offset=4440)
    obj=StepGuardedNativeAs(c);obj.loss(model,'cpu');y=torch.tensor([0,0,1,1,2,2])
    a=model.forward_identity_only(x)['tx_logits'];b=a+.05*torch.randn_like(a)
    original=torch.autograd.grad;calls=[]
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    with obj.labeled_context(),patch.object(obj.parameter_guard,'capture') as capture:
        parts=[t1_model.pair(a,b,y,kl_weight=0. if i==2 else 1.)[0] for i in range(4)]
        labeled=.5*parts[0]+.25*parts[1]+.25*parts[2]+.25*parts[3]+a.square().mean()*.01
        with patch('torch.autograd.grad',side_effect=counted):value=obj.guide(labeled,model)
        value.backward();capture.assert_called_once()
    assert len(calls)==4 and model.teacher_calls==1 and model.student_calls==7
    assert obj.metrics['HPL_GP_U_to_L_norm_cap']==0 and not obj.metrics['HPL_GP_relative_norm_cap_enabled']
    assert obj.state_dict()['U_branch_guard']['recipe']['U_to_L_norm_cap'] is None
    for detail in obj.smoke_guard_details.values():
        assert not detail['capped'] and detail['applied_U_norm']<=detail['U_norm']*(1+1e-6)+1e-8


def test_two_segments_keep_original_forward_and_reference_optimizer_budgets():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);save(p/'progress.json',dict(PL_seen=396900,PL_accepted=390000));v=states()
        for s in v:
            s['U_gradient_guard']['recipe']['U_to_L_norm_cap']=None
            s['U_branch_guard']['recipe']['U_to_L_norm_cap']=None
        actual=aggregate(v,[dict(folder=tmp)],weight=6.)
        assert actual['U_gradient_guard']['recipe']['U_to_L_norm_cap'] is None
        assert actual['U_branch_guard']['gradient_calls']==26640 and actual['U_parameter_step_guard']['reference_optimizer_steps']==6660
        assert actual['student_U_views']==10602900 and actual['teacher_eval_U_views']==2097900


def test_release_has_three_seed_dual_readout_matrix_and_scorer_after_SG():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['step_guard_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_queue_waits_BG_GPU_completion_and_reserves_both_pre_CUDA_pids():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},branch_guard_GPU_owner_state='bg',step_guard_GPU_owner_state='sg')
    source={k:dict(jobs={}) for k in ('bg','sg')}
    for seed in (2026092701,2026092702,2026092703):
        source['bg']['jobs'][BG[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        source['sg']['jobs'][SG[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:source[p]):
        assert not slot_ready(rt,2026092702)
        source['bg']['jobs'][BG[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        source['sg']['jobs'][SG[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_real_original_checkpoint_smoke_precedes_any_formal_row():
    from experiments.pamf_ssl_research.control import run
    class Reached(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_step_no_cap_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            raise Reached
        with patch('experiments.pamf_ssl_research.hard_pl_step_no_cap.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_step_no_cap.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(Reached):run(rt)
            launch.assert_not_called()


def test_target_report_keeps_raw_goal_and_separate_DURA_reference():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS));rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHODS[0],row_id=METHODS[0]+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            p=Path(c['output_root'])/'scores.json';v=json.loads(p.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(p,v)
        report(rt,root/'report');d=json.loads((root/'report/analysis.json').read_text())
        assert len(d['verdicts'])==2 and {v['method'] for v in d['verdicts']}==set(METHODS) and not d['goal_met']
