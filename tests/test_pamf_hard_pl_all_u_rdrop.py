from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace,MethodType
from unittest.mock import patch
import pytest
import torch
from experiments.pamf_ssl_research.hard_pl_all_u_rdrop import RUN,METHODS,RECIPE,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_all_u_rdrop_publish import prepare
from experiments.pamf_ssl_research.hard_pl_all_u_rdrop_objective import AllURDrop
from experiments.pamf_ssl_research.hard_pl_step_guard_objective import StepGuardedNativeAs
from experiments.pamf_ssl_research.hard_pl_step_guard import METHODS as SG
from experiments.pamf_ssl_research.hard_pl_step_no_cap import METHODS as NC,configuration as nc_configuration
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from experiments.pamf_ssl_research.rejected_rdrop import rejected_rdrop
from tests.test_pamf_hard_pl_step_guard import states
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def config():return configuration(fixture()['2026092701']['F21']['train'],'/p','new')


def stochastic_student(model):
    original=model.forward_identity_only
    def forward(self,*args,**kwargs):
        value=original(*args,**kwargs)
        return dict(value,tx_logits=value['tx_logits']+.03*torch.randn_like(value['tx_logits']))
    model.forward_identity_only=MethodType(forward,model)
    return model


def make_objective(cls,allrd=True):
    from tests.test_pamf_hard_pl_native_as import batch
    model,_,x=batch(smoke=True);model=stochastic_student(model)
    c=(configuration if allrd else nc_configuration)(fixture()['2026092701']['F21']['train'],'/p','new')
    c.update(hard_pl_smoke=True,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_only_rejected_kl_recipe_changes_from_fixed_step_no_cap():
    c=config();assert validate(c) is c and c['hard_pl_all_u_rdrop']==RECIPE
    old=nc_configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    ignored={'run_id','row_id','algorithm_id','display_method','output_root','hard_pl_all_u_rdrop'}
    assert {k:v for k,v in c.items() if k not in ignored}=={k:v for k,v in old.items() if k not in ignored}
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95 and c['hard_pl_parent_epoch']==20
    assert c['hard_pl_gradient_guard']['U_to_L_norm_cap'] is None and c['optimizer_updates']==11100


@pytest.mark.parametrize('change',[{'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},
    {'hard_pl_all_u_rdrop':dict(RECIPE,physical_As_KL_weight=1.)},
    {'hard_pl_all_u_rdrop':dict(RECIPE,hard_label_gate=.9)},
    {'hard_pl_all_u_rdrop':dict(RECIPE,hard_CE_or_contrast_on_rejected=True)}])
def test_invalid_permission_or_changed_hard_label_recipe_rejected(change):
    c=config();c.update(change)
    with pytest.raises(ValueError):validate(c)


def test_full_current_U_loss_adds_exactly_one_weighted_rejected_term_without_extra_forward():
    torch.manual_seed(51);old,base,_=make_objective(StepGuardedNativeAs,False)
    torch.manual_seed(51);new,obj,_=make_objective(AllURDrop)
    torch.manual_seed(52);base.loss(old,'cpu')
    torch.manual_seed(52);returned=obj.loss(new,'cpu')
    extra,counts=rejected_rdrop(obj.smoke_allrd_logits,obj.smoke_allrd_mask,obj.smoke_allrd_physical)
    torch.testing.assert_close(obj.pending_gradient_loss,base.pending_gradient_loss+6*extra)
    assert returned==0 and obj.accepted==base.accepted==4 and old.teacher_calls==new.teacher_calls==1
    assert old.student_calls==new.student_calls==6 and obj.allrd_current is None
    for a,b in zip(old.student_inputs,new.student_inputs):torch.testing.assert_close(a,b)
    assert extra>0 and counts['rejected_U']==2
    q=obj.state_dict()['U_all_RDrop'];assert q['rejected']==2 and q['extra_KL_evaluations']==3
    assert not q['hard_label_or_CE_on_rejected'] and not q['raw_acceptance_changed'] and q['physical_As_KL_weight']==0


def test_all_rejected_U_receives_consistency_but_no_hard_label_CE():
    torch.manual_seed(53);model,obj,_=make_objective(AllURDrop);model.teacher.zero_();obj.c['hard_pl_smoke']=False
    # Normal training permits zero accepted classes;the source smoke separately
    # requires a nonzero accepted cross-RX signature derivative.
    obj.loss(model,'cpu');extra=obj.metrics['HPL_ALLRD_extra_base_loss']
    assert obj.accepted==0 and extra>0
    torch.testing.assert_close(obj.pending_gradient_loss,6*extra)
    values=torch.autograd.grad(obj.pending_gradient_loss,tuple(model.parameters()),allow_unused=True)
    assert any(v is not None and v.norm()>0 for v in values)


def test_full_guard_consumes_added_loss_once_with_same_four_derivatives_and_default_head():
    from experiments.cric_dura_comparison import t1_model
    model,obj,x=make_objective(AllURDrop);obj.loss(model,'cpu');y=torch.tensor([0,0,1,1,2,2])
    a=model.forward_identity_only(x)['tx_logits'];b=a+.05*torch.randn_like(a)
    original=torch.autograd.grad;calls=[]
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    with obj.labeled_context(),patch.object(obj.parameter_guard,'capture') as capture:
        parts=[t1_model.pair(a,b,y,kl_weight=0. if i==2 else 1.)[0] for i in range(4)]
        labeled=.5*parts[0]+.25*parts[1]+.25*parts[2]+.25*parts[3]+a.square().mean()*.01
        with patch('torch.autograd.grad',side_effect=counted):value=obj.guide(labeled,model)
        value.backward();capture.assert_called_once()
    assert len(calls)==4 and model.teacher_calls==1 and model.student_calls==7 and obj.pending_gradient_loss is None
    assert obj.metrics['HPL_GP_U_to_L_norm_cap']==0 and not obj.metrics['HPL_GP_relative_norm_cap_enabled']
    for name,p in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected)


def rd_state(s):
    rejected=s['seen']-s['accepted'];nonphysical=rejected//2;pairs=2*rejected+nonphysical
    return dict(recipe=deepcopy(RECIPE),calls=s['calls'],seen=s['seen'],rejected=rejected,nonphysical_rejected=nonphysical,
        pairs=pairs,reused_prediction_views=2*pairs,extra_KL_evaluations=3*s['calls'],hard_label_or_CE_on_rejected=False,
        raw_acceptance_changed=False,physical_As_KL_weight=0.,pending_logits=False,
        extra_teacher=0,extra_student=0,extra_head=0,new_head=False,U_truth_reads=0,target_reads=0)


def test_both_native_segments_keep_counts_permissions_and_unchanged_forward_budgets():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);save(p/'progress.json',dict(PL_seen=396900,PL_accepted=390000));v=states()
        for s in v:
            s['U_gradient_guard']['recipe']['U_to_L_norm_cap']=None;s['U_branch_guard']['recipe']['U_to_L_norm_cap']=None
            s['U_all_RDrop']=rd_state(s)
        actual=aggregate(v,[dict(folder=tmp)],weight=6.);q=actual['U_all_RDrop']
        assert q['calls']==6660 and q['seen']==1701000 and q['extra_KL_evaluations']==19980
        assert q['rejected']==sum(s['seen']-s['accepted'] for s in v) and q['pairs']==2*q['rejected']+q['nonphysical_rejected']
        assert actual['student_U_views']==10602900 and actual['teacher_eval_U_views']==2097900
        assert actual['U_branch_guard']['gradient_calls']==26640 and actual['U_parameter_step_guard']['reference_optimizer_steps']==6660
        v[-1]['U_all_RDrop']['hard_label_or_CE_on_rejected']=True
        with pytest.raises(ValueError,match='Rejected U'):aggregate(v,[dict(folder=tmp)],weight=6.)


def test_release_keeps_three_seeds_and_seven_views_per_readout_scorer_after_NoCap():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['step_no_cap_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and len(p['views'])==7
        assert 'truth' not in p['tests']['main']


def test_queue_waits_SG_predictions_and_reserves_both_pre_CUDA_PIDs():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},step_guard_GPU_owner_state='sg',step_no_cap_GPU_owner_state='nc')
    data={k:dict(jobs={}) for k in ('sg','nc')}
    for seed in (2026092701,2026092702,2026092703):
        data['sg']['jobs'][SG[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        data['nc']['jobs'][NC[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:data[p]):
        assert not slot_ready(rt,2026092702)
        data['sg']['jobs'][SG[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        data['nc']['jobs'][NC[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_real_original_E20_smoke_precedes_formal_dispatch():
    from experiments.pamf_ssl_research.control import run
    class Reached(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_all_u_rdrop_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            raise Reached
        with patch('experiments.pamf_ssl_research.hard_pl_all_u_rdrop.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_all_u_rdrop.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(Reached):run(rt)
            launch.assert_not_called()


def test_raw_goal_and_fixed_DURA_are_reported_separately():
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
