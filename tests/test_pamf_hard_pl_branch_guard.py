from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from experiments.pamf_ssl_research.hard_pl_branch_guard import RUN,METHODS,BRANCH_GUARD,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_branch_guard_objective import BranchGuardedNativeAs,protected_branch_loss,labeled_context_active
from experiments.pamf_ssl_research.hard_pl_branch_guard_publish import prepare
from experiments.pamf_ssl_research.hard_pl_native_cap2 import METHODS as CAP
from experiments.pamf_ssl_research.hard_pl_balanced_ce import METHODS as BCE
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_native_as import batch as native_batch
from tests.test_pamf_hard_pl_balanced_ce import native_states
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def config():return configuration(fixture()['2026092701']['F21']['train'],'/p','new')


def test_explicit_parameter_bridge_keeps_full_L_once_and_removes_component_conflict():
    p=nn.Parameter(torch.tensor([.2,.3],dtype=torch.float64));named=[('encoder.t2.weight',p)]
    clean=p[0];physical=p[1];labeled=clean+physical;unlabeled=-p[0]+2*p[1]
    original=torch.autograd.grad;calls=[]
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    with patch('torch.autograd.grad',side_effect=counted):
        value,details,expected=protected_branch_loss(labeled,unlabeled,named,clean,physical)
    assert len(calls)==4 and p.grad is None
    value.backward();torch.testing.assert_close(p.grad,torch.tensor([1.,3.],dtype=torch.float64))
    assert torch.equal(p.grad,expected[0]) and details['t2']['constraint_dots_after']['L_clean']==pytest.approx(0.)


def test_zero_U_keeps_the_entire_L_gradient():
    p=nn.Parameter(torch.tensor([.2,.3]));clean=p[0].square();physical=p[1].square();labeled=clean+physical+p.sum()*.1
    expected=torch.autograd.grad(labeled,p,retain_graph=True)[0]
    value,_,_=protected_branch_loss(labeled,p.sum()*0.,[('x',p)],clean,physical)
    value.backward();torch.testing.assert_close(p.grad,expected)


def test_configuration_preserves_original_parent_recipe_and_permissions():
    c=config()
    assert c['hard_pl_branch_guard']==BRANCH_GUARD and c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95
    assert c['hard_pl_native_cap2']['extra_autograd_passes']==2 and c['hard_pl_gradient_guard']['U_to_L_norm_cap']==2
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    assert c['hard_pl_native_as']['extra_As_teacher']==0 and not c['posthoc_head_fit']
    from experiments.pamf_ssl_research.hard_pl_native_cap2 import validate as cap_validate
    with pytest.raises(ValueError):cap_validate(c)


@pytest.mark.parametrize('change',[
    {'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},{'hard_pl_no_uas':True},
    {'hard_pl_branch_guard':dict(BRANCH_GUARD,extra_autograd_passes=0)},
    {'hard_pl_gradient_guard':dict(config()['hard_pl_gradient_guard'],anchor='complete_current_L_objective')},
])
def test_invalid_source_permissions_or_operator_are_rejected(change):
    c=config();c.update(change)
    with pytest.raises(ValueError):validate(c)


def test_actual_six_U_forwards_then_four_L_pair_capture_and_one_explicit_backward():
    torch.manual_seed(120);model,_,x=native_batch(smoke=True)
    c=config();c.update(hard_pl_smoke=True,source_rxs=[1,3],step_offset=4440);obj=BranchGuardedNativeAs(c)
    before=tuple(id(p) for p in model.parameters());obj.loss(model,'cpu')
    assert model.student_calls==6
    from experiments.cric_dura_comparison import t1_model
    y=torch.tensor([0,0,1,1,2,2]);a=model.forward_identity_only(x)['tx_logits'];b=a+.05*torch.randn_like(a)
    pair_function=t1_model.pair
    with obj.labeled_context():
        parts=[t1_model.pair(a,b,y,kl_weight=0. if i==2 else 1.)[0] for i in range(4)]
        labeled=.5*parts[0]+.25*parts[1]+.25*parts[2]+.25*parts[3]+a.square().mean()*.01
        value=obj.guide(labeled,model)
    assert t1_model.pair is pair_function and not labeled_context_active()
    value.backward();state=obj.state_dict();branch=state['U_branch_guard']
    assert model.teacher_calls==1 and model.student_calls==7 and state['U_native_As']['generated']==3
    assert branch['calls']==1 and branch['gradient_calls']==4 and branch['extra_autograd_calls']==2
    assert branch['L_pair_captures']==4 and not branch['pending_capture'] and not branch['pending_U_loss']
    for group,detail in obj.smoke_guard_details.items():
        assert detail['applied_U_norm']<=2*detail['L_norm']*(1+1e-6)+1e-12
        assert all(cos is None or cos>=-1.000001e-6 for name,cos in detail['anchor_cosines_after'].items()
                   if name not in detail['near_zero_component_anchors_omitted'])
    for name,p in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected)
    assert tuple(id(p) for p in model.parameters())==before and all(not m._forward_hooks for m in model.modules())


def test_capture_is_removed_on_failure_and_incomplete_L_pairs_cannot_train():
    model,_,_=native_batch(smoke=False);obj=BranchGuardedNativeAs(config())
    from experiments.cric_dura_comparison import t1_model
    original=t1_model.pair
    with pytest.raises(RuntimeError):
        with obj.labeled_context():raise RuntimeError('source failure')
    assert t1_model.pair is original and not labeled_context_active()
    with pytest.raises(ValueError,match='four original'):
        with obj.labeled_context():obj.guide(next(model.parameters()).sum(),model)
    assert t1_model.pair is original and not labeled_context_active()


def test_installed_native_hook_captures_only_its_four_L_pairs_and_always_restores():
    from experiments.cric_dura_comparison import t1_model
    from experiments.pamf_ssl_research.hard_pl_train import installed
    original_pair=t1_model.pair
    class FakeObjective:
        def __init__(self,c):self.real=BranchGuardedNativeAs(c);self.metrics={}
        def labeled_context(self):return self.real.labeled_context()
        def guide(self,labeled,student):
            from experiments.pamf_ssl_research.labeled_branch_capture import components
            clean,physical=components(labeled)
            assert clean==pytest.approx(1.) and physical==pytest.approx(.25)
            return labeled
    def old_hook(**kwargs):
        for _ in range(4):t1_model.pair(None,None,None)
        return torch.tensor(1.25),{}
    model=SimpleNamespace(_source_labeled_feature_hook=old_hook)
    with patch.object(t1_model,'install',side_effect=lambda m,c:m),patch.object(t1_model,'pair',return_value=(torch.tensor(1.),None,None)):
        pair=t1_model.pair
        with installed(config(),{},objective_cls=FakeObjective):
            t1_model.install(model,config());model._source_labeled_feature_hook(identity_loss=torch.tensor(0.))
        assert t1_model.pair is pair and not labeled_context_active()
    assert t1_model.pair is original_pair


def test_U_truth_rejected_before_teacher_and_student_forward():
    model,_,x=native_batch(smoke=False);obj=BranchGuardedNativeAs(config())
    bad=(x,torch.zeros(6,dtype=torch.long),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[bad];model._cd_U_iter=iter([bad])
    with pytest.raises(ValueError,match='U truth'):obj.loss(model,'cpu')
    assert model.teacher_calls==model.student_calls==0


def branch_states():
    from experiments.pamf_ssl_research.hard_pl_guard import GUARD
    states=native_states()
    for s in states:
        s.pop('U_balanced_CE');s['U_gradient_guard']=dict(recipe=dict(GUARD,U_to_L_norm_cap=2.,
            projection='simultaneous_three_L_anchors_per_group',anchor='complete_L_nonphysical_L_physical_L'),
            calls=s['calls'],conflicts={g:0 for g in ('t2','cls_head','other')},caps={g:0 for g in ('t2','cls_head','other')},pending_U_loss=False)
        s['U_branch_guard']=dict(recipe=deepcopy(BRANCH_GUARD),calls=s['calls'],gradient_calls=4*s['calls'],
            extra_autograd_calls=2*s['calls'],L_pair_captures=4*s['calls'],pending_capture=False,pending_U_loss=False,
            anchor_conflicts={g:{a:0 for a in ('L_full','L_clean','L_As')} for g in ('t2','cls_head','other')},
            extra_teacher_or_student_forward=0,new_head=False,U_truth_reads=0,target_reads=0)
    return states


def test_two_segments_keep_exact_component_derivative_and_native_view_budgets():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);save(p/'progress.json',dict(PL_seen=396900,PL_accepted=390000));states=branch_states()
        actual=aggregate(states,[dict(folder=str(p))],weight=6.)
        branch=actual['U_branch_guard']
        assert branch['calls']==6660 and branch['gradient_calls']==26640 and branch['extra_autograd_calls']==13320
        assert branch['L_pair_captures']==26640 and actual['student_U_views']==10602900 and actual['teacher_eval_U_views']==2097900
        assert actual['physical_student_U_views']==1701000
        states[-1]['U_branch_guard']['extra_autograd_calls']-=1
        with pytest.raises(ValueError,match='branch'):aggregate(states,[dict(folder=str(p))],weight=6.)


def test_release_has_three_seed_raw_DURA_matrix_and_scorer_after_cap2():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['native_cap2_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_queue_waits_matching_balancedCE_and_reserves_cap2_pre_CUDA_PID():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},balanced_ce_GPU_owner_state='b',native_cap2_GPU_owner_state='c')
    states={k:dict(jobs={}) for k in ('b','c')}
    for seed in (2026092701,2026092702,2026092703):
        states['b']['jobs'][BCE[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['c']['jobs'][CAP[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092702)
        states['b']['jobs'][BCE[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        states['c']['jobs'][CAP[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert slot_ready(rt,2026092702) and smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_matching_original_E20_smoke_is_the_first_GPU_action():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_branch_guard_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_branch_guard.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_branch_guard.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_full_target_report_keeps_raw_goal_and_DURA_reference():
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
