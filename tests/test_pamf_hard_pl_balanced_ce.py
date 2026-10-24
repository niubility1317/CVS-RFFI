from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_balanced_ce import RUN,METHODS,BALANCED_CE,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_balanced_ce_objective import BalancedCE,GuardedBalancedCE,ce_reallocation
from experiments.pamf_ssl_research.hard_pl_balanced_ce_publish import prepare
from experiments.pamf_ssl_research.hard_pl_native_as import NATIVE_AS,METHODS as NAS,configuration as native_configuration
from experiments.pamf_ssl_research.hard_pl_native_as_objective import NativeAs
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation import METHODS as RXE
from experiments.pamf_ssl_research.source_balanced_pseudo_ce import contribution_weights
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_native_as import batch as native_batch
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def batch(cls=BalancedCE,smoke=False):
    model,old,x=native_batch(smoke=smoke)
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    model.teacher[2]=model.teacher[0]
    return model,cls(c),x


def test_balancing_preserves_native_As_parent_permissions_and_all_original_budgets():
    from experiments.pamf_ssl_research.hard_pl_native_as import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_balanced_ce']==BALANCED_CE and c['hard_pl_native_as']==NATIVE_AS and not c['hard_pl_no_uas']
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95 and c['hard_pl_parent_epoch']==20
    assert c['hard_pl_continuation_epochs']==30 and c['epochs']==50 and c['optimizer_updates']==11100
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},{'hard_pl_no_uas':True},
                   {'hard_pl_balanced_ce':dict(BALANCED_CE,maximum=2.)},
                   {'hard_pl_balanced_ce':dict(BALANCED_CE,persistent_statistics=True)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def test_six_view_CE_correction_matches_direct_weighted_loss_and_all_logit_gradients():
    torch.manual_seed(107);zs=[torch.randn(6,6,requires_grad=True,dtype=torch.float64) for _ in range(6)]
    y=torch.tensor([0,0,0,1,1,2]);a=torch.tensor([True]*5+[False]);rx=torch.tensor([1,3,1,1,3,1])
    w=contribution_weights(y,a,rx);delta,before,after=ce_reallocation(zs,y,a,w)
    expected_before=sum(c*(F.cross_entropy(z,y,reduction='none',label_smoothing=.01)*a).mean()
                        for c,z in zip((.2,.2,.2,.2,.1,.1),zs))
    expected_after=sum(c*(F.cross_entropy(z,y,reduction='none',label_smoothing=.01)*a*w).mean()
                       for c,z in zip((.2,.2,.2,.2,.1,.1),zs))
    torch.testing.assert_close(before,expected_before);torch.testing.assert_close(after,expected_after)
    torch.testing.assert_close(before+delta,after)
    got=torch.autograd.grad(before+delta,zs,retain_graph=True);expected=torch.autograd.grad(expected_after,zs)
    for g,e in zip(got,expected):
        torch.testing.assert_close(g,e);assert g[-1].count_nonzero()==0


def test_unit_weights_preserve_native_As_loss_and_parameter_gradients():
    torch.manual_seed(109);first,base,x=native_batch(smoke=False)
    torch.manual_seed(109);second,new,_=batch();second.teacher=first.teacher.clone()
    torch.manual_seed(110);old=base.loss(first,'cpu')
    with patch('experiments.pamf_ssl_research.hard_pl_balanced_ce_objective.contribution_weights',
               side_effect=lambda p,a,r:torch.ones(len(p),dtype=torch.float64,device=p.device)):
        torch.manual_seed(110);changed=new.loss(second,'cpu')
    torch.testing.assert_close(old,changed)
    ga=torch.autograd.grad(old,list(first.parameters()),allow_unused=True)
    gb=torch.autograd.grad(changed,list(second.parameters()),allow_unused=True)
    for a,b in zip(ga,gb):
        if a is None:assert b is None
        else:torch.testing.assert_close(a,b)


def test_actual_reweighting_reuses_six_logits_and_enters_full_L_guard_once():
    torch.manual_seed(111);model,obj,x=batch(GuardedBalancedCE,True);parameters=tuple(id(p) for p in model.parameters())
    obj.loss(model,'cpu')
    assert obj.guard_calls==0 and model.teacher_calls==1 and model.student_calls==6 and model._cd_practical.calls==1
    w=obj.smoke_bce_weights;a=obj.smoke_bce_accepted
    assert w[a].min()<1<w[a].max() and float(w[a].sum())==pytest.approx(int(a.sum()))
    assert model._cd_practical.packets==3 and obj.state_dict()['U_native_As']['generated']==3
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward();s=obj.state_dict();r=s['U_balanced_CE']
    assert r['calls']==1 and r['captures']==6 and r['extra_CE_kernels']==6 and r['captured_packet_views']==36
    assert r['accepted']==s['accepted']==4 and r['accepted_weight_mass']==pytest.approx(4)
    assert obj.guard_calls==1 and not s['U_gradient_guard']['pending_U_loss']
    assert not r['extra_teacher'] and not r['extra_head'] and not r['extra_backbone'] and not r['new_head']
    for name,p in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected)
    assert model.training and tuple(id(p) for p in model.parameters())==parameters
    assert obj._bce_logits is None and obj._bce_current is None and all(not m._forward_hooks for m in model.modules())


def test_all_rejected_samples_have_zero_label_gradient_and_finite_neutral_weights():
    model,obj,_=batch();model.teacher.zero_();value=obj.loss(model,'cpu');value.backward()
    r=obj.state_dict()['U_balanced_CE']
    assert value==0 and torch.isfinite(value) and r['accepted']==r['accepted_weight_mass']==0
    assert r['minimum_weight']==r['maximum_weight']==1 and model.student_calls==6
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in model.parameters())


def test_failed_student_forward_clears_captured_logits_and_existing_feature_hooks():
    model,obj,_=batch();original=model.forward_identity_only
    def failed(x,domain_labels=None):
        if model.student_calls==3:raise RuntimeError('source student failure')
        return original(x,domain_labels)
    model.forward_identity_only=failed
    with pytest.raises(RuntimeError,match='source student failure'):obj.loss(model,'cpu')
    assert obj._bce_logits is None and obj._bce_current is None
    assert not obj.style_active and not obj.joint_active and not obj.capture_active
    assert 'features' not in model.id_backbone.encoder.__dict__ and all(not m._forward_hooks for m in model.modules())


@pytest.mark.parametrize('case',['missing_view','wrong_mass','rejected_non_neutral'])
def test_bad_logit_or_contribution_policy_is_rejected(case):
    zs=[torch.randn(4,6) for _ in range(6)];p=torch.zeros(4,dtype=torch.long);a=torch.tensor([True,True,True,False]);w=torch.ones(4,dtype=torch.float64)
    if case=='missing_view':zs.pop()
    elif case=='wrong_mass':w[0]=1.2
    else:w[-1]=1.2
    with pytest.raises(ValueError):ce_reallocation(zs,p,a,w)


def native_states():
    ss=[state(e,True) for e in (10,20)]
    for s in ss:
        s.update(physical_seen=s['seen']//2,physical_student_views=s['seen'],raw_As_seen=0)
        s['U_native_As']=dict(recipe=deepcopy(NATIVE_AS),calls=s['calls'],generated=s['seen']//2,
            student_physical_views=s['seen'],raw_accepted_half=s['accepted']//2,accepted_student_views=2*(s['accepted']//2),
            raw_fallback=0,extra_As_teacher=0,all_raw_accepted_retained=True,extra_student_forwards=0,
            new_head=False,EMA_updates=0,U_truth_read=False,CE_dose_normalizer=1.25)
        s['U_balanced_CE']=dict(recipe=deepcopy(BALANCED_CE),calls=s['calls'],captures=6*s['calls'],captured_packet_views=6*s['seen'],
            extra_CE_kernels=6*s['calls'],accepted=s['accepted'],accepted_weight_mass=float(s['accepted']),minimum_weight=.7,maximum_weight=1.5,
            max_batch_dose_error=1e-12,all_raw_accepted_retained=True,raw_acceptance_changed=False,extra_teacher=0,extra_backbone=0,
            extra_head=0,new_head=False,persistent_context=False,EMA_updates=0,prototype_updates=0,U_truth_reads=0,target_reads=0,CE_dose_normalizer=1.25)
    return ss


def test_two_segments_preserve_actual_mass_and_original_native_As_teacher_student_costs():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);save(p/'progress.json',dict(PL_seen=396900,PL_accepted=390000));ss=native_states()
        r=aggregate(ss,[dict(folder=str(p))],weight=6.);b=r['U_balanced_CE']
        assert b['calls']==6660 and b['captures']==b['extra_CE_kernels']==39960 and b['captured_packet_views']==10206000
        assert b['accepted_weight_mass']==b['accepted']==1700800 and b==r['hard_PL_tail']['U_balanced_CE']
        assert r['teacher_eval_U_views']==2097900 and r['student_U_views']==10602900
        assert r['physical_U_seen']==850500 and r['physical_student_U_views']==1701000 and r['raw_U_As_seen']==0
        ss[-1]['U_balanced_CE']['accepted_weight_mass']-=1
        with pytest.raises(ValueError,match='Bounded CE contribution'):aggregate(ss,[dict(folder=str(p))],weight=6.)
        ss=native_states();ss[-1].pop('U_balanced_CE')
        with pytest.raises(ValueError,match='Matching nativeAs bounded'):aggregate(ss,[dict(folder=str(p))],weight=6.)


def test_release_keeps_all_three_seeds_readouts_and_single_scorer_after_nativeAs():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and rt['prior_score_owner_state']==rt['native_as_GPU_owner_state']
    assert len(rt['rows'])==rt['max_total_workers']==3
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_queue_requires_finished_RXE_and_reserves_dispatched_nativeAs_before_CUDA():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},receiver_extrapolation_GPU_owner_state='r',native_as_GPU_owner_state='n')
    states={k:dict(jobs={}) for k in ('r','n')}
    for seed in (2026092701,2026092702,2026092703):
        states['r']['jobs'][RXE[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['n']['jobs'][NAS[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092702)
        states['r']['jobs'][RXE[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        states['n']['jobs'][NAS[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert slot_ready(rt,2026092702) and smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_first_matching_seed_real_E20_smoke_precedes_any_formal_dispatch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_balanced_ce_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_balanced_ce.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_balanced_ce.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_complete_both_readout_report_keeps_raw_goal_separate_from_DURA():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS));rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHODS[0],row_id=METHODS[0]+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            p=Path(c['output_root'])/'scores.json';v=json.loads(p.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(p,v)
        report(rt,root/'report');r=json.loads((root/'report/analysis.json').read_text())
        assert len(r['verdicts'])==2 and {v['method'] for v in r['verdicts']}==set(METHODS) and not r['goal_met']
