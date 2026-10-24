from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_native_cap2 import RUN,METHODS,CAP2,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_native_cap2_publish import prepare
from experiments.pamf_ssl_research.hard_pl_native_as import METHODS as NAS
from experiments.pamf_ssl_research.hard_pl_balanced_ce import METHODS as BCE
from experiments.pamf_ssl_research.hard_pl_native_as_objective import GuardedNativeAs
from experiments.pamf_ssl_research.u_gradprotect_objective import projected_gradients,protected_loss
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_native_as import batch as native_batch
from tests.test_pamf_hard_pl_balanced_ce import native_states
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def batch(smoke=True):
    model,old,x=native_batch(smoke=smoke)
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,GuardedNativeAs(c),x


def test_default_cap1_is_unchanged_and_cap2_preserves_conflict_projection():
    p=nn.Parameter(torch.zeros(2));named=[('encoder.t2.weight',p)];gl=(torch.tensor([1.,0]),);gu=(torch.tensor([-2.,3.]),)
    default,_=projected_gradients(named,gl,gu);explicit,_=projected_gradients(named,gl,gu,max_u_ratio=1.)
    torch.testing.assert_close(default[0],torch.tensor([1.,1.]));assert torch.equal(default[0],explicit[0])
    new,d=projected_gradients(named,gl,gu,max_u_ratio=2.)
    torch.testing.assert_close(new[0],torch.tensor([1.,2.]));assert d['t2']['dot_after']==0 and d['t2']['applied_U_norm']==pytest.approx(2)
    assert d['t2']['conflict'] and d['t2']['capped']


@pytest.mark.parametrize('ratio',[0.,-1.,float('nan')])
def test_invalid_norm_limits_cannot_enter_parameter_updates(ratio):
    p=nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError,match='positive U/L'):projected_gradients([('encoder.t2.weight',p)],[p.detach()],[p.detach()],max_u_ratio=ratio)


def test_tinyL_zeroU_and_uncapped_gradients_are_retained_under_cap2():
    p=nn.Parameter(torch.zeros(2));named=[('encoder.t2.weight',p)]
    actual,d=projected_gradients(named,[torch.zeros(2)],[torch.ones(2)],max_u_ratio=2.)
    assert not actual[0].count_nonzero() and d['t2']['tiny_L']
    actual,d=projected_gradients(named,[torch.tensor([1.,0])],[torch.tensor([0.,.5])],max_u_ratio=2.)
    torch.testing.assert_close(actual[0],torch.tensor([1.,.5]));assert not d['t2']['capped']


def test_real_explicit_backward_receives_cap2_combined_parameter_gradient_once():
    p=nn.Parameter(torch.tensor([.2,.1]));named=[('encoder.t2.weight',p)]
    labeled=(p*torch.tensor([1.,0])).sum();unlabeled=(p*torch.tensor([-2.,3.])).sum()
    value,detail,expected=protected_loss(labeled,unlabeled,named,max_u_ratio=2.);value.backward()
    torch.testing.assert_close(p.grad,torch.tensor([1.,2.]));torch.testing.assert_close(p.grad,expected[0])
    assert detail['t2']['dot_after']==0


def test_cap2_configuration_changes_only_fixed_norm_limit_and_preserves_permissions():
    from experiments.pamf_ssl_research.hard_pl_native_as import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_native_cap2']==CAP2 and c['hard_pl_gradient_guard']['U_to_L_norm_cap']==2
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95 and not c['hard_pl_no_uas']
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},{'hard_pl_no_uas':True},
       {'hard_pl_gradient_guard':dict(c['hard_pl_gradient_guard'],U_to_L_norm_cap=3.)},
       {'hard_pl_native_cap2':dict(CAP2,U_to_L_norm_cap=1.5)}):
        v=deepcopy(c);v.update(change)
        with pytest.raises(ValueError):validate(v)


def test_native_As_forwards_and_fullL_cap2_gradient_are_actual_not_nominal():
    torch.manual_seed(113);model,obj,x=batch();params=tuple(id(p) for p in model.parameters());obj.loss(model,'cpu')
    assert model.student_calls==6 and model.teacher_calls==1 and model._cd_practical.calls==1 and model._cd_practical.packets==3
    assert obj.guard_calls==0 and obj.pending_gradient_loss is not None
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward();s=obj.state_dict();g=s['U_gradient_guard']
    assert g['calls']==1 and g['recipe']['U_to_L_norm_cap']==2 and not g['pending_U_loss']
    assert obj.metrics['HPL_GP_U_to_L_norm_cap']==2 and s['U_native_As']['generated']==3 and s['physical_student_views']==6
    assert s['raw_As_seen']==0 and not s['U_native_As']['extra_As_teacher']
    assert any(v['applied_U_norm']>v['L_norm']+1e-5 for v in obj.smoke_guard_details.values())
    for v in obj.smoke_guard_details.values():
        assert v['applied_U_norm']<=2*v['L_norm']*(1+1e-6)+1e-8
        assert v['dot_after']>=-1e-5*(v['L_norm']*v['applied_U_norm']+1e-12)
    for name,p in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected)
    assert model.training and tuple(id(p) for p in model.parameters())==params
    assert all(not m._forward_hooks for m in model.modules()) and not obj.joint_active and not obj.style_active


def test_U_truth_exposure_is_rejected_before_any_teacher_or_student_forward():
    model,obj,x=batch(False);bad=(x,torch.zeros(6,dtype=torch.long),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[bad];model._cd_U_iter=iter([bad])
    with pytest.raises(ValueError,match='U truth'):obj.loss(model,'cpu')
    assert model.teacher_calls==model.student_calls==0


def test_closed_cap2_segments_keep_original_native_As_cost_and_one_matching_guard():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);save(p/'progress.json',dict(PL_seen=396900,PL_accepted=390000));ss=native_states()
        from experiments.pamf_ssl_research.hard_pl_guard import GUARD
        for s in ss:
            s.pop('U_balanced_CE');s['U_gradient_guard']=dict(recipe=dict(GUARD,U_to_L_norm_cap=2.),calls=s['calls'],
                conflicts={g:0 for g in ('t2','cls_head','other')},caps={g:0 for g in ('t2','cls_head','other')},pending_U_loss=False)
        r=aggregate(ss,[dict(folder=str(p))],weight=6.)
        assert r['teacher_eval_U_views']==2097900 and r['student_U_views']==10602900 and r['physical_U_seen']==850500
        assert r['physical_student_U_views']==1701000 and r['U_gradient_guard']['calls']==6660
        assert r['U_gradient_guard']['recipe']['U_to_L_norm_cap']==2
        ss[-1]['U_gradient_guard']['recipe']['U_to_L_norm_cap']=1
        with pytest.raises(ValueError,match='exactly one fixed'):aggregate(ss,[dict(folder=str(p))],weight=6.)


def test_release_keeps_three_seed_fullreadouts_and_single_scorer_after_balancedCE():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['balanced_ce_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_queue_requires_finished_nativeAs_and_reserves_dispatched_balancedCE():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},native_as_GPU_owner_state='n',balanced_ce_GPU_owner_state='b')
    states={k:dict(jobs={}) for k in ('n','b')}
    for seed in (2026092701,2026092702,2026092703):
        states['n']['jobs'][NAS[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['b']['jobs'][BCE[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092702)
        states['n']['jobs'][NAS[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        states['b']['jobs'][BCE[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert slot_ready(rt,2026092702) and smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_matching_seed_real_E20_cap2_smoke_precedes_any_formal_dispatch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_native_cap2_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_native_cap2.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_native_cap2.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_full_target_report_keeps_raw_goal_separate_from_DURA():
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
