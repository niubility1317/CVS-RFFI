from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_native_as import RUN,METHODS,NATIVE_AS,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_native_as_objective import NativeAs,GuardedNativeAs
from experiments.pamf_ssl_research.hard_pl_native_as_publish import prepare
from experiments.pamf_ssl_research.hard_pl_joint import configuration as joint_configuration
from experiments.pamf_ssl_research.hard_pl_joint_objective import JointNoAsMixStyle
from experiments.pamf_ssl_research.hard_pl_path_reliable_as import METHODS as COMBO
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation import METHODS as RXE
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_reliable_as import RasStudent
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def batch(cls=NativeAs,smoke=True,base=False):
    model=RasStudent().train();model.fail_As=True;x=torch.randn(6,2,8)
    data=(x,torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[data];model._cd_U_iter=iter([data])
    c=(joint_configuration if base else configuration)(fixture()['2026092701']['F21']['train'],'/p','new')
    c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_native_As_keeps_raw_pseudogate_and_every_original_loss_budget_permission():
    from experiments.pamf_ssl_research.hard_pl_joint import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert not c['hard_pl_no_uas'] and c['hard_pl_native_as']==NATIVE_AS
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95 and c['hard_pl_parent_epoch']==20
    assert c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},{'hard_pl_no_uas':True},
                   {'hard_pl_native_as':dict(NATIVE_AS,augmented_view_gate=.95)},
                   {'hard_pl_native_as':dict(NATIVE_AS,extra_As_teacher=1)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def test_all_native_half_views_are_used_even_when_As_classifier_would_fail():
    torch.manual_seed(101);model,obj,x=batch(GuardedNativeAs);params=tuple(id(p) for p in model.parameters())
    obj.loss(model,'cpu');s=obj.state_dict();q=s['U_native_As']
    assert model.teacher_calls==1 and model.student_calls==6 and model._cd_practical.calls==1 and model._cd_practical.packets==3
    assert q['generated']==3 and q['raw_accepted_half']==2 and q['accepted_student_views']==4
    assert s['physical_seen']==3 and s['physical_student_views']==6 and s['raw_As_seen']==0
    assert q['extra_As_teacher']==0 and q['raw_fallback']==0 and s['accepted']==4
    torch.testing.assert_close(model.student_inputs[2],model.student_inputs[3])
    torch.testing.assert_close(model.student_inputs[2][[0,2,4]],x[[0,2,4]]+.25)
    torch.testing.assert_close(model.student_inputs[4],x);torch.testing.assert_close(model.student_inputs[5],x)
    assert obj.smoke_native_As_matches and obj.smoke_native_nonphysical_ordinary_matches and obj.smoke_joint_signature_gradient>0
    assert model.training and tuple(id(p) for p in model.parameters())==params
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward()
    for name,param in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert param.grad is None
        else:torch.testing.assert_close(param.grad,expected)
    assert all(not m._forward_hooks for m in model.modules())


def test_identity_native_generator_preserves_Joint_loss_and_raw_acceptance():
    torch.manual_seed(103);first,base,_=batch(JointNoAsMixStyle,False,True)
    torch.manual_seed(103);second,native,_=batch(NativeAs,False)
    second._cd_practical=lambda x,*a:x
    torch.manual_seed(104);old=base.loss(first,'cpu');torch.manual_seed(104);new=native.loss(second,'cpu')
    torch.testing.assert_close(old,new)
    assert base.accepted==native.accepted and first.teacher_calls==second.teacher_calls==1


def test_rejected_raw_U_has_zero_label_gradient_even_with_native_As():
    model,obj,_=batch(smoke=False);model.teacher.zero_();loss=obj.loss(model,'cpu');loss.backward()
    assert loss==0 and torch.isfinite(loss) and obj.state_dict()['accepted']==0
    assert obj.state_dict()['U_native_As']['accepted_student_views']==0
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())


def test_closed_segments_count_physical_views_without_inventing_teacher_or_rawfallback():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);save(p/'progress.json',dict(PL_seen=396900,PL_accepted=390000));ss=[state(e,True) for e in (10,20)]
        for s in ss:
            s.update(physical_seen=s['seen']//2,physical_student_views=s['seen'],raw_As_seen=0)
            s['U_native_As']=dict(recipe=deepcopy(NATIVE_AS),calls=s['calls'],generated=s['seen']//2,
                student_physical_views=s['seen'],raw_accepted_half=s['accepted']//2,accepted_student_views=2*(s['accepted']//2),
                raw_fallback=0,extra_As_teacher=0,all_raw_accepted_retained=True,extra_student_forwards=0,
                new_head=False,EMA_updates=0,U_truth_read=False,CE_dose_normalizer=1.25)
        r=aggregate(ss,[dict(folder=str(p))],weight=6.)
        assert r['physical_U_seen']==850500 and r['physical_student_U_views']==1701000 and r['raw_U_As_seen']==0
        assert r['teacher_eval_U_views']==2097900 and r['student_U_views']==10602900 and r['U_native_As']['generated']==850500
        ss[-1]['U_native_As']['extra_As_teacher']=1
        with pytest.raises(ValueError,match='Native As'):aggregate(ss,[dict(folder=str(p))],weight=6.)


def test_release_keeps_all_seeds_readouts_and_single_scorer_after_receiver_extrapolation():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and rt['prior_score_owner_state']==rt['receiver_extrapolation_GPU_owner_state']
    assert len(rt['rows'])==rt['max_total_workers']==3
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_queue_requires_finished_combination_and_reserves_dispatched_extrapolation():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},path_reliable_as_GPU_owner_state='c',receiver_extrapolation_GPU_owner_state='r')
    states={k:dict(jobs={}) for k in ('c','r')}
    for seed in (2026092701,2026092702,2026092703):
        states['c']['jobs'][COMBO[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['r']['jobs'][RXE[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092702)
        states['c']['jobs'][COMBO[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        states['r']['jobs'][RXE[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert slot_ready(rt,2026092702) and smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_matching_seed_real_smoke_precedes_formal_dispatch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_native_as_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_native_as.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_native_as.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_target_report_requires_fullmatrix_and_keeps_raw_goal_separate():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS));rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHODS[0],row_id=METHODS[0]+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            p=Path(c['output_root'])/'scores.json';v=json.loads(p.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(p,v)
        report(rt,root/'report');result=json.loads((root/'report/analysis.json').read_text())
        assert len(result['verdicts'])==2 and {v['method'] for v in result['verdicts']}==set(METHODS) and not result['goal_met']
