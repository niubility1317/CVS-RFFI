from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation import RUN,METHODS,RECEIVER_EXTRAPOLATION,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation,GuardedReceiverExtrapolation
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_publish import prepare
from experiments.pamf_ssl_research.hard_pl_joint import configuration as joint_configuration
from experiments.pamf_ssl_research.hard_pl_joint_objective import JointNoAsMixStyle
from experiments.pamf_ssl_research.hard_pl_reliable_as import METHODS as RAS
from experiments.pamf_ssl_research.hard_pl_path_reliable_as import METHODS as COMBO
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_path import objective_batch as native_batch
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def batch(cls=ReceiverExtrapolation,smoke=True,base=False,same_rx=False):
    model,_,x=native_batch(JointNoAsMixStyle,smoke=smoke)
    if same_rx:model._cd_U_loader[0][3]['rx_i'].fill_(1)
    c=(joint_configuration if base else configuration)(fixture()['2026092701']['F21']['train'],'/p','new')
    c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_source_fixed_scale_full_features_permissions_parent_and_CE_dose():
    from experiments.pamf_ssl_research.hard_pl_joint import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    assert c['hard_pl_no_uas'] and c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95
    assert RECEIVER_EXTRAPOLATION['alpha']==.25
    assert RECEIVER_EXTRAPOLATION['ordinary_CE_coefficient']+RECEIVER_EXTRAPOLATION['extrapolated_CE_coefficient']==.5
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},{'hard_pl_no_uas':False},
                   {'hard_pl_receiver_extrapolation':dict(RECEIVER_EXTRAPOLATION,alpha=.5)},
                   {'hard_pl_receiver_extrapolation':dict(RECEIVER_EXTRAPOLATION,extrapolated_CE_coefficient=.5)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def test_full_feature_new_CE_trains_all_existing_branches_and_exact_fullL_guard():
    torch.manual_seed(91);model,obj,x=batch(GuardedReceiverExtrapolation);params=tuple(id(p) for p in model.parameters())
    obj.loss(model,'cpu');s=obj.state_dict();q=s['U_receiver_extrapolation']
    assert model.teacher_calls==1 and model.student_calls==6 and tuple(id(p) for p in model.parameters())==params
    assert min(obj.smoke_rxe_gradients.values())>0 and obj.smoke_joint_signature_gradient>0
    assert q['calls']==1 and q['extra_head_calls']==2 and q['extra_head_packet_views']==12
    assert 0<q['eligible_packet_views']<=2*s['accepted'] and q['all_raw_accepted_retained']
    assert s['student_views']==36 and s['physical_seen']==0 and s['raw_As_seen']==3
    assert q['full_deployment_signature_retained'] and q['full_deployment_response_retained']
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward()
    for name,param in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert param.grad is None
        else:torch.testing.assert_close(param.grad,expected)
    assert obj._rxe_current is None and all(not m._forward_hooks for m in model.modules())


def test_zero_displacement_conserves_old_CE_value_instead_of_adding_dose():
    torch.manual_seed(93);first,base,_=batch(JointNoAsMixStyle,False,True)
    torch.manual_seed(93);second,new,_=batch(ReceiverExtrapolation,False)
    torch.manual_seed(94);a=base.loss(first,'cpu')
    with patch('experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective.extrapolate',side_effect=lambda h,*args:h):
        torch.manual_seed(94);b=new.loss(second,'cpu')
    torch.testing.assert_close(a,b)
    assert first.teacher_calls==second.teacher_calls==1 and first.student_calls==second.student_calls==6


def test_unpaired_accepted_samples_keep_original_loss_and_are_not_dropped():
    torch.manual_seed(95);first,base,_=batch(JointNoAsMixStyle,False,True,True)
    torch.manual_seed(95);second,new,_=batch(ReceiverExtrapolation,False,False,True)
    torch.manual_seed(96);a=base.loss(first,'cpu');torch.manual_seed(96);b=new.loss(second,'cpu')
    torch.testing.assert_close(a,b)
    assert new.state_dict()['accepted']==base.state_dict()['accepted']>0
    assert new.state_dict()['U_receiver_extrapolation']['eligible_packet_views']==0


def test_raw_rejected_U_has_zero_gradient_and_failed_head_cleans_transient_graphs():
    model,obj,_=batch(smoke=False);model.teacher.zero_();loss=obj.loss(model,'cpu');loss.backward()
    assert loss==0 and all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())
    model,obj,_=batch(smoke=False)
    with patch.object(model.id_backbone.encoder,'classify_features',side_effect=RuntimeError('existing head failure')):
        with pytest.raises(RuntimeError,match='existing head failure'):obj.loss(model,'cpu')
    assert obj._rxe_current is None and obj._joint_current is None and all(not m._forward_hooks for m in model.modules())


def test_two_native_segments_keep_actual_extra_head_cost_and_eligible_bounds():
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000));ss=[state(e,True) for e in (10,20)]
        for s in ss:s['U_receiver_extrapolation']=dict(recipe=deepcopy(RECEIVER_EXTRAPOLATION),calls=s['calls'],extra_head_calls=2*s['calls'],
            extra_head_packet_views=2*s['seen'],eligible_packet_views=2*s['accepted'],extra_backbone_forwards=0,teacher_or_L_head_calls=0,
            new_head=False,persistent_hooks=False,all_raw_accepted_retained=True,full_deployment_signature_retained=True,full_deployment_response_retained=True,CE_dose_normalizer=1.25)
        r=aggregate(ss,[dict(folder=str(parent))],weight=6.)
        assert r['U_receiver_extrapolation']['extra_head_calls']==13320 and r['U_receiver_extrapolation']['extra_head_packet_views']==3402000
        assert r['student_U_views']==10602900 and r['teacher_eval_U_views']==2097900 and r['physical_U_seen']==0
        ss[-1]['U_receiver_extrapolation']['eligible_packet_views']=2*ss[-1]['accepted']+1
        with pytest.raises(ValueError,match='Full-feature'):aggregate(ss,[dict(folder=str(parent))],weight=6.)


def test_release_keeps_all3seeds_original_head_both_readouts_and_scorer_after_combination():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and rt['prior_score_owner_state']==rt['path_reliable_as_GPU_owner_state']
    assert len(rt['rows'])==rt['max_total_workers']==3
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_slot_requires_finished_reliableAs_predictions_and_dispatched_combination():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},reliable_as_GPU_owner_state='r',path_reliable_as_GPU_owner_state='c')
    states={k:dict(jobs={}) for k in ('r','c')}
    for seed in (2026092701,2026092702,2026092703):
        states['r']['jobs'][RAS[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['c']['jobs'][COMBO[0]+'-s'+str(seed)]=dict(status='QUEUED')
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092702)
        states['r']['jobs'][RAS[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert not slot_ready(rt,2026092702)
        states['c']['jobs'][COMBO[0]+'-s2026092702'].update(status='RUNNING',pid=123)
        assert slot_ready(rt,2026092702) and smoke_slot(rt,lambda *a:(2,18000))==(2026092702,6)
        assert pending_gpu_pids(rt,METHODS,2026092702,[dict(gpu=6,child=SimpleNamespace(pid=99))])=={123,99}
        assert smoke_slot(rt,lambda *a:(3,18000)) is None


def test_first_eligible_matching_seed_smoke_must_precede_any_formal_child_launch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            assert {r['model_seed']:r['gpu'] for r in c['rows']}=={2026092701:5,2026092702:6,2026092703:7}
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_receiver_extrapolation.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_receiver_extrapolation.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_complete_report_keeps_both_control_methods_and_raw_goal_independent_of_DURA():
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
