from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_complement import RUN,METHODS,COMPLEMENT,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_complement_objective import Complement,GuardedComplement
from experiments.pamf_ssl_research.hard_pl_complement_publish import prepare
from experiments.pamf_ssl_research.hard_pl_joint import METHODS as JOINT,JOINT as JOINT_RECIPE
from experiments.pamf_ssl_research.hard_pl_guard import METHODS as GUARD_METHODS,GUARD
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from experiments.pamf_ssl_research.hard_pl_joint_objective import JointNoAsMixStyle
from tests.test_pamf_hard_pl_joint import Student
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_same_parent_head_and_CE_dose_are_fixed_with_negative_permissions():
    from experiments.pamf_ssl_research.hard_pl_joint import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    assert c['hard_pl']['weight']==6 and c['hard_pl']['threshold']==.95 and c['hard_pl_joint_contrast']==JOINT_RECIPE
    assert COMPLEMENT['ordinary_CE_coefficient']+COMPLEMENT['masked_CE_coefficient']==COMPLEMENT['ordinary_CE_coefficient_before']
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},
                   {'hard_pl_complement':dict(COMPLEMENT,masked_CE_coefficient=.5)},{'hard_pl_complement':dict(COMPLEMENT,new_head=True)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def objective_batch(cls,smoke=True,zero_signature=False):
    model=Student().train();ref=model.id_backbone.encoder
    class NativeContribution(torch.nn.Module):
        def __init__(self,linear):super().__init__();self.linear=linear
        def forward(self,x):return .2*self.linear(x)
    # Actual CompositePool returns the scaled additive contribution;CompositeReference adds it directly.
    ref.sequence_pool=NativeContribution(ref.sequence_pool)
    class NativeReference(type(ref)):
        def features(self,x):
            core=self.core.id_backbone;a=core.time_down(x);h=core.t2.pool(a)
            return core.projection(h.mean(-1))+self.sequence_pool(a.square().mean(-1))
        def classify_features(self,h):return self.core.id_backbone.cls_head.classify(h)
    ref.__class__=NativeReference
    if zero_signature:
        with torch.no_grad():
            for param in ref.sequence_pool.parameters():param.zero_()
    x=torch.randn(6,2,8);batch=(x,torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[batch];model._cd_U_iter=iter([batch])
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new');c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_real_mask_bypasses_signature_but_trains_core_and_same_head_through_guard():
    torch.manual_seed(51);model,obj,x=objective_batch(GuardedComplement);head=model.id_backbone.encoder.core.id_backbone.cls_head
    before=tuple(id(p) for p in model.parameters());obj.loss(model,'cpu')
    assert model.teacher_calls==1 and model.student_calls==6 and tuple(id(p) for p in model.parameters())==before
    assert obj.smoke_complement_gradients['signature']<=1e-7 and obj.smoke_complement_gradients['head']>0 and obj.smoke_complement_gradients['core']>0
    assert obj.smoke_joint_signature_gradient>0 and model.id_backbone.encoder.core.id_backbone.cls_head is head
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward()
    for name,param in model.named_parameters():torch.testing.assert_close(param.grad,obj.smoke_guard_expected[name])
    status=obj.state_dict();comp=status['U_complement']
    assert comp['calls']==1 and comp['extra_head_calls']==2 and comp['extra_head_packet_views']==12 and comp['CE_dose_normalizer']==1.25
    assert status['student_views']==36 and status['U_style']['calls']==6 and status['U_joint_contrast']['captures']==2
    assert all(h.grad is not None and torch.isfinite(h.grad).all() and h.grad.norm()>0 for h in obj.smoke_complement_features)
    assert all(not m._forward_hooks for m in model.modules()) and not obj.complement_active


def test_zero_signature_view_preserves_objective_value_instead_of_adding_CE_dose():
    torch.manual_seed(53);first,base,_=objective_batch(JointNoAsMixStyle,smoke=False,zero_signature=True)
    torch.manual_seed(53);second,comp,_=objective_batch(Complement,smoke=False,zero_signature=True)
    torch.manual_seed(54);old=base.loss(first,'cpu')
    torch.manual_seed(54);new=comp.loss(second,'cpu')
    torch.testing.assert_close(new,old)
    torch.testing.assert_close(comp.metrics['HPL_complement_masked_CE'],comp.metrics['HPL_complement_ordinary_CE'])
    assert comp.state_dict()['U_complement']['extra_head_calls']==2 and first.student_calls==second.student_calls==6


def test_all_rejected_complement_stays_zero_with_no_U_parameter_gradient():
    model,obj,_=objective_batch(Complement,smoke=False);model.teacher.zero_()
    loss=obj.loss(model,'cpu');loss.backward()
    assert loss==0 and torch.isfinite(loss) and obj.state_dict()['accepted']==0
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())


def test_two_segments_count_only_extra_existing_head_calls_not_backbone_views():
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000));values=[state(e,True) for e in (10,20)]
        for s in values:
            s['U_gradient_guard']=dict(recipe=deepcopy(GUARD),calls=s['calls'],pending_U_loss=False,
                conflicts={g:1 for g in GUARD['groups']},caps={g:1 for g in GUARD['groups']})
            s['U_joint_contrast']=dict(recipe=deepcopy(JOINT_RECIPE),calls=s['calls'],captures=2*s['calls'],captured_packets=2*s['seen'],
                anchors=2*s['accepted'],extra_forward_calls=0,teacher_or_L_captures=0,persistent_feature_wrapper=False)
            s['U_complement']=dict(recipe=deepcopy(COMPLEMENT),calls=s['calls'],extra_head_calls=2*s['calls'],extra_head_packet_views=2*s['seen'],
                extra_backbone_forwards=0,teacher_or_L_head_calls=0,new_head=False,persistent_hooks=False,full_deployment_signature_retained=True,CE_dose_normalizer=1.25)
        result=aggregate(values,[dict(folder=str(parent))],weight=6.)
        assert result['U_complement']['extra_head_calls']==13320 and result['U_complement']['extra_head_packet_views']==3402000
        assert result['student_U_views']==10602900 and result['hard_PL_tail']['U_complement']==result['U_complement']
        values[-1]['U_complement']['extra_backbone_forwards']=1
        with pytest.raises(ValueError,match='same dose'):aggregate(values,[dict(folder=str(parent))],weight=6.)


def test_complete_release_keeps_three_seeds_and_both_readouts_after_Joint_scorer():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['joint_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_resource_handoff_reserves_both_starting_peers_and_uses_first_matching_ready_card():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},guard_GPU_owner_state='g',joint_GPU_owner_state='j')
    states={k:dict(jobs={}) for k in ('g','j')}
    for seed in (2026092701,2026092702,2026092703):
        states['g']['jobs'][GUARD_METHODS[1]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['j']['jobs'][JOINT[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed+10)
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092701)
        assert pending_gpu_pids(rt,METHODS,2026092701,[dict(gpu=5,child=SimpleNamespace(pid=99))])=={99,2026092701,2026092711}
        states['g']['jobs'][GUARD_METHODS[1]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert slot_ready(rt,2026092702)
        calls=[]
        def capacity(gpu,pending):calls.append((gpu,pending));return 2,18000
        assert smoke_slot(rt,capacity)==(2026092702,6) and calls==[(6,(2026092712,))]
        assert smoke_slot(rt,lambda *args:(3,18000)) is None


def test_first_ready_seed_smoke_keeps_its_GPU_and_precedes_any_formal_Popen():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_complement_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            assert {r['model_seed']:r['gpu'] for r in c['rows']}=={2026092701:5,2026092702:6,2026092703:7}
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_complement.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_complement.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_full_comp_report_preserves_original_raw_goal_separate_from_DURA():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS));rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHODS[0],row_id=METHODS[0]+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';value=json.loads(path.read_text());value.update(run_id=RUN,row_id=row['row_id']);save(path,value)
        report(rt,root/'report');result=json.loads((root/'report/analysis.json').read_text())
        assert len(result['verdicts'])==2 and {v['method'] for v in result['verdicts']}==set(METHODS) and not result['goal_met']
