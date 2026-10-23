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
from experiments.pamf_ssl_research.hard_pl_reliable_as import RUN,METHODS,RELIABLE_AS,configuration,validate,slot_ready,smoke_slot
from experiments.pamf_ssl_research.hard_pl_reliable_as_objective import ReliableAs,GuardedReliableAs,view_mask
from experiments.pamf_ssl_research.hard_pl_reliable_as_publish import prepare
from experiments.pamf_ssl_research.hard_pl_joint import configuration as joint_configuration,JOINT
from experiments.pamf_ssl_research.hard_pl_joint_objective import JointNoAsMixStyle
from experiments.pamf_ssl_research.hard_pl_complement import METHODS as COMP
from experiments.pamf_ssl_research.hard_pl_path import METHODS as PATH
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_joint import Student
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_fixed_reliable_As_parent_permissions_gate_and_unchanged_loss_recipe():
    from experiments.pamf_ssl_research.hard_pl_joint import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    assert c['hard_pl']['threshold']==.95 and c['hard_pl']['weight']==6 and c['hard_pl_joint_contrast']==JOINT
    assert not c['hard_pl_no_uas'] and RELIABLE_AS['CE_and_KL_coefficients_unchanged']
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},{'hard_pl_no_uas':True},
                   {'hard_pl_reliable_as':dict(RELIABLE_AS,augmented_view_gate=.8)},{'hard_pl_reliable_as':dict(RELIABLE_AS,new_pseudo_labels=True)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


def test_view_gate_requires_original_acceptance_same_label_and_inclusive_confidence():
    q=torch.full((4,6),.01,dtype=torch.float64);q[0,0]=.95;q[1,0]=.94;q[2,1]=.99;q[3,0]=.99
    actual=view_mask(q,torch.zeros(4,dtype=torch.long),torch.tensor([True,True,True,False]),.95)
    assert actual.tolist()==[True,False,False,False]
    with pytest.raises(ValueError):view_mask(q,torch.zeros(4,dtype=torch.long),torch.ones(4),.95)
    with pytest.raises(ValueError):view_mask(q*float('nan'),torch.zeros(4,dtype=torch.long),torch.ones(4,dtype=torch.bool),.95)


class Generator:
    def __init__(self):self.calls=self.packets=0
    def __call__(self,x,*args):self.calls+=1;self.packets+=len(x);return x+.25


class RasStudent(Student):
    def __init__(self):
        super().__init__();self.eval_norm=nn.BatchNorm1d(2);self._cd_practical=Generator();self.student_inputs=[];self.fail_As=False
        self.As_teacher=torch.tensor([[10.,0,0,0,0,0],[0.,0,10,0,0,0],[10.,0,0,0,0,0]])
    def forward(self,x):
        assert not self.training and 'features' not in self.id_backbone.encoder.__dict__
        self.teacher_calls+=1;self.eval_norm(x)
        if len(x)==6:return self.teacher
        if self.fail_As:raise RuntimeError('source As eval failure')
        assert len(x)==3;return self.As_teacher
    def forward_identity_only(self,x,domain_labels=None):
        self.student_inputs.append(x.detach().clone());return super().forward_identity_only(x,domain_labels)


def objective_batch(cls=ReliableAs,smoke=True,joint=False):
    model=RasStudent().train();x=torch.randn(6,2,8)
    batch=(x,torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[batch];model._cd_U_iter=iter([batch]);template=fixture()['2026092701']['F21']['train']
    c=(joint_configuration if joint else configuration)(template,'/p','new');c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_actual_half_view_uses_As_or_exactraw_without_dropping_any_raw_accepted_U():
    torch.manual_seed(71);model,obj,x=objective_batch(GuardedReliableAs);params=tuple(id(p) for p in model.parameters())
    norm={k:v.clone() for k,v in model.eval_norm.state_dict().items()};obj.loss(model,'cpu');r=obj.state_dict()['U_reliable_As']
    assert model.teacher_calls==2 and model.student_calls==6 and model._cd_practical.calls==1 and model._cd_practical.packets==3
    assert (r['generated'],r['teacher_As_views'],r['used'],r['raw_fallback'],r['raw_accepted_half'])==(3,3,1,2,2)
    assert obj.state_dict()['accepted']==4 and obj.state_dict()['physical_student_views']==2 and obj.state_dict()['raw_As_seen']==2
    assert model.training and tuple(id(p) for p in model.parameters())==params
    for key,value in norm.items():torch.testing.assert_close(model.eval_norm.state_dict()[key],value)
    torch.testing.assert_close(model.student_inputs[2],model.student_inputs[3])
    torch.testing.assert_close(model.student_inputs[2][0],x[0]+.25)
    torch.testing.assert_close(model.student_inputs[2][[2,4]],x[[2,4]])
    torch.testing.assert_close(model.student_inputs[4],x);torch.testing.assert_close(model.student_inputs[5],x)
    assert obj.smoke_joint_signature_gradient>0 and obj.smoke_ras_exact_fallback and obj.smoke_ras_eval_norm_unchanged
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    obj.guide(labeled,model).backward()
    for name,param in model.named_parameters():
        expected=obj.smoke_guard_expected[name]
        if expected is None:assert param.grad is None
        else:torch.testing.assert_close(param.grad,expected)
    assert all(not m._forward_hooks for m in model.modules()) and not obj.ras_active


def test_all_view_rejected_fallback_matches_joint_loss_and_full_U_exposure():
    torch.manual_seed(73);first,base,_=objective_batch(JointNoAsMixStyle,smoke=False,joint=True)
    torch.manual_seed(73);second,ras,_=objective_batch(ReliableAs,smoke=False);second.As_teacher.zero_()
    torch.manual_seed(74);old=base.loss(first,'cpu')
    torch.manual_seed(74);new=ras.loss(second,'cpu')
    torch.testing.assert_close(new,old)
    assert first.student_calls==second.student_calls==6 and first._cd_practical.calls==0 and second._cd_practical.calls==1
    assert ras.state_dict()['U_reliable_As']['used']==0 and ras.state_dict()['U_reliable_As']['raw_fallback']==3
    assert base.state_dict()['accepted']==ras.state_dict()['accepted']==4


def test_all_raw_rejected_still_falls_back_and_has_no_U_parameter_gradient():
    model,obj,_=objective_batch(smoke=False);model.teacher.zero_();loss=obj.loss(model,'cpu');loss.backward()
    assert loss==0 and torch.isfinite(loss) and obj.state_dict()['accepted']==0
    assert obj.state_dict()['U_reliable_As']['used']==0 and obj.state_dict()['raw_As_seen']==3
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())


def test_failed_extra_eval_restores_training_and_transient_feature_contexts():
    model,obj,_=objective_batch(smoke=False);model.fail_As=True
    with pytest.raises(RuntimeError,match='source As eval failure'):obj.loss(model,'cpu')
    assert model.training and not obj.ras_active and not obj.style_active and not obj.joint_active
    assert all(not m._forward_hooks for m in model.modules()) and 'features' not in model.id_backbone.encoder.__dict__


def test_two_segments_report_actual_As_use_fallback_and_total_teacher_cost():
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000));values=[state(e,True) for e in (10,20)]
        for s in values:
            generated=s['seen']//2;used=generated//2;s.update(physical_seen=generated,physical_student_views=2*used,raw_As_seen=generated-used)
            s['U_reliable_As']=dict(recipe=deepcopy(RELIABLE_AS),calls=s['calls'],generated=generated,teacher_As_views=generated,
                used=used,raw_fallback=generated-used,raw_accepted_half=generated,view_confident=generated,view_argmax_agree=used,
                all_raw_accepted_retained=True,extra_student_forwards=0,new_head=False,EMA_updates=0,U_truth_read=False,teacher_eval_norm_updates=0)
        r=aggregate(values,[dict(folder=str(parent))],weight=6.)
        assert r['physical_U_seen']==850500 and r['teacher_eval_U_raw_views']==2097900 and r['teacher_eval_U_As_views']==850500
        assert r['teacher_eval_U_views']==2948400 and r['student_U_views']==10602900
        assert r['U_reliable_As']['raw_fallback']==425250 and r['hard_PL_tail']['U_reliable_As']==r['U_reliable_As']
        values[-1]['U_reliable_As']['all_raw_accepted_retained']=False
        with pytest.raises(ValueError,match='retain all'):aggregate(values,[dict(folder=str(parent))],weight=6.)


def test_reliable_As_release_has_same_defaulthead_budget_both_readouts_and_chained_single_scorer():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['path_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and p['source_output']==c['output_root'] and p['source_run_id']==RUN
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_reliable_As_waits_for_completed_Complement_GPU_work_and_reserves_Path_PID():
    from experiments.pamf_ssl_research.control import pending_gpu_pids
    rt=dict(gpu_by_seed={'2026092701':5,'2026092702':6,'2026092703':7},complement_GPU_owner_state='c',path_GPU_owner_state='p')
    states={k:dict(jobs={}) for k in ('c','p')}
    for seed in (2026092701,2026092702,2026092703):
        states['c']['jobs'][COMP[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed)
        states['p']['jobs'][PATH[0]+'-s'+str(seed)]=dict(status='RUNNING',pid=seed+10)
    with patch('experiments.cric_dura_comparison.io.read',side_effect=lambda p:states[p]):
        assert not slot_ready(rt,2026092701)
        assert pending_gpu_pids(rt,METHODS,2026092701,[dict(gpu=5,child=SimpleNamespace(pid=99))])=={99,2026092701,2026092711}
        states['c']['jobs'][COMP[0]+'-s2026092702']['status']='PREDICTIONS_COMPLETE'
        assert slot_ready(rt,2026092702) and smoke_slot(rt,lambda *args:(2,18000))==(2026092702,6)
        assert smoke_slot(rt,lambda *args:(3,18000)) is None


def test_reliable_As_smoke_uses_first_matching_released_seed_card_before_formal_launch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_reliable_as_worker' in command and kwargs['env']['CUDA_VISIBLE_DEVICES']=='6'
            c=json.loads(Path(command[-1]).read_text());assert c['rows'][0]['model_seed']==2026092702
            assert {r['model_seed']:r['gpu'] for r in c['rows']}=={2026092701:5,2026092702:6,2026092703:7}
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_reliable_as.smoke_slot',return_value=(2026092702,6)), \
             patch('experiments.pamf_ssl_research.hard_pl_reliable_as.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,18000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_reliable_As_report_keeps_raw_goal_separate_from_DURA_and_both_matched_baselines():
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
