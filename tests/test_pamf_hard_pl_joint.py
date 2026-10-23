from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_ssl_research.hard_pl_joint import RUN,METHODS,JOINT,configuration,validate,slot_ready
from experiments.pamf_ssl_research.hard_pl_joint_objective import capture_joint,JointNoAsMixStyle,GuardedJointNoAs
from experiments.pamf_ssl_research.hard_pl_joint_publish import prepare
from experiments.pamf_ssl_research.hard_pl_guard import GUARD,METHODS as OLD
from experiments.pamf_ssl_research.hard_pl_early_train import aggregate
from tests.test_pamf_hard_pl_early import state
from tests.test_pamf_hard_pl_release import fixture
from tests.test_pamf_ssl_goal_counts import save


def test_joint_configuration_preserves_parent_permissions_and_existing_mechanisms():
    from experiments.pamf_ssl_research.hard_pl_guard import validate as old_validate
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new')
    assert c['hard_pl']['threshold']==.95 and c['hard_pl']['weight']==6
    assert c['hard_pl_parent_epoch']==20 and c['hard_pl_continuation_epochs']==30 and c['optimizer_updates']==11100
    assert c['hard_pl_no_uas'] and c['hard_pl_shared_style_plan'] and c['hard_pl_gradient_guard']==GUARD
    with pytest.raises(ValueError):old_validate(c)
    for change in ({'target_inputs':'x'},{'posthoc_head_fit':True},{'hard_pl_parent_epoch':30},
                   {'hard_pl_joint_contrast':dict(JOINT,weight=.1)},{'hard_pl_joint_contrast':dict(JOINT,new_head=True)}):
        bad=deepcopy(c);bad.update(change)
        with pytest.raises(ValueError):validate(bad)


@pytest.mark.parametrize('instance_method',[False,True])
def test_capture_restores_exact_reference_method_on_success_and_exception(instance_method):
    class Reference(nn.Module):
        def features(self,x):return x.repeat(1,80)
    model=Reference()
    if instance_method:model.features=MethodType(lambda self,x:x.repeat(1,80)+1,model)
    before=model.__dict__.get('features');x=torch.randn(4,2)
    with capture_joint(model) as values:
        output=model.features(x)
    assert len(values)==1 and values[0] is output and model.__dict__.get('features') is before
    with pytest.raises(RuntimeError,match='source failure'):
        with capture_joint(model):
            model.features(x)
            raise RuntimeError('source failure')
    assert model.__dict__.get('features') is before
    with pytest.raises(ValueError,match='Exactly one'):
        with capture_joint(model):pass
    assert model.__dict__.get('features') is before


class DirectHead(nn.Linear):
    def classify(self,x):return F.linear(x,self.weight,self.bias)


class JointReference(nn.Module):
    def __init__(self):
        super().__init__();self.core=nn.Module();core=nn.Module();self.core.id_backbone=core
        core.time_down=nn.Conv1d(2,4,1);core.t2=nn.Module();core.t2.pool=nn.Conv1d(4,4,1)
        core.projection=nn.Linear(4,160);core.cls_head=DirectHead(160,6)
        self.sequence_pool=nn.Linear(4,160)
    def features(self,x):
        core=self.core.id_backbone;a=core.time_down(x);h=core.t2.pool(a)
        return core.projection(h.mean(-1))+.2*self.sequence_pool(a.square().mean(-1))


class Student(nn.Module):
    def __init__(self):
        super().__init__();self.id_backbone=nn.Module();self.id_backbone.encoder=JointReference()
        self._cd_U_seen=self._cd_U_accepted=self.student_calls=self.teacher_calls=0;self._feature_l_steps=1
        self.teacher=torch.tensor([[10.,0,0,0,0,0],[10.,0,0,0,0,0],[0.,10,0,0,0,0],
                                   [0.,10,0,0,0,0],[0.,0,0,0,0,0],[0.,0,0,0,0,0]])
        self._cd_practical=lambda *args:(_ for _ in ()).throw(AssertionError('U physical generator forbidden'))
    def forward(self,x):
        assert not self.training and 'features' not in self.id_backbone.encoder.__dict__
        self.teacher_calls+=1;return self.teacher
    def forward_identity_only(self,x,domain_labels=None):
        self.student_calls+=1;ref=self.id_backbone.encoder
        return dict(tx_logits=ref.core.id_backbone.cls_head.classify(ref.features(x)))


def objective_batch(cls,smoke=True):
    model=Student().train();x=torch.randn(6,2,8)
    batch=(x,torch.full((6,),-1),torch.zeros(6,dtype=torch.long),dict(rx_i=torch.tensor([1,3]*3)))
    model._cd_U_loader=[batch];model._cd_U_iter=iter([batch])
    c=configuration(fixture()['2026092701']['F21']['train'],'/p','new');c.update(hard_pl_smoke=smoke,source_rxs=[1,3],step_offset=4440)
    return model,cls(c),x


def test_actual_joint_features_reach_signature_with_no_new_forward_or_head():
    torch.manual_seed(49);model,obj,x=objective_batch(GuardedJointNoAs);head_calls=[]
    handle=model.id_backbone.encoder.core.id_backbone.cls_head.register_forward_hook(lambda *args:head_calls.append(1))
    zero=obj.loss(model,'cpu')
    assert model.teacher_calls==1 and model.student_calls==6 and zero==0 and not head_calls
    assert obj.smoke_joint_signature_gradient>0 and len(obj.smoke_joint_features)==2
    assert all(h.shape==(6,160) and torch.isfinite(h.grad).all() and h.grad.norm()>0 for h in obj.smoke_joint_features)
    assert all(p.grad is None for p in model.parameters())
    labeled=F.cross_entropy(model.forward_identity_only(x)['tx_logits'],torch.tensor([0,0,1,1,2,2]))
    corrected=obj.guide(labeled,model);corrected.backward();handle.remove()
    assert all(torch.isfinite(h.grad).all() and h.grad.norm()>0 for h in obj.smoke_joint_features)
    for name,param in model.named_parameters():
        torch.testing.assert_close(param.grad,obj.smoke_guard_expected[name])
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.id_backbone.encoder.sequence_pool.parameters())
    status=obj.state_dict();joint=status['U_joint_contrast']
    assert joint['calls']==1 and joint['captures']==2 and joint['captured_packets']==12 and joint['anchors']==8
    assert status['student_views']==36 and status['U_style']['plan_calls']==1 and status['physical_seen']==0
    assert not joint['extra_forward_calls'] and not joint['teacher_or_L_captures'] and not joint['persistent_feature_wrapper']
    assert 'features' not in model.id_backbone.encoder.__dict__ and not obj.joint_active
    assert all(not m._forward_hooks for m in model.modules())


def test_all_rejected_joint_has_no_U_gradient_or_nonfinite_empty_statistics():
    model,obj,x=objective_batch(JointNoAsMixStyle,smoke=False);model.teacher.zero_()
    loss=obj.loss(model,'cpu');loss.backward()
    assert torch.isfinite(loss) and loss==0 and obj.state_dict()['U_joint_contrast']['anchors']==0
    assert all(p.grad is None or not p.grad.count_nonzero() for p in model.parameters())
    assert 'features' not in model.id_backbone.encoder.__dict__


def test_both_native_segments_account_for_reused_joint_packets():
    with tempfile.TemporaryDirectory() as tmp:
        parent=Path(tmp);save(parent/'progress.json',dict(PL_seen=396900,PL_accepted=390000))
        values=[state(e,True) for e in (10,20)]
        for value in values:
            value['U_gradient_guard']=dict(recipe=deepcopy(GUARD),calls=value['calls'],pending_U_loss=False,
                conflicts={g:1 for g in GUARD['groups']},caps={g:2 for g in GUARD['groups']})
            value['U_joint_contrast']=dict(recipe=deepcopy(JOINT),calls=value['calls'],captures=2*value['calls'],
                captured_packets=2*value['seen'],anchors=2*value['accepted'],extra_forward_calls=0,
                teacher_or_L_captures=0,persistent_feature_wrapper=False)
        result=aggregate(values,[dict(folder=str(parent))],weight=6.)
        joint=result['U_joint_contrast']
        assert joint['calls']==6660 and joint['captures']==13320 and joint['captured_packets']==3402000
        assert result['student_U_views']==10602900 and result['hard_PL_tail']['U_joint_contrast']==joint
        values[1]['U_joint_contrast']['captured_packets']+=1
        with pytest.raises(ValueError,match='reuse exactly two'):aggregate(values,[dict(folder=str(parent))],weight=6.)


def test_three_rows_keep_both_readouts_and_predecessor_scoring_chain():
    rt,blobs=prepare(fixture(),'/p','/p/releases/'+RUN,'new')
    assert rt['methods']==list(METHODS) and len(rt['rows'])==rt['max_total_workers']==3
    assert rt['prior_score_owner_state']==rt['guard_GPU_owner_state']
    for row in rt['rows']:
        stem='configs/'+row['row_id'];c,p,s=(blobs[stem+'-'+k+'.json'] for k in ('train','predict','score'))
        assert validate(c) is c and c['run_id']==p['source_run_id']==RUN and p['source_output']==c['output_root']
        assert list(p['readouts'])==s['readouts']==['raw','DURA-LU'] and 'truth' not in p['tests']['main']


def test_slot_waits_for_LS_GPU_release_and_NAMS_dispatch_including_starting_PID():
    from experiments.pamf_ssl_research.control import pending_gpu_pids,capacity
    seed=2026092701;rt=dict(gpu_by_seed={str(seed):5},guard_GPU_owner_state='old')
    jobs={m+'-s'+str(seed):dict(status='RUNNING',pid=99+i) for i,m in enumerate(OLD)}
    with patch('experiments.cric_dura_comparison.io.read',return_value=dict(jobs=jobs)):
        assert not slot_ready(rt,seed)
        jobs[OLD[0]+'-s'+str(seed)]['status']='PREDICTIONS_COMPLETE'
        assert slot_ready(rt,seed) and not slot_ready(rt,2026092702)
        active=[dict(gpu=5,child=SimpleNamespace(pid=101))]
        assert pending_gpu_pids(rt,METHODS,seed,active)=={100,101}
        with patch('experiments.pamf_ssl_research.control.subprocess.check_output',side_effect=['12\n13\n','19000']):
            assert capacity(5,pending_gpu_pids(rt,METHODS,seed))==(3,19000)
        jobs[OLD[1]+'-s'+str(seed)]['status']='QUEUED'
        assert not slot_ready(rt,seed)


def test_real_joint_smoke_precedes_formal_launch():
    from experiments.pamf_ssl_research.control import run
    class ReachedSmoke(Exception):pass
    with tempfile.TemporaryDirectory() as tmp:
        rt,_=prepare(fixture(),'/p',tmp,'new');rt.update(run_root=tmp+'/output',runtime_file=tmp+'/runtime.json')
        def reached(command,**kwargs):
            assert 'experiments.pamf_ssl_research.hard_pl_joint_worker' in command
            raise ReachedSmoke
        with patch('experiments.pamf_ssl_research.hard_pl_joint.slot_ready',return_value=True), \
             patch('experiments.pamf_ssl_research.hard_pl_joint.reserved_predecessor_pids',return_value=(99,)), \
             patch('experiments.pamf_ssl_research.control.capacity',return_value=(2,19000)), \
             patch('experiments.pamf_ssl_research.control.subprocess.run',side_effect=reached), \
             patch('experiments.pamf_ssl_research.control.subprocess.Popen') as launch:
            with pytest.raises(ReachedSmoke):run(rt)
            launch.assert_not_called()


def test_complete_joint_report_preserves_raw_goal_and_DURA_reference():
    from experiments.pamf_ssl_research.dura_reference_report import report
    from tests.test_pamf_ssl_dura_reference import report_fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS))
        rt['rows']=[r for r in rt['rows'] if r['method']=='PSM']
        for row in rt['rows']:
            row.update(method=METHODS[0],row_id=METHODS[0]+'-s'+str(row['model_seed']))
            c=json.loads(Path(row['score_config']).read_text());c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
            path=Path(c['output_root'])/'scores.json';v=json.loads(path.read_text());v.update(run_id=RUN,row_id=row['row_id']);save(path,v)
        report(rt,root/'report');value=json.loads((root/'report/analysis.json').read_text())
        assert len(value['verdicts'])==2 and {v['method'] for v in value['verdicts']}==set(METHODS)
        assert not value['goal_met']
