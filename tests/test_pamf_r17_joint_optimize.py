import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from experiments.pamf_r17_joint_optimize import design as d,models as m,runtime,teacher as helper,objective
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_unrolled_restore.models import UnrolledRestorer

torch.set_num_threads(2)


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6)
    def features(self,x):return self.encoder(x.flatten(1)).tanh()
    def classify_features(self,x):return self.head(x)


@pytest.mark.parametrize('arm',d.ARMS)
def test_complete_compact_models_and_all_context_prior_parameters_are_trainable(arm):
    model=m.SharedFactorRestorer(arm);runtime.validate_aliases(model)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    assert sum(p.numel() for p in model.frozen.parameters())==1397619
    assert all(p.requires_grad for p in model.parameters())
    model.train();assert model.frozen.training and model.prior.training


@pytest.mark.parametrize('arm',d.ARMS)
def test_joint_loss_updates_coarse_action_prior_but_not_original_teacher_or_F(arm):
    torch.manual_seed(17);original=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for core in original.prior:core.out.weight.normal_(std=.001)
        original.frozen.coarse.temporal.out.weight.normal_(std=.001)
        original.frozen.estimator.core.out.weight.normal_(std=.001)
        original.frozen.estimator.global_head.weight.normal_(std=.001)
    original_before=copy.deepcopy(original.state_dict());student=m.compact_from_teacher(original,arm);student.train()
    before=copy.deepcopy(student.state_dict());reader=Reader().eval().requires_grad_(False);reader_before=copy.deepcopy(reader.state_dict())
    x=torch.randn(2,2,256)*.1;r=x+torch.randn_like(x)*.005
    desired,context=helper.teacher_and_student_context(original,student,r)
    assert all(not value.requires_grad for value in desired) and context[0].requires_grad
    with torch.no_grad():assert all(torch.equal(a,b) for a,b in zip(desired,original.stage_outputs(r)[0]))
    output=student.stage_outputs(r,context)[0];loss,terms=objective.joint_objective(output,desired,x,reader,clean_reference=d.CLEAN_OPT[arm])
    expected=10*terms['teacher_stage_nmse']+terms['task_KL']+terms['task_feature_distance']+float(d.CLEAN_OPT[arm])*terms['iq_nmse']
    assert torch.allclose(loss,expected)
    optimizer=torch.optim.AdamW(student.parameters(),lr=1e-4);loss.mean().backward()
    for group in (student.frozen.coarse,student.frozen.estimator,student.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert all(p.grad is None for p in original.parameters()) and all(p.grad is None for p in reader.parameters())
    optimizer.step()
    for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
        assert any(not torch.equal(value,before[key]) for key,value in student.state_dict().items() if key.startswith(prefix))
    assert all(torch.equal(value,original_before[key]) for key,value in original.state_dict().items())
    assert all(torch.equal(value,reader_before[key]) for key,value in reader.state_dict().items())


@pytest.mark.parametrize('clean_reference',(False,True))
def test_corrective_clean_target_and_direct_IQ_term_are_explicit(monkeypatch,clean_reference):
    clean=torch.ones(2,2,256);desired=[torch.ones_like(clean)*.5 for _ in range(3)];output=[torch.zeros_like(clean) for _ in range(3)]
    def task(student,reference,reader):
        assert reference is (clean if clean_reference else desired[-1])
        return torch.ones(2)*.5,dict(task_KL=torch.ones(2)*.4,task_feature_distance=torch.ones(2)*.1)
    monkeypatch.setattr(objective,'frozen_task_fidelity',task)
    value,terms=objective.joint_objective(output,desired,clean,None,clean_reference=clean_reference)
    assert torch.equal(value,torch.ones(2)*(3.+float(clean_reference)))


def test_teacher_no_grad_does_not_detach_student_context(monkeypatch):
    parameter=nn.Parameter(torch.ones(()));seen=[]
    class TeacherContext:
        def estimate(self,x):assert not torch.is_grad_enabled();seen.append('teacher');return (1,)*3
    class Student:
        def estimate_context(self,x):assert torch.is_grad_enabled();seen.append('student');return (x*parameter,)*3
    def stages(prior,x,context):assert prior=='original' and context==(1,1,1);return ['originaltarget'],None,None,None
    monkeypatch.setattr(helper,'stages',stages)
    desired,context=helper.teacher_and_student_context(SimpleNamespace(frozen=TeacherContext(),prior='original'),Student(),torch.ones(1,2,256))
    context[0].sum().backward();assert parameter.grad==512 and desired==['originaltarget'] and seen==['teacher','student']


def test_objectives_share_architecture_and_budget_but_clean_is_optimized_only_in_corrective_arms():
    assert d.PARAMETERS[d.ARMS[0]]==d.PARAMETERS[d.ARMS[1]] and d.PARAMETERS[d.ARMS[2]]==d.PARAMETERS[d.ARMS[3]]
    assert list(d.CLEAN_OPT.values())==[False,True,False,True]
    assert d.configuration()['cumulative_component_epoch_sum']==120 and d.configuration()['cumulative_serial_training_passes']==80
    assert d.GPU_PRECEDING_RUN.endswith('m4-r07') and 'frozen_coarse_epochs' not in d.configuration()
