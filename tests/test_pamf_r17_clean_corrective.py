import copy
from pathlib import Path
import pytest
import torch
from torch import nn
from experiments.pamf_r17_clean_corrective import design as d,models as m,runtime,objective
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_unrolled_restore.models import UnrolledRestorer

torch.set_num_threads(2)


class Reader(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(512,160);self.head=nn.Linear(160,6)
    def features(self,x):return self.encoder(x.flatten(1)).tanh()
    def classify_features(self,x):return self.head(x)


def parent_model():
    model=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for core in model.prior:core.out.weight.normal_(std=.001)
        model.frozen.coarse.temporal.out.weight.normal_(std=.001)
        model.frozen.estimator.core.out.weight.normal_(std=.001)
        model.frozen.estimator.global_head.weight.normal_(std=.001)
    return model


@pytest.mark.parametrize('arm',d.ARMS)
def test_compact_context_prior_all_receive_gradients_and_update_without_parent_or_F_updates(arm):
    torch.manual_seed(17);original=parent_model();original_before=copy.deepcopy(original.state_dict())
    student=m.compact_from_teacher(original,arm);student.train();runtime.validate_aliases(student)
    assert sum(p.numel() for p in student.parameters())==d.PARAMETERS[arm]
    assert all(p.requires_grad for p in student.parameters())
    before=copy.deepcopy(student.state_dict());reader=Reader().eval().requires_grad_(False);reader_before=copy.deepcopy(reader.state_dict())
    clean=(torch.randn(2,2,256)*.1).requires_grad_(True);received=clean.detach()+torch.randn_like(clean)*.005
    output=student.stage_outputs(received)[0];loss,terms=objective.clean_objective(output,clean,reader)
    assert set(terms)=={'iq_nmse','task_KL','task_feature_distance'}
    assert torch.allclose(loss,terms['iq_nmse']+terms['task_KL']+terms['task_feature_distance'])
    optimizer=torch.optim.AdamW(student.parameters(),lr=1e-4);loss.mean().backward()
    for group in (student.frozen.coarse,student.frozen.estimator,student.prior):
        grads=[p.grad for p in group.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    assert clean.grad is None and all(p.grad is None for p in original.parameters()) and all(p.grad is None for p in reader.parameters())
    optimizer.step()
    for prefix in ('frozen.coarse.','frozen.estimator.','prior.'):
        assert any(not torch.equal(value,before[key]) for key,value in student.state_dict().items() if key.startswith(prefix))
    assert all(torch.equal(value,original_before[key]) for key,value in original.state_dict().items())
    assert all(torch.equal(value,reader_before[key]) for key,value in reader.state_dict().items())


def test_direct_clean_objective_uses_only_final_stage_and_detached_source_clean(monkeypatch):
    clean=torch.ones(2,2,256,requires_grad=True);first=torch.full_like(clean,float('nan'));final=torch.zeros_like(clean,requires_grad=True)
    def task(student,reference,reader):
        assert student is final and torch.equal(reference,clean) and not reference.requires_grad
        return torch.ones(2)*.5,dict(task_KL=torch.ones(2)*.4,task_feature_distance=torch.ones(2)*.1)
    monkeypatch.setattr(objective,'frozen_task_fidelity',task)
    value,terms=objective.clean_objective([first,first,final],clean,None)
    assert torch.equal(value,torch.ones(2)*1.5)
    value.mean().backward();assert clean.grad is None and final.grad.abs().sum()>0


@pytest.mark.parametrize('bad',['stages','shape'])
def test_invalid_paired_clean_shape_rejects_before_F_access(monkeypatch,bad):
    clean=torch.ones(2,2,256);output=[torch.zeros_like(clean) for _ in range(3)]
    if bad=='stages':output.pop()
    else:output[-1]=torch.zeros(2,2,255)
    monkeypatch.setattr(objective,'frozen_task_fidelity',lambda *a:pytest.fail('No F access'))
    with pytest.raises(ValueError):objective.clean_objective(output,clean,None)


def test_zero_energy_clean_has_finite_loss_and_student_gradient():
    clean=torch.zeros(2,2,256);final=torch.full_like(clean,.01,requires_grad=True);reader=Reader().eval().requires_grad_(False)
    loss,_=objective.clean_objective([final]*3,clean,reader);loss.mean().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(final.grad).all() and final.grad.abs().sum()>0


def test_fixed_paired_clean_only_budget_and_queue_do_not_inherit_teacher_training():
    cfg=d.configuration()
    assert d.EPOCHS==20 and d.UPDATES==4940 and len(d.ARMS)==2
    assert cfg['waveform_teacher_weight']==0 and cfg['R17_training_forward_examples']==0
    assert cfg['teacher_context_forward_examples']==0 and cfg['student_context_forward_examples']==1260000
    assert cfg['cumulative_serial_training_passes']==80 and cfg['cumulative_component_epoch_sum']==120
    assert list(d.GPUS.values())==[4,5] and d.GPU_PRECEDING_RUN.endswith('m2-r08')
    assert d.PARAMETERS[d.ARMS[0]]==1822977 and d.PARAMETERS[d.ARMS[1]]==1592461
    assert cfg['target_feedback'] is False and cfg['F_parameter_updates']==0
