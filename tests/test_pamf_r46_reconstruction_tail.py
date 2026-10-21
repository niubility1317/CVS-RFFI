import math
import pytest
import torch
from experiments.pamf_r46_reconstruction_tail import design as d
from experiments.pamf_r46_reconstruction_tail.losses import reconstruction_risk,objective
from experiments.pamf_r46_multipoint_fast.models import objective as original


@pytest.mark.parametrize('arm,expected,weights',[('tail_half',3.,[.5,1.5,.5,1.5]),('tail_quarter',3.25,[.5,2.5,.5,.5])])
def test_exact_mixed_tail_value_and_piecewise_gradient(arm,expected,weights):
    values=torch.tensor([1.,4.,2.,3.],requires_grad=True);risk,info=reconstruction_risk(values,arm)
    assert risk.mean().item()==expected and not info['weights'].requires_grad
    risk.mean().backward();assert torch.equal(values.grad,torch.tensor(weights)/4)
    assert torch.equal(info['weights'],torch.tensor(weights)) and int(info['selected'].sum())==math.ceil(4*d.FRACTION[arm])


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('n',[1,3,24,256])
def test_all_examples_retained_and_actual_last_batch_normalized(arm,n):
    values=torch.linspace(.1,2.,n,requires_grad=True);risk,info=reconstruction_risk(values,arm)
    assert torch.all(info['weights']>=.5) and torch.isclose(info['weights'].sum(),torch.tensor(float(n)))
    assert info['batch']==n and info['selected_count']==math.ceil(d.FRACTION[arm]*n)
    expected=.5*values.mean()+.5*values.detach().sort(descending=True).values[:info['selected_count']].mean()
    assert torch.isclose(risk.mean(),expected)
    risk.mean().backward();assert torch.all(values.grad>0)


@pytest.mark.parametrize('arm',d.ARMS)
def test_ties_stable_by_current_batch_position_without_state(arm):
    first,one=reconstruction_risk(torch.ones(8),arm);second,two=reconstruction_risk(torch.ones(8),arm)
    assert torch.equal(first,second) and torch.equal(one['selected'],two['selected'])
    assert one['selected'].tolist()==[True]*one['selected_count']+[False]*(8-one['selected_count'])


@pytest.mark.parametrize('value',[torch.tensor([]),torch.ones(2,2),torch.ones(2,dtype=torch.long),torch.tensor([-1.,2.]),torch.tensor([float('nan')]),torch.tensor([float('inf')])])
def test_invalid_reconstruction_rejected(value):
    with pytest.raises(ValueError):reconstruction_risk(value,'tail_half')


@pytest.mark.parametrize('arm',d.ARMS)
def test_real_objective_uses_only_reconstruction_for_focus_and_keeps_teacher_mean(arm):
    torch.manual_seed(68)
    clean=torch.randn(4,2,256,requires_grad=True)
    student=[torch.randn(4,2,256,requires_grad=True) for _ in range(3)]
    teacher=[torch.randn(4,2,256,requires_grad=True) for _ in range(3)]
    loss,terms,focus=objective(student,teacher,clean,11,arm)
    base,old=original(student,teacher,clean,11)
    for name in old:assert torch.equal(terms[name],old[name])
    assert torch.allclose(loss.mean(),.5*terms['reconstruction_mean'].mean()+.5*terms['reconstruction_tail'].mean()+d.teacher_weight(11)*old['teacher_stage_nmse'].mean())
    assert torch.allclose(loss,terms['reconstruction_risk']+d.teacher_weight(11)*old['teacher_stage_nmse'])
    # Large teacher changes affect the teacher term, never reconstruction ranks.
    changed=[x.detach()+100 for x in teacher]
    _,_,other=objective(student,changed,clean,11,arm)
    assert torch.equal(focus['weights'],other['weights'])
    loss.mean().backward()
    assert clean.grad is None and all(t.grad is None for t in teacher)
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in student)


def test_fixed_synthetic_scope_no_claim_of_source_training_or_new_inference_graph():
    cfg=d.configuration()
    assert cfg['parameters']==273765 and cfg['actual_optimizer_updates']==cfg['actual_source_epochs']==cfg['real_IQ_reads']==cfg['GPU_launches']==0
    assert cfg['new_trainable_parameters']==cfg['new_calibration_IQ']==cfg['extra_training_G_forwards']==cfg['extra_inference_operations']==0
    assert cfg['TX_labels_read']==cfg['F_weights_read']==cfg['F_outputs_read']==cfg['V_reads']==cfg['target_reads']==0
    assert cfg['control']['row']=='final' and cfg['control']['run'].endswith('-r66') and not cfg['goal_complete']
