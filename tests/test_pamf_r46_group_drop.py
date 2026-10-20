"""Source-free structural checks; these are not recognition/fidelity evidence."""
import copy
import torch
from torch import nn
import pytest

from experiments.pamf_r46_group_drop import models as m
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor as Parent

torch.set_num_threads(2)


@pytest.mark.parametrize('score',[
    torch.ones(14),torch.tensor([float('nan')]*12),-torch.ones(12),torch.ones(12,dtype=torch.int64)])
def test_invalid_group_scores_fail(score):
    with pytest.raises(ValueError):m.select_groups(score)


def test_maps_choose_complete_original_groups_with_stable_ties():
    assert m.select_groups(torch.ones(12)).tolist()==list(range(9))
    score=torch.tensor([1.]*3+[9.]*3+[3.]*3+[5.]*3)
    assert m.select_groups(score).tolist()==list(range(3,12))


def test_explicit_group_constructor_preserves_original_GN4_computation():
    source=m.Core(incoming=4,outgoing=4,widths=(8,12,20,32)).eval()
    with torch.no_grad():
        source.out.weight.normal_(std=.02);source.out.bias.normal_(std=.02)
    target=m.GroupCore(incoming=4,outgoing=4,widths=(8,12,20,32),groups=4).eval()
    target.load_state_dict(source.state_dict(),strict=True)
    x=torch.randn((2,4,256))
    with torch.no_grad():assert torch.equal(source(x),target(x))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_retained_group_normalization_has_same_mathematical_partition(dtype):
    generator=torch.Generator().manual_seed(62)
    x=torch.randn((2,12,31),generator=generator,dtype=dtype)
    x=x*torch.arange(1,13,dtype=dtype)[None,:,None]+torch.arange(12,dtype=dtype)[None,:,None]
    source=nn.GroupNorm(4,12).to(dtype)
    with torch.no_grad():
        source.weight.copy_(torch.randn(12,generator=generator,dtype=dtype))
        source.bias.copy_(torch.randn(12,generator=generator,dtype=dtype))
    indices=torch.tensor([0,1,2,6,7,8,9,10,11])
    retained=nn.GroupNorm(3,9,eps=source.eps).to(dtype)
    regrouped=nn.GroupNorm(1,9,eps=source.eps).to(dtype)
    for target in (retained,regrouped):
        with torch.no_grad():
            target.weight.copy_(source.weight[indices]);target.bias.copy_(source.bias[indices])
    tolerance=2e-6 if dtype==torch.float32 else 1e-12
    assert torch.allclose(source(x)[:,indices],retained(x[:,indices]),atol=tolerance,rtol=tolerance)
    assert not torch.allclose(source(x)[:,indices],regrouped(x[:,indices]),atol=1e-4,rtol=1e-4)


@pytest.fixture(scope='module')
def source():
    torch.manual_seed(62)
    model=Parent('prune_refine').eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==350521
    return model


def test_two_rows_start_from_identical_parameters_and_differ_only_in_context_groups(source):
    before={k:v.clone() for k,v in source.state_dict().items()}
    retained=m.compact_from_parent(source,'group_preserve')
    control=m.compact_from_parent(source,'regroup_control')
    assert sum(p.numel() for p in retained.parameters())==sum(p.numel() for p in control.parameters())<273765
    assert tuple(retained.state_dict())==tuple(control.state_dict())
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in retained.state_dict().items())
    assert all(torch.equal(v,before[k]) for k,v in source.state_dict().items())
    for model,groups in ((retained,3),(control,1)):
        context=[v for v in model.frozen.modules() if isinstance(v,nn.GroupNorm)]
        prior=[v for v in model.prior.modules() if isinstance(v,nn.GroupNorm)]
        assert len(context)==len(prior)==84
        assert all(v.num_groups==groups for v in context)
        assert all(v.num_groups==4 for v in prior)
        assert model.initialization_report['IQ_examples']==0
        assert model.initialization_report['F_reads']==model.initialization_report['TX_labels_read']==0
        assert model.initialization_report['parent_unchanged'] and not model.initialization_report['exact_function_equivalence']
        assert all(p.requires_grad for p in model.parameters())
        assert all(torch.equal(v,source.prior.state_dict()[k]) for k,v in model.prior.state_dict().items())
        assert next(model.prior.parameters()).data_ptr()!=next(source.prior.parameters()).data_ptr()


@pytest.mark.parametrize('arm',['group_preserve','regroup_control'])
def test_shared_prior_weights_and_stage_biases_survive_compression(source,arm):
    model=m.compact_from_parent(source,arm)
    long=[(name,layer) for name,layer in model.prior[0].named_modules() if isinstance(layer,nn.Conv1d) and layer.kernel_size[0]>1]
    assert len(long)==28
    for name,layer in long:
        for index in (1,2):
            other=model.prior[index].get_submodule(name)
            assert layer.weight is other.weight and layer.bias is not other.bias


@pytest.mark.parametrize('arm',['group_preserve','regroup_control'])
def test_native_IQ_interface_and_portable_strict_state(source,arm):
    model=m.compact_from_parent(source,arm).eval()
    portable=m.CompactSuccessor(arm).eval();portable.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        for x in (torch.zeros((1,2,256)),torch.randn((2,2,256))):
            output=model(x)
            assert output.shape==x.shape and torch.isfinite(output).all()
            assert torch.equal(output,portable(x))


def test_parent_and_arm_rejection_precedes_any_transplant(source):
    with pytest.raises(ValueError):m.compact_from_parent(source,'unknown')
    wrong=copy.deepcopy(source);wrong.arm='retain_refine'
    with pytest.raises(ValueError):m.compact_from_parent(wrong,'group_preserve')


def test_original_waveform_objective_has_no_F_or_TX_inputs():
    clean=torch.randn((2,2,256));student=[torch.randn_like(clean,requires_grad=True) for _ in range(3)]
    teacher=[torch.randn_like(clean) for _ in range(3)]
    loss,terms=m.objective(student,teacher,clean,20)
    expected=terms['iq_nmse']+.05*terms['spectral_error']+.1*terms['teacher_stage_nmse']
    assert torch.allclose(loss,expected)
    loss.mean().backward();assert all(s.grad is not None and torch.isfinite(s.grad).all() for s in student)
