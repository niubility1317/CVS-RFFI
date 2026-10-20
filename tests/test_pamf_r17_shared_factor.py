import copy
import io
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from experiments.pamf_r17_shared_factor import design as d
from experiments.pamf_r17_shared_factor.models import SharedFactorRestorer,share_factor_long,compact_from_teacher,distillation_loss
from experiments.pamf_r17_compact.models import FactorConv

torch.set_num_threads(2)


def tiny_priors():
    torch.manual_seed(20261020)
    priors=nn.ModuleList([nn.Sequential(nn.Conv1d(4,8,3,padding=1),nn.GroupNorm(2,8),nn.SiLU(),nn.Conv1d(8,2,1)) for _ in range(3)])
    with torch.no_grad():
        for i,core in enumerate(priors):core[1].weight.add_(i*.1);core[1].bias.add_(i*.02);core[3].weight.add_(i*.03)
    return priors


@pytest.mark.parametrize('arm',d.ARMS)
def test_complete_cost_and_actual_factor_aliasing(arm):
    model=SharedFactorRestorer(arm)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    maps=[{n:m for n,m in core.named_modules() if isinstance(m,FactorConv)} for core in model.prior]
    assert len(maps[0])==28 and all(set(v)==set(maps[0]) for v in maps)
    for name in maps[0]:
        assert len({id(v[name].left.weight) for v in maps})==1
        assert len({id(v[name].right.weight) for v in maps})==1
        assert len({id(v[name].right.bias) for v in maps})==3
    model.train()
    assert not model.frozen.training and all(not p.requires_grad for p in model.frozen.parameters())
    assert all(p.requires_grad for p in model.prior.parameters())


@pytest.mark.parametrize('fraction',[.5,.25])
def test_mean_kernel_factorization_preserves_per_step_bias_norm_and_pointwise_state(fraction):
    priors=tiny_priors();before=[copy.deepcopy(core.state_dict()) for core in priors]
    mean=torch.stack([v['0.weight'] for v in before]).mean(0).flatten(1)
    reports=share_factor_long(priors,fraction,initialize=True)
    assert len(reports)==1 and reports[0]['factored']
    rank=max(1,int(8*fraction));u,s,vh=torch.linalg.svd(mean,full_matrices=False)
    expected=(u[:,:rank]*s[:rank])@vh[:rank]
    for i,core in enumerate(priors):
        assert isinstance(core[0],FactorConv)
        actual=core[0].right.weight[:,:,0]@core[0].left.weight.flatten(1)
        assert torch.allclose(actual,expected,rtol=2e-5,atol=2e-6)
        assert torch.equal(core[0].right.bias,before[i]['0.bias'])
        for key in ('1.weight','1.bias','3.weight','3.bias'):assert torch.equal(core.state_dict()[key],before[i][key])
    assert len({id(core[0].right.bias) for core in priors})==3
    assert len({id(core[1].weight) for core in priors})==3
    assert len({id(core[3].weight) for core in priors})==3


def test_original_teacher_is_unchanged_and_shared_factor_gradients_are_live_once():
    teacher=SimpleNamespace(candidate='unroll_unshared',prior=tiny_priors().requires_grad_(False),frozen=nn.Linear(1,1))
    before=[copy.deepcopy(core.state_dict()) for core in teacher.prior]
    model=compact_from_teacher(teacher,'shared_half');x=torch.randn(2,4,16);clean=torch.randn(2,2,16)
    with torch.no_grad():desired=[core(x) for core in teacher.prior]
    output=[core(x) for core in model.prior];loss,terms=distillation_loss(output,desired,clean);loss.mean().backward()
    assert torch.allclose(loss,10*terms['teacher_stage_nmse'])
    left=model.prior[0][0].left.weight;right=model.prior[0][0].right.weight
    assert sum(p is left for p in model.prior.parameters())==sum(p is right for p in model.prior.parameters())==1
    for value in (left,right):assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.norm()>0
    assert all(p.grad is None for core in teacher.prior for p in core.parameters())
    for core,old in zip(teacher.prior,before):assert all(torch.equal(v,old[k]) for k,v in core.state_dict().items())
    assert all(not p.requires_grad for p in model.frozen.parameters())
    assert model.initialization_report['IQ_examples']==model.initialization_report['target_reads']==0


@pytest.mark.parametrize('arm',d.ARMS)
def test_synthetic_state_roundtrip_preserves_shared_graph_and_exact_output(arm):
    model=SharedFactorRestorer(arm).eval();value=torch.zeros(1,2,256)
    with torch.no_grad():expected=model(value)
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored=SharedFactorRestorer(arm).eval();restored.load_state_dict(torch.load(stream,map_location='cpu',weights_only=True),strict=True)
    with torch.no_grad():actual=restored(value)
    assert torch.isfinite(actual).all() and torch.equal(actual,expected)
    assert sum(p.numel() for p in restored.parameters())==d.PARAMETERS[arm]


def test_non_saving_small_kernel_keeps_mean_shared_weight_without_factor_expansion():
    priors=nn.ModuleList([nn.Sequential(nn.Conv1d(1,1,3,padding=1)) for _ in range(3)])
    before=[v[0].bias.detach().clone() for v in priors]
    reports=share_factor_long(priors,.25,initialize=True)
    assert not reports[0]['factored'] and reports[0]['shared_factor_parameters']==3
    assert len({id(v[0].weight) for v in priors})==1
    for i,core in enumerate(priors):assert torch.equal(core[0].bias,before[i])


@pytest.mark.parametrize('bad',['fraction','steps','stride','teacher'])
def test_unregistered_or_incompatible_compositions_are_rejected(bad):
    priors=tiny_priors()
    if bad=='fraction':
        with pytest.raises(ValueError):share_factor_long(priors,.75,initialize=True)
    elif bad=='steps':
        with pytest.raises(ValueError):share_factor_long(priors[:2],.5,initialize=True)
    elif bad=='stride':
        priors[1][0].stride=(2,)
        with pytest.raises(ValueError):share_factor_long(priors,.5,initialize=True)
    else:
        teacher=SimpleNamespace(candidate='unroll_shared',prior=priors,frozen=nn.Linear(1,1))
        with pytest.raises(ValueError):compact_from_teacher(teacher,'shared_half')
