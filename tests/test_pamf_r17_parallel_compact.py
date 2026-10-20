import copy
import pytest
import torch
from torch import nn
from experiments.pamf_r17_parallel_compact import design as d,models as m,runtime
from experiments.pamf_reception_restore.models import Core
from experiments.pamf_action_restore.models import ActionRestorer
from experiments.pamf_reception_restore.models import ComplexTFUNet
from experiments.pamf_unrolled_restore.models import UnrolledRestorer
from experiments.pamf_r17_compact.models import FactorConv

torch.set_num_threads(2)


@pytest.mark.parametrize('arm',d.ARMS)
def test_all_four_complete_models_have_fixed_cost_aliases_and_independent_norm_bias(arm):
    model=m.SharedFactorRestorer(arm);assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    maps=[dict(core.named_modules()) for core in model.prior]
    for name,module in maps[0].items():
        if isinstance(module,FactorConv):
            assert all(values[name].left.weight is module.left.weight and values[name].right.weight is module.right.weight for values in maps)
            if module.right.bias is not None:assert len({id(values[name].right.bias) for values in maps})==3
        if isinstance(module,nn.GroupNorm):assert len({id(values[name].weight) for values in maps})==3
        if isinstance(module,nn.Conv1d) and module.kernel_size==(1,) and not name.endswith('.right'):
            assert len({id(values[name].weight) for values in maps})==(1 if d.POINTWISE[arm] else 3)
    runtime.validate_aliases(model);model.train();assert not model.frozen.training and all(not p.requires_grad for p in model.frozen.parameters())


def test_original_pointwise_means_are_shared_without_changing_bias_norm_or_teacher():
    torch.manual_seed(3);priors=[Core(incoming=9) for _ in range(3)];before=[copy.deepcopy(core.state_dict()) for core in priors]
    reports=m.share_pointwise(priors);assert reports
    maps=[dict(core.named_modules()) for core in priors]
    for report in reports:
        name=report['name'];key=name+'.weight';expected=torch.stack([state[key] for state in before]).mean(0)
        assert torch.equal(maps[0][name].weight,expected)
        assert all(values[name].weight is maps[0][name].weight for values in maps)
        if maps[0][name].bias is not None:
            assert all(torch.equal(values[name].bias,state[name+'.bias']) for values,state in zip(maps,before))
    for core,state in zip(priors,before):
        for name,module in core.named_modules():
            if isinstance(module,nn.GroupNorm):assert torch.equal(module.weight,state[name+'.weight'])


@pytest.mark.parametrize('arm',d.ARMS)
def test_source_teacher_factory_preserves_teacher_and_stage_gradients(arm):
    torch.manual_seed(7);teacher=UnrolledRestorer('unroll_unshared',ActionRestorer(ComplexTFUNet())).eval().requires_grad_(False)
    with torch.no_grad():
        for core in teacher.prior:core.out.weight.normal_(std=.001)
    before=copy.deepcopy(teacher.state_dict());model=m.compact_from_teacher(teacher,arm)
    assert sum(p.numel() for p in model.parameters())==d.PARAMETERS[arm]
    assert model.initialization_report['pointwise_weights_shared']==d.POINTWISE[arm]
    assert bool(model.initialization_report['pointwise_layers'])==d.POINTWISE[arm]
    assert all(torch.equal(value,before[key]) for key,value in teacher.state_dict().items())
    x=torch.randn(2,2,256)*.1
    with torch.no_grad():context=model.frozen.estimate(x);desired=m.stages(teacher.prior,x,context)[0]
    output=model.stage_outputs(x,context)[0];loss,_=m.distillation_loss(output,desired,x);loss.mean().backward()
    assert all(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in core.parameters()) for core in model.prior)
    assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in model.frozen.parameters())


def test_parallel_queue_keeps_current_R06_and_serializes_all_own_scoring():
    from experiments.pamf_r17_shared_factor import design as old
    assert set(d.GPUS.values())=={2,3,6,7} and not set(d.GPUS.values())&set(old.GPUS.values())
    assert d.SCORING_RUN==old.RUN and d.configuration()['target_acceptance_allowance_pp']==1.
    assert all(d.PARAMETERS[a]<3118001 for a in d.ARMS)
