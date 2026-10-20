import copy
import pytest
import torch
from torch import nn
from experiments.pamf_r44_pruned_refine.models import CompactSuccessor
from experiments.pamf_r17_compact.models import FactorConv
from tools.r17_activation_factorization import calibrated_factor
from tools.r46_activation_factorization import SharedInputMoments, calibrated_copy, eligible_groups

torch.set_num_threads(2)


def parent():
    torch.manual_seed(38)
    model=CompactSuccessor('prune_refine').eval().requires_grad_(False)
    with torch.no_grad():
        for index, core in enumerate(model.prior):
            core.out.weight.normal_(std=.001)
            core.encoder.blocks[0][0].net[0].bias.fill_(.01*index)
        model.frozen.coarse.temporal.out.weight.normal_(std=.001)
        model.frozen.estimator.core.out.weight.normal_(std=.001)
        model.frozen.estimator.global_head.weight.normal_(std=.001)
    return model


def test_weighted_factor_minimizes_regularized_activation_error():
    torch.manual_seed(7);conv=nn.Conv1d(3,8,3,padding=1)
    moment=torch.diag(torch.logspace(-3,2,9,dtype=torch.float64))
    calibrated, report=calibrated_factor(conv,moment,.5,1e-3)
    ordinary=FactorConv.from_conv(conv,.5)
    covariance=moment+report['ridge']*torch.eye(9,dtype=torch.float64)
    def error(factor):
        difference=conv.weight.double().flatten(1)-factor.right.weight.double()[:,:,0]@factor.left.weight.double().flatten(1)
        return ((difference@covariance)*difference).sum()
    assert error(calibrated)<=error(ordinary)+1e-8
    assert torch.equal(calibrated.right.bias,conv.bias)


@pytest.mark.parametrize('fraction',[.625,.5])
def test_full_context_factorization_pools_shared_inputs_preserves_GN_bias_and_parent(fraction):
    teacher=parent();before=copy.deepcopy(teacher.state_dict());x=torch.randn(2,2,256)*.1
    with SharedInputMoments(teacher,.5,positions=8) as collector:
        collector.observe(x)
        with pytest.raises(ValueError):collector.finish()
    moments, report=collector.finish()
    assert report['teacher_examples']==2 and all(v['patch_count']>0 for v in report['groups'].values())
    assert all(v['shared_stage_uses']==(3 if key.startswith('prior::') else 1) for key,v in report['groups'].items())
    assert all(not module._forward_pre_hooks for module in teacher.modules())
    model, audit=calibrated_copy(teacher,moments,fraction)
    count=sum(p.numel() for p in model.parameters())
    assert count==(278180 if fraction==.625 else 236355)
    assert len(audit)==len(eligible_groups(teacher,fraction))
    assert all(torch.equal(v,before[key]) for key,v in teacher.state_dict().items())
    old=dict(teacher.named_modules());new=dict(model.named_modules())
    for name,value in old.items():
        if isinstance(value,nn.GroupNorm):
            assert new[name].num_groups==value.num_groups and new[name].num_channels==value.num_channels
            assert torch.equal(new[name].weight,value.weight) and torch.equal(new[name].bias,value.bias)
    for key,value in eligible_groups(teacher,fraction).items():
        if value['group']=='prior':
            factors=[core.get_submodule(value['name']) for core in model.prior]
            assert factors[0].left.weight is factors[1].left.weight is factors[2].left.weight
            assert factors[0].right.weight is factors[1].right.weight is factors[2].right.weight
            for factor,original in zip(factors,value['members']):
                assert torch.equal(factor.right.bias,original.bias)
            assert factors[0].right.bias is not factors[1].right.bias
    model(x).square().mean().backward()
    for component in (model.frozen.coarse,model.frozen.estimator,*model.prior):
        grads=[p.grad for p in component.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)


@pytest.mark.parametrize('observed_before_failure',[False,True])
def test_calibration_hooks_removed_on_failure_and_partial_observations_not_accepted(observed_before_failure):
    teacher=parent();collector=SharedInputMoments(teacher)
    with pytest.raises(RuntimeError):
        with collector:
            if observed_before_failure:collector.observe(torch.randn(1,2,256)*.1)
            raise RuntimeError('source reader failed')
    assert all(not module._forward_pre_hooks for module in teacher.modules())
    with pytest.raises(ValueError):collector.finish()
    with pytest.raises(ValueError):collector.__enter__()


def test_trainable_or_wrong_parent_rejected_without_IQ():
    teacher=parent();teacher.requires_grad_(True)
    with pytest.raises(ValueError):SharedInputMoments(teacher)
    teacher.requires_grad_(False);teacher.arm='other'
    with pytest.raises(ValueError):SharedInputMoments(teacher)


def test_distinct_prior_teachers_not_silently_averaged():
    teacher=parent();core=teacher.prior[1].encoder.blocks[0][0].net[0]
    core.weight=nn.Parameter(core.weight.detach().clone(),requires_grad=False)
    with pytest.raises(ValueError):eligible_groups(teacher,.5)


def test_missing_moments_rejected_before_parent_mutation():
    teacher=parent();before=copy.deepcopy(teacher.state_dict())
    with pytest.raises(ValueError):calibrated_copy(teacher,{},.625)
    assert all(torch.equal(value,before[key]) for key,value in teacher.state_dict().items())
