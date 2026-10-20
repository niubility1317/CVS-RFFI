import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.pamf_r17_compact.models import FactorConv
from tools.r17_activation_factorization import InputSecondMoment,calibrated_factor,factor_rank,input_patches


def test_covariance_weighting_preserves_dominant_output_instead_of_dominant_kernel():
    conv=nn.Conv1d(1,3,5,bias=True).double()
    with torch.no_grad():
        conv.weight.zero_();conv.weight[0,0,0]=4;conv.weight[1,0,1]=3;conv.weight[2,0,2]=1
        conv.bias.copy_(torch.tensor([.1,.2,.3]))
    before={key:value.clone() for key,value in conv.state_dict().items()}
    moment=torch.diag(torch.tensor([.001,.001,100.,1.,1.],dtype=torch.float64))
    factor,report=calibrated_factor(conv,moment,fraction=1/3)
    ordinary=FactorConv.from_conv(conv,fraction=1/3)
    def output_error(value):
        matrix=value.right.weight[:,:,0]@value.left.weight.flatten(1)
        delta=conv.weight.flatten(1)-matrix
        return torch.sum((delta@moment)*delta)
    assert output_error(factor)<output_error(ordinary)*.001
    assert report['rank']==1 and report['factor_kernel_parameters']<report['original_kernel_parameters']
    assert report['regularized_output_relative_squared_error']<.01
    assert torch.equal(conv.bias,factor.right.bias)
    assert all(torch.equal(value,before[key]) for key,value in conv.state_dict().items())


def test_second_moments_are_uncentered_and_streaming_counts_match_actual_patches():
    conv=nn.Conv1d(2,4,3,padding=2,dilation=2,stride=2)
    values=torch.arange(2*2*7,dtype=torch.float32).reshape(2,2,7)/10
    stats=InputSecondMoment(conv,positions=8)
    with pytest.raises(ValueError):stats.value()
    stats.update(values[:1]);stats.update(values[1:])
    patches=F.unfold(values.unsqueeze(2),(1,3),dilation=(1,2),padding=(0,2),stride=(1,2)).permute(1,0,2).reshape(6,-1)
    assert stats.count==patches.shape[1]
    assert torch.equal(stats.value(),stats.value().T)
    assert torch.allclose(stats.value(),((patches@patches.T)/patches.shape[1]).double(),rtol=1e-6,atol=1e-6)
    assert float(stats.value().mean())>0 and not stats.value().requires_grad
    subset=input_patches(conv,values,positions=2)
    assert subset.shape==(6,4) and not subset.requires_grad


def test_singular_moments_are_regularized_without_changing_teacher_or_backpropagating():
    conv=nn.Conv1d(4,8,3,padding=1)
    before=conv.weight.detach().clone()
    moment=torch.zeros(12,12,requires_grad=True)
    factor,report=calibrated_factor(conv,moment)
    assert report['ridge']==1e-8 and torch.isfinite(factor(torch.ones(2,4,7))).all()
    assert torch.equal(before,conv.weight) and moment.grad is None
    assert factor.left.weight.requires_grad and factor.right.weight.requires_grad


def test_calibrated_weighted_error_matches_measured_linear_output_error():
    torch.manual_seed(10)
    conv=nn.Conv1d(3,8,3,padding=1,bias=True).double()
    x=torch.randn(3,3,11,dtype=torch.float64)+2
    patches=input_patches(conv,x,positions=11)
    moment=patches@patches.T/patches.shape[1]
    factor,report=calibrated_factor(conv,moment)
    direct=(conv(x)-factor(x)).square().sum()/patches.shape[1]
    w=conv.weight.flatten(1);d=w-factor.right.weight[:,:,0]@factor.left.weight.flatten(1)
    assert torch.allclose(direct,torch.sum((d@moment)*d),rtol=1e-10,atol=1e-10)
    assert report['uncentered_second_moment']


def test_invalid_or_non_saving_factorization_is_rejected_before_modification():
    conv=nn.Conv1d(4,8,3)
    with pytest.raises(ValueError):factor_rank(conv,1.)
    with pytest.raises(ValueError):factor_rank(nn.Conv1d(4,4,1),.5)
    with pytest.raises(ValueError):calibrated_factor(conv,torch.eye(3))
    with pytest.raises(ValueError):calibrated_factor(conv,torch.full((12,12),float('nan')))
    moment=torch.eye(12);moment[0,1]=1
    with pytest.raises(ValueError):calibrated_factor(conv,moment)
    with pytest.raises(ValueError):input_patches(conv,torch.ones(2,4,8),positions=0)
