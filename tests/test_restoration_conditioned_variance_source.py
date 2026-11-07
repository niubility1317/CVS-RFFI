import numpy as np
import pytest
import torch

from experiments.pamf_phase3_restoration.conditioned_variance_source import probe,select_source


def test_saturated_raw_derivative_can_hide_a_helpful_variance_direction():
    action=torch.ones(1,1,1,dtype=torch.float64)
    observed=torch.ones(1,1,dtype=torch.float64)
    mean=torch.zeros(1,dtype=torch.float64)
    covariance=.01*torch.ones(1,1,dtype=torch.float64)
    noise=.1*torch.ones(1,dtype=torch.float64)
    raw=-50*torch.ones(1,1,dtype=torch.float64)
    result,_=probe(action,observed,mean,covariance,noise,raw,
                   lambda restored:((restored-1).square().mean(1),{}))
    assert result['zero']['uniform_direction_derivative'][0]<0
    assert result['initial_fraction']['loss'][0]<result['zero']['loss'][0]
    assert result['learned']['gradient_abs_mean'][0]<1e-20
    assert torch.equal(raw,-50*torch.ones_like(raw))


def test_probe_also_reports_when_extra_variance_amplifies_realized_noise():
    action=torch.ones(1,1,1,dtype=torch.float64)
    result,_=probe(action,4*torch.ones(1,1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),
                   .01*torch.ones(1,1,dtype=torch.float64),.1*torch.ones(1,dtype=torch.float64),
                   torch.zeros(1,1,dtype=torch.float64),lambda x:((x-.3).square().mean(1),{}))
    assert result['zero']['uniform_direction_derivative'][0]>0
    assert result['moderate_fraction']['loss'][0]>result['zero']['loss'][0]


def test_missing_receiver_or_scene_is_rejected_before_neural_work():
    with pytest.raises(ValueError,match='five-receiver'):
        select_source(np.ones(63000,dtype=np.int64),np.full(63000,'clean'))
    rx=np.resize(np.asarray([1,3,4,6,8]),63000)
    with pytest.raises(ValueError,match='per fixed receiver/scene'):
        select_source(rx,np.full(63000,'clean'))
