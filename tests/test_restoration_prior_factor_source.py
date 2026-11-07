import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.prior_factor_source import cross_priors


def test_crossing_reuses_each_actual_mean_and_covariance():
    same = (torch.zeros(3),torch.eye(3))
    other = (torch.ones(3),2*torch.eye(3))
    priors = cross_priors(same,other)
    assert priors['same'][0] is same[0] and priors['same'][1] is same[1]
    assert priors['mean_other'][0] is other[0] and priors['mean_other'][1] is same[1]
    assert priors['covariance_other'][0] is same[0] and priors['covariance_other'][1] is other[1]
    assert priors['other'][0] is other[0] and priors['other'][1] is other[1]


def test_equal_covariance_isolates_the_exact_mean_projection():
    action = torch.tensor([[[1.,.2],[0.,.1]]],dtype=torch.float64)
    observed = torch.tensor([[.4,.3]],dtype=torch.float64)
    variance = torch.tensor([.1],dtype=torch.float64)
    same = (torch.zeros(2,dtype=torch.float64),torch.eye(2,dtype=torch.float64))
    other = (torch.tensor([.3,-.1],dtype=torch.float64),same[1])
    priors = cross_priors(same,other)
    outputs = {key:gaussian_posterior(action,observed,*prior,variance)[0] for key,prior in priors.items()}
    torch.testing.assert_close(outputs['same'],outputs['covariance_other'],rtol=0,atol=0)
    torch.testing.assert_close(outputs['mean_other'],outputs['other'],rtol=0,atol=0)
    assert not torch.allclose(outputs['same'],outputs['mean_other'])
