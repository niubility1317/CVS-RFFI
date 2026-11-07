import torch

from experiments.pamf_phase3_restoration.coherent_main_phase import phase_weights, rotate_anchor


def test_coherent_phase_scores_match_independent_gaussian_likelihoods():
    rng = torch.Generator().manual_seed(51)
    def rand(*shape):
        return torch.randn(*shape,generator=rng,dtype=torch.float64)
    observed,main,image = rand(2,6),rand(2,6),rand(2,6)
    factor = rand(2,6,6)
    covariance = factor @ factor.transpose(-1,-2) + .2*torch.eye(6,dtype=torch.float64)
    scalar = covariance.diagonal(dim1=-2,dim2=-1).mean(1)
    chol = torch.linalg.cholesky(covariance)
    phase = torch.linspace(-.8,.8,17,dtype=torch.float64)
    prior = -.5*(phase/.3).square()
    residual = torch.stack((observed-image,main,rotate_anchor(main)),dim=2)
    for normal,whitened in [(covariance,torch.linalg.solve_triangular(chol,residual,upper=False)),
        (scalar[:,None,None]*torch.eye(6,dtype=torch.float64),residual/scalar[:,None,None].sqrt())]:
        actual = phase_weights(whitened,phase,prior)
        scores = []
        for a,p in zip(phase,prior):
            mean = a.cos()*main+a.sin()*rotate_anchor(main)+image
            dist = torch.distributions.MultivariateNormal(mean,normal)
            scores.append(dist.log_prob(observed)+p)
        expected = torch.stack(scores,dim=1).softmax(1)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-11)


def test_main_anchor_rotation_keeps_image_reference_fixed():
    main = torch.tensor([[1.,-2.,3.,-4.]],dtype=torch.float64)
    image = torch.tensor([[3.,2.,-1.,.7]],dtype=torch.float64)
    phase = torch.tensor([-.5,0.,.5],dtype=torch.float64)
    theta = phase[-1]
    observed = theta.cos()*main+theta.sin()*rotate_anchor(main)+image
    basis = 20*torch.stack((observed-image,main,rotate_anchor(main)),dim=2)
    weights = phase_weights(basis,phase,torch.zeros_like(phase))
    assert weights.argmax(1).tolist() == [2]


def test_blind_source_reader_rejects_target_touched_cache_before_iq(tmp_path):
    import pytest
    from experiments.pamf_phase3_restoration.source_coherent_main_phase import received_only

    (tmp_path/'complete.json').write_text('{"count":1800,"TX_labels_read":0,"target_reads":1}',encoding='utf-8')
    with pytest.raises(ValueError,match='Original complete source-only observations'):
        received_only(tmp_path,[0,1],1800)
