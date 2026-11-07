from unittest.mock import patch

import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.phase_ramp_posterior import calibrated_action, ramp_posteriors


def test_factor_reuse_matches_independent_conditioned_posteriors():
    rng = torch.Generator().manual_seed(61)
    action = torch.randn(2,6,6,generator=rng,dtype=torch.float64)
    observed = torch.randn(2,6,generator=rng,dtype=torch.float64)
    z = torch.randn(6,6,generator=rng,dtype=torch.float64)
    cov = z @ z.T + torch.eye(6,dtype=torch.float64)
    mean = torch.randn(6,generator=rng,dtype=torch.float64)
    variance = torch.tensor([.03,.4],dtype=torch.float64)
    phase = torch.tensor([-.4,0.,.4],dtype=torch.float64)
    drift = torch.tensor([-.5,0.,.5],dtype=torch.float64)
    angles = phase[None,:,None] + drift[:,None,None]*torch.linspace(-.5,.5,3,dtype=torch.float64)
    log_prior = -phase[None,:].square() - 2*drift[:,None].square()
    outputs, evidence = [], []
    for drift_index in range(3):
        for phase_index in range(3):
            angle = angles[drift_index,phase_index]
            i,q = torch.diag(angle.cos()),torch.diag(angle.sin())
            rotation = torch.cat((torch.cat((i,-q),dim=1),torch.cat((q,i),dim=1)),dim=0)
            a = rotation @ action
            values, _ = gaussian_posterior(a,observed,mean,cov,variance)
            normal = a @ cov @ a.transpose(-1,-2) + torch.diag_embed(variance[:,None].expand(-1,6))
            log_prob = torch.distributions.MultivariateNormal(a@mean,normal).log_prob(observed)
            outputs.append(values)
            evidence.append(log_prob + log_prior[drift_index,phase_index])
    output = torch.stack(outputs,dim=1)
    logits = torch.stack(evidence,dim=1)
    joint_weights = logits.softmax(dim=1)
    phase_weights = logits[:,3:6].softmax(dim=1)
    with patch.object(torch.linalg,'cholesky',wraps=torch.linalg.cholesky) as cholesky:
        actual,calibration = ramp_posteriors(action,observed,mean,cov,variance,phase,drift,log_prior)
        assert cholesky.call_count==1
    torch.testing.assert_close(actual[:,0],output[:,4])
    torch.testing.assert_close(actual[:,1],(output[:,3:6]*phase_weights[:,:,None]).sum(1))
    torch.testing.assert_close(actual[:,2],(output*joint_weights[:,:,None]).sum(1))
    gamma = torch.exp(1j*angles).flatten(0,1)
    torch.testing.assert_close(calibration[:,0],phase_weights.to(torch.complex128) @ gamma[3:6])
    torch.testing.assert_close(calibration[:,1],joint_weights.to(torch.complex128) @ gamma)


def test_calibrated_action_maps_receiver_rows_and_preserves_iq_order():
    action = torch.arange(36,dtype=torch.float64).reshape(1,6,6)
    factor = torch.tensor([[1j,-1.,.5+.2j]],dtype=torch.complex128)
    i,q = torch.diag(factor.real[0]),torch.diag(factor.imag[0])
    rotation = torch.cat((torch.cat((i,-q),dim=1),torch.cat((q,i),dim=1)),dim=0)
    torch.testing.assert_close(calibrated_action(action,factor),rotation @ action)
