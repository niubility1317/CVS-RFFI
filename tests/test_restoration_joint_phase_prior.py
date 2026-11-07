import math

import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.joint_phase_prior import joint_phase_mean
from experiments.pamf_phase3_restoration.phase_ramp_posterior import calibrated_action, ramp_posteriors
from experiments.pamf_phase3_restoration.waveform_mixture_prior import mixture_posterior


def fixture():
    rng = torch.Generator().manual_seed(713)
    def rand(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float64)
    action, observed, mean, offsets = rand(2, 6, 6), rand(2, 6), rand(6), rand(4, 6)
    factor = rand(6, 6)
    within = factor @ factor.T + .3 * torch.eye(6, dtype=torch.float64)
    weights = torch.tensor([.21, .25, .17, .37], dtype=torch.float64)
    variance = torch.tensor([.15, 1.2], dtype=torch.float64)
    phase = torch.linspace(-math.pi/2, math.pi/2, 13, dtype=torch.float64)
    return action, observed, mean, within, offsets, weights, variance, phase, -2*phase.square()


def test_joint_matches_independent_phase_mode_enumeration():
    action, observed, mean, within, offsets, weights, variance, phase, log_prior = values = fixture()
    actual, probabilities, moments = joint_phase_mean(*values)
    scores, outputs = [], []
    for offset, weight in zip(offsets, weights):
        for angle, prior in zip(phase, log_prior):
            gain = torch.exp(1j*angle).expand(2, 3)
            model = calibrated_action(action, gain)
            center = mean + offset
            cov = model @ within @ model.transpose(-1,-2) + variance[:,None,None]*torch.eye(6, dtype=torch.float64)
            likelihood = torch.distributions.MultivariateNormal(model @ center, cov).log_prob(observed)
            scores.append(likelihood + weight.log() + prior)
            outputs.append(gaussian_posterior(model, observed, center, within, variance)[0])
    w = torch.stack(scores, dim=1).softmax(dim=1)
    expected = (w[:,:,None]*torch.stack(outputs, dim=1)).sum(1)
    mode_weights = w.reshape(2,4,len(phase)).sum(2)
    phase_weights = w.reshape(2,4,len(phase)).sum(1)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(probabilities, mode_weights, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(moments, phase_weights @ torch.stack((phase.cos(),phase.sin()),dim=1))


def test_degenerate_paths_equal_existing_single_mechanisms():
    action, observed, mean, within, offsets, weights, variance, phase, log_prior = fixture()
    zero = phase[6:7].clone().zero_()
    from_prior, p, _ = joint_phase_mean(action, observed, mean, within, offsets, weights, variance, zero, zero)
    expected, q = mixture_posterior(action, observed, mean, within, offsets, weights, variance)
    torch.testing.assert_close(from_prior, expected)
    torch.testing.assert_close(p, q)
    from_phase, _, _ = joint_phase_mean(action, observed, mean, within, offsets[:1]*0,
        weights[:1]*0+1, variance, phase, log_prior)
    ramp, _ = ramp_posteriors(action, observed, mean, within, variance, phase, zero, log_prior[None])
    torch.testing.assert_close(from_phase, ramp[:,1])



def test_prior_provenance_rejects_target_exposure_before_reading_weights(tmp_path):
    import json
    import pytest
    from experiments.pamf_phase3_restoration.joint_phase_source import WAVEFORM_COMMIT, prior_path

    result = dict(status='SOURCE_WAVEFORM_MIXTURE_COMPLETE',code_commit=WAVEFORM_COMMIT,
        TX_labels_read=0,target_inputs=1,OE_reads=0,novel_reads=0,optimizer_updates=0,
        source_prior=dict(source_Ls=6300,source_Us=56700))
    (tmp_path/'results.json').write_text(json.dumps(result),encoding='utf-8')
    (tmp_path/'complete.json').write_text('{"status":"ARTIFACTS_COMPLETE"}',encoding='utf-8')
    with pytest.raises(ValueError,match='Original IQ-only'):
        prior_path(tmp_path)
