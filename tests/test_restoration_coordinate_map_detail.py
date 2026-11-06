from unittest.mock import patch

import pytest
import torch
from torch import nn

from experiments.pamf_phase3_restoration import coordinate_map_detail as model
from experiments.pamf_phase3_restoration import coordinate_map_detail_trial as trial
from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior


def rotation(n, quarter):
    angles = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    cosine, sine = angles[quarter]
    return torch.kron(torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.float64),
                      torch.eye(n, dtype=torch.float64))


def test_operator_conjugation_uses_inverse_output_and_forward_input():
    matrix = torch.arange(72, dtype=torch.float64).reshape(2, 6, 6)
    for quarter in range(4):
        r = rotation(3, quarter)
        torch.testing.assert_close(model.operator_in_original_frame(matrix, quarter), r.T @ matrix @ r)


def test_source_prior_matches_four_component_moments():
    generator = torch.Generator().manual_seed(8)
    a = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    covariance = a @ a.T + torch.eye(6, dtype=torch.float64)
    mean = torch.arange(6, dtype=torch.float64) / 10
    rotated_mean, rotated_covariance = model.coordinate_prior(mean, covariance)
    second_moment = covariance + torch.outer(mean, mean)
    rotations = [rotation(3, quarter) for quarter in range(4)]
    expected = sum(r @ second_moment @ r.T for r in rotations) / 4
    torch.testing.assert_close(rotated_mean, torch.zeros_like(mean), atol=0, rtol=0)
    torch.testing.assert_close(rotated_covariance, expected)
    torch.testing.assert_close(rotations[1] @ rotated_covariance, rotated_covariance @ rotations[1])
    assert torch.linalg.eigvalsh(rotated_covariance).min() > 0


class CoordinateDependentPhysics(nn.Module):
    def __init__(self):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(.1))
        self.register_buffer('prior_mean', torch.zeros(512))
        self.register_buffer('prior_covariance', torch.eye(512))
        self.calls = 0

    def estimate(self, iq):
        self.calls += 1
        terms = iq.flatten(1).double()
        gain = 1 + self.strength * terms.square() + .05 * terms
        matrix = torch.diag_embed(gain) + .001 * terms[:, :, None] * terms[:, None, :].roll(1, dims=2)
        variance = .01 + self.strength.square() * iq[:, 0, :].square().mean(1)
        return dict(action_matrix=matrix, variance=variance)


class CoordinateDependentDetail(nn.Module):
    def __init__(self):
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(.2))
        self.calls = 0

    def forward(self, base, context):
        self.calls += 1
        i, q, noise = context.unbind(1)
        delta = torch.stack((i.square() + .3 * q, -i * q + i), 1) * noise[:, None]
        return base + self.strength * delta


def new_toy_model():
    restorer = model.CoordinateMAPDetail.__new__(model.CoordinateMAPDetail)
    nn.Module.__init__(restorer)
    restorer.physical = CoordinateDependentPhysics()
    restorer.detail = CoordinateDependentDetail()
    return restorer


def test_nontrivial_group_model_with_one_real_posterior_per_call():
    restorer = new_toy_model()
    iq = torch.randn(1, 2, 256, generator=torch.Generator().manual_seed(16))
    with patch.object(model, 'gaussian_posterior', wraps=gaussian_posterior) as solve:
        output, aux = restorer(iq, return_aux=True)
        assert solve.call_count == 1
        assert restorer.physical.calls == restorer.detail.calls == 4
        assert (output - aux['base_iq']).abs().max() > .01
        assert (output - iq).abs().max() > .01
        for quarter in range(1, 4):
            r = rotation(256, quarter)
            rotated_input = (iq.flatten(1).double() @ r.T).float().reshape_as(iq)
            expected_output = (output.flatten(1).double() @ r.T).float().reshape_as(iq)
            rotated, rotated_aux = restorer(rotated_input, return_aux=True)
            torch.testing.assert_close(rotated, expected_output, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(rotated_aux['variance'], aux['variance'])
            torch.testing.assert_close(rotated_aux['action_matrix'], r @ aux['action_matrix'] @ r.T)
        assert solve.call_count == 4


def test_physical_and_detail_gradients_remain_separate():
    restorer = new_toy_model()
    iq = torch.randn(1, 2, 256, generator=torch.Generator().manual_seed(23))
    output, aux = restorer(iq, return_aux=True)
    output.square().mean().backward()
    assert restorer.physical.strength.grad is None
    assert restorer.detail.strength.grad.abs() > 0
    restorer.zero_grad(set_to_none=True)
    (aux['action_matrix'].square().mean() + aux['variance'].mean()).backward()
    assert restorer.detail.strength.grad is None
    assert restorer.physical.strength.grad.abs() > 0


def test_fixed_scratch30_and_full_frozen_main_contract():
    with pytest.raises(ValueError):
        trial.configuration('CovarianceDetail')
    cfg = trial.configuration(trial.ARM)
    assert cfg['run_id'] == trial.RUN
    assert cfg['source_draws'] == list(range(1, 21)) + list(range(1, 11))
    assert cfg['training_stages'] == 1
    assert cfg['training_posterior_calls'] == 59070
    assert cfg['training_posterior_rows'] == cfg['training_F0_forward_rows'] == 1_890_000
    assert cfg['training_physical_branch_rows'] == cfg['training_detail_branch_rows'] == 7_560_000
    assert cfg['F0_guidance_for_G_updates'] is True
    assert cfg['deployment_inputs'].startswith('currentreceivedIQonly')
    assert cfg['checkpoint_selection'] == 'final30_only'
    restorer = trial.factory(trial.ARM)
    assert restorer.initialization_report['ancestor_checkpoints'] == []
    assert sum(p.numel() for p in restorer.parameters()) == 5_481_223
