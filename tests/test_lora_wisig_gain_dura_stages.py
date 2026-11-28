import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.gain_dura_screen import FACTOR_COUNT, STAGE_MODES, score_stage_panel
from experiments.pamf_lora_main.gain_dura_stage_fit import phase_sensitive_loss, source_training_pair
from experiments.pamf_lora_main.gain_dura_stages import StageGain
from experiments.pamf_r65_four_route.stage_polyphase import _stages
from experiments.pamf_restoration.models import complex_iq, pair


def example(points):
    rng = torch.Generator().manual_seed(351)
    recorded = torch.randn(2, points, dtype=torch.complex64, generator=rng)
    gain = .8 * recorded + .25j * recorded.roll(1, dims=-1)
    base = pair(.3 * recorded)
    k = torch.tensor([[1., -.2 + .1j]]).expand(2, -1)
    action = Action(k, torch.ones_like(recorded), .06j * torch.ones_like(recorded))
    context = (base, action, torch.full((len(recorded),), .1))
    priors = nn.ModuleList([nn.Conv1d(9, 2, 3, padding=1), nn.Conv1d(9, 2, 3, padding=1)])
    return recorded, gain, context, priors.requires_grad_(False)


@pytest.mark.parametrize('points', [256, 8192])
def test_common_real_gain_inside_dura_reaches_both_frozen_endpoints(points):
    observed, gain, context, priors = example(points)
    reference, _, _, _ = _stages(priors, pair(observed), context, points=points)
    model = StageGain()
    model.coefficient = lambda observed, *args: torch.zeros_like(observed.real)
    for z in model(pair(observed), gain, context, priors):
        torch.testing.assert_close(z, gain, atol=1e-6, rtol=1e-6)
    model.coefficient = lambda observed, *args: torch.ones_like(observed.real)
    stages = model(pair(observed), gain, context, priors)
    for z, control in zip(stages, reference, strict=True):
        torch.testing.assert_close(z, complex_iq(control), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('packet_only', [False, True])
def test_both_matched_gate_arms_learn_from_paired_source_without_changing_experts(packet_only):
    observed, gain, context, priors = example(256)
    model = StageGain(packet_only=packet_only)
    rng = torch.Generator().manual_seed(51)
    clean, _ = source_training_pair(observed, rng)
    stages = model(pair(observed), gain, context, priors)
    phase_sensitive_loss(stages, clean).backward()
    assert torch.isfinite(stages[-1]).all()
    assert model.network[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in priors.parameters())


def test_independent_screen_counts_corrections_when_both_frozen_experts_are_wrong():
    n = FACTOR_COUNT
    labels = np.zeros(n, np.int16)
    predictions = np.zeros((2, len(STAGE_MODES), n), np.int16)
    predictions[1, 1:3, :12] = 1
    predictions[1, 0, :12] = 1
    predictions[1, 4, :7] = 1
    result = score_stage_panel(predictions, labels)
    assert result['both_experts_wrong']['practical_low_urban'] == 12
    assert result['corrected_when_both_wrong']['stage_frequency']['practical_low_urban'] == 12
    assert result['corrected_when_both_wrong']['stage_packet']['practical_low_urban'] == 5
    assert result['candidates']['stage_frequency']['promote']
    assert not result['candidates']['stage_packet']['promote']
