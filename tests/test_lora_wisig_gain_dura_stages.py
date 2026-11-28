import numpy as np
import pytest
import torch
from torch import nn

from experiments.pamf_action_restore.operators import Action
from experiments.pamf_lora_main.gain_dura_screen import (
    COMPLEX_MODES, FACTOR_COUNT, PAIRED_MODES, PAIRED_SEED, STAGE_MODES,
    panel_dimensions, receiver_index, score_stage_panel,
)
from experiments.pamf_lora_main.gain_dura_stage_fit import (
    fit_plan, make_branch, paired_detail_loss, phase_sensitive_loss, source_training_pair,
)
from experiments.pamf_lora_main.gain_dura_stages import ComplexPriorResidual, StageGain
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


@pytest.mark.parametrize('points', [256, 8192])
def test_paired_detail_rejects_common_collapse_and_preserves_relative_complex_phase(points):
    clean, _, _, _ = example(points)
    phase = torch.polar(torch.ones(8), torch.linspace(-.4, .6, 8)).repeat_interleave(points // 8)
    distorted = (clean * phase).detach().requires_grad_(True)
    assert paired_detail_loss(clean*1j, -clean*.5, clean) < 1e-10
    anchored = paired_detail_loss(distorted, distorted, clean)
    assert anchored > .01
    assert paired_detail_loss(clean, distorted, clean) > .5*anchored
    anchored.backward()
    assert torch.isfinite(distorted.grad).all() and distorted.grad.norm() > 0


def test_paired_detail_uses_two_frequency_arms_and_new_fixed_wifi_receiver():
    runtime = {'paired_from': 'frozen'}
    assert fit_plan(runtime) == (PAIRED_SEED, PAIRED_MODES[3:])
    assert panel_dimensions(runtime) == (FACTOR_COUNT, PAIRED_MODES)
    assert receiver_index('wisig', PAIRED_MODES) == 4


def test_complex_head_adds_an_independent_quadrature_with_same_budget_as_aligned_control():
    received, _, _, _ = example(256)
    fields = (received,) * 5
    variance = received.real.new_full((len(received), 1), .1)
    direction = torch.ones_like(received)
    free = ComplexPriorResidual()
    aligned = ComplexPriorResidual(aligned_only=True)
    with torch.no_grad():
        free.spectral[-1].bias[1] = 1.
    aligned.load_state_dict(free.state_dict(), strict=True)
    delta = free(fields, fields, variance, direction)
    projected = aligned(fields, fields, variance, direction)
    assert torch.count_nonzero(delta.imag) > 0
    assert torch.count_nonzero(delta.real) == 0
    assert torch.count_nonzero(projected) == 0
    assert (delta.abs().square().mean(-1).sqrt()
            <= .45 * received.abs().square().mean(-1).sqrt() + 1e-6).all()


@pytest.mark.parametrize('mode', COMPLEX_MODES[3:])
def test_residual_heads_are_live_inside_both_dura_stages(mode):
    observed, gain, context, priors = example(256)
    baseline = StageGain()
    branch = make_branch(mode)
    branch.network.load_state_dict(baseline.network.state_dict())
    outputs = branch(pair(observed), gain, context, priors)
    control = baseline(pair(observed), gain, context, priors)
    torch.testing.assert_close(outputs[-1], control[-1])
    loss = phase_sensitive_loss(outputs, observed)
    for value in torch.autograd.grad(loss, (branch.correction.temporal[-1].weight,
                                            branch.correction.spectral[-1].weight)):
        assert torch.isfinite(value).all() and value.norm() > 0
