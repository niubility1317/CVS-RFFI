import pytest
import torch

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.gain_dura_nuisance import add_packet_nuisance
from experiments.pamf_lora_main.gain_dura_stage_fit import fit_plan, make_branch
from experiments.pamf_lora_main.w24_native_dura import capture_scale


@pytest.mark.parametrize('size', [256, 8192])
def test_nuisance_replays_independently_and_preserves_identity_pairs(size):
    raw = torch.randn((16, size), dtype=torch.complex64, generator=torch.Generator().manual_seed(411))
    clean = raw / capture_scale(raw)
    observed = torch.cat((clean[:3], -.8j * clean[3:]))
    base = observed.clone()
    a, y = add_packet_nuisance(clean, observed, torch.Generator().manual_seed(4113))
    b, z = add_packet_nuisance(clean, observed, torch.Generator().manual_seed(4113))
    assert torch.equal(a, b) and torch.equal(y, z)
    assert torch.equal(observed, base)
    assert torch.equal(y[:3], observed[:3]) and torch.equal(a[:3], clean[:3])
    active = (y[3:] != observed[3:]).any(-1)
    assert 0 < active.sum() < 13
    torch.testing.assert_close(capture_scale(y[3:][active]), torch.ones_like(capture_scale(y[3:][active])))
    assert torch.isfinite(y).all() and torch.isfinite(a).all()


def test_drift_pair_uses_the_same_residual_capacity_and_fixed_receiver():
    rt = {'drift_from': 'older-screen', 'train_bound_from': 'source-calibration'}
    assert fit_plan(rt) == (screen.DRIFT_SEED, screen.DRIFT_MODES[3:])
    assert screen.panel_dimensions(rt) == (160, screen.DRIFT_MODES)
    assert screen.receiver_index('wisig', screen.DRIFT_MODES) == 8
    candidate, baseline = [make_branch(name) for name in screen.DRIFT_MODES[3:]]
    assert candidate.state_dict().keys() == baseline.state_dict().keys()
    assert len(list(candidate.parameters())) == len(list(baseline.parameters()))
    assert candidate.correction.aligned_only == baseline.correction.aligned_only is False
