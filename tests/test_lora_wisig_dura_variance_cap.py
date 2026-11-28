import numpy as np
import torch
from torch import nn

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.gain_dura_variance import (
    CALIBRATION_PACKETS, calibrate_values, project_output,
)
from experiments.pamf_restoration.models import pair


def test_calibration_freezes_quantiles_from_matched_source_only_values():
    clean = np.linspace(-12, -8, CALIBRATION_PACKETS)
    impaired = np.linspace(-5, 0, CALIBRATION_PACKETS)
    edit = np.linspace(.1, 1.2, CALIBRATION_PACKETS)
    curve = calibrate_values(clean, impaired, edit)
    assert curve['lower_logvar'] == np.quantile(clean, .995)
    assert curve['upper_logvar'] >= curve['lower_logvar'] + np.log(4.)
    assert curve['max_radius'] == .50
    assert 32 <= curve['strong_samples'] < CALIBRATION_PACKETS


def test_full_output_trust_radius_preserves_raw_clean_exactly_and_never_overedits():
    raw = torch.full((4, 256), 2 + 0j, dtype=torch.complex64)
    normalized = raw / 2
    curve = dict(lower_logvar=-10., upper_logvar=-4., max_radius=.5)
    estimate = torch.tensor([-15., -7., -4., 0.]).exp()
    full = raw + 20j
    output = project_output(raw, full, normalized, estimate, curve)
    assert torch.equal(output[0], raw[0])
    torch.testing.assert_close((output - raw).abs().mean(-1), torch.tensor([0., .5, 1., 1.]))
    torch.testing.assert_close(project_output(raw * 3, full * 3, normalized * 3, estimate * 9, curve), output * 3)


def test_variance_readout_returns_a_matched_unbounded_arm_from_one_frozen_call():
    class Gate(nn.Module):
        calls = 0

        def forward(self, received, real_gain, context, priors):
            self.calls += 1
            return [real_gain + .25j]

    gate = Gate()
    raw = torch.full((2, 256), 2 + 0j, dtype=torch.complex64)
    norm = raw / 2
    curve = dict(lower_logvar=-10., upper_logvar=-4., max_radius=.5)
    context = (None, None, torch.tensor([-12., -4.]).exp())
    capped, full = screen.variance_outputs(gate, pair(norm), norm, context, None, raw, 2., curve, None)
    assert gate.calls == 1
    torch.testing.assert_close(full, raw + .5j)
    assert torch.equal(capped[0], raw[0])
    assert torch.equal(capped[1], full[1])


def test_variance_screen_keeps_same_label_free_mode_and_fixed_receiver():
    rt = {'variance_from': 'prior'}
    assert screen.panel_dimensions(rt) == (160, screen.VARIANCE_MODES)
    assert screen.receiver_index('wisig', screen.VARIANCE_MODES) == 6
