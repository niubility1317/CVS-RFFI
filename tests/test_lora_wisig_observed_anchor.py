import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.received_quality import capture_features, clean_geometry_weight, fit_clean_geometry
from experiments.pamf_lora_main.shared_gain_dura import SharedGainDura, retain_observed_component
from experiments.pamf_lora_main.shared_gain_dura_fit import aligned_waveform_error


@pytest.mark.parametrize('length', (256, 8192))
def test_relative_stage_projection_preserves_received_component(length):
    torch.manual_seed(11)
    receive = torch.randn(2, length, dtype=torch.complex64)
    proposal = (-3. + 8.j) * receive
    projected = retain_observed_component(receive, proposal)
    torch.testing.assert_close(projected, (.8 + .75j) * receive)
    torch.testing.assert_close(retain_observed_component(receive, torch.zeros_like(receive)), .8 * receive)
    torch.testing.assert_close(retain_observed_component(receive, receive), receive)
    stages, _, _, _ = SharedGainDura(retain_received=True).restoration(receive)
    assert all((out.norm(dim=-1) >= .799 * receive.norm(dim=-1)).all() for out in stages)


def test_clean_geometry_depends_only_on_scale_free_receive_features():
    torch.manual_seed(21)
    phase = torch.arange(256) * .10
    reference = torch.polar(torch.ones_like(phase), phase)
    waves = reference + .1 * torch.randn(1024, 256, dtype=torch.complex64)
    training, calibration = capture_features(waves).chunk(2)
    gate = fit_clean_geometry(training, calibration)
    weights = clean_geometry_weight(waves[512:], gate)
    assert int((weights == 0).sum()) >= 508
    torch.testing.assert_close(capture_features(waves[:3] * (2j)), capture_features(waves[:3]))
    severe = torch.randn_like(waves[512:520])
    assert (clean_geometry_weight(severe, gate) > .99).all()


def test_waveform_objective_is_invariant_to_unobservable_global_phase():
    clean = torch.randn(2, 256, dtype=torch.complex64)
    assert aligned_waveform_error(clean * 1j, clean) < 1e-12
    assert aligned_waveform_error(clean * .2j, clean) > .6


def test_candidate_reuses_exact_blinded_physical_ids(tmp_path):
    runs = tmp_path / 'runs'
    parent = runs / screen.SHARED_RUN
    source = parent / 'lora'
    source.mkdir(parents=True)
    screen.write(source/'complete.json', screen.panel_receipt('lora', 160, screen.SHARED_MODES))
    ids = np.arange(200)
    np.savez(source/'predictions.npz', ids=ids[:160])
    rt = {'observed_anchor': True, 'run_root': str(runs / screen.ANCHOR_RUN),
          'shared_panel_from': str(parent)}
    selected = screen.panel_rows(rt, 'lora', ids, np.ones(200, dtype=int))
    np.testing.assert_array_equal(selected, np.arange(160))
    with pytest.raises(ValueError, match='incompatible physical records'):
        screen.panel_rows(rt, 'lora', ids[::-1], np.ones(200, dtype=int))
