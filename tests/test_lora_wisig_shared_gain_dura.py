import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.shared_gain_dura import SharedGainDura
from experiments.pamf_lora_main.shared_gain_dura_fit import objective


@pytest.mark.parametrize('length', (256, 8192))
def test_shared_graph_preserves_length_and_honors_clean_bypass(length):
    torch.manual_seed(7)
    model = SharedGainDura()
    with torch.no_grad():
        model.priors[-1][-1].bias.fill_(.4)
        raw = torch.randn(2, length, dtype=torch.complex64)
        restored = model(raw, clean_cut=-20.)
        scaled = model(2. * raw, clean_cut=-20.)
        bypassed = model(raw, clean_cut=20.)
    assert restored.shape == raw.shape
    assert torch.isfinite(restored).all()
    assert not torch.equal(restored, raw)
    torch.testing.assert_close(scaled, 2. * restored)
    assert torch.equal(bypassed, raw)


def test_same_initialization_updates_both_stages_action_and_gain():
    torch.manual_seed(12)
    model = SharedGainDura()
    control = SharedGainDura(packet_gain=True)
    control.load_state_dict(model.state_dict(), strict=True)
    raw = torch.randn(2, 256, dtype=torch.complex64).repeat(2, 1)
    observation = raw + .4 * torch.randn_like(raw)
    assert torch.equal(model(observation, clean_cut=-2.), control(observation, clean_cut=-2.))
    for arm in (model, control):
        heads = (arm.coarse[-1].weight, arm.channel.weight, arm.local_phase.weight,
                 arm.gain[-1].weight, *(prior[-1].weight for prior in arm.priors))
        gradients = torch.autograd.grad(objective(arm, raw, observation), heads)
        assert all(torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in gradients)


def test_shared_panel_selects_only_registered_receivers():
    rt = {'shared_model': True}
    for dataset, receiver in (('lora', 1), ('wisig', 3)):
        rxs = np.repeat([receiver, 0], [200, 200])
        rows = screen.panel_rows(rt, dataset, np.arange(len(rxs)), rxs)
        assert len(rows) == 160
        assert (rxs[rows] == receiver).all()


def test_wisig_panel_uses_all_unexposed_rows_before_reusing(tmp_path):
    runs = tmp_path / 'runs'
    rt = {'shared_model': True, 'run_root': str(runs / screen.SHARED_RUN)}
    panels = (('shared_prior_frequency', screen.FREQUENCY_RETRY, screen.FREQUENCY_MODES),
              ('shared_prior_stage', screen.STAGE_RUN, screen.STAGE_MODES))
    for i, (key, run, modes) in enumerate(panels):
        rt[key] = str(runs / run)
        folder = runs / run / 'wisig'
        folder.mkdir(parents=True)
        screen.write(folder/'complete.json', screen.panel_receipt('wisig', 160, modes))
        np.savez(folder/'predictions.npz', ids=np.arange(i * 160, (i + 1) * 160))
    source_rx = np.full(360, 3)
    rows = screen.panel_rows(rt, 'wisig', np.arange(360), source_rx)
    assert len(rows) == len(np.unique(rows)) == 160
    assert (source_rx[rows] == 3).all()
    assert np.array_equal(rows[-40:], np.arange(320, 360))


def test_short_inputs_outside_the_native_contract_fail():
    with pytest.raises(ValueError, match='native complex IQ'):
        SharedGainDura()(torch.randn(2, 255, dtype=torch.complex64), clean_cut=0.)
