import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.repeated_edit import MAX_EDIT, PREAMBLE, repeated_edit_outputs


@pytest.mark.parametrize('length', (256, 8192))
def test_projection_preserves_private_period_difference_and_matches_edit_power(length):
    torch.manual_seed(5)
    start, count, period = PREAMBLE[length]
    stop = start + count * period
    packet = torch.randn(1, length, dtype=torch.complex64)
    phase = torch.arange(count) * .7
    rotation = torch.polar(torch.ones_like(phase), phase).unsqueeze(-1)
    common = torch.randn(1, period, dtype=torch.complex64)
    packet[:, start:stop] = (rotation * common).flatten()
    proposal = packet + .5 * torch.randn_like(packet) + .4 * packet
    coherent, control = repeated_edit_outputs(packet, proposal)
    for result in (coherent, control):
        assert torch.equal(result[:, :start], packet[:, :start])
        assert torch.equal(result[:, stop:], packet[:, stop:])
    delta = coherent - packet
    assert 0 < delta.norm() / packet[:, start:stop].norm() <= MAX_EDIT + 1e-6
    torch.testing.assert_close(delta.norm(), (control-packet).norm(), rtol=1e-5, atol=1e-5)
    aligned = delta[:, start:stop].reshape(count, period) * rotation.conj()
    torch.testing.assert_close(aligned, aligned[0].expand_as(aligned))
    assert (coherent - control).norm() > .01


def test_original_packet_exact_when_frozen_model_bypasses():
    y = torch.randn(3, 256, dtype=torch.complex64)
    assert all(torch.equal(x, y) for x in repeated_edit_outputs(y,y))


def test_new_stage_reuses_only_matching_preamble_screen_physical_rows(tmp_path):
    prior = tmp_path / screen.ANCHOR_RUN
    folder = prior / 'wisig'
    folder.mkdir(parents=True)
    ids = np.arange(200)
    screen.write(folder/'complete.json', screen.panel_receipt('wisig',160,screen.ANCHOR_MODES))
    np.savez(folder/'predictions.npz', ids=ids[:160])
    rt = {'repeat_edit_from': str(prior), 'run_root':str(tmp_path/screen.REPEAT_RUN)}
    picked = screen.panel_rows(rt,'wisig',ids,np.full(200,3))
    np.testing.assert_equal(picked,np.arange(160))
    with pytest.raises(ValueError,match='incompatible physical'):
        screen.panel_rows(rt,'wisig',ids,np.full(200,9))
