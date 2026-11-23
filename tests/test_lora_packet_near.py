import torch

from experiments.pamf_lora_main.broadband_vector import supported
from experiments.pamf_lora_main.complex_budget import chirp_rms
from experiments.pamf_lora_main.denoise_components import bands
from experiments.pamf_lora_main.near_spectrum import near_error
from experiments.pamf_lora_main.packet_near import PacketNear, phase_transfer


def test_phase_transfer_moves_companion_to_reference_frame():
    spectrum = torch.tensor([1+1j, 2-1j])
    chirps = torch.stack((spectrum, 1j*spectrum, torch.zeros_like(spectrum)))[None]
    phase, reliability = phase_transfer(chirps)
    torch.testing.assert_close(phase[0, 0, 1]*chirps[0, 1], chirps[0, 0])
    torch.testing.assert_close(phase[0, 1, 0]*chirps[0, 0], chirps[0, 1])
    torch.testing.assert_close(reliability[0, :2, :2], torch.ones(2, 2))
    assert torch.isfinite(phase).all()
    assert reliability[0, 2].count_nonzero() == 0


def test_context_uses_same_reception_companions_and_preserves_band_doses():
    torch.manual_seed(17)
    y = torch.randn(2, 6, 256, dtype=torch.complex64)
    frozen = supported(torch.randn_like(y), True)
    quality = .8*torch.ones(2, 6, 1)
    model = PacketNear(packet=True)
    first = model(y, .8*y, quality, frozen)
    assert torch.equal(first, frozen)
    near_error(first, .8*y, y).backward()
    assert model.context.out.weight.grad.norm() > 0
    with torch.no_grad():
        model.context.out.weight.fill_(.1)
    modified = y.clone()
    modified[:, 1] *= .1
    for packet in (True, False):
        before = model.context(y, .8*y, quality, packet=packet)[:, 0]
        after = model.context(modified, .8*modified, quality, packet=packet)[:, 0]
        assert torch.allclose(before, after) is not packet
    output = model(y, .8*y, quality, frozen)
    a, b = bands(256, y.device)
    assert torch.equal(output[..., ~(a | b)], frozen[..., ~(a | b)])
    for selected in (a, b):
        torch.testing.assert_close(chirp_rms(output[..., selected]), chirp_rms(frozen[..., selected]))
