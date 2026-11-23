import torch

from experiments.pamf_lora_main.broadband_vector import supported
from experiments.pamf_lora_main.complex_budget import chirp_rms
from experiments.pamf_lora_main.denoise_components import bands
from experiments.pamf_lora_main.dura_repair import PacketRestorer
from experiments.pamf_lora_main.near_spectrum import NearSpectrum, near_error, same_signal
from experiments.pamf_lora_main.near_spectrum_source import MODES, readouts


def test_refiner_preserves_outer_coefficients_and_each_near_band_norm():
    torch.manual_seed(73)
    y = torch.randn(2, 6, 256, dtype=torch.complex64)
    frozen = supported(torch.randn_like(y), True)
    quality = .8*torch.ones(2, 6, 1)
    model = NearSpectrum()
    first = model(y, .8*y, quality, frozen)
    assert torch.equal(first, frozen)
    near_error(first, .8*y, y).backward()
    assert all(b.out.weight.grad.norm() > 0 for b in (model.spectral, model.temporal))
    with torch.no_grad():
        model.spectral.out.bias.fill_(.6)
    output = model(y, .8*y, quality, frozen)
    a, b = bands(256, y.device)
    assert not torch.equal(output, frozen)
    assert torch.equal(output[..., ~(a | b)], frozen[..., ~(a | b)])
    for selected in (a, b):
        torch.testing.assert_close(chirp_rms(output[..., selected]), chirp_rms(frozen[..., selected]))


def test_independent_draw_consistency_uses_common_physical_coordinates():
    torch.manual_seed(49)
    y = torch.randn(2, 6, 256, dtype=torch.complex64)
    angle = .4+2.*torch.pi*11*torch.arange(256)/256
    phase = torch.polar(torch.ones_like(angle), angle).expand_as(y)
    to_first = torch.stack((torch.ones_like(phase), phase), dim=1)
    truth = torch.stack((y, y/phase), dim=1)
    pair = torch.fft.fft(torch.stack((.04*y, .04*y/phase), dim=1), norm='ortho')
    common = same_signal(pair, truth, truth, to_first)
    pair[:, 1] *= -1.
    assert common < 1e-10
    assert same_signal(pair, truth, truth, to_first) > common + 1e-5


def test_true_noise_only_changes_oracle_outputs():
    torch.manual_seed(91)
    y, noise = torch.randn(2, 6, 2, 256), .06*torch.randn(2, 6, 2, 256)
    mask = torch.tensor([True, False])

    def gate(iq):
        return .9*iq, mask

    def wide(x, b, _):
        return supported(torch.fft.fft(x-b, norm='ortho'), True)

    models = tuple(NearSpectrum().eval() for _ in range(2))
    tracker = PacketRestorer('consensus').eval()
    a, n = readouts(y, noise, gate, tracker, models, wide=wide)
    b, _ = readouts(y, -noise, gate, tracker, models, wide=wide)
    assert n == 1 and a.shape == (len(MODES), *y.shape)
    assert torch.equal(a[:6], b[:6])
    assert torch.equal(a[1:, 1], (.9*y[1]).expand_as(a[1:, 1]))
