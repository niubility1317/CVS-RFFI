import torch

from experiments.pamf_lora_main.complex_budget import BUDGETS, chirp_rms, known_vectors
from experiments.pamf_lora_main.complex_budget_source import MODES, readouts
from experiments.pamf_lora_main.complex_vector_source import readouts as earlier_readouts
from experiments.pamf_lora_main.dura_repair import PacketRestorer


def test_bandwidth_pair_has_same_whole_chirp_energy():
    torch.manual_seed(56)
    x = torch.randn(3, 6, 256, dtype=torch.complex64)
    x[0, 0] = 0.
    result = known_vectors(x)
    f = torch.fft.fftfreq(256).abs()
    for name, cap in BUDGETS.items():
        n, b = (result[f'{key}_{name}'] for key in ('narrow', 'broad'))
        torch.testing.assert_close(chirp_rms(n), chirp_rms(b))
        assert (chirp_rms(n) <= cap + 1e-7).all()
        assert torch.count_nonzero(n[..., (f == 0) | (f > 1/8)]) == 0
        assert torch.count_nonzero(b[..., f == 0]) == 0
    assert torch.equal(result['all_exact'], x)
    assert torch.equal(result['broad_exact'][..., 1:], x[..., 1:])


def test_learned_scales_do_not_depend_on_source_noise_and_keep_old_readout():
    torch.manual_seed(78)
    y, noise = torch.randn(2, 6, 2, 256), .05*torch.randn(2, 6, 2, 256)
    gate = torch.tensor([True, False])

    def baseline(wave):
        return .9*wave, gate

    def model(obs, b, _quality):
        return torch.fft.fft((obs-b)*.06, norm='ortho')

    tracker = PacketRestorer('consensus').eval()
    a, active, _, gap = readouts(y, noise, baseline, tracker, (model, model))
    b, _, _, _ = readouts(y, -noise, baseline, tracker, (model, model))
    earlier, _ = earlier_readouts(y, noise, baseline, tracker, (model, model))
    assert active == 1 and gap < 1e-6
    assert torch.equal(a[:8], b[:8])
    assert torch.equal(a[MODES.index('direct_1x')], earlier[2])
    assert torch.equal(a[MODES.index('balanced_1x')], earlier[3])
    assert torch.equal(a[1:, ~gate], (.9*y[~gate]).expand_as(a[1:, ~gate]))
