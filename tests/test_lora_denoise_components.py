import torch

from experiments.pamf_lora_main.denoise_components import bands, component_readouts, observed_gain, split_along
from experiments.pamf_lora_main.denoise_components_source import MODES, readouts
from experiments.pamf_lora_main.denoise_source import readouts as previous_readouts
from experiments.pamf_lora_main.dura_repair import PacketRestorer


def test_complex_projection_is_orthogonal_and_handles_silent_bins():
    torch.manual_seed(52)
    direction = torch.randn(3, 6, 16, dtype=torch.complex64)
    direction[0, 0] = 0
    value = torch.randn_like(direction)
    parallel, residual = split_along(value, direction)
    torch.testing.assert_close(parallel+residual, value)
    torch.testing.assert_close((residual*direction.conj()).sum(-1), torch.zeros_like(direction[..., 0]),
                               atol=3e-6, rtol=0)
    assert torch.count_nonzero(parallel[0, 0]) == 0


def test_source_swaps_match_the_physical_components_and_dose():
    torch.manual_seed(13)
    y = torch.randn(3, 6, 256, dtype=torch.complex64)
    b, a = y+.5*torch.randn_like(y), y+.15*torch.randn_like(y)
    result, gap = component_readouts(b, a, y)
    assert gap < 1e-6
    near, mid = bands(256, y.device)
    for selected in (near, mid):
        ref = y[..., selected]
        p, n = split_along(b[..., selected]-ref, ref)
        pp, nn = split_along(a[..., selected]-ref, ref)
        torch.testing.assert_close(result['retention_gain'][..., selected], ref+pp+n)
        torch.testing.assert_close(result['retention_residual'][..., selected], ref+p+nn)
        torch.testing.assert_close(result['retention_bands'][..., selected], a[..., selected])
    for waveform in (*result.values(), observed_gain(b, a)):
        assert torch.equal(waveform[..., ~(near | mid)], b[..., ~(near | mid)])


def test_delivered_rows_keep_previous_controls_and_match_dose_without_noise_leak():
    torch.manual_seed(34)
    tracker = PacketRestorer('consensus')
    iq = torch.randn(3, 6, 2, 256)
    noise = .2*torch.randn_like(iq)
    chosen = torch.tensor([True, False, True])

    def gate(y):
        return y, chosen

    def ordinary(y, _q, *, packet):
        assert packet
        return .23*y

    def retained(y, _q, *, packet):
        assert packet
        return .11*y

    models = [ordinary, retained]
    real, n, gap = readouts(iq, noise, gate, tracker, models)
    prior, _ = previous_readouts(iq, noise, gate, tracker, models, (True, True))
    changed, _, _ = readouts(iq, -noise, gate, tracker, models)
    assert n == 2 and gap < 1e-6
    for pos, existing in (('ordinary', 2), ('retained', 3), ('known_noise', 5)):
        assert torch.equal(real[MODES.index(pos)], prior[existing])
    assert torch.equal(real[MODES.index('observed_gain')], changed[MODES.index('observed_gain')])
    assert torch.equal(real[:, ~chosen], iq[~chosen].expand_as(real[:, ~chosen]))
