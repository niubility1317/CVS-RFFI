import torch

from experiments.pamf_lora_main.alias_curvature import alias_probe, nonlinear_change, orient_pair, private, substitute
from experiments.pamf_lora_main.chirp_band_power import frequency_groups


def test_probe_uses_only_private_near_band_alias_bins():
    y = torch.randn(3, 6, 256, dtype=torch.complex128)
    raw = torch.fft.fft(y, norm='ortho')
    bins = frequency_groups(256, y.device)[0]
    actual = torch.fft.fft(alias_probe(y), norm='ortho')
    expected = torch.zeros_like(raw)
    expected[..., bins] = private(raw[..., (bins + 128) % 256])
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_plus_minus_removes_first_order_and_is_negative_quadratic_response():
    anchor = torch.randn(3, 6, 256, dtype=torch.complex128)
    probe = alias_probe(anchor)
    change = nonlinear_change(anchor.square(), (anchor + probe).square(), (anchor - probe).square())
    torch.testing.assert_close(change, -probe.square(), atol=1e-12, rtol=1e-12)


def test_keep_physical_energy_and_match_opposite_edits():
    y = torch.randn(3, 6, 256, dtype=torch.complex128)
    d = torch.randn_like(y)
    rms = torch.rand(3, 6, 1, dtype=torch.float64) + .2
    raw = torch.fft.fft(y, norm='ortho')
    bins = frequency_groups(256, y.device)
    a, b, wide = (torch.fft.fft(v,norm='ortho') for v in substitute(y,d,rms))
    assert not torch.equal(a,b)
    assert not torch.equal(a,wide)
    for output in (a,b,wide):
        torch.testing.assert_close(output.mean(1),raw.mean(1),atol=1e-12,rtol=1e-12)
        for sub in bins:
            old = private(raw[...,sub])*rms
            after = private(output[...,sub])*rms
            torch.testing.assert_close(after.abs().square().sum((-1,-2)),old.abs().square().sum((-1,-2)),
                                       atol=1e-12,rtol=1e-12)
    s = lambda x: (rms*(x-raw)).abs().square().sum((-2,-1))
    torch.testing.assert_close(s(a),s(b),atol=1e-12,rtol=1e-12)
    zero = torch.zeros_like(raw[..., bins[0]])
    out = orient_pair(zero, d[..., bins[0]], rms)
    assert not out[0].count_nonzero() and not out[1].count_nonzero()
