import torch

from experiments.pamf_phase3_restoration.packet_inverse_filter import (
    PacketInverseFilter,
    apply_inverse,
    repeat_context,
    scratch_packet_filter,
)


def test_reference_keeps_phase_increment_and_handles_zero_iq():
    wave = torch.randn(20, generator=torch.Generator().manual_seed(19))
    short = torch.exp(0.41j * torch.arange(4))[:, None] * wave[None, :]
    iq = torch.zeros(2, 2, 256)
    iq[0, :, 80:160] = torch.stack((short.real.flatten(), short.imag.flatten()))
    ctx = repeat_context(iq)
    torch.testing.assert_close(ctx[0, :20], wave)
    torch.testing.assert_close(ctx[0, 20:40], torch.zeros(20), atol=1e-6, rtol=0)
    torch.testing.assert_close(ctx[0, 40:], torch.tensor([0.41]).mul(1j).exp().view(torch.float32))
    assert torch.equal(ctx[1], torch.zeros_like(ctx[1]))


def test_widely_linear_filter_matches_complex_formula_with_reflect_padding():
    g = torch.Generator().manual_seed(23)
    iq = torch.randn(3, 2, 256, generator=g)
    k = 5
    taps = torch.randn(3, 4, k, 256, generator=g)
    innovation = torch.randn(3, 2, 256, generator=g)
    actual = apply_inverse(iq, taps, innovation)
    padded = torch.nn.functional.pad(iq, (2, 2), mode='reflect')
    u = torch.complex(padded[:, 0], padded[:, 1])
    a = torch.complex(taps[:, 0], taps[:, 1])
    b = torch.complex(taps[:, 2], taps[:, 3])
    expected = torch.complex(iq[:, 0] + innovation[:, 0], iq[:, 1] + innovation[:, 1])
    for j in range(k):
        observed = u[:, j:j+256]
        expected += a[:, j]*observed + b[:, j]*observed.conj()
    torch.testing.assert_close(actual, torch.stack((expected.real, expected.imag), 1), atol=4e-6, rtol=3e-6)


def test_identity_scratch_and_exact_partition():
    iq = torch.randn(2, 2, 256)
    model = scratch_packet_filter(2026092701).eval()
    torch.testing.assert_close(model(iq), iq, atol=0, rtol=0)
    torch.testing.assert_close(model.region_weights.sum(1), torch.ones(1, 256))
    assert torch.equal(model.region_weights[0, :, 0], torch.tensor([1., 0., 0.]))
    assert torch.equal(model.region_weights[0, :, 120], torch.tensor([0., 1., 0.]))
    assert torch.equal(model.region_weights[0, :, 255], torch.tensor([0., 0., 1.]))
    target = PacketInverseFilter()
    target.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(target(iq), iq, atol=0, rtol=0)


def test_active_filter_trains_all_paths_and_is_record_local():
    torch.manual_seed(29)
    iq = torch.randn(2, 2, 256)
    model = PacketInverseFilter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target = .7*iq + .2
    (model(iq)-target).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    restored = model(iq)
    (restored-target).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad.abs().sum() > 0 for p in model.parameters())
    assert not torch.equal(restored, iq)
    with torch.inference_mode():
        torch.testing.assert_close(model(iq[1:]), restored[1:], atol=2e-6, rtol=2e-6)
