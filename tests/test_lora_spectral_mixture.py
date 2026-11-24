import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.spectral_mixture import SpectralMixture, gaussian_mixture
from experiments.pamf_lora_main.spectral_mixture_source import FIELDS, MODES, recover
from experiments.pamf_lora_main.variance_transport import physical_norm


def test_fixed_u_mixture_fit_keeps_finite_monotone_likelihood_and_normalized_weights():
    generator = torch.Generator().manual_seed(325)
    examples = torch.randn(203, 12, generator=generator) * .22
    examples[:99, :5] += 2.
    centers, covariance, weights, responsibilities, trace = gaussian_mixture(examples)
    assert torch.isfinite(centers).all() and (covariance > 0).all()
    torch.testing.assert_close(weights.sum(), torch.ones(()))
    torch.testing.assert_close(responsibilities.sum(-1), torch.ones(len(examples)))
    assert (trace.diff() >= -1e-4).all()


def test_route_swaps_real_halves_and_preserves_edit_comparison():
    generator = torch.Generator().manual_seed(53)
    waves = torch.randn(42, 6, 256, dtype=torch.complex64, generator=generator)
    model = SpectralMixture()
    model.fit(lambda: waves.split(14))
    received = waves[:3]
    weights, confidence = model.routing(received)
    swapped, confidence_swapped = model.routing(received[:, [1, 0, 3, 2, 5, 4]])
    torch.testing.assert_close(swapped, weights[[1, 0, 2]], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(confidence_swapped, confidence, atol=1e-6, rtol=1e-5)
    contaminated = received.clone()
    contaminated[:, ::2] *= 5j
    changed, _ = model.routing(contaminated)
    torch.testing.assert_close(changed[1], weights[1], atol=1e-7, rtol=1e-6)
    assert not torch.allclose(changed[0], weights[0])
    assert confidence.min() >= 0 and confidence.max() <= 1
    rms = torch.linspace(.7, 1.4, 6)[None, :, None]
    edits, _, _ = model.changes(received, rms)
    analytic, _ = super(SpectralMixture, model).changes(received, rms)
    torch.testing.assert_close(edits[0], analytic[2], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(edits.mean(-1), torch.zeros_like(edits.mean(-1)), atol=1e-7, rtol=0)
    doses = physical_norm(edits, rms)
    torch.testing.assert_close(doses[6], doses[7], atol=1e-7, rtol=1e-5)

    iq = real_iq(received)
    tracker = PacketRestorer('consensus').eval()
    out, n, info = recover(iq, tracker, model, 0.)
    assert out.shape == (len(MODES) - 2, *iq.shape) and n == len(iq) and info.shape == (n, len(FIELDS))
    bypass, n, _ = recover(iq, tracker, None, float('inf'))
    assert n == 0 and torch.equal(bypass, iq.expand_as(bypass))
