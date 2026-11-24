import torch

from experiments.pamf_lora_main.band_allocation import (
    BandAllocation, band_energy, band_masks, equal_budget, ifft,
)
from experiments.pamf_lora_main.band_allocation_diagnostic import oracle_readouts, substitutions
from experiments.pamf_lora_main.band_allocation_source import MODES, RECIPES, objective, recover
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates


def test_energy_projection_respects_fixed_bands_and_keeps_gradients():
    rng = torch.Generator().manual_seed(375)
    base = torch.rand((256, 3), generator=rng, dtype=torch.float64)
    base[::9, 0] = 0.
    select = (torch.rand(base.shape, generator=rng) > .3) & (base > 0)
    proposal = (base * torch.rand(base.shape, generator=rng) ** 3).requires_grad_()
    gain = equal_budget(base, proposal, select)
    assert torch.allclose((base * gain.square()).sum(-1), base.sum(-1), rtol=2e-12, atol=2e-12)
    assert torch.equal(gain[~select], torch.ones_like(gain[~select]))
    assert gain.min() >= .5 and gain.max() <= 1.5 + 1e-14
    gain.square().sum().backward()
    assert torch.isfinite(proposal.grad).all()
    assert proposal.grad.abs().max() > 0


def test_exact_band_controls_separate_energy_and_direction():
    rng = torch.Generator().manual_seed(52)
    bands = band_masks().to(torch.float64)
    frame = torch.rand((4, 6, 1), generator=rng)
    anchor = torch.randn((4, 6, 256), generator=rng, dtype=torch.complex128)
    exact = torch.randn(anchor.shape, generator=rng, dtype=anchor.dtype)
    anchor, exact = anchor * bands.sum(0), exact * bands.sum(0)
    edits, evidence = substitutions(anchor, exact, frame, bands)
    e0, e1 = (band_energy(spectrum, frame, bands) for spectrum in (anchor, exact))
    assert evidence[:, 0].bool().all()
    assert torch.allclose(band_energy(edits[1], frame, bands) / e1, e0.sum(-1, keepdim=True) / e1.sum(-1, keepdim=True))
    assert torch.allclose(band_energy(edits[2], frame, bands), e0)
    assert torch.allclose((edits[1] * anchor.conj()).imag, torch.zeros_like(anchor.real), atol=1e-12)
    assert torch.allclose((edits[2] * exact.conj()).imag, torch.zeros_like(anchor.real), atol=1e-12)
    assert torch.equal(edits[4][..., ~bands[0].bool()], anchor[..., ~bands[0].bool()])
    assert torch.allclose((ifft(edits)*frame).abs().square().sum((-1,-2)), e0.sum(-1).expand(8,-1))


def test_real_source_objective_trains_and_clean_uses_bitwise_bypass():
    rng = torch.Generator().manual_seed(103)
    tracker = PacketRestorer('consensus').eval()
    clean = tracker.reference.chirp[None, None, :].expand(4, 6, -1).clone()
    noise = torch.randn(clean.shape, dtype=clean.dtype, generator=rng)
    iq = real_iq(clean + .1 * noise)
    model = BandAllocation()
    model.learn_source_statistics([coordinates(tracker, iq)[0]])
    recipe = RECIPES['band']
    loss, risk = objective(model, tracker, iq, rng, with_identity=True, recipe=recipe, step=1)
    loss.backward()
    assert torch.isfinite(risk).all()
    assert sum(parameter.grad.abs().sum() for parameter in model.parameters()) > 0
    model.eval()
    with torch.no_grad():
        output, triggered, _ = recover(iq, tracker, (model,), float('inf'))
        assert triggered == 0
        assert torch.equal(output[0], iq)
        output, triggered, rows = recover(iq, tracker, (model,), 0., matrix=True)
        assert triggered == len(iq)
        assert output.shape == (len(MODES) - 2, *iq.shape)
        assert torch.isfinite(rows).all()
        assert torch.equal(output[3], output[7])

        class FrozenDirection:
            def changes(self, y, rms):
                anchor = model.prior.changes(y, rms)[0][2] * .5
                return torch.stack((anchor, anchor)), None, None

        result, count, stats = oracle_readouts(iq, real_iq(.1*noise), tracker, FrozenDirection(), 0.)
        assert count == len(iq)
        assert result.shape == (9, *iq.shape)
        assert stats[:, -1].max() < 5e-6
