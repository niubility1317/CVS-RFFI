import torch

from experiments.pamf_lora_main.band_allocation import band_energy, band_masks, fft, ifft
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates
from experiments.pamf_lora_main.weak_sideband import limited_turn, source_reference
from experiments.pamf_lora_main.weak_sideband_source import MODES, RECIPES, objective, recover


def test_remove_only_the_clean_carrier_before_changing_coordinate():
    rng = torch.Generator().manual_seed(12)
    clean_frame = torch.ones((1, 6, 256), dtype=torch.complex128)
    rx_phase = torch.arange(256, dtype=torch.float64)*.018
    received_frame = torch.polar(torch.ones_like(rx_phase), rx_phase)[None, None]
    clean_tone = .8*torch.ones_like(clean_frame[..., :1])
    weak = torch.randn(clean_frame.shape, generator=rng, dtype=clean_frame.dtype)*.03
    weak = weak - weak.mean(-1, keepdim=True)
    source = (clean_tone+weak) * clean_frame
    received = .7*torch.ones_like(source)
    reference = source_reference(source, clean_frame, clean_tone, received, received_frame, separated=True)
    unreduced = source_reference(source, clean_frame, clean_tone, received, received_frame, separated=False)
    assert torch.allclose(reference.mean(-1), received.mean(-1))
    transported = weak/received_frame
    assert torch.allclose(reference-received, transported-transported.mean(-1, keepdim=True))
    assert (fft(reference-unreduced)[..., 1:].abs().square().mean() > .001)


def test_weak_turn_keeps_physical_band_energy_and_far_spectrum():
    rng = torch.Generator().manual_seed(55)
    basis = band_masks().to(torch.float64)
    raw = torch.randn((5,6,256), generator=rng, dtype=torch.complex128)
    source = raw*basis.sum(0)
    rms = torch.rand((5,6,1), generator=rng, dtype=torch.float64) + .2
    proposal = torch.randn(source.shape, generator=rng, dtype=source.dtype)
    output = fft(limited_turn(ifft(source), proposal, rms, basis[:2].bool()))
    assert torch.allclose(band_energy(output, rms, basis), band_energy(source, rms, basis), atol=1e-10)
    assert torch.allclose(output[..., basis[2].bool()], source[..., basis[2].bool()], atol=1e-12)
    assert output[...,0].abs().max() < 1e-12


def test_active_two_arm_predictions_and_source_gradient():
    rng = torch.Generator().manual_seed(33)
    tracker = PacketRestorer('consensus').eval()
    pure = tracker.reference.chirp.expand(4,6,-1)
    source = real_iq(pure + .12*torch.randn(pure.shape, dtype=pure.dtype, generator=rng))
    models = []
    for recipe in RECIPES.values():
        model = recipe.model()
        model.prior.calibrate([coordinates(tracker,source)[0]])
        loss, fields = objective(model, tracker, source, rng, with_identity=True, recipe=recipe, step=1)
        loss.backward()
        assert torch.isfinite(fields).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        assert sum(p.grad.abs().sum() for p in model.parameters()) > 0
        models.append(model.eval())
    with torch.no_grad():
        matrix, count, telemetry = recover(source, tracker, models, 0., compare=True)
        assert count == len(source) and matrix.shape == (len(MODES) - 2, *source.shape)
        assert telemetry[:, -1].max() < 1e-4
        matrix, count, _ = recover(source, tracker, models, float('inf'), compare=True)
        assert count == 0 and torch.equal(matrix, source.expand_as(matrix))
