import torch

from experiments.pamf_lora_main.angular_transport import AngularTransport, initialize, inner, objective
from experiments.pamf_lora_main.angular_transport_source import MODES, RECIPES, recover
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq


def test_learned_rotation_holds_physical_dose_and_bypasses_clean():
    generator = torch.Generator().manual_seed(17)
    all_iq = torch.randn(31, 6, 256, dtype=torch.complex64, generator=generator)
    model = AngularTransport(angular=True).eval()
    model.prior.calibrate(all_iq.split(12))
    sample = all_iq[:4]
    rms = torch.linspace(.8, 1.2, 6)[None, :, None].expand(4, -1, -1)
    context = model.observe(sample, torch.ones_like(rms), rms)
    coarse, main = model.stages(sample, rms, context)
    _, parallel = model.stages(sample, rms, context, angular=False)
    analytic = context[0]
    built, _ = model.prior.changes(sample, rms)
    torch.testing.assert_close(analytic, .5 * built[2], rtol=1e-5, atol=1e-6)
    expected = inner(analytic, analytic, rms)
    for edit in (coarse, main, parallel):
        torch.testing.assert_close(inner(edit, edit, rms), expected, rtol=1e-5, atol=1e-7)
        assert edit.mean(dim=-1).abs().max() < 1e-6
    assert not torch.allclose(main, parallel)

    iq = real_iq(sample)
    tracker = PacketRestorer('consensus').eval()
    modes, active, telemetry = recover(iq, tracker, (model, model), 0., compare=True)
    assert modes.shape == (len(MODES) - 2, *iq.shape) and active == len(iq)
    torch.testing.assert_close(telemetry[:, 5:7], telemetry[:, 6:8], rtol=1e-5, atol=1e-7)
    bypass, active, _ = recover(iq, tracker, (model, model), float('inf'), compare=True)
    assert active == 0 and torch.equal(bypass, iq.expand_as(bypass))


def test_both_arms_optimize_coarse_and_fine_from_source_waveforms():
    torch.manual_seed(51)
    iq = torch.randn(23, 6, 2, 256)
    tracker = PacketRestorer('consensus').eval()
    for name, recipe in RECIPES.items():
        model = recipe.model()
        initialize(model, tracker, iq, None)
        loss, losses = objective(model, tracker, iq[:5], torch.Generator().manual_seed(89),
                                 with_identity=True, recipe=recipe, step=330)
        assert losses.shape == (7,) and torch.isfinite(losses).all()
        loss.backward()
        for stage in (model.coarse, model.fine):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stage.parameters()), name
            assert sum(p.grad.abs().sum() for p in stage.parameters()) > 0
