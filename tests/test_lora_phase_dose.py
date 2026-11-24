import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.phase_dose import PhaseDose, objective
from experiments.pamf_lora_main.phase_dose_source import MODES, RECIPES, recover


@torch.no_grad()
def test_phase_dose_readouts_preserve_magnitude_and_match_actual_edit():
    rng = torch.Generator().manual_seed(729)
    wave = torch.randn((5, 6, 256), dtype=torch.complex64, generator=rng)
    iq = real_iq(wave)
    tracker = PacketRestorer('consensus').eval()
    outputs, active, evidence = recover(iq, tracker, (PhaseDose(False).eval(), PhaseDose(True).eval()),
                                        0., compare=True)
    assert active == 5 and outputs.shape == (len(MODES) - 2, *iq.shape)
    assert torch.isfinite(evidence).all()
    torch.testing.assert_close(outputs.norm(dim=-2), iq.norm(dim=-2).expand_as(outputs[:, :, :, 0]),
                               atol=5e-7, rtol=5e-7)
    i = MODES.index('local_equal') - 2
    j = MODES.index('pooled_equal') - 2
    a, b = ((outputs[k] - iq).flatten(1).norm(dim=1) for k in (i, j))
    torch.testing.assert_close(a, b, rtol=4e-5, atol=2e-5)
    unchanged, active, _ = recover(iq, tracker, (PhaseDose(True).eval(),), float('inf'))
    assert not active and torch.equal(unchanged, iq.expand_as(unchanged))


def test_two_fit_arms_have_finite_source_gradients():
    rng = torch.Generator().manual_seed(584)
    iq = real_iq(torch.randn((6, 6, 256), dtype=torch.complex64, generator=rng))
    tracker = PacketRestorer('consensus').eval()
    for recipe in RECIPES.values():
        model = recipe.model()
        loss, metrics = objective(model, tracker, iq, rng, with_identity=True, recipe=recipe, step=1)
        assert torch.isfinite(metrics).all()
        loss.backward()
        for layer in (model.coarse[-1], model.fine[-1]):
            assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.norm() > 0.
