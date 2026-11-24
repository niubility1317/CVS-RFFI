import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.neighbor_phase import NeighborPhaseDose, neighbor_angle, remove_parallel
from experiments.pamf_lora_main.neighbor_phase_source import MODES, RECIPES, recover
from experiments.pamf_lora_main.phase_dose import objective


def test_neighbor_angle_preserves_edges_and_linear_ramp():
    t = torch.linspace(-1., 1., 256)
    linear = torch.polar(torch.ones_like(t), 1.7 * t).expand(2, 6, -1)
    angles, signal = neighbor_angle(linear)
    assert angles[..., :4].count_nonzero() == angles[..., -4:].count_nonzero() == 0
    assert angles.abs().max() < 1e-6 and (signal[..., 4:-4] > .999).all()
    changed = linear.clone()
    changed[1, 2, 101] *= 1j
    measured, _ = neighbor_angle(changed)
    assert measured[1, 2, 101] < -1.5
    assert measured[0].abs().max() < 1e-6


def test_weighted_projection_removes_only_parallel_component():
    rng = torch.Generator().manual_seed(72)
    ref = torch.rand((3, 6, 256), generator=rng) - .5
    other = torch.rand(ref.shape, generator=rng)
    w = torch.rand(ref.shape, generator=rng)
    value = remove_parallel(other + 2. * ref, ref, w)
    assert (value * ref * w).sum(-1).abs().max() < 5e-6
    torch.testing.assert_close(value, remove_parallel(other, ref, w), rtol=2e-6, atol=1e-6)


def test_neighbor_fit_and_matched_inference_keep_waveform_contract():
    rng = torch.Generator().manual_seed(249)
    wave = torch.randn((5, 6, 256), dtype=torch.complex64, generator=rng)
    iq = real_iq(wave)
    tracker = PacketRestorer('consensus').eval()
    for recipe in RECIPES.values():
        model = recipe.model()
        loss, metric = objective(model, tracker, iq, rng, with_identity=True, recipe=recipe, step=12000)
        assert torch.isfinite(metric).all()
        loss.backward()
        assert all(layer.weight.grad.norm() > 0 for layer in (model.coarse[-1], model.fine[-1]))
        if model.use_neighbor:
            assert all(layer.weight.grad[1].norm() > 0 for layer in (model.coarse[-1], model.fine[-1]))
    models = (NeighborPhaseDose(False).eval(), NeighborPhaseDose(True).eval())
    with torch.no_grad():
        output, active, evidence = recover(iq, tracker, models, 0., compare=True)
        unchanged, count, _ = recover(iq, tracker, models[-1:], float('inf'))
    assert count == 0 and torch.equal(unchanged, iq.expand_as(unchanged))
    assert active == 5 and len(output) == len(MODES) - 2 and torch.isfinite(evidence).all()
    torch.testing.assert_close(output.norm(dim=-2), wave.abs().expand_as(output.select(-2, 0)),
                               rtol=5e-7, atol=4e-7)
    a = output[MODES.index('neighbor_equal') - 2] - iq
    b = output[MODES.index('repeat_equal') - 2] - iq
    torch.testing.assert_close(a.flatten(1).norm(dim=-1), b.flatten(1).norm(dim=-1), rtol=4e-5, atol=3e-5)
