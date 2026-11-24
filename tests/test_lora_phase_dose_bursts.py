import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.phase_dose import objective
from experiments.pamf_lora_main.phase_dose_bursts_source import RECIPES, burst_observation


def test_burst_observation_is_same_draw_replayable_on_legal_iq():
    rng = torch.Generator().manual_seed(54211)
    source = torch.randn((24, 6, 256), dtype=torch.complex64, generator=rng)
    seed = 824941
    rng.manual_seed(seed)
    first = burst_observation(source, rng)
    rng.manual_seed(seed)
    second = burst_observation(source, rng)
    assert first.shape == source.shape and torch.isfinite(first).all()
    assert torch.equal(first, second) and not torch.equal(first, source)


def test_local_burst_fit_keeps_same_budget_and_joint_gradients():
    rng = torch.Generator().manual_seed(5212)
    iq = real_iq(torch.randn((6, 6, 256), dtype=torch.complex64, generator=rng))
    tracker = PacketRestorer('consensus').eval()
    for recipe in RECIPES.values():
        assert recipe.steps == 40000 and recipe.batch == 24
        model = recipe.model()
        loss, metrics = objective(model, tracker, iq, rng, with_identity=True, recipe=recipe, step=18000)
        loss.backward()
        assert torch.isfinite(metrics).all()
        assert all(torch.count_nonzero(layer.weight.grad) > 0 for layer in (model.coarse[-1], model.fine[-1]))
