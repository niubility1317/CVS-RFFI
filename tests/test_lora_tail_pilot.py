import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.natural_detail import natural_objective, paired_echo_inputs
from experiments.pamf_lora_main.tail_pilot_source import RECIPES, TAIL_WEIGHT
from experiments.pamf_lora_main.tail_risk import stratified_upper_quarter


def test_tail_loss_gives_each_received_noise_half_equal_weight():
    disturbance = torch.arange(8)
    risk = torch.tensor([[[4., 3., 2., 1., 11., 12., 13., 14.]]], requires_grad=True)
    value = stratified_upper_quarter(risk, disturbance)
    torch.testing.assert_close(value, torch.tensor([9.]))
    value.sum().backward()
    assert risk.grad[0, 0].tolist() == [.5, 0., 0., 0., 0., 0., 0., .5]


def test_matched_first_step_and_tail_updates_the_compact_restorer():
    recipe = RECIPES['tail']
    generator = torch.Generator().manual_seed(23)
    model = recipe.model()
    clean = real_iq(torch.randn(4, 6, 256, dtype=torch.complex64, generator=generator))
    tracker = PacketRestorer('consensus').eval()
    common = dict(with_identity=True, recipe=recipe, step=1,
                  preserve=.35, make_pair=paired_echo_inputs)
    ordinary, measures = natural_objective(
        model, tracker, clean, torch.Generator().manual_seed(17), **common)
    control, observed = natural_objective(
        model, tracker, clean, torch.Generator().manual_seed(17), tail_weight=0., **common)
    candidate, recorded = natural_objective(
        model, tracker, clean, torch.Generator().manual_seed(17), tail_weight=TAIL_WEIGHT, **common)
    torch.testing.assert_close(control, ordinary)
    torch.testing.assert_close(observed[:4], measures)
    torch.testing.assert_close(observed, recorded)
    assert torch.isfinite(candidate) and candidate > control
    candidate.backward()
    for head in (model.coarse.temporal, model.prior):
        grad = head.out.weight.grad
        assert torch.isfinite(grad).all() and grad.norm() > 0
