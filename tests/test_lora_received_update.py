import torch

from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.received_update import (
    ReceivedUpdate, masked_center, objective, solve_gains,
)
from experiments.pamf_lora_main.received_update_source import MODES, RECIPES, recover, render


def test_directional_box_solution_retains_only_the_supported_direction():
    precision = torch.tensor([[[1., .2], [.2, 1.]], [[1., .2], [.2, 1.]]])
    evidence = torch.tensor([[-.1, .5], [-.1, -.1]])
    active = torch.ones_like(evidence, dtype=torch.bool)
    assert torch.equal(solve_gains(evidence, precision, active, scalar=False),
                       torch.tensor([[0., .5], [0., 0.]]))
    torch.testing.assert_close(solve_gains(evidence, precision, active, scalar=True),
                               torch.tensor([[1/6, 1/6], [0., 0.]]))
    active[0, 1] = False
    assert not solve_gains(evidence, precision, active, scalar=False).any()


def test_centering_preserves_zero_sum_and_masked_samples():
    rng = torch.Generator().manual_seed(11)
    z = torch.randn((7, 6, 256), dtype=torch.complex64, generator=rng)
    valid = torch.rand(z.shape, generator=rng) > .5
    centered = masked_center(z, valid)
    torch.testing.assert_close(centered.sum(1), torch.zeros_like(z[:, 0]), atol=7e-7, rtol=0)
    assert not centered[~valid].any()


def test_waveform_round_trip_and_both_heads_receive_source_gradient():
    rng = torch.Generator().manual_seed(233)
    iq = real_iq(torch.randn((5, 6, 256), dtype=torch.complex64, generator=rng))
    tracker = PacketRestorer('consensus').eval()
    for name, recipe in RECIPES.items():
        model = recipe.model()
        loss, metrics = objective(model, tracker, iq, rng, with_identity=True, recipe=recipe, step=1)
        loss.backward()
        assert torch.isfinite(metrics).all()
        for block in (model.coarse[-1], model.refine[-1], model.risk):
            assert block.weight.grad.norm() > 0, name
    with torch.no_grad():
        output, active, _ = recover(iq, tracker, (ReceivedUpdate(True).eval(),), float('inf'))
    assert active == 0
    assert torch.equal(output, iq.expand_as(output))


def test_active_matrix_records_evidence_and_equal_physical_edits():
    rng = torch.Generator().manual_seed(541)
    iq = real_iq(torch.randn((4, 6, 256), dtype=torch.complex64, generator=rng))
    tracker = PacketRestorer('consensus').eval()
    models = tuple(recipe.model().eval() for recipe in RECIPES.values())
    evidence = []
    with torch.no_grad():
        output, active = render(iq, None, lambda x: (x, None), tracker, models,
                                threshold=0., measurements=evidence)
        single, single_active, _ = recover(iq, tracker, models[-1:], 0.)
    assert active == 4
    assert single_active == 4 and torch.equal(single, output[4:6])
    assert output.shape == (len(MODES), *iq.shape)
    assert torch.isfinite(output).all() and evidence[0].shape == (4, 10)
    edits = output[6:8] - iq
    torch.testing.assert_close(edits.flatten(2).norm(dim=-1)[0], edits.flatten(2).norm(dim=-1)[1],
                               atol=5e-6, rtol=2e-5)
