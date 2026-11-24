import torch

from experiments.pamf_lora_main.dura_prior_width import PacketDura
from experiments.pamf_lora_main.dura_stage_step_target import DOSE, accepted_step, stage_reads
from experiments.pamf_lora_main.relative_carrier import mix


def test_post_normalization_check_rejects_a_raw_chord_optimum_that_increases_residual():
    first = torch.tensor([[[1+0j, 0j]]])
    second = torch.tensor([[[0j, 1+0j]]])
    received = first + .5 * second
    weights = torch.tensor([[[4.,1.]]])
    observe = lambda x: x
    emitted, alpha, before, after = accepted_step(received, first, second, first, second, observe,
                                                   weights, noncarrier=False)
    # A small positive analytic step is accepted in this example.
    assert 0 < alpha.item() < 1
    assert after.item() < before.item()
    torch.testing.assert_close(emitted.abs().square().sum(), first.abs().square().sum())

    # The raw chord passes through zero. Restoring its power jumps to either endpoint.
    opposite = -first
    received = torch.zeros_like(first)
    restored, alpha, before, after = accepted_step(received, first, opposite, first, opposite, observe,
                                                     weights, noncarrier=False)
    torch.testing.assert_close(restored, first, atol=0, rtol=0)
    assert alpha.item() == 0 and before.item() == after.item()


def test_carrier_projection_prevents_a_constant_only_change_from_deciding():
    first = torch.tensor([[[1+0j, -1+0j]]])
    second = torch.tensor([[[2**.5+0j, 0j]]])
    received = second.clone()
    weights = torch.ones_like(first.real)
    observe = lambda x: x.mean(dim=-1,keepdim=True).expand_as(x)
    before_y = observe(first)
    after_y = observe(second)
    broadband, b, _, _ = accepted_step(received, first, second, before_y, after_y, observe, weights,
                                       noncarrier=False)
    detail, d, _, _ = accepted_step(received, first, second, before_y, after_y, observe, weights,
                                    noncarrier=True)
    assert b.item() == 1 and d.item() == 0
    torch.testing.assert_close(broadband, second)
    torch.testing.assert_close(detail, first)


@torch.no_grad()
def test_staged_readouts_share_the_original_second_quarter_and_power():
    torch.manual_seed(978)
    model = PacketDura().eval()
    received = torch.randn((2,6,256), dtype=torch.complex64)
    rotation = torch.polar(torch.ones(256), torch.linspace(0., 5., 256)).expand_as(received)
    restored, checks = stage_reads(received, rotation, model)
    assert len(restored) == 5
    torch.testing.assert_close(restored[1], mix(received, model(received,None,rotation)[-1], DOSE),
                               atol=0, rtol=0)
    for wave in restored:
        torch.testing.assert_close(wave.abs().square().sum((-2,-1)), received.abs().square().sum((-2,-1)))
    for alpha, before, after in checks:
        assert ((alpha >= 0) & (alpha <= 1) & (after <= before)).all()
