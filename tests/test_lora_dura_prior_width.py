import torch

from experiments.pamf_lora_main.dura_prior_width import PacketDura, WIDE_PRIOR_WIDTHS
from experiments.pamf_r65_four_route.stage_polyphase import PRIOR_WIDTHS


def test_dura_expands_only_the_two_priors():
    torch.manual_seed(821)
    compact = PacketDura()
    torch.manual_seed(821)
    wide = PacketDura(wide=True)
    for a,b in zip(compact.restorer.frozen.parameters(), wide.restorer.frozen.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0., rtol=0.)
    for model, widths in ((compact, PRIOR_WIDTHS), (wide, WIDE_PRIOR_WIDTHS)):
        assert len(model.restorer.prior) == 2
        for prior in model.restorer.prior:
            assert tuple(block[0].net[0].out_channels for block in prior.encoder.blocks) == widths
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in compact.parameters())


def test_packet_dura_joint_gradients_and_power():
    torch.manual_seed(822)
    model = PacketDura()
    received = torch.randn(1, 6, 256, dtype=torch.complex64)
    rotation = torch.polar(torch.ones(256), torch.linspace(0., 6., 256)).expand_as(received)
    coarse, final = model(received, None, rotation)
    for wave in (coarse, final):
        assert wave.shape == received.shape and wave.isfinite().all()
        torch.testing.assert_close(wave.abs().square().mean(), received.abs().square().mean())
    target = torch.randn_like(received)
    ((coarse-target).abs().square().mean()+(final-target).abs().square().mean()).backward()
    for block in (model.restorer.frozen.coarse, model.restorer.frozen.estimator, *model.restorer.prior):
        gradients = [p.grad for p in block.parameters() if p.grad is not None]
        assert gradients and all(g.isfinite().all() for g in gradients)
        assert any(g.abs().sum()>0 for g in gradients)
