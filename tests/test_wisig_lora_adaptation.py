import torch

from experiments.pamf_unrolled_restore.wisig_lora_adaptation import expanded_readout, rank_readout


class IdentityExpanded:
    def stages(self, y, quality, scale, *, receiver_variance):
        assert y.shape[1:] == (6, 256)
        assert torch.equal(y[:, 0], y[:, 5])
        assert receiver_variance.shape == (len(y), 6, 1)
        return (y, y, y), None, None, None, None


class IdentityRank:
    training = False

    def propose(self, y, quality, *, mixed_conv):
        assert y.shape[1:] == (1, 256)
        return y, y, torch.ones_like(quality)


def test_expanded_identity_keeps_received_iq():
    iq = torch.randn(3, 2, 256)
    assert torch.allclose(expanded_readout(IdentityExpanded(), iq), iq, atol=1e-6)


def test_rank_identity_keeps_received_iq():
    iq = torch.randn(3, 2, 256)
    out = rank_readout(IdentityRank(), iq, torch.ones(512), torch.zeros(512, 4))
    assert torch.allclose(out, iq, atol=1e-6)


def test_expanded_uses_supplied_preamble_variance():
    from experiments.pamf_lora_main.dura_capacity import DuraCapacity
    torch.set_num_threads(2)
    model = DuraCapacity('base').eval()
    y = torch.randn(1, 6, 256, dtype=torch.complex64)
    q, scale = torch.ones(1, 6, 1), torch.ones(1, 6, 1)
    observed = torch.full((1, 6, 1), .25)
    with torch.no_grad():
        *_, noise, _ = model.stages(y, q, scale, receiver_variance=observed)
    assert torch.equal(noise, observed)
