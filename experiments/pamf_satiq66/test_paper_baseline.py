import math

import torch

from experiments.pamf_satiq66.paper_baseline import PaperSatIQ, angular_triplet, paper_loss


def test_semihard_triplet_uses_cosine_and_falls_back_to_farthest_negative():
    angles = torch.deg2rad(torch.tensor([0., 60., 30., 90.], dtype=torch.float64))
    embedding = torch.stack((angles.cos(), angles.sin()), dim=1)
    iq = torch.ones(4, 2, 3, dtype=torch.float64)
    labels = torch.tensor([4, 4, 9, 17])
    loss, components = paper_loss(embedding, torch.zeros_like(iq), iq, labels)

    # Anchor 0 selects the 90-degree negative. Anchor 60 has only closer negatives.
    triplet = torch.tensor((.5 + .5 + math.sqrt(3) / 2) / 2, dtype=iq.dtype)
    torch.testing.assert_close(components["triplet_loss"], triplet)
    torch.testing.assert_close(components["reconstruction_loss"], torch.ones((), dtype=iq.dtype))
    torch.testing.assert_close(loss, 1 + triplet)
    narrow_margin = torch.tensor((.3 - .5 + math.sqrt(3) / 2) / 2, dtype=iq.dtype)
    torch.testing.assert_close(angular_triplet(embedding, labels, margin=.3), narrow_margin)


def test_baseline_full_forward_has_unit_embedding_and_trains_both_iq_branches():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = PaperSatIQ()
        iq = torch.rand(4, 2, 11000, generator=torch.Generator().manual_seed(42))
        embedding, recon = model(iq)
        assert embedding.shape == (4, 512)
        assert recon.shape == iq.shape
        torch.testing.assert_close(embedding.norm(dim=1), torch.ones(4))
        loss, _ = paper_loss(embedding, recon, iq, torch.tensor([4, 4, 9, 9]))
        loss.backward()

        for branch in (*model.encoders, *model.decoders):
            assert any(p.grad is not None and p.grad.norm() > 0 for p in branch.parameters())
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model.embed(iq), model(iq)[0])
    finally:
        torch.set_num_threads(previous)
