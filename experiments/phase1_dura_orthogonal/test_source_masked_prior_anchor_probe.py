import torch

from experiments.phase1_dura_orthogonal.source_masked_prior_anchor_probe import prior_wave, ideal_probe


class Ones(torch.nn.Module):
    def forward(self, observed, mask, scale):
        return torch.ones_like(observed)


def test_stitching_covers_interior_and_preserves_endpoints():
    x = torch.randn(2, 2, 256)
    output = prior_wave(Ones(), x)
    torch.testing.assert_close(output[:, :, 1:-1], torch.ones_like(output[:, :, 1:-1]))
    torch.testing.assert_close(output[:, :, 0], x[:, :, 0])
    torch.testing.assert_close(output[:, :, -1], x[:, :, -1])


def test_positive_and_signed_source_oracle():
    anchor = torch.ones(2, 2, 256)
    prediction = torch.zeros_like(anchor)
    clean = anchor + torch.tensor([.25, -.25])[:, None, None]*(prediction-anchor)
    values = ideal_probe(anchor, clean, prediction)
    torch.testing.assert_close(values[:, 3], torch.tensor([.25, -.25]))
    assert values[0, 1] < 1e-10 and values[1, 1] == values[1, 0]
    assert values[1, 2] < 1e-10
