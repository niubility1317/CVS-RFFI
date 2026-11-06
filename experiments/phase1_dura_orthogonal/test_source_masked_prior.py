import torch

from experiments.phase1_dura_orthogonal.source_masked_prior import masked_inputs, errors, SPAN


def test_mask_and_linear_interpolation_use_only_visible_neighbours():
    t = torch.arange(256, dtype=torch.float32)
    clean = torch.stack((t, -2*t))[None]
    starts = torch.tensor([40])
    observed, mask, scale, linear = masked_inputs(clean, starts)
    assert mask.sum() == SPAN and torch.count_nonzero(observed[:, :, 40:72]) == 0
    torch.testing.assert_close(errors(linear, clean, mask), torch.zeros(1), atol=1e-9, rtol=0)
    changed = clean.clone()
    changed[:, :, 40:72] += 10000
    observed2, mask2, scale2, linear2 = masked_inputs(changed, starts)
    torch.testing.assert_close(observed, observed2)
    torch.testing.assert_close(mask, mask2)
    torch.testing.assert_close(scale, scale2)
    torch.testing.assert_close(linear, linear2)
