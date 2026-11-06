import torch

from experiments.phase1_dura_orthogonal.source_masked_prior import masked_inputs, errors
from experiments.phase1_dura_orthogonal.source_masked_prior_compare import lag_prediction


def test_lag_baseline_reconstructs_periodic_iq_without_mask_leakage():
    torch.manual_seed(7)
    clean = torch.randn(2, 2, 40).repeat(1, 1, 7)[..., :256]
    starts = torch.tensor([1, 223])
    observed, mask, _, _ = masked_inputs(clean, starts)
    prediction = lag_prediction(observed, mask)
    torch.testing.assert_close(errors(prediction, clean, mask), torch.zeros(2), atol=1e-10, rtol=0)
    altered = clean.clone()
    altered[mask[:, None].expand_as(altered)] += 1000
    observed2, mask2, _, _ = masked_inputs(altered, starts)
    torch.testing.assert_close(prediction, lag_prediction(observed2, mask2))
