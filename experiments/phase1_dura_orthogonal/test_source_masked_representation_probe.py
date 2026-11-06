import torch

from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior
from experiments.phase1_dura_orthogonal.source_masked_representation_probe import features, raw_features, MASK_START


def test_masked_features_ignore_hidden_samples():
    torch.manual_seed(5)
    model = MaskedIQPrior().eval()
    x = torch.randn(4, 2, 256)
    altered = x.clone()
    altered[:, :, MASK_START:MASK_START+32] += 1000
    torch.testing.assert_close(features(model, x), features(model, altered), rtol=0, atol=0)
    torch.testing.assert_close(raw_features(x), raw_features(altered), rtol=0, atol=0)
