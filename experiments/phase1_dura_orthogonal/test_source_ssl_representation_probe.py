import torch
from experiments.phase1_dura_orthogonal.source_ssl_representation_probe import matching_metrics, feature_statistics


def test_record_matching_and_collapsed_features():
    clean = torch.eye(3)
    matched = matching_metrics(clean, clean)
    assert matched['record_top1'] == 1.
    assert matched['matched_cosine'] == 1.
    assert matched['other_cosine'] == 0.
    collapsed = feature_statistics(torch.ones(3, 4))
    assert collapsed == {'feature_std_mean': 0., 'covariance_effective_rank': 0.}
