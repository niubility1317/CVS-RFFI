import pytest
import torch
from experiments.phase1_dura_orthogonal.variance_factor_source_probe import summarize_factors, FACTORS


def test_group_oracle_contains_unmodified_baseline_and_counts():
    assert FACTORS.index(1.) == 3
    errors = torch.tensor([[.3, .2, .4, .5, .6, .7, .8], [.8, .7, .6, .5, .4, .3, .2]])
    result = summarize_factors(errors)
    assert result['baseline_mean'] == pytest.approx(.5)
    assert result['per_record_oracle_mean'] == pytest.approx(.2)
    assert result['oracle_improved_count'] == 2
    assert result['best_fixed_factor'] == .25


def test_group_summary_rejects_nonfinite():
    with pytest.raises(ValueError):
        summarize_factors(torch.full((2, len(FACTORS)), float('nan')))
