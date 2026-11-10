import pytest
from experiments.pamf_branch_training.paired_analysis import cells, merge_disjoint


def test_combined_target_macro_f1_uses_summed_confusion_not_weighted_f1():
    a = dict(count=3, confusion=[[2, 0], [1, 0]])
    b = dict(count=2, confusion=[[0, 1], [0, 1]])
    result = merge_disjoint(a, b)
    assert result['count'] == 5
    assert result['accuracy'] == pytest.approx(.6)
    assert result['macro_f1'] == pytest.approx((4 / 6 + 2 / 4) / 2)


def test_comparison_refuses_unscored_or_duplicate_cells():
    with pytest.raises(ValueError):
        cells({'status': 'SCORED', 'truth_last': False, 'results': []})
    result = dict(dimension='overall', test='cross_rx', readout='raw', view='clean')
    with pytest.raises(ValueError, match='Duplicate'):
        cells({'status': 'SCORED', 'truth_last': True, 'results': [result, result]})
