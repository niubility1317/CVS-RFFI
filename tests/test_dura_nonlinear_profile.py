import pytest
from experiments.pamf_dura_efficiency.nonlinear_profile import cost_decision


def row(ratio):return dict(timings=dict(cached5=[.1]*12,exact10=[.1*ratio]*12))


def test_both_views_must_meet_cost_bound_no_mean_masking():
    assert cost_decision({'1':row(1.2),'20':row(1.4)})['eligible']
    assert not cost_decision({'1':row(1.),'20':row(1.51)})['eligible']


def test_paired_cost_rule_is_checked_independently_of_marginal_medians():
    old=[.1]*6+[1.]*6;new=[.16]*6+[1.6]+[.1]*5
    result=cost_decision({'1':dict(timings=dict(cached5=old,exact10=new)),'20':row(1.)})
    assert result['views'][0]['median_ratio']<1 and result['views'][0]['paired_median_ratio']>1.5
    assert not result['eligible']


def test_partial_or_nonfinite_timing_matrix_is_rejected():
    with pytest.raises(ValueError):cost_decision({'1':row(1.)})
    with pytest.raises(ValueError):cost_decision({'1':row(1.),'20':dict(timings=dict(cached5=[.1]*11,exact10=[.1]*11))})
    with pytest.raises(ValueError):cost_decision({'1':row(1.),'20':row(float('nan'))})
