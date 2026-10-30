import numpy as np
from experiments.pamf_dura_efficiency.enhancement_probability_audit import statistics,pair_statistics


def test_uniform_probabilities_have_known_scores():
    s=statistics(np.zeros((2,6)),np.array([0,1]))
    assert s['correct']==1
    np.testing.assert_allclose(s['NLL_nats'],np.log(6))
    np.testing.assert_allclose(s['Brier_sum6'],5/6)
    np.testing.assert_allclose(s['ECE_10_equal_width'],1/3)
    assert sum(b['count'] for b in s['calibration_bins'])==2


def test_extreme_logits_are_stable_and_confidence_can_rise_while_wrong():
    y=np.array([0,1])
    before=np.array([[2,0,0,0,0,0],[0,1,0,0,0,0]],dtype=float)
    after=np.array([[0,1000,0,0,0,0],[0,1000,0,0,0,0]],dtype=float)
    s=statistics(after,y);pair=pair_statistics(before,after,y)
    assert np.isfinite(s['NLL_nats']) and s['NLL_nats']==500
    assert pair['damaged']['count']==1 and pair['both_correct']['count']==1
    assert pair['more_confident_and_wrong_after']==1
