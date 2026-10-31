import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr import group_calibrated as m
from experiments.pamf_phase2_practical.milkr import support_calibrated as b
from tests.test_milkr_semantic_residual import example


def test_offset_matches_exhaustive_group_decisions_and_keeps_inputs():
    rng=np.random.default_rng(81);scores=rng.normal(size=(40,5));original=scores.copy();labels=np.repeat(np.arange(5),8)
    delta,audit=m.choose_offset(scores,labels,2)
    gaps=np.unique(scores[:,2:].max(1)-scores[:,:2].max(1))
    candidates=np.r_[gaps[0]-1.,gaps[:-1]+np.diff(gaps)/2,gaps[-1]+1.,0.]
    optimum=max(b.target_and_score(labels,m.add_offset(scores,d,2).argmax(1),2) for d in candidates)
    assert audit['conditional_LOO_H_after']==optimum
    assert np.array_equal(original,scores)
    assert b.target_and_score(labels,m.add_offset(scores,delta,2).argmax(1),2)==optimum


def test_k1_source_zero_equals_backend_and_query_batch_independent():
    x,_,y,ids=example(k=1);rule=dict(config=['historical','nearest',0.,'single'],ridge=.1)
    with pytest.raises(ValueError,match='source old/new'):m.fit(x,y,ids,fixed=rule)
    ref=b.fit(x,y,ids,fixed=rule);state=m.fit(x,y,ids,fixed=dict(rule,old_new_beta=0.))
    assert np.array_equal(ref.alpha,state.alpha)
    assert np.array_equal(ref.intercept,state.intercept)
    all_scores=state.score(x)
    individual=np.concatenate([state.score({k:v[i:i+1] for k,v in x.items()}) for i in range(len(y))])
    assert np.allclose(all_scores,individual)
