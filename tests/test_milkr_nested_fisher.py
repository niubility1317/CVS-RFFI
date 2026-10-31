import numpy as np
import json
import pytest
from tests.test_milkr_semantic_residual import example
from experiments.pamf_phase2_practical.milkr import nested_fisher as n,signature_kernel as s


def test_folds_are_physical_disjoint_balanced_and_cover_support_once():
    y=np.repeat(np.arange(8),5);ids=[f'id-{(i*7)%41:03}' for i in range(len(y))];all_held=[]
    for train,held in n.folds(y,ids):
        assert not set(train)&set(held)
        assert len(set(np.bincount(y[train])))==1 and len(set(np.bincount(y[held])))==1
        all_held.extend(held)
    assert sorted(all_held)==list(range(len(y)))
    with pytest.raises(ValueError,match='physical'):n.folds(y,['duplicate']*len(y))


def test_control_exact_R9_and_nested_K2_tie_uniform_query_independent():
    x,_,y,ids=example(k=2);x['sig_views']=np.random.default_rng(72).normal(size=(len(y),336))
    rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1,outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    ref=s.fit(x,y,ids,fixed=rule)
    control=n.fit(x,y,ids,fixed=dict(rule,force_uniform=True))
    assert np.array_equal(ref.alpha,control.alpha) and np.array_equal(ref.score(x),control.score(x))
    state=n.fit(x,y,ids,fixed=rule)
    json.dumps(state.selection,allow_nan=False)
    json.dumps(control.selection,allow_nan=False)
    assert state.selection['outer_support_selection']['selected_strength']==0.
    all_scores=state.score(x);individual=np.concatenate([state.score({k:v[i:i+1] for k,v in x.items()}) for i in range(len(y))])
    assert np.allclose(all_scores,individual)
