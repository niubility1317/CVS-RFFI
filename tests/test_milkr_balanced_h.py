import json
import numpy as np
from tests.test_milkr_semantic_residual import example
from experiments.pamf_phase2_practical.milkr import balanced_h as b,nested_fisher as n


def test_none_and_K1_exact_R14_and_calibrated_queries_are_independent():
    rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1,outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    for k in (1,3):
        x,_,y,ids=example(k=k);x['sig_views']=np.random.default_rng(92).normal(size=(len(y),336))
        reference=n.fit(x,y,ids,fixed=rule);grid=b.fit_grid(x,y,ids,fixed=rule)
        assert np.array_equal(reference.alpha,grid['none'].alpha)
        assert np.array_equal(reference.score(x),grid['none'].score(x))
        for kap,state in grid.items():
            json.dumps(state.selection,allow_nan=False)
            one=np.concatenate([state.score({name:value[i:i+1] for name,value in x.items()}) for i in range(len(y))])
            assert np.allclose(state.score(x),one)
            if k==1:assert np.array_equal(reference.score(x),state.score(x))


def test_asymmetric_six_class_source_groups_have_defined_old_bandwidth():
    rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1,outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    for k in (1,2):
        x,_,y,ids=example(k=k);positions=np.arange(6*k);x={name:value[positions] for name,value in x.items()}
        x['sig_views']=np.random.default_rng(29).normal(size=(6*k,336));y=y[positions];ids=ids[:6*k]
        for old in (2,3,4,5):
            grid=b.fit_grid(x,y,ids,old_count=old,fixed=rule)
            assert all(np.isfinite(state.score(x)).all() for state in grid.values())


def test_B_all_old_classes_exact_R14():
    x,_,y,ids=example(k=3);x['sig_views']=np.random.default_rng(113).normal(size=(len(y),336))
    rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1,outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    old=int(y.max())+1;reference=n.fit(x,y,ids,old_count=old,fixed=rule)
    for state in b.fit_grid(x,y,ids,old_count=old,fixed=rule).values():
        assert np.array_equal(reference.alpha,state.alpha) and np.array_equal(reference.score(x),state.score(x))
