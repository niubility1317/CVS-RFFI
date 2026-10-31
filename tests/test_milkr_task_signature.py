import numpy as np
import pytest
from tests.test_milkr_semantic_residual import example
from experiments.pamf_phase2_practical.milkr import task_signature as t,signature_kernel as s


def data():
    x,_,y,ids=example(k=2);x['sig_views']=np.random.default_rng(92).normal(size=(len(y),336))
    return x,y,ids


def test_frozen_support_statistics_psd_and_query_independence():
    x,y,ids=data();rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1,task_mode='std0.1')
    state=t.fit(x,y,ids,fixed=rule,conditional_loo=False);mean=state.normalizer.location.copy();scale=state.normalizer.scale.copy()
    full=state.score(x);one=np.concatenate([state.score({k:v[i:i+1] for k,v in x.items()}) for i in range(len(y))])
    assert np.allclose(full,one)
    assert np.array_equal(mean,state.normalizer.location) and np.array_equal(scale,state.normalizer.scale)
    assert not state.normalizer.location.flags.writeable and not state.normalizer.scale.flags.writeable
    for geometry in ('signature','signature_identity','signature_historical'):
        K=s.similarity(state.features,state.features,geometry)
        assert np.allclose(K.diagonal(),1) and np.linalg.eigvalsh(K).min()>-1e-12


def test_no_normalization_exactly_recovers_R9_and_missing_mode_rejected():
    x,y,ids=data();rule=dict(config=['signature','nearest',0.,'single'],ridge=.1)
    with pytest.raises(ValueError,match='Frozen source'):t.fit(x,y,ids,fixed=rule)
    a=s.fit(x,y,ids,fixed=rule,conditional_loo=False)
    b=t.fit(x,y,ids,fixed=dict(rule,task_mode='none'),conditional_loo=False)
    assert np.array_equal(a.alpha,b.alpha) and np.array_equal(a.score(x),b.score(x))
