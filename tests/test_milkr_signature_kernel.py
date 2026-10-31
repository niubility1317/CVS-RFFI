import numpy as np
import pytest
from tests.test_milkr_semantic_residual import example
from experiments.pamf_phase2_practical.milkr import signature_kernel as s
from experiments.pamf_phase2_practical.milkr import support_calibrated as base


def test_three_view_kernel_psd_and_query_independence():
    x,_,y,ids=example(k=2);x['sig_views']=np.random.default_rng(19).normal(size=(len(y),336));n=s.normalize(x)
    for geometry in ('signature','signature_identity','signature_historical'):
        K=s.similarity(n,n,geometry)
        assert np.linalg.eigvalsh(K).min()>-1e-12
        assert np.allclose(np.diag(K),1)
    state=s.fit(x,y,ids,fixed=dict(config=['signature','all',.5,'single'],ridge=.1),conditional_loo=False)
    full=state.score(x);one=np.concatenate([state.score({k:v[i:i+1] for k,v in x.items()}) for i in range(len(y))])
    assert np.allclose(full,one)
    with pytest.raises(ValueError,match='schema'):s.normalize({k:v for k,v in x.items() if k!='sig_views'})


def test_default_backend_still_uses_its_original_schema():
    x,_,y,ids=example(k=1);state=base.fit(x,y,ids,fixed=dict(config=['identity','all',.5,'single'],ridge=.1))
    assert state.score(x).shape==(len(y),8)
    assert 'sig_views' not in state.features
