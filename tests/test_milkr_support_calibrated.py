import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.support_calibrated import decompose,coefficients,fit
from tests.test_milkr_semantic_residual import example


def test_conditional_loo_matches_explicit_block_solve():
    rng=np.random.default_rng(7);x=rng.normal(size=(12,7));K=x@x.T;y=np.repeat(np.arange(3),4);lam=.1
    _,_,loo=coefficients(decompose(K,y),lam)
    for i in range(len(y)):
        keep=np.arange(len(y))!=i;n=int(keep.sum());A=np.zeros((n+1,n+1))
        A[:n,:n]=K[np.ix_(keep,keep)]+lam*np.eye(n);A[n,:n]=1;A[:n,n]=1
        rhs=np.zeros((n+1,3));rhs[:n]=np.eye(3)[y[keep]];solution=np.linalg.solve(A,rhs)
        assert np.allclose(loo[i],K[i,keep]@solution[:n]+solution[n],atol=1e-10)


def test_k1_requires_source_and_query_independence():
    x,l,y,ids=example(k=1)
    with pytest.raises(ValueError,match='frozen source'):fit(x,y,ids)
    state=fit(x,y,ids,fixed=dict(config=['historical','nearest',0.,'single'],ridge=.1))
    full=state.score(x)
    single=np.concatenate([state.score({k:v[i:i+1] for k,v in x.items()}) for i in range(len(y))])
    assert np.allclose(full,single)
    assert state.selection['rule']=='source_fixed'
