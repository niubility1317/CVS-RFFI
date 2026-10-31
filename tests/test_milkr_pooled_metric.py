import numpy as np
from experiments.pamf_phase2_practical.milkr.pooled_metric import metric_parts,solve_metric


def test_scatter_and_centroid_downdate_matches_explicit_leave_one_out():
    rng=np.random.default_rng(14);x=rng.normal(size=(12,8));y=np.repeat(np.arange(3),4);K=x@x.T;ridge=.3
    alpha,b,loo=solve_metric(K,y,metric_parts(K,y),ridge)
    means=np.stack([x[y==c].mean(0) for c in range(3)])
    residual=x-means[y];A=np.linalg.inv(np.eye(8)+residual.T@residual/ridge)
    expected=2*x@A@means.T-np.sum((means@A)*means,axis=1)
    assert np.allclose(K@alpha+b,expected,atol=1e-10)
    for i in range(len(y)):
        keep=np.arange(len(y))!=i;fold=x[keep];fy=y[keep]
        mu=np.stack([fold[fy==c].mean(0) for c in range(3)])
        r=fold-mu[fy];inv=np.linalg.inv(np.eye(8)+r.T@r/ridge)
        score=2*x[i]@inv@mu.T-np.sum((mu@inv)*mu,axis=1)
        assert np.allclose(loo[i],score,atol=1e-10)
