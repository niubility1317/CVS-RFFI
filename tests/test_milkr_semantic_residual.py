import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.semantic_residual import (
    fit, gram, unit_features, NAMES, base)


def example(classes=8, k=2):
    rng = np.random.default_rng(91)
    n = classes*k
    x = {s:rng.normal(size=(n,96 if s=='fft' else 160)) for s in NAMES}
    return x, rng.normal(size=(n,6)), np.repeat(np.arange(classes),k), list(map(str,range(n)))


def test_physical_duplicate_views_and_batch_independence():
    x,l,y,ids=example()
    a=fit([x],[l],y,ids)
    b=fit([x,x],[l,l],y,ids)
    assert np.allclose(a.alpha,b.alpha,atol=1e-12)
    scores=a.score(x,l)
    assert np.allclose(scores,np.concatenate([a.score({s:v[i:i+1] for s,v in x.items()},l[i:i+1]) for i in range(len(y))]))
    assert not a.alpha.flags.writeable


def test_kernel_psd_and_merged_inheritance_normal_equation():
    x,l,y,ids=example()
    ox={s:v[:12] for s,v in x.items()}
    b=fit([ox],[l[:12]],y[:12],ids[:12])
    c=fit([x],[l],y,ids,previous=b)
    K=gram(c.views,c.views,c.tau)
    assert np.linalg.eigvalsh(K).min()>-1e-12
    coef=np.zeros_like(c.alpha);coef[:12,:6]=b.alpha
    for col in range(6,8):
        coef[y==col,col]+=1/2;coef[:12,col]-=1/12
    coef-=coef.mean(1,keepdims=True)
    rhs=np.eye(8)[y]-1/8-base(l,6,8,True)-K@coef
    expected=coef+np.linalg.solve(K+.02*np.eye(len(y)),rhs)
    assert np.allclose(c.alpha,expected,atol=1e-12)
    assert c.tau==b.tau
    assert c.residual<1e-12


def test_wrong_episode_or_view_or_labels_rejected():
    x,l,y,ids=example()
    b=fit([{s:v[:12] for s,v in x.items()}],[l[:12]],y[:12],ids[:12])
    with pytest.raises(ValueError,match='identity'):
        fit([x],[l],y,['other']+ids[1:],previous=b)
    with pytest.raises(ValueError,match='identity'):
        fit([x,x],[l,l],y,ids,previous=b)
    with pytest.raises(ValueError,match='equal K'):
        fit([x],[l],np.zeros(len(y),int),ids)
    bad=dict(x);bad['query_truth']=y
    with pytest.raises(ValueError,match='Unexpected'):
        unit_features(bad)
