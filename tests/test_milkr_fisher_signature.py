import numpy as np
from tests.test_milkr_semantic_residual import example
from experiments.pamf_phase2_practical.milkr import fisher_signature as f,signature_kernel as s


def data(k):
    x,_,y,ids=example(k=k);x['sig_views']=np.random.default_rng(18).normal(size=(len(y),336));return x,y,ids


def test_zero_strength_and_K1_exactly_recover_R9():
    for k,strength in ((1,.75),(2,0.)):
        x,y,ids=data(k);rule=dict(config=['signature_identity','all',.5,'single'],ridge=.1)
        a=s.fit(x,y,ids,fixed=rule,conditional_loo=False);b=f.fit(x,y,ids,fixed=dict(rule,fisher_strength=strength),conditional_loo=False)
        assert np.array_equal(a.alpha,b.alpha) and np.array_equal(a.score(x),b.score(x))


def test_support_weights_formula_frozen_and_query_independent():
    x,y,ids=data(2);eta=.5;normalizer,_=f.make_normalizer(x,y,eta)
    raw=s.normalize(x)['sig_views'].reshape(-1,3,112);means=np.stack([raw[y==c].mean(0) for c in range(8)])
    b=((means-means.mean(0))**2).mean(0,keepdims=True);w=((raw-means[y])**2).sum(0,keepdims=True)/(len(y)-8)
    ratio=b/(w+w.mean(2,keepdims=True)+np.finfo(np.float64).eps)
    assert np.allclose(normalizer.weights,1-eta+eta*ratio/ratio.mean(2,keepdims=True))
    state=f.fit(x,y,ids,fixed=dict(config=['signature','all',.5,'multi'],ridge=.1,fisher_strength=eta),conditional_loo=False)
    before=state.normalizer.weights.copy();full=state.score(x)
    one=np.concatenate([state.score({k:v[i:i+1] for k,v in x.items()}) for i in range(len(y))])
    assert np.allclose(full,one) and np.array_equal(before,state.normalizer.weights)
    assert not state.normalizer.weights.flags.writeable
