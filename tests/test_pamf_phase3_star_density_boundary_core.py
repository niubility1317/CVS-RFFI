from copy import deepcopy
import numpy as np
import pytest
import torch
from experiments.pamf_phase3_open_world import star_density_boundary_core as core
from experiments.pamf_phase3_open_world import star_class_density_core as numerical
from test_pamf_phase3_star_class_density_core import state


def tensor(a,dtype=torch.float64,requires_grad=False):return torch.tensor(np.asarray(a).tolist(),dtype=dtype,requires_grad=requires_grad)


def test_ratio_matches_independent_numpy_deployment_and_only_frozen_predicted_class_routes():
    s=state();before=deepcopy(s);model=core.FrozenDensity(s)
    x=np.random.default_rng(11).normal(size=(6,160));c=np.arange(6,dtype=np.int64)
    actual=model(tensor(x),tensor(c,torch.long)).detach().tolist();expected=numerical.raw_scores(s,x,c)[:,3]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-12)
    changed=deepcopy(s);changed['known_means'][15:]+=20
    np.testing.assert_allclose(core.FrozenDensity(changed)(tensor(x[:1]),tensor(c[:1],torch.long)).detach().tolist(),actual[:1],rtol=0,atol=1e-12)
    assert not list(model.parameters()) and all(not b.requires_grad for b in model.buffers())
    assert all(np.array_equal(v,before[k]) for k,v in s.items())


def test_balanced_BCE_equals_numpy_and_repeating_OE_does_not_change_role_weight():
    model=core.FrozenDensity(state());rng=np.random.default_rng(17)
    left=rng.normal(size=(4,160));right=rng.normal(size=(2,160));lc=np.array([0,1,2,3]);rc=np.array([4,5])
    result=model.loss(tensor(left),tensor(right),tensor(lc,torch.long),tensor(rc,torch.long))
    a=numerical.raw_scores(state(),left,lc)[:,3]/160;b=numerical.raw_scores(state(),right,rc)[:,3]/160
    expected=.5*(np.logaddexp(0,-a).mean()+np.logaddexp(0,b).mean())
    assert float(result['loss'])==pytest.approx(expected,abs=1e-13)
    repeated=model.loss(tensor(left),tensor(np.tile(right,(3,1))),tensor(lc,torch.long),tensor(np.tile(rc,3),torch.long))
    assert float(repeated['loss'])==pytest.approx(expected,abs=1e-13)


def test_finite_difference_gradient_reaches_live_features_without_changing_constants():
    model=core.FrozenDensity(state());buffers={k:v.clone() for k,v in model.named_buffers()};rng=np.random.default_rng(19)
    x=rng.normal(size=(2,160));other=rng.normal(size=(3,160));a=tensor(x,requires_grad=True);b=tensor(other,requires_grad=True)
    c=tensor([0,2],torch.long);d=tensor([1,3,5],torch.long);value=model.loss(a,b,c,d)['loss'];value.backward()
    assert float(a.grad.norm())>0 and float(b.grad.norm())>0
    epsilon=1e-5;plus=x.copy();minus=x.copy();plus[0,7]+=epsilon;minus[0,7]-=epsilon
    expected=(float(model.loss(tensor(plus),tensor(other),c,d)['loss'])-float(model.loss(tensor(minus),tensor(other),c,d)['loss']))/(2*epsilon)
    assert float(a.grad[0,7])==pytest.approx(expected,rel=1e-5,abs=1e-10)
    assert all(torch.equal(value,dict(model.named_buffers())[key]) for key,value in buffers.items())


def test_invalid_live_or_frozen_identity_inputs_are_rejected(subtests):
    model=core.FrozenDensity(state());x=tensor(np.ones((2,160)));c=tensor([0,1],torch.long)
    cases=[(torch.zeros_like(x),c),(x.float().half(),c),(x,tensor([0.,1.])),(x,tensor([0,6],torch.long)),(x,tensor([0],torch.long)),(x[:,:159],c)]
    bad=x.clone();bad[0,0]=float('nan');cases.append((bad,c))
    for z,closed in cases:
        with subtests.test(shape=tuple(z.shape),dtype=str(closed.dtype)),pytest.raises(ValueError):model(z,closed)
