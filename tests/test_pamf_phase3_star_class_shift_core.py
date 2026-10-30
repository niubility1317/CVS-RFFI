from copy import deepcopy
import numpy as np
import pytest
from experiments.pamf_phase3_open_world import star_class_shift_core as core
from experiments.pamf_phase3_open_world import star_mixture_uda_core as shared
from test_pamf_phase3_star_mixture_uda_core import state


def prior(radius=1.,q=1):
    rng=np.random.default_rng(501);b=np.linalg.qr(rng.normal(size=(960,q)))[0].T.reshape(q,6,160)*np.sqrt(6.)
    return dict(schema=np.asarray(core.PRIOR_SCHEMA),basis=b,variances=np.arange(1,q+1,dtype=np.float64),radius=np.asarray(radius,np.float64))


def test_source_tensor_preserves_opposing_class_nuisance_cancelled_by_global_mean():
    s=state();mu=s['known_means'].reshape(6,15,160);offset=np.linspace(-.7,.7,15)
    mu[0,:,0]=2.;mu[1,:,0]=-2.;mu[0,:,1]=offset;mu[1,:,1]=-offset
    p=core.build_prior(s)
    assert len(p['basis'])==1 and np.allclose(p['basis'].mean(axis=1),0,atol=1e-12)
    assert float(p['radius'])==pytest.approx(.7/np.sqrt(3.))
    assert np.linalg.norm((mu-mu.mean(axis=1,keepdims=True)).mean(axis=0))==0
    shifted=deepcopy(s);shifted['known_means']+=np.repeat(np.arange(6)[:,None,None]*np.eye(160)[None,:1,:],15,axis=1).reshape(90,160)
    after=core.build_prior(shifted)
    assert np.allclose(p['variances'],after['variances'],atol=1e-12,rtol=0)
    assert np.allclose(p['basis'].reshape(1,-1).T@p['basis'].reshape(1,-1),after['basis'].reshape(1,-1).T@after['basis'].reshape(1,-1),atol=1e-12,rtol=0)
    assert all(not a.flags.writeable for a in p.values())


def test_zero_and_shared_shifts_match_old_density_without_double_whitening():
    s=state();s['whitener']*=1.2;s['origin'][:]=.1;s['log_abs_det_whitener']=np.asarray(160*np.log(1.2))
    x=np.eye(160,dtype=np.float32)[:3];d=np.zeros(160);d[0]=.4
    np.testing.assert_allclose(core.scores(s,x,np.tile(d,(6,1))),shared.scores(s,x,d),rtol=0,atol=1e-12)
    a=core.scores(s,x,np.zeros((6,160)));b=shared.scores(s,x,np.zeros(160))
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    nonuniform=np.zeros((6,160));nonuniform[0,0]=.2;nonuniform[1,0]=-.1
    batch=core.scores(s,x,nonuniform)
    np.testing.assert_allclose(batch[1:2],core.scores(s,x[1:2],nonuniform),rtol=0,atol=1e-12)


def explicit_system(s,w,r,p):
    means=np.concatenate((s['known_means'],s['background_means']));v=np.r_[s['known_variances'],s['background_variances']]
    h=np.diag(1/p['variances']);n=np.zeros(len(p['basis']))
    for j in range(len(means)):
        b=p['basis'][:,j//15,:] if j<90 else p['basis'].mean(axis=1)
        precision=r[:,j]/v[j];h+=precision.sum()*(b@b.T)
        for i in range(len(w)):n+=precision[i]*(b@(w[i]-means[j]))
    return h,n


def test_coupled_M_step_matches_explicit_precision_system_and_trust_region_KKT():
    s=state();p=prior(1e6,2);rng=np.random.default_rng(22);w=rng.normal(size=(7,160))
    r=rng.dirichlet(np.ones(91),size=7);h,n=explicit_system(s,w,r,p)
    a,m=core._maximization(s,w,r,p)
    np.testing.assert_allclose(a,np.linalg.solve(h,n),atol=1e-12,rtol=1e-12);assert m==0
    bounded=deepcopy(p);bounded['radius']=np.asarray(1e-4)
    a,m=core._maximization(s,w,r,bounded)
    assert m>0 and np.linalg.norm(a)==pytest.approx(1e-4,abs=1e-12)
    np.testing.assert_allclose(h@a+m*a,n,atol=1e-10,rtol=1e-12)


def test_soft_U_fit_recovers_opposing_offsets_with_one_latent_state_and_preserves_source():
    s=state();mu=s['known_means'].reshape(6,15,160);mu[0,:,0]=1.;mu[1,:,0]=-1.
    for c in range(2,6):mu[c,:,c]=20.
    s['known_variances'][:]=.0001;b=np.zeros((1,6,160));b[0,0,1]=np.sqrt(3.);b[0,1,1]=-np.sqrt(3.)
    p=dict(schema=np.asarray(core.PRIOR_SCHEMA),basis=b,variances=np.ones(1),radius=np.asarray(1.))
    x=np.zeros((384,160),np.float32);x[:192,0]=np.sqrt(.96);x[192:,0]=-np.sqrt(.96);x[:192,1]=.2;x[192:,1]=-.2
    before=deepcopy(s);a,d=core.fit_adapter(s,x,p)
    assert a['delta'][0,1]==pytest.approx(.2,abs=1e-6) and a['delta'][1,1]==pytest.approx(-.2,abs=1e-6)
    np.testing.assert_array_equal(a['background_delta'],np.zeros(160))
    assert d['iterations']==20 and [t['iteration'] for t in d['trace']]==list(range(21))
    assert min(np.diff([t['objective'] for t in d['trace']]))>=-1e-7
    for t in d['trace']:assert t['shift_rms']==pytest.approx(t['coefficient_norm'])
    assert d['U_only'] and not d['Q_inputs_used'] and not d['true_U_labels_used'] and not d['default_logits_used']
    assert all(np.array_equal(v,before[k]) for k,v in s.items()) and all(not v.flags.writeable for v in a.values())


def test_adapter_rejects_unlinked_background_and_outside_source_RMS_ball():
    p=prior();coef=np.asarray([.5]);delta=np.einsum('k,kcd->cd',coef,p['basis'])
    a=dict(schema=np.asarray(core.ADAPTER_SCHEMA),delta=delta,background_delta=delta.mean(axis=0),coefficients=coef)
    assert core.validate_adapter(a,p) is a
    bad=deepcopy(a);bad['background_delta'][0]+=.1
    with pytest.raises(ValueError):core.validate_adapter(bad,p)
    bad=deepcopy(a);bad['coefficients']*=3;bad['delta']*=3;bad['background_delta']*=3
    with pytest.raises(ValueError):core.validate_adapter(bad,p)
    with pytest.raises(ValueError):core.fit_adapter(state(),np.zeros((383,160)),p)
    with pytest.raises(ValueError):core.fit_adapter(state(),np.full((384,160),np.nan),p)
