from copy import deepcopy
import numpy as np
import pytest
from experiments.pamf_phase3_open_world import star_mixture_uda_core as core


def state():
    background=np.zeros((1,160),np.float64);background[0,0]=10.
    return dict(schema=np.asarray(core.density.SCHEMA),origin=np.zeros(160),whitener=np.eye(160),
        log_abs_det_whitener=np.asarray(0.),pooled_L_variance=np.asarray(1.),known_means=np.zeros((90,160)),
        known_variances=np.ones(90),known_weights=np.full(90,1/90),background_means=background,
        background_variances=np.ones(1),background_weights=np.ones(1))


def prior(radius=1.):
    return dict(schema=np.asarray(core.PRIOR_SCHEMA),basis=np.eye(160,dtype=np.float64)[:1],
                variances=np.ones(1),radius=np.asarray(radius,np.float64))


def test_source_prior_cancels_class_offsets_and_bounds_only_source_nuisance():
    s=state();rng=np.random.default_rng(5);classes=rng.normal(size=(6,160));offset=np.zeros((15,160))
    offset[:14,:14]=.1*np.eye(14);offset[14]=-offset[:14].sum(0)
    s['known_means']=(classes[:,None,:]+offset[None]).reshape(90,160)
    a=core.build_prior(s);assert len(a['basis'])==14
    assert float(a['radius'])==pytest.approx(np.linalg.norm(offset,axis=1).max())
    changed=deepcopy(s);changed['known_means']+=(rng.normal(size=(6,1,160))*3).repeat(15,axis=1).reshape(90,160)
    b=core.build_prior(changed)
    assert np.allclose(a['basis'].T@a['basis'],b['basis'].T@b['basis'],atol=1e-12,rtol=0)
    assert np.allclose(a['variances'],b['variances'],atol=1e-12,rtol=0)
    assert float(a['radius'])==pytest.approx(float(b['radius']))
    assert all(not v.flags.writeable for v in a.values())


def test_one_gaussian_MAP_has_closed_form_and_fixed_monotone_iterations():
    s=state();p=prior();w=np.zeros((384,160));w[:,0]=.2
    a,report=core.fit_whitened(s,w,p,'DA_K')
    assert a['delta'][0]==pytest.approx(384*.2/385,abs=1e-12)
    assert np.count_nonzero(a['delta'][1:])==0 and report['iterations']==20
    assert [v['iteration'] for v in report['trace']]==list(range(21))
    assert np.min(np.diff([v['objective'] for v in report['trace']]))>=-1e-7
    assert report['default_logits_used'] is False and report['true_unknown_count_used'] is False


def test_background_components_prevent_unregistered_mass_from_dragging_known_fit():
    s=state();p=prior();w=np.zeros((384,160));w[:,0]=.2;w[288:,0]+=10
    known,k=core.fit_whitened(s,w,p,'DA_K');mixed,m=core.fit_whitened(s,w,p,'DA_M')
    assert known['delta'][0]==pytest.approx(1.,abs=1e-12)
    assert mixed['delta'][0]==pytest.approx(.2,abs=.01)
    assert k['group_prior']==[1.,0.] and m['group_prior']==[.5,.5]
    assert all(v['coefficient_norm']<=1+1e-10 for v in k['trace']+m['trace'])
    assert np.min(np.diff([v['objective'] for v in m['trace']]))>=-1e-7


def test_score_uses_exactly_one_whitening_and_no_query_fit_or_renormalization():
    s=state();s['whitener']*=1.2;s['log_abs_det_whitener']=np.asarray(160*np.log(1.2));s['origin'][:]=.1
    x=np.eye(160,dtype=np.float32)[:3];delta=np.zeros(160);delta[0]=.4
    out=core.scores(s,x,delta);h=(x.astype(np.float64)-.1)*1.2-delta
    expected=-80*np.log(2*np.pi)-.5*np.square(h).sum(1)+160*np.log(1.2)
    assert np.allclose(out[:,0],expected,rtol=0,atol=1e-12)
    assert np.allclose(out[1:2],core.scores(s,x[1:2],delta),rtol=0,atol=1e-12)
    zero=core.scores(s,x,np.zeros(160));known={k:s[k] for k in core.density.KNOWN_FIELDS}
    assert np.array_equal(zero[:,0],core.density.score(known,x,'known_only'))
    assert np.array_equal(zero[:,1],core.density.score(s,x,'density_ratio'))


def test_adapter_rejects_out_of_source_subspace_or_radius():
    p=prior();a=dict(schema=np.asarray(core.ADAPTER_SCHEMA),delta=np.zeros(160),coefficients=np.zeros(1))
    assert core.validate_adapter(a,p) is a
    bad=deepcopy(a);bad['delta'][1]=.1
    with pytest.raises(ValueError):core.validate_adapter(bad,p)
    bad=deepcopy(a);bad['coefficients'][0]=2;bad['delta'][0]=2
    with pytest.raises(ValueError):core.validate_adapter(bad,p)
    with pytest.raises(ValueError):core.fit_whitened(state(),np.zeros((383,160)),p,'DA_M')
    with pytest.raises(ValueError):core.fit_whitened(state(),np.full((384,160),np.nan),p,'DA_M')


def test_fit_adapters_preserves_source_and_target_pair_differences():
    s=state();p=prior();rng=np.random.default_rng(10);x=rng.normal(size=(384,160)).astype(np.float32);before=deepcopy(s)
    adapters,diag=core.fit_adapters(s,x,p)
    assert set(adapters)=={'DA_K','DA_M'} and diag['U_only'] and not diag['Q_inputs_used']
    z=(core.density.unit(x)-s['origin'])@s['whitener']
    for a in adapters.values():
        shifted=z-a['delta']
        assert np.allclose(shifted[1:]-shifted[:-1],z[1:]-z[:-1],rtol=0,atol=1e-15)
        assert all(not v.flags.writeable for v in a.values())
    assert all(np.array_equal(v,before[k]) for k,v in s.items())
