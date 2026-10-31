"""Known-only Jacobian, fractional posterior mass, and frozen background oracles."""
from copy import deepcopy
import importlib
import math
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_selective_uda_core')


@pytest.fixture(scope='module',autouse=True)
def one_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def test_complete_mixture_E_step_includes_known_Jacobian_and_keeps_OE_original():
    from test_pamf_phase3_star_diagonal_uda_core import gaussian
    m=api();state,x=gaussian();w=m.density.unit(x)*2;a=np.full(160,1.03);b=np.linspace(-.001,.001,160)
    objective,r=m._expectation(state,w,a,b);h=a*w+b;loga=np.log(a).sum()
    known=-80*math.log(2*math.pi*.2)-np.square(h).sum(1)/.4+loga-math.log(2)
    background=-80*math.log(2*math.pi*.2)-np.square(w).sum(1)/.4-math.log(2)
    total=np.logaddexp(known,background);expected=np.exp(known-total)
    np.testing.assert_allclose(r.sum(1),expected,rtol=0,atol=2e-14)
    np.testing.assert_allclose(r,np.repeat((expected/90)[:,None],90,axis=1),rtol=0,atol=2e-15)
    expected_objective=total.sum()+len(w)*160*math.log(2)-80*(np.square(a-1).sum()+np.square(b).sum())
    np.testing.assert_allclose(objective,expected_objective,rtol=0,atol=1e-8)
    assert r.sum(1).mean()>.9  # dropping the known-only Jacobian changes this E step


def test_fractional_known_mass_M_step_stationary_not_N_and_zero_mass_identity():
    m=api();w=np.zeros((384,160));w[:,0]=2;means=np.zeros((90,160));variance=np.full(90,.2)
    r=np.full((384,90),.25/90);a,b=m._m_step(w,r,means,variance);neff=r.sum()
    residual=a*w+b;np.testing.assert_allclose(-1.25*residual.sum(0)-160*b,0,atol=2e-10)
    gradient=-1.25*(w*residual).sum(0)+neff/a-160*(a-1)
    np.testing.assert_allclose(gradient,0,atol=2e-10)
    assert np.linalg.norm(gradient+(384-neff)/a)>100
    zero_a,zero_b=m._m_step(w,np.zeros_like(r),means,variance)
    np.testing.assert_array_equal(zero_a,np.ones(160));np.testing.assert_array_equal(zero_b,np.zeros(160))
    means[:,0]=-10;negative_a,negative_b=m._m_step(w,r,means,variance)
    residual=negative_a*w+negative_b-means[0]
    np.testing.assert_allclose(-1.25*(w*residual).sum(0)+neff/negative_a-160*(negative_a-1),0,atol=1e-9)
    assert 0<negative_a[0]<.1


def test_twenty_iterations_monotonic_independent_identity_start_zero_known_and_permutation():
    from test_pamf_phase3_star_diagonal_uda_core import gaussian
    from test_pamf_phase3_star_class_competition_core import fixture
    m=api();state,_,_=fixture();_,x=gaussian();before=deepcopy(state);xx=x.copy()
    adapter,diag=m.fit_adapter(state,x);order=np.random.default_rng(6).permutation(len(x));again,_=m.fit_adapter(state,x[order])
    for key in ('inverse_scale','offset'):np.testing.assert_allclose(adapter[key],again[key],rtol=0,atol=1e-12)
    assert diag['iterations']==20 and diag['ridge']==160. and diag['U_count']==384
    assert diag['known_only_transform'] and diag['background_frozen'] and diag['group_prior']==[.5,.5]
    assert [r['iteration'] for r in diag['trace']]==list(range(21))
    assert np.all(np.diff([r['objective'] for r in diag['trace']])>=-1e-8)
    assert diag['trace'][0]['scale_min']==diag['trace'][0]['scale_max']==1 and diag['trace'][0]['offset_norm']==0
    for key in state:np.testing.assert_array_equal(state[key],before[key])
    np.testing.assert_array_equal(x,xx)
    zero,_=gaussian();zero['known_means'][:]=1000;identity,report=m.fit_adapter(zero,x)
    np.testing.assert_array_equal(identity['inverse_scale'],np.ones(160));np.testing.assert_array_equal(identity['offset'],np.zeros(160))
    assert all(r['known_posterior_mass']==0 for r in report['trace']) and len(report['trace'])==21


def test_raw_ratio_oracle_preserves_affine_Jacobian_and_does_not_renormalize(monkeypatch):
    from test_pamf_phase3_star_class_competition_core import fixture,logsum
    from experiments.pamf_phase3_open_world import star_class_density_core as original
    m=api();state,x,closed=fixture();a=dict(schema=np.asarray(m.ADAPTER_SCHEMA),kind=np.asarray('SELECTIVE'),inverse_scale=np.ones(160),offset=np.zeros(160))
    np.testing.assert_array_equal(m.raw_scores(state,x,closed,a),original.raw_scores(state,x,closed)[:,[1,3]])
    a['inverse_scale']=np.linspace(.8,1.3,160);a['offset']=np.linspace(-.05,.04,160);expected=[]
    for point,c in zip(x,closed):
        unit=point.astype(np.float64)/np.linalg.norm(point.astype(np.float64));w=(unit-state['origin'])@state['whitener'];h=a['inverse_scale']*w+a['offset']
        known=[-80*math.log(2*math.pi*v)-np.dot(h-mean,h-mean)/(2*v) for mean,v in zip(state['known_means'],state['known_variances'])]
        background=logsum([math.log(p)-80*math.log(2*math.pi*v)-np.dot(w-mean,w-mean)/(2*v)
            for mean,v,p in zip(state['background_means'],state['background_variances'],state['background_weights'])])
        jac=np.log(a['inverse_scale']).sum();expected.append([logsum(known)-math.log(90)-background+jac,logsum(known[c*15:(c+1)*15])-math.log(15)-background+jac])
    actual=m.raw_scores(state,x,closed,a);np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    monkeypatch.setattr(m,'fit_adapter',lambda *a:pytest.fail('Query prediction must not fit'))
    np.testing.assert_allclose(np.concatenate([m.raw_scores(state,x[i:i+1],closed[i:i+1],a) for i in range(len(x))]),actual,rtol=0,atol=1e-12)


def test_invalid_counts_features_state_and_adapter_rejected(subtests):
    from test_pamf_phase3_star_diagonal_uda_core import gaussian
    m=api();state,x=gaussian();a=dict(schema=np.asarray(m.ADAPTER_SCHEMA),kind=np.asarray('SELECTIVE'),inverse_scale=np.ones(160),offset=np.zeros(160))
    for defect in ('count','zero','nan','variance','scale','kind','private','closed'):
        with subtests.test(defect=defect):
            s=deepcopy(state);xx=x.copy();adapter=deepcopy(a);c=np.zeros(len(x),np.int64)
            if defect=='count':xx=xx[:-1]
            elif defect=='zero':xx[0]=0
            elif defect=='nan':xx[0,0]=np.nan
            elif defect=='variance':s['known_variances'][0]=0
            elif defect=='scale':adapter['inverse_scale'][0]=-1
            elif defect=='kind':adapter['kind']=np.asarray('DIAG')
            elif defect=='private':adapter['y']=np.zeros(1,np.int64)
            else:c[0]=6
            with pytest.raises(ValueError):
                if defect in ('count','zero','nan','variance'):m.fit_adapter(s,xx)
                else:m.raw_scores(s,xx,c,adapter)
