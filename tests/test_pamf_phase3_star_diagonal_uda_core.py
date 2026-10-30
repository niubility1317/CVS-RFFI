"""Gaussian EM oracle, exact stationary coordinate updates, and fixed score routing."""
from copy import deepcopy
import importlib
import math
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_diagonal_uda_core')


@pytest.fixture(scope='module',autouse=True)
def one_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def gaussian():
    from test_pamf_phase3_star_signal_register_core import model
    state=model();state['known_variances'][:]=.2;state['background_means']=np.zeros((1,160));state['background_variances']=np.asarray([.2]);state['background_weights']=np.ones(1)
    state['whitener']*=2;state['log_abs_det_whitener']=np.asarray(160*math.log(2.))
    x=np.random.default_rng(301).normal(size=(384,160));x/=np.linalg.norm(x,axis=1,keepdims=True)
    return state,x.astype(np.float32)


def test_single_Gaussian_Jacobian_coordinate_optimum_and_fixed_twenty_iterations():
    m=api();state,x=gaussian();adapters,diag=m.fit_adapters(state,x);white=m.density.unit(x)*2;n=len(x);lam=160.;v=.2
    sx=white.sum(0)/v;D=n/v+lam;H=np.square(white).sum(0)/v+lam-sx*sx/D
    expected=(lam+np.sqrt(lam*lam+4*H*n))/(2*H)
    for kind in m.ARMS:
        a=adapters[kind]['inverse_scale'];b=adapters[kind]['offset']
        np.testing.assert_allclose(a,1 if kind=='SHIFT' else expected,rtol=0,atol=2e-14)
        np.testing.assert_allclose(b,-a*sx/D,rtol=0,atol=2e-14)
        def objective(scale,offset):
            h=scale*white+offset
            return float((-80*np.log(2*np.pi*v)-np.square(h).sum(1)/(2*v)).sum()+n*np.log(scale).sum()
                +n*160*math.log(2.)-lam/2*(np.square(scale-1).sum()+np.square(offset).sum()))
        trace=diag['arms'][kind]['trace'];assert [row['iteration'] for row in trace]==list(range(21))
        np.testing.assert_allclose(trace[0]['objective'],objective(np.ones(160),np.zeros(160)),rtol=0,atol=1e-8)
        np.testing.assert_allclose(trace[-1]['objective'],objective(a,b),rtol=0,atol=1e-8)
        assert min(np.diff([row['objective'] for row in trace]))>=-1e-8 and trace[-1]['objective']>trace[0]['objective']
        np.testing.assert_allclose(-(a*white+b).sum(0)/v-lam*b,0,atol=2e-10)
        if kind=='DIAG':
            np.testing.assert_allclose(-(white*(a*white+b)).sum(0)/v+n/a-lam*(a-1),0,atol=2e-10)
            perturbed=a.copy();perturbed[0]+=.01;assert objective(a,b)>objective(perturbed,b)
    assert diag['iterations']==20 and diag['ridge']==160. and diag['U_count']==384
    assert diag['arms']['SHIFT']['group_prior']==diag['arms']['DIAG']['group_prior']==[.5,.5]


def test_fixed_mixture_monotonic_labelblind_permutation_and_source_not_mutated():
    from test_pamf_phase3_star_class_competition_core import fixture
    m=api();state,_,_=fixture();_,x=gaussian();original=deepcopy(state);xx=x.copy()
    first,diag=m.fit_adapters(state,x);order=np.random.default_rng(2).permutation(len(x));again,_=m.fit_adapters(state,x[order])
    for arm in m.ARMS:
        for key in ('inverse_scale','offset'):np.testing.assert_allclose(again[arm][key],first[arm][key],rtol=0,atol=1e-12)
        assert np.all(first[arm]['inverse_scale']>0)
        values=[row['objective'] for row in diag['arms'][arm]['trace']];assert np.all(np.diff(values)>=-1e-8)
    for key in state:np.testing.assert_array_equal(state[key],original[key])
    np.testing.assert_array_equal(x,xx)
    assert diag['U_only'] and not diag['true_U_labels_used'] and not diag['true_unknown_count_used'] and not diag['Q_inputs_used']


def test_negative_linear_term_has_the_positive_stationary_root():
    m=api();w=np.zeros((384,160));w[:,0]=2;means=np.zeros((1,160));means[0,0]=-10
    a,b=m._m_step(w,np.ones((384,1)),means,np.asarray([.2]),'DIAG')
    # Independent derivatives of N log(a) minus quadratic Gaussian/ridge cost.
    residual=a*w+b-means[0]
    np.testing.assert_allclose(-residual.sum(0)/.2-160*b,0,atol=1e-9)
    np.testing.assert_allclose(-(w*residual).sum(0)/.2+384/a-160*(a-1),0,atol=1e-9)
    assert 0<a[0]<.2 and np.all(a>0)


def test_identity_exact_old_ratio_and_affine_not_unit_renormalized(monkeypatch):
    from test_pamf_phase3_star_class_competition_core import fixture,logsum
    from experiments.pamf_phase3_open_world import star_class_density_core as old
    m=api();state,x,closed=fixture();identity=dict(schema=np.asarray(m.ADAPTER_SCHEMA),kind=np.asarray('DIAG'),inverse_scale=np.ones(160),offset=np.zeros(160))
    np.testing.assert_array_equal(m.raw_scores(state,x,closed,identity),old.raw_scores(state,x,closed)[:,[1,3]])
    a=np.linspace(.7,1.3,160);b=np.linspace(-.03,.05,160);adapter=dict(identity,inverse_scale=a,offset=b)
    actual=m.raw_scores(state,x,closed,adapter);expected=[]
    for point,c in zip(x,closed):
        unit=point.astype(np.float64)/np.linalg.norm(point.astype(np.float64));h=a*((unit-state['origin'])@state['whitener'])+b
        known=[-80*math.log(2*math.pi*v)-float(np.dot(h-mean,h-mean))/(2*v) for mean,v in zip(state['known_means'],state['known_variances'])]
        bg=logsum([math.log(p)-80*math.log(2*math.pi*v)-float(np.dot(h-mean,h-mean))/(2*v)
            for mean,v,p in zip(state['background_means'],state['background_variances'],state['background_weights'])])
        expected.append([logsum(known)-math.log(90)-bg,logsum(known[c*15:(c+1)*15])-math.log(15)-bg])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    monkeypatch.setattr(m,'fit_adapters',lambda *args:pytest.fail('Query scoring must not fit'))
    pieces=np.concatenate([m.raw_scores(state,x[i:i+1],closed[i:i+1],adapter) for i in range(len(x))])
    np.testing.assert_allclose(actual,pieces,rtol=0,atol=1e-12)


def test_invalid_role_rows_features_adapter_and_predicted_class(subtests):
    m=api();state,x=gaussian();adapter=dict(schema=np.asarray(m.ADAPTER_SCHEMA),kind=np.asarray('DIAG'),inverse_scale=np.ones(160),offset=np.zeros(160))
    for defect in ('count','zero','nan','width','variance','scale','dtype','kind','closed'):
        with subtests.test(defect=defect):
            s=deepcopy(state);xx=x.copy();a=deepcopy(adapter);c=np.zeros(len(x),np.int64)
            if defect=='count':xx=xx[:-1]
            elif defect=='zero':xx[0]=0
            elif defect=='nan':xx[0,0]=np.nan
            elif defect=='width':xx=xx[:,:159]
            elif defect=='variance':s['known_variances'][0]=0
            elif defect=='scale':a['inverse_scale'][0]=0
            elif defect=='dtype':a['offset']=a['offset'].astype(np.float32)
            elif defect=='kind':a['kind']=np.asarray('foreign')
            else:c[0]=6
            with pytest.raises(ValueError):
                if defect in ('count','zero','nan','width','variance'):m.fit_adapters(s,xx)
                else:m.raw_scores(s,xx,c,a)
