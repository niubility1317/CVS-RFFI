"""Unbalanced transport stationarity and fixed-source prior-only density tests."""
from copy import deepcopy
import importlib
import math
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_uot_prior_core')


@pytest.fixture(scope='module',autouse=True)
def one_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def test_scalar_mass_analytic_UOT_KKT_and_cost_offset_normalized_mass():
    m=api();cost=np.asarray([[2.4]]);a=np.asarray([.3]);b=np.asarray([.7]);rows,diag=m._sinkhorn(cost,a,b)
    exact=(math.log(.3)+math.log(.7)-2.4)/2.05
    np.testing.assert_allclose(rows,[exact],atol=1e-8,rtol=0)
    assert abs(2.4+.05*rows[0]+rows[0]-math.log(.3)+rows[0]-math.log(.7))<2e-8
    assert diag['kkt_stationarity_residual']<2e-8 and diag['iterations']==200
    cost=np.asarray([[.1,.7,.9],[.5,.2,.4]]);a=np.asarray([.35,.65]);b=np.asarray([.2,.3,.5])
    original,d=m._sinkhorn(cost,a,b);shifted,sd=m._sinkhorn(cost+3,a,b)
    np.testing.assert_allclose(np.exp(original-d['log_total_mass']),np.exp(shifted-sd['log_total_mass']),rtol=0,atol=1e-12)
    np.testing.assert_allclose(sd['log_total_mass']-d['log_total_mass'],-3/2.05,atol=1e-8,rtol=0)
    assert d['kkt_stationarity_residual']<1e-7
    assert [r['iteration'] for r in d['trace']]==list(range(1,201))


def test_U_permutation_prior_floor_aggregate_only_and_source_unchanged():
    from test_pamf_phase3_star_tied_density_core import inputs
    m=api();state,_,_=inputs();rng=np.random.default_rng(411);x=rng.normal(size=(384,160)).astype(np.float32)
    before=deepcopy(state);xx=x.copy();adapter,diag=m.fit_adapter(state,x);order=rng.permutation(384);again,_=m.fit_adapter(state,x[order])
    assert set(adapter)=={'schema','domain_weights','background_weights'} and adapter['domain_weights'].shape==(15,)
    assert adapter['background_weights'].shape==(3,) and adapter['schema'].item()==m.ADAPTER_SCHEMA
    for key in ('domain_weights','background_weights'):np.testing.assert_allclose(again[key],adapter[key],atol=1e-12,rtol=0)
    assert np.all(adapter['domain_weights']>=.5/15) and np.all(adapter['background_weights']>=.5*state['background_weights'])
    assert diag['epsilon']==.05 and diag['rho_source']==diag['rho_target']==1 and diag['blend']==.5 and diag['U_count']==384
    assert diag['U_only'] and not diag['Q_inputs_used'] and not diag['true_unknown_count_used'] and not diag['true_U_labels_used']
    np.testing.assert_allclose(adapter['domain_weights'],.5/15+.5*np.asarray(diag['learned_domain_weights']),rtol=0,atol=1e-15)
    np.testing.assert_allclose(adapter['background_weights'],.5*state['background_weights']+.5*np.asarray(diag['learned_background_weights']),rtol=0,atol=1e-15)
    assert 0<=diag['known_transport_fraction']<=1 and len(diag['trace'])==200
    assert all(np.isfinite(diag[k]) for k in ('log_total_mass','kkt_stationarity_residual','max_log_scaling_change'))
    for key in state:np.testing.assert_array_equal(state[key],before[key])
    np.testing.assert_array_equal(x,xx)


def test_identity_exact_and_manual_shared_domain_conditional_Gaussian_weights():
    from test_pamf_phase3_star_tied_density_core import inputs,logsum
    from experiments.pamf_phase3_open_world import star_class_density_core as old
    m=api();state,x,c=inputs();identity=dict(schema=np.asarray(m.ADAPTER_SCHEMA),domain_weights=np.full(15,1/15),background_weights=state['background_weights'].copy())
    np.testing.assert_array_equal(m.raw_scores(state,x,c,identity),old.raw_scores(state,x,c)[:,[1,3]])
    domain=.5/15+.5*np.arange(1,16)/np.arange(1,16).sum();bg=.5*state['background_weights']+.5*np.asarray([.7,.2,.1])
    adapter=dict(identity,domain_weights=domain,background_weights=bg);expected=[]
    for point,label in zip(x,c):
        q=point.astype(np.float64);z=(q/np.linalg.norm(q)-state['origin'])@state['whitener']
        def gaussian(mean,v):return -80*math.log(2*math.pi*v)-float(np.dot(z-mean,z-mean))/(2*v)
        known=[gaussian(mu,v) for mu,v in zip(state['known_means'],state['known_variances'])]
        background=logsum([math.log(weight)+gaussian(mu,v) for weight,mu,v in zip(bg,state['background_means'],state['background_variances'])])
        expected.append([logsum([math.log(domain[j%15]/6)+value for j,value in enumerate(known)])-background,
            logsum([math.log(domain[j])+known[int(label)*15+j] for j in range(15)])-background])
    result=m.raw_scores(state,x,c,adapter);np.testing.assert_allclose(result,expected,atol=1e-12,rtol=0)
    rerouted=m.raw_scores(state,x,(c+1)%6,adapter);np.testing.assert_array_equal(rerouted[:,0],result[:,0]);assert np.any(abs(rerouted[:,1]-result[:,1])>1e-4)
    np.testing.assert_allclose(np.concatenate([m.raw_scores(state,x[i:i+1],c[i:i+1],adapter) for i in range(len(x))]),result,atol=1e-12,rtol=0)
    assert m.METHODS==('pooled_ratio','predicted_ratio','UOT_pooled_ratio','UOT_predicted_ratio') and m.PRIMARY_INDEX==3


def test_invalid_transport_data_prior_support_or_original_class(subtests):
    from test_pamf_phase3_star_tied_density_core import inputs
    m=api();state,x,c=inputs();adapter=dict(schema=np.asarray(m.ADAPTER_SCHEMA),domain_weights=np.full(15,1/15),background_weights=state['background_weights'].copy())
    for defect in ('field','floor','nan','shape','BG','class','features','U_count','cost','measure'):
        with subtests.test(defect=defect):
            a=deepcopy(adapter);xx=x.copy();cc=c.copy()
            if defect=='field':a['kind']=np.asarray('forbidden')
            elif defect=='floor':a['domain_weights'][0]=.01;a['domain_weights'][1]+=(1/15-.01)
            elif defect=='nan':a['domain_weights'][0]=np.nan
            elif defect=='shape':a['domain_weights']=a['domain_weights'][:-1]
            elif defect=='BG':a['background_weights'][0]=0
            elif defect=='class':cc[0]=6
            elif defect=='features':xx[0]=0
            with pytest.raises(ValueError):
                if defect=='U_count':m.fit_adapter(state,np.ones((383,160),np.float32))
                elif defect=='cost':m._sinkhorn(np.asarray([[-1.]]),np.ones(1),np.ones(1))
                elif defect=='measure':m._sinkhorn(np.zeros((1,1)),np.zeros(1),np.ones(1))
                else:m.raw_scores(state,xx,cc,a)
