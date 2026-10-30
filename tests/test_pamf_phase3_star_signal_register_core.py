"""All-U density clusters, true-class macro calibration and joint decisions."""
from copy import deepcopy
import importlib
import math
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_register_core')


@pytest.fixture(scope='module',autouse=True)
def single_blas_thread():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1,user_api='blas'):yield


def model():
    m=api()
    return dict(schema=np.asarray(m.density.SCHEMA),origin=np.zeros(160),whitener=np.eye(160),log_abs_det_whitener=np.asarray(0.),pooled_L_variance=np.asarray(1.),
        known_means=np.zeros((90,160)),known_variances=np.ones(90),known_weights=np.full(90,1/90),
        background_means=np.stack((np.zeros(160),np.eye(160)[0]*.4)),background_variances=np.asarray([.7,1.3]),background_weights=np.asarray([.4,.6]))


def test_source_eps_is_seventh_other_neighbor_and_exact_source_grid(subtests):
    m=api();state=model();before=deepcopy(state);theta=np.arange(70)*2*np.pi/70
    cell=np.zeros((70,160));cell[:,0]=np.cos(theta);cell[:,1]=np.sin(theta);x=np.tile(cell,(90,1))
    rows=np.asarray([(c,r,d) for c in range(6) for r in m.RECEIVERS for d in (1,2,3) for _ in range(70)],np.int64)
    source,diag=m.source_radius(state,x,*rows.T)
    assert set(source)=={'schema','eps'} and diag['source_count']==6300
    np.testing.assert_allclose(source['eps'],2*np.sin(4*np.pi/70),atol=1e-14)
    for key,value in state.items():np.testing.assert_array_equal(value,before[key])
    for defect in ('rows','label','RX'):
        with subtests.test(defect=defect):
            xx=x;meta=rows.copy()
            if defect=='rows':xx=x[:-1];meta=meta[:-1]
            elif defect=='label':meta[0,0]=6
            else:meta[0,1]=26
            with pytest.raises(ValueError):m.source_radius(state,xx,*meta.T)


def cluster_input():
    rng=np.random.default_rng(81);x=rng.normal(size=(384,160));x/=np.linalg.norm(x,axis=1,keepdims=True)
    for start,size,axis,sign in ((0,16,0,1),(16,16,0,-1),(32,15,1,1)):
        x[start:start+size]=0;x[start:start+size,axis]=sign;x[start:start+size,3]=np.linspace(-.001,.001,size)
    ids=np.asarray([('b' if i<16 else 'a' if i<32 else 'c' if i<47 else 'noise')+'-'+str(i).zfill(4) for i in range(384)])
    scores=np.ones((384,2));scores[:8,1]=-1;scores[16:25,1]=-1;scores[32:47,1]=-1
    return x.astype(np.float32),scores,ids


def test_all_U_DBSCAN_strict_majority_min16_stable_IDs_and_full_member_moments():
    m=api();state=model();x,scores,ids=cluster_input();source=dict(schema=np.asarray(m.SOURCE_SCHEMA),eps=np.asarray(.01))
    anonymous,assignment,diag=m.discover(state,source,x,scores,ids)
    assert diag['all_U_rows_used'] and diag['registered_count']==1 and len(diag['clusters'])==3
    assert anonymous['anonymous_ids'].tolist()==['anon-0001'] and np.all(assignment['assignments'][16:32]==0)
    assert np.all(assignment['assignments'][:16]==-1) and np.all(assignment['assignments'][32:47]==-1)
    assert int((assignment['assignments']>=0).sum())==16 and int(assignment['rejected'][16:32].sum())==9
    assert any(row['size']==16 and row['rejected_fraction']==.5 and not row['qualified'] for row in diag['clusters'])
    white=m.density.unit(x[16:32]);mean=white.mean(0);variance=(16/176)*np.square(white-mean).sum()/16/160+(160/176)
    np.testing.assert_allclose(anonymous['means'][0],mean,atol=1e-15);np.testing.assert_allclose(anonymous['variances'][0],variance,atol=1e-15)
    order=np.random.default_rng(82).permutation(384);again,rows,_=m.discover(state,source,x[order],scores[order],ids[order])
    for key,v in anonymous.items():np.testing.assert_array_equal(again[key],v)
    inverse=np.argsort(order)
    for key in ('assignments','dbscan_labels','rejected'):np.testing.assert_array_equal(rows[key][inverse],assignment[key])
    zero,none,_=m.discover(state,source,x,np.ones_like(scores),ids)
    assert zero['means'].shape==(0,160) and zero['variances'].shape==(0,) and not (none['assignments']>=0).any()


def test_anonymous_logpdf_no_one_over_K_mass_and_joint_acceptance_ties_K0():
    m=api();state=model();state['whitener']*=2;state['log_abs_det_whitener']=np.asarray(160*math.log(2.))
    x=np.eye(160)[:2];a=dict(schema=np.asarray(m.ANONYMOUS_SCHEMA),means=np.zeros((2,160)),variances=np.asarray([.5,.5]),anonymous_ids=np.asarray(['anon-0001','anon-0002']))
    ratios=m.anonymous_ratios(state,a,x)
    for i,point in enumerate(x*2):
        lognew=-80*math.log(2*math.pi*.5)-np.dot(point,point)
        components=[math.log(p)-80*math.log(2*math.pi*v)-np.dot(point-center,point-center)/(2*v) for center,v,p in zip(state['background_means'],state['background_variances'],state['background_weights'])]
        peak=max(components);background=peak+math.log(sum(math.exp(v-peak) for v in components))
        np.testing.assert_allclose(ratios[i],[lognew-background]*2,atol=1e-12)
    closed=np.asarray([3,1,4,0]);base=np.asarray([[1,1],[-1,-2],[-1,-1],[0,0]],np.float64)
    new=np.asarray([[2,2],[2,1],[0,0],[1,1]],np.float64);tau=np.asarray([1.,1.])
    pred,decision=m.predict(closed,base,new,tau)
    np.testing.assert_array_equal(pred[:,2],[3,6,-1,0]);np.testing.assert_array_equal(pred[:,3],[3,6,-1,0])
    assert decision[3,2]==0 and pred[3,2]==0
    p0,d0=m.predict(closed,base,np.empty((4,0)),np.zeros(2))
    np.testing.assert_array_equal(p0[:,:2],p0[:,2:]);np.testing.assert_array_equal(d0[:,:2],d0[:,2:])


def known_V(m):
    rows=[]
    for rx in m.RECEIVERS:
        for scene in m.SCENES:
            for label in range(6):
                for j in range(200 if label==1 else 5):rows.append((label,rx,scene,label==0 and j==0))
    y=np.asarray([r[0] for r in rows],np.int64);rx=np.asarray([r[1] for r in rows],np.int64);scene=np.asarray([r[2] for r in rows])
    raw=np.where(np.asarray([r[3] for r in rows]),10.,0.)[:,None]
    return np.asarray(['V-'+str(i) for i in range(len(y))]),y,y.copy(),y.copy(),np.ones((len(y),2)),raw,rx,scene


def test_calibration_complete_ties_minimal_tau_true_class_macro_and42group_report():
    m=api();args=known_V(m);tau,report=m.calibrate(*args)
    np.testing.assert_array_equal(tau,[9.,9.]);assert report['constraints_passed'] and report['group_count']==42 and len(report['hard_names'])==12
    for name in ('pooled_REG1','predicted_REG1'):
        row=report['methods'][name];assert len(row['groups'])==42 and row['previous_threshold']==-1 and not row['previous_constraints_passed']
        assert row['groups']['overall']['known_macro_acceptance']==1
        failed=row['previous_failure_groups']['overall'];np.testing.assert_allclose(failed['original_closed_minus_new_open'],1/30)
        assert failed['known_macro_acceptance']==1  # wrong anonymous accepts are not AK
    # No slots means an exact zero-threshold registration no-op.
    zero=list(args);zero[5]=np.empty((len(args[0]),0));t,r=m.calibrate(*zero)
    np.testing.assert_array_equal(t,[0.,0.]);assert r['constraints_passed']
    missing=[a[args[1]!=5] for a in args];_,r=m.calibrate(*missing)
    assert not r['constraints_passed'] and not r['coverage']['hard_groups_complete']


def test_invalid_parameter_shapes_nonfinite_labels_and_anonymous_state(subtests):
    m=api();state=model();x,scores,ids=cluster_input();source=dict(schema=np.asarray(m.SOURCE_SCHEMA),eps=np.asarray(.01))
    for defect in ('duplicate','dtype','nan','shape'):
        with subtests.test(defect=defect):
            xx=x.copy();ss=scores.copy();ii=ids.copy()
            if defect=='duplicate':ii[0]=ii[1]
            elif defect=='dtype':ss=ss.astype(np.float32)
            elif defect=='nan':xx[0,0]=np.nan
            else:xx=xx[:,:159]
            with pytest.raises(ValueError):m.discover(state,source,xx,ss,ii)
    with pytest.raises(ValueError):m.predict(np.asarray([6]),np.ones((1,2)),np.zeros((1,0)),np.zeros(2))
    bad=dict(schema=np.asarray(m.ANONYMOUS_SCHEMA),means=np.zeros((1,160)),variances=np.asarray([-1.]),anonymous_ids=np.asarray(['anon-0001']))
    with pytest.raises(ValueError):m.anonymous_ratios(state,bad,np.eye(160)[:1])
    args=list(known_V(m));args[1]=args[1].copy();args[1][0]=6
    with pytest.raises(ValueError):m.calibrate(*args)
