from fractions import Fraction
import numpy as np
import pytest
from experiments.pamf_phase3_open_world import ground_open_core as core


def sample():
    rows=[(c,r,d,j) for c in range(6) for r in core.RECEIVERS for d in core.DAYS for j in range(10)]
    y=np.asarray([a[0] for a in rows],np.int64);rx=np.asarray([a[1] for a in rows],np.int64);day=np.asarray([a[2] for a in rows],np.int64)
    raw=np.asarray([10*((core.RECEIVERS.index(r)*3+d-1-2*c)%15)+j for c,r,d,j in rows],np.float64)
    ids=np.asarray(['opaque'+str(i) for i in range(len(y))]);return ids,y,y.copy(),y.copy(),raw,rx,day


def test_exact_common_tail_protects_all_receiver_date_cells():
    args=sample();state,report=core.calibrate_known(*args)
    assert report['constraints_passed'] and report['known_only'] and not report['unknown_inputs_used'] and not report['test_inputs_used']
    assert set(report['groups'])==set(core.GROUP_NAMES) and len(report['groups'])==24
    assert Fraction(report['common_tail_fraction'])==Fraction(1,150)
    assert report['rejected_counts_by_prediction']==[1]*6
    assert all(v['C_K']>=.95 and v['original_closed_minus_open']<=.02 for v in report['groups'].values())
    margin=core.apply(args[4],args[2],state)
    assert int((margin<0).sum())==6 and state['conditional_thresholds'].dtype==np.float64


def test_complete_ties_and_failed_identity_are_not_hidden():
    args=list(sample());args[4][:]=7.;state,report=core.calibrate_known(*args)
    assert report['constraints_passed'] and np.all(core.apply(args[4],args[2],state)==0)
    args=list(sample());args[2][args[1]==0]=1
    state,report=core.calibrate_known(*args)
    assert not report['constraints_passed'] and not report['zero_tail_feasible']
    assert report['predicted_class_missing']==[0] and Fraction(report['common_tail_fraction'])==0
    assert np.isfinite(core.apply(args[4],args[2],state)).all()


def test_only_known_calibration_and_exact_deployment_state(subtests):
    args=sample();state,_=core.calibrate_known(*args)
    for index,value in ((0,np.repeat('repeat',len(args[0]))),(1,np.full(len(args[0]),-1)),(2,args[2].astype(np.float64)),(4,np.full(len(args[0]),np.nan)),(5,np.full(len(args[0]),34)),(6,np.zeros(len(args[0]),np.int64))):
        with subtests.test(index=index),pytest.raises(ValueError):
            changed=list(args);changed[index]=value;core.calibrate_known(*changed)
    for bad in (dict(state,U_truth=[]),dict(state,schema='foreign'),dict(state,conditional_thresholds=np.zeros(6,np.float32))):
        with subtests.test(state=str(bad.keys())),pytest.raises(ValueError):core.apply(args[4],args[2],bad)
    with pytest.raises(TypeError):core.calibrate_known(*args,unknown_scores=np.ones(10))


def test_energy_is_stable_per_record_and_not_softmax_confidence():
    logits=np.asarray([[30.,0.,-30.,1.,-1.,0.],[2.,3.,4.,5.,6.,7.]],np.float32)
    expected=np.logaddexp.reduce(logits.astype(np.float64),axis=1)
    np.testing.assert_array_equal(core.energy(logits),expected)
    np.testing.assert_allclose(core.energy(logits+4)-expected,4.,rtol=0,atol=1e-14)
    for i in range(2):np.testing.assert_array_equal(core.energy(logits[i:i+1]),expected[i:i+1])
    with pytest.raises(ValueError):core.energy(logits[:,:5])
