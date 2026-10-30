"""Exact known-only C12 risks; no features, models or unknown inputs."""
from fractions import Fraction
import importlib
import inspect
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_macro12_risk')


def data(m,n=50):
    rows=[(c,r,s,k) for c in range(6) for r in m.RECEIVERS for s in m.SCENES for k in range(n)]
    y=np.asarray([c for c,r,s,k in rows],np.int64)
    return dict(ids=np.asarray(['V'+str(i) for i in range(len(rows))]),y=y,closed=y.copy(),baseline=y.copy(),
        score=np.asarray([10000*c+k+(0 if r==1 and s==m.SCENES[0] else 1000) for c,r,s,k in rows],np.float64),
        rx=np.asarray([r for c,r,s,k in rows],np.int64),scenario=np.asarray([s for c,r,s,k in rows])),rows


def call(m,a):return m.calibrate_known(**a)


def test_macro12_maximal_fraction_keeps_rx_scene_constraints_and_reports_cross_damage():
    m=api();a,rows=data(m,100);state,report=call(m,a)
    assert report['common_tail_rate']==dict(numerator=1,denominator=300)
    np.testing.assert_array_equal(state['conditional_thresholds'],10000*np.arange(6)+10)
    assert report['constraints_passed'] and len(report['hard_names'])==12 and len(report['diagnostic_names'])==30
    assert len(report['groups'])==42 and report['groups']['rxscene/1/'+m.SCENES[0]]['original_closed_minus_new_open']==.1
    assert all(report['groups'][k]['constraints_passed'] for k in report['hard_names'])
    assert not report['next_constraints_passed'] and set(report['next_failure_groups'])<=set(report['hard_names'])
    from experiments.pamf_phase3_open_world import star_class_risk as old
    previous,_=old.calibrate_known(**a)
    assert np.all(state['conditional_thresholds']>=previous['conditional_thresholds'])
    assert set(state)=={'schema','conditional_thresholds'}


def test_whole_ties_predicted_class_routing_and_acceptance_is_not_correct_accuracy():
    m=api();a,rows=data(m);a['closed']=(a['y']+1)%6;a['baseline']=a['closed'].copy()
    a['score']=np.asarray([100*c+(0 if row[3]<2 else row[3]) for c,row in zip(a['closed'],rows)],np.float64)
    state,r=call(m,a);assert r['common_tail_rate']==dict(numerator=1,denominator=10)
    assert r['groups']['overall']['known_macro_acceptance']==.9 and r['groups']['overall']['new_open_macro_accuracy']==0
    np.testing.assert_array_equal(state['conditional_thresholds'],np.arange(6)*100+5)
    score=np.full(6,250.);closed=np.arange(6,dtype=np.int64)
    np.testing.assert_array_equal(m.apply(score,closed,state),score-state['conditional_thresholds'][closed])
    assert list(inspect.signature(m.apply).parameters)==['score','closed','calibration']
    with pytest.raises(TypeError):m.apply(score,closed,state,y=closed)
    a['score']=a['closed'].astype(np.float64);_,r=call(m,a)
    assert r['common_tail_rate']==dict(numerator=0,denominator=1) and r['next_common_tail_rate']==dict(numerator=1,denominator=1)


def test_missing_cross_is_diagnostic_but_mandatory_or_predicted_coverage_and_closed_loss_fail(subtests):
    m=api();a,_=data(m)
    for defect in ('cross','RX','predicted','closed_loss'):
        with subtests.test(defect=defect):
            b={k:v.copy() for k,v in a.items()}
            if defect in ('cross','RX'):
                remove=(b['y']==0)&(b['rx']==1)
                if defect=='cross':remove&=b['scenario']==m.SCENES[0]
                b={k:v[~remove] for k,v in b.items()}
            elif defect=='predicted':b['closed'][b['closed']==5]=4;b['baseline']=b['closed'].copy()
            else:b['closed']=(b['y']+1)%6
            state,r=call(m,b)
            if defect=='cross':
                assert r['constraints_passed'] and not r['groups']['rxscene/1/'+m.SCENES[0]]['certified']
            else:
                assert not r['constraints_passed'] and r['zero_rejection_fallback']
                assert np.all(m.apply(b['score'],b['closed'],state)>=0)


def test_illegal_known_inputs_and_finite_scalar_state_rejected(subtests):
    m=api();a,_=data(m)
    for defect in ('score','nan','unknown','ids','rx','scene'):
        with subtests.test(defect=defect),pytest.raises(ValueError):
            b={k:v.copy() for k,v in a.items()}
            if defect=='score':b['score']=b['score'][:,None]
            elif defect=='nan':b['score'][0]=np.nan
            elif defect=='unknown':b['y'][0]=6
            elif defect=='ids':b['ids'][0]=b['ids'][1]
            elif defect=='rx':b['rx'][0]=26
            else:b['scenario'][0]='invalid'
            call(m,b)
    state,_=call(m,a)
    with pytest.raises(ValueError):m.apply(np.array([np.inf]),np.array([0]),state)
    with pytest.raises(ValueError):m.apply(np.ones(1),np.array([6]),state)
