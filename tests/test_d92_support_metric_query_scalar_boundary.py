"""Exact scalar-boundary regression on the production nonzero trial branch.

The branch is compiled from source so this narrow test cannot accidentally
exercise a reimplemented tolerance check or a zero-direction/no-trial path.
This is not an end-to-end fitting or complete-head certificate test.
"""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import score_d92_support_metric_joint_benchmark as original
import score_d92_support_metric_joint_benchmark_scalar_r02 as repaired

ORIGINAL=ROOT/'tools/score_d92_support_metric_joint_benchmark.py'
REPAIRED=ROOT/'tools/score_d92_support_metric_joint_benchmark_scalar_r02.py'
OLD=b"_equal(trial['comparison_tolerance']/tol,1.,'Trial float tolerance differs')"
NEW=b"_equal(float(trial['comparison_tolerance']/tol),1.,'Trial float tolerance differs')"


def test_clone_has_exactly_one_scalar_cast_and_no_other_change():
    before=ORIGINAL.read_bytes();after=REPAIRED.read_bytes()
    assert before.count(OLD)==1 and before.count(NEW)==0
    assert after==before.replace(OLD,NEW)
    assert repaired.EPS==original.EPS
    assert repaired.FROZEN_ALGORITHM==original.FROZEN_ALGORITHM
    assert repaired.FROZEN_RESOURCES==original.FROZEN_RESOURCES


class LiteralTrialArchive:
    def __init__(self,values):self.values=values;self.calls=[]
    def reference(self,ref,coords,state,key):
        assert ref==dict(key='trial_0') and coords==dict(split_id='literal')
        assert state=='B_SUPPORT_METRIC' and key=='trial_0'
        self.calls.append(key);return ref
    def load(self,ref):
        assert ref==dict(key='trial_0')
        return self.values


def actual_trial_branch(module,source,*,tolerance_factor=1.):
    """Execute exactly the actual trial loop, including real RMSCE readback."""
    tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_candidate_numeric')
    loops=[n for n in ast.walk(function) if isinstance(n,ast.For) and
        isinstance(n.target,ast.Tuple) and [v.id for v in n.target.elts]==['i','trial']]
    assert len(loops)==1
    body=ast.parse("accepted=0\nlast=anchor\nloss=a['initial_objective']['RMSCE']").body
    body.append(deepcopy(loops[0]))
    body.extend(ast.parse('return accepted,last,loss').body)
    wrapper=ast.FunctionDef(name='run_trial',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],
        kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    compiled=compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(source),'exec')

    anchor=np.zeros(1);direction=np.array([.125]);gradient=np.array([-1.]);M=np.array([[2.]])
    rhs=1.+1e-4*float(gradient@direction);after=math.log(2.)
    tolerance=128*module.EPS*max(1.,abs(1.),abs(after),abs(rhs))
    assert type(tolerance) is np.float64
    # JSON emits/loads a Python float, while recomputing the denominator uses
    # EPS=np.float64. That exact boundary caused the original rejection.
    trial=json.loads(json.dumps(dict(trial=0,step_size=1.,loss_before=1.,loss_after=after,
        comparison_tolerance=float(tolerance)*tolerance_factor,armijo_rhs=rhs,
        metric_ball_value=float(direction@M@direction),accepted=True,
        real_inequality_holds=True,observed_objective_increase=False,state_ref=dict(key='trial_0')),
        allow_nan=False))
    assert type(trial['comparison_tolerance']) is float
    assert type(trial['comparison_tolerance']/tolerance) is np.float64
    audit=dict(initial_objective=dict(RMSCE=1.),trial_count=1,trials=[trial])
    assert audit['trial_count']>0 and len(audit['trials'])==1 and np.any(direction)
    saved=dict(theta=direction.copy(),scores=np.zeros((2,2)),labels=np.array([0,1]),probabilities=np.full((2,2),.5))
    archives=LiteralTrialArchive(saved);observed=[]
    def equal(actual,expected,label):
        if label=='Trial float tolerance differs':observed.append((type(actual),float(actual)))
        return module._equal(actual,expected,label)
    environment=dict(module.__dict__,_equal=equal,a=audit,trials=audit['trials'],anchor=anchor,
        direction=direction,M=M,initial=dict(gradient=gradient),archives=archives,
        coords=dict(split_id='literal'),stage=dict(state='B_SUPPORT_METRIC'),rank=1,
        train=['i0','i1'],classes=['a','b'],label_map=dict(i0='a',i1='b'))
    exec(compiled,environment)
    result=environment['run_trial']()
    assert archives.calls==['trial_0']
    return result,observed


def test_original_reproduces_npfloat_type_rejection_in_real_nonzero_trial_branch():
    with pytest.raises(ValueError,match='Trial float tolerance differs'):
        actual_trial_branch(original,ORIGINAL)


def test_repaired_nonzero_accepted_trial_preserves_exact_numeric_tolerance():
    (accepted,theta,loss),observed=actual_trial_branch(repaired,REPAIRED)
    assert accepted==1 and np.array_equal(theta,np.array([.125]))
    assert loss==math.log(2.) and observed==[(float,1.)]


@pytest.mark.parametrize('factor',[0.,.5,2.])
def test_incorrect_tolerance_is_still_rejected_in_actual_trial_branch(factor):
    with pytest.raises(ValueError,match='Trial float tolerance differs'):
        actual_trial_branch(repaired,REPAIRED,tolerance_factor=factor)
