"""Only synthetic score expansions; no training, benchmark, or real inputs."""
if __name__ == '__main__':
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in (Path(__file__),root/'code/cvsrffi/d92_margin_joint_local_ridge.py'):
        text=path.read_text(encoding='utf-8')
        assert '\ufffd' not in text
        ast.parse(text,filename=str(path))
    print('AST_UTF8_ONLY_OK')
    raise SystemExit(0)

from dataclasses import replace
import json
import numpy as np
import pytest
from cvsrffi import d92_margin_joint_local_ridge as mj


def feature_rows(n,seed):
    rng=np.random.default_rng(seed)
    return {key:rng.normal(size=(n,d)) for key,d in zip(mj._NAMES,(160,96,160,160,160))}


def artificial_state(classes,*,prior=None,tau=1.3,gamma=.7,seed=64):
    """A deterministic valid scoring expansion; no claims about fitted KKT state."""
    n=len(classes);raw=feature_rows(n,seed);rng=np.random.default_rng(seed+1)
    b,a=mj.interaction._blocks(**raw);context=mj.fcr._context(b,a)
    hb={key:value[:0] for key,value in context.items()}
    U=rng.normal(size=(736,8))*.002
    ab,aa,_=mj.fcr._adapt(context,U)
    old=classes if prior is None else prior.classes
    oi=np.array([i for i,name in enumerate(classes) if name in old],dtype=np.int64)
    ni=np.array([i for i,name in enumerate(classes) if name not in old],dtype=np.int64)
    p=mj.MarginJointProblem(context,hb,np.arange(n),np.empty(0,dtype=np.int64),tuple(classes),
        np.ones(n)/n,oi,ni,np.zeros((n,n)),np.empty((0,n)),tau,gamma,2.,
        np.zeros((n,n)),np.empty((0,n)),'B' if prior is None else 'C_seq',128,1000000,{})
    intercept=np.linspace(-.23,.41,n)
    num=dict(alpha=rng.normal(size=(n,n))*.05)
    if prior is None:
        num.update(intercept=intercept,reference=np.zeros(n),reference_self=np.array(0.),mean=np.zeros(n),grand=np.array(0.))
    else:num['b']=intercept
    return mj.MarginJointState(U,np.empty((736,0)),np.empty((8,0)),np.zeros_like(U),
        np.zeros((n,8)),np.zeros(8),raw,np.arange(n),tuple('support-'+str(i) for i in range(n)),
        tuple(classes),tuple(old),p,dict(numeric=num,b=ab,a=aa),prior,{},dict(mode=p.mode))


def sequence(**kwargs):
    b=artificial_state(('old-a','old-b'),seed=11)
    c=artificial_state(('new','old-b','old-a'),prior=b,seed=13,**kwargs)
    return b,c,feature_rows(1,71)


def test_one_B_call_then_only_C_residual_bitwise_and_mapping(monkeypatch):
    b,c,x=sequence();expected,old_audit=c.score_with_audit(**x)
    calls=[];real=mj._score_residual
    def traced(problem,*args,**kwargs):
        calls.append(problem)
        return real(problem,*args,**kwargs)
    monkeypatch.setattr(mj,'_score_residual',traced)
    bscore,ba,packet=b.score_single_for_reuse(**x)
    assert len(calls)==1 and calls[0] is b.problem
    actual,ca=c.score_single_with_reused_prior(packet,**x)
    assert len(calls)==2 and calls[1] is c.problem
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(packet.scores,bscore)
    assert ca['prior_column_indices']==[2,1]
    assert ca['prior']=={} and ca['prior_work_executed'] is False and ca['prior_reused'] is True
    assert ca['kernel_pair_count']==ca['residual']['kernel_pair_count']==len(c.ids)
    assert old_audit['kernel_pair_count']==len(c.ids)+len(b.ids)
    assert ca['prior_score_seconds'] is None and ca['prior_score_physical_count']==0
    assert ba['single_query_api_seconds']>=ba['score_seconds']
    json.dumps(ca,allow_nan=False);json.dumps(ba,allow_nan=False)


def test_input_and_score_capture_immune_to_caller_mutation():
    b,c,x=sequence();original={key:value.copy() for key,value in x.items()}
    score,_,packet=b.score_single_for_reuse(**x)
    wanted=c.score(**original)
    score[:]=99
    for value in x.values():value[:]=100
    np.testing.assert_array_equal(c.score_single_with_reused_prior(packet,**original)[0],wanted)
    with pytest.raises(ValueError,match='different single input'):
        c.score_single_with_reused_prior(packet,**x)
    for array in (packet.scores,packet.background,packet.auxiliary,*packet.features.values()):
        with pytest.raises(ValueError):array.setflags(write=True)
    assert all(v.dtype==np.float64 for v in packet.features.values()) and packet.scores.dtype==np.float64


@pytest.mark.parametrize('n',[0,2])
def test_multi_or_empty_sample_rejected(n):
    b,c,x=sequence();_,_,packet=b.score_single_for_reuse(**x)
    bad=feature_rows(n,24)
    with pytest.raises(ValueError,match='exactly one'):
        b.score_single_for_reuse(**bad)
    with pytest.raises(ValueError,match='exactly one'):
        c.score_single_with_reused_prior(packet,**bad)


def test_shape_dtype_and_wrong_actual_B_rejected_before_residual(monkeypatch):
    b,c,x=sequence();_,_,packet=b.score_single_for_reuse(**x)
    other=artificial_state(b.classes,seed=11)
    wrong=replace(c,prior=other)
    def forbidden(*args,**kwargs):raise AssertionError('Residual must not run for invalid packet')
    monkeypatch.setattr(mj,'_score_residual',forbidden)
    with pytest.raises(ValueError,match='actual B'):
        wrong.score_single_with_reused_prior(packet,**x)
    with pytest.raises(ValueError,match='exactly one'):
        c.score_single_with_reused_prior(packet,**dict(x,z_id=x['z_id'][:,:159]))
    bad=replace(packet,scores=np.zeros((1,3)))
    with pytest.raises(ValueError,match='column shape'):
        c.score_single_with_reused_prior(bad,**x)
    # Metadata mutation on a sealed NumPy array must be rejected as well.
    packet.scores.dtype=np.float32
    with pytest.raises(ValueError,match='column shape'):
        c.score_single_with_reused_prior(packet,**x)


def test_source_replacement_expires_packet_and_unsealed_source_rejected():
    b,c,x=sequence();_,_,packet=b.score_single_for_reuse(**x)
    with pytest.raises(ValueError):b.U.setflags(write=True)
    object.__setattr__(b,'U',b.U.copy())
    with pytest.raises(ValueError,match='Expired'):
        c.score_single_with_reused_prior(packet,**x)
    # Do not make a full-support copy to support manually unsealed states.
    with pytest.raises(ValueError,match='sealed B numeric storage'):
        b.score_single_for_reuse(**x)


def test_mutable_dictionary_values_expire_packet(monkeypatch):
    b,c,x=sequence()
    dictionary=mj.fcr._V0.copy()
    monkeypatch.setattr(mj.fcr,'_V0',dictionary)
    _,_,packet=b.score_single_for_reuse(**x)
    dictionary[0,0]+=.01
    with pytest.raises(ValueError,match='numeric values changed'):
        c.score_single_with_reused_prior(packet,**x)


@pytest.mark.parametrize('scalar_type',[float,np.float64])
def test_immutable_centering_scalars_bound_by_actual_type_and_value(scalar_type):
    b,c,x=sequence()
    numeric=dict(b.final_cache['numeric'],reference_self=scalar_type(0.),grand=scalar_type(0.))
    b=replace(b,final_cache=dict(b.final_cache,numeric=numeric));c=replace(c,prior=b)
    assert type(b.final_cache['numeric']['reference_self']) is scalar_type
    _,_,packet=b.score_single_for_reuse(**x)
    np.testing.assert_array_equal(c.score_single_with_reused_prior(packet,**x)[0],c.score(**x))
    old=mj._scoring_numeric_binding(scalar_type(0.))
    other_type=np.float64 if scalar_type is float else float
    for changed in (scalar_type(1.),scalar_type(-0.),other_type(0.)):
        with pytest.raises(ValueError,match='numeric scalar type or value changed'):
            mj._check_scoring_numeric_binding(mj._scoring_numeric_binding(changed),old,scope='synthetic')


def test_source_numeric_metadata_and_column_order_changes_expire_packet():
    b,c,x=sequence();_,_,packet=b.score_single_for_reuse(**x)
    b.final_cache['numeric']['alpha'].shape=(4,)
    with pytest.raises(ValueError,match='numeric storage'):
        c.score_single_with_reused_prior(packet,**x)
    b,c,x=sequence();_,_,packet=b.score_single_for_reuse(**x)
    object.__setattr__(b,'classes',tuple(reversed(b.classes)))
    with pytest.raises(ValueError,match='class column order'):
        c.score_single_with_reused_prior(packet,**x)


@pytest.mark.parametrize('target',['alpha','original'])
def test_same_source_array_shape_dtype_but_changed_strides_expires_packet(target,monkeypatch):
    b,c,x=sequence();_,_,packet=b.score_single_for_reuse(**x)
    array=b.final_cache['numeric']['alpha'] if target=='alpha' else b.problem.train_context['original']
    shape,dtype,address=array.shape,array.dtype,array.__array_interface__['data'][0]
    array.strides=(array.strides[0],0)
    assert array.shape==shape and array.dtype==dtype and array.__array_interface__['data'][0]==address
    def forbidden(*args,**kwargs):raise AssertionError('Expired view must fail before scoring')
    monkeypatch.setattr(mj,'_score_residual',forbidden)
    with pytest.raises(ValueError,match='Expired prior score cache: B: numeric storage changed'):
        c.score_single_with_reused_prior(packet,**x)


@pytest.mark.parametrize('target',['scores','background','auxiliary'])
def test_packet_sealed_array_changed_strides_rejected_before_scoring(target,monkeypatch):
    b,c,x=sequence();_,_,packet=b.score_single_for_reuse(**x)
    array=getattr(packet,target)
    shape,dtype,address=array.shape,array.dtype,array.__array_interface__['data'][0]
    array.strides=(array.strides[0],0)
    assert array.shape==shape and array.dtype==dtype and array.__array_interface__['data'][0]==address
    def forbidden(*args,**kwargs):raise AssertionError('Changed packet view must fail before scoring')
    monkeypatch.setattr(mj,'_score_residual',forbidden)
    with pytest.raises(ValueError,match='Invalid prior packet column shape or view: numeric storage changed'):
        c.score_single_with_reused_prior(packet,**x)


def test_new0_same_actual_B_returns_capture_without_any_scoring(monkeypatch):
    b,_,x=sequence();scores,_,packet=b.score_single_for_reuse(**x)
    def forbidden(*args,**kwargs):raise AssertionError('new0 must not score again')
    monkeypatch.setattr(mj,'_score_residual',forbidden)
    actual,audit=b.score_single_with_reused_prior(packet,**x)
    np.testing.assert_array_equal(actual,scores)
    assert audit['new0_exact_B_reuse'] and audit['residual']==audit['prior']=={}
    assert 'kernel_pair_count' not in audit


@pytest.mark.parametrize('tau,gamma',[(1.3,None),(1.3,0.),(0.,.7)])
def test_degenerate_C_still_executes_its_full_free_intercept(tau,gamma,monkeypatch):
    b,c,x=sequence(tau=tau,gamma=gamma)
    # Each class has one synthetic support record and r=0: never skip C's head.
    assert c.Z.shape[1]==0 and len(c.ids)==len(c.classes)
    expected=c.score(**x);_,_,packet=b.score_single_for_reuse(**x)
    calls=[];real=mj._score_residual
    def traced(problem,*args,**kwargs):
        calls.append(problem);return real(problem,*args,**kwargs)
    monkeypatch.setattr(mj,'_score_residual',traced)
    actual,audit=c.score_single_with_reused_prior(packet,**x)
    assert len(calls)==1 and calls[0] is c.problem
    np.testing.assert_array_equal(actual,expected)
    assert audit['residual']['intercept_addition_count']==len(c.classes)
    if gamma is None or gamma==0:
        assert actual[0,0]==c.final_cache['numeric']['b'][0]
    if gamma is None:assert 'kernel_pair_count' not in audit['residual']


def test_default_score_with_audit_remains_on_original_two_call_path(monkeypatch):
    b,c,x=sequence();before=dict(mj.FROZEN_CONFIG);calls=[];real=mj._score_residual
    def traced(problem,*args,**kwargs):
        calls.append(problem);return real(problem,*args,**kwargs)
    monkeypatch.setattr(mj,'_score_residual',traced)
    _,audit=c.score_with_audit(**x)
    assert len(calls)==2 and calls[0] is c.problem and calls[1] is b.problem
    assert 'prior_reused' not in audit and audit['prior']['kernel_pair_count']==len(b.ids)
    assert mj.FROZEN_CONFIG==before


def test_residual_failure_keeps_partial_work_and_no_complete_C(monkeypatch):
    b,c,x=sequence();bs,ba,packet=b.score_single_for_reuse(**x)
    def failed(p,cache,U,b0,a0,progress=None,H=None):
        progress['raw_distance_pair_count']=3
        raise FloatingPointError('synthetic C residual failure')
    monkeypatch.setattr(mj,'_score_residual',failed)
    with pytest.raises(FloatingPointError) as caught:
        c.score_single_with_reused_prior(packet,**x)
    a=caught.value.single_query_reuse_audit
    assert a['score_physical_count'] is None and a['residual']=={'raw_distance_pair_count':3}
    assert not a['prior_work_executed'] and a['prior']=={}
    np.testing.assert_array_equal(bs,packet.scores)
    assert ba['score_physical_count']==1


def test_frozen_production_K1_states_and_actual_new0():
    # Small, legal synthetic support; actual production prepare/fit constructs
    # the immutable B/C state, with no adapter optimization at K1.
    from test_d92_margin_joint_local_ridge import fixture,old_args,scope,LIMITS
    args=fixture(k=1);oa=old_args(args)
    b=mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**oa,**LIMITS,context=scope()))
    # The real production centering stores these as immutable Python scalars,
    # unlike the literal zero-dimensional arrays in artificial_state above.
    assert type(b.final_cache['numeric']['reference_self']) is float
    assert type(b.final_cache['numeric']['grand']) is float
    args['old_classes']=('c0','c1')
    c=mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**args,**LIMITS,inherited=b,context=scope()),mode='C_seq')
    new0=mj.fit_margin_joint_local_ridge(mj.prepare_margin_joint_training(**oa,**LIMITS,inherited=b,context=scope()),mode='C_seq')
    assert new0 is b
    x=feature_rows(1,802)
    bs,_,packet=b.score_single_for_reuse(**x)
    np.testing.assert_array_equal(c.score_single_with_reused_prior(packet,**x)[0],c.score(**x))
    np.testing.assert_array_equal(new0.score_single_with_reused_prior(packet,**x)[0],bs)
