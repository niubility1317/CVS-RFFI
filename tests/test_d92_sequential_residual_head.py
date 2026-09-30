"""Independent synthetic math/identity checks. No external scores or target files."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi import d92_sequential_residual_head as core
from cvsrffi.d92_branch_local_ridge import FROZEN_CONFIG as BASE_CONFIG


def inputs(c=3,k=2,seed=91):
    rng=np.random.default_rng(seed);n=c*k
    return dict(z_id=rng.normal(size=(n,160)),base_scores=rng.normal(size=(n,c)),
        support_labels=np.repeat(np.arange(c),k),support_ids=tuple('id%04d'%i for i in range(n)),
        classes=tuple('class%d'%i for i in range(c)))


def independent_loss(p,U,V,aU,aV):
    # GELU expressed through math.erf rather than the implementation's vector helper.
    x=math.sqrt(160)*p.z@U
    h=np.array([[a*.5*(1+math.erf(a/math.sqrt(2)))/math.sqrt(8) for a in row] for row in x])
    logits=p.base_logits+h@V
    ce=np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(logits)),p.labels])
    return ce+(np.square(U-aU).sum()+np.square(V-aV).sum())/(2*len(logits))


def test_frozen_config_matches_json_and_original_base():
    doc=json.loads((ROOT/'configs/d92_sequential_residual_head_frozen_20260930.json').read_text(encoding='utf-8'))
    assert doc=={'algorithm':core.FROZEN_CONFIG}
    assert core.FROZEN_CONFIG['base_algorithm']==BASE_CONFIG
    assert core.FROZEN_CONFIG['channel']['equalization_enabled'] is False
    assert core.FROZEN_CONFIG['steps']==64 and core.FROZEN_CONFIG['initialization_seed'] is None


def test_analytic_gradient_independent_finite_differences():
    p=core.prepare_residual_training(**inputs());rng=np.random.default_rng(11)
    aU=core._initial_u();aV=rng.normal(size=(8,3))*.03
    U=aU+rng.normal(size=aU.shape)*.01;V=aV+rng.normal(size=aV.shape)*.02
    stats,gU,gV=core._objective_gradient(p,U,V,aU,aV)
    assert stats['loss_total']==pytest.approx(independent_loss(p,U,V,aU,aV),abs=2e-14)
    assert stats['loss_total']==pytest.approx(stats['loss_ce']+stats['loss_proximal_U']+stats['loss_proximal_V'])
    for which,index in [('U',(0,0)),('U',(79,4)),('U',(159,7)),('V',(0,0)),('V',(3,1)),('V',(7,2))]:
        delta=1e-6;plus_U=U.copy();minus_U=U.copy();plus_V=V.copy();minus_V=V.copy()
        (plus_U if which=='U' else plus_V)[index]+=delta
        (minus_U if which=='U' else minus_V)[index]-=delta
        numeric=(independent_loss(p,plus_U,plus_V,aU,aV)-independent_loss(p,minus_U,minus_V,aU,aV))/(2*delta)
        assert (gU if which=='U' else gV)[index]==pytest.approx(numeric,abs=3e-9,rel=3e-6)


def test_physical_sum_regularizer_not_mean_parameter_norm():
    p=core.prepare_residual_training(**inputs());U=core._initial_u();V=np.ones((8,3))*.2
    aU=U.copy();aV=np.zeros_like(V)
    s,gU,gV=core._objective_gradient(p,U,V,aU,aV)
    s2,gU2,gV2=core._objective_gradient(p,U,V,U,V)
    assert s['loss_proximal_V']==pytest.approx(np.sum(V*V)/(2*len(p.z)))
    np.testing.assert_allclose(gV-gV2,V/len(p.z),rtol=1e-13,atol=1e-15)
    np.testing.assert_array_equal(gU,gU2)


def test_first_step_gradients_and_exact_adam_initial_update():
    p=core.prepare_residual_training(**inputs());U=core._initial_u();V=np.zeros((8,3))
    _,gU,gV=core._objective_gradient(p,U,V,U,V)
    np.testing.assert_array_equal(gU,np.zeros_like(U));assert np.linalg.norm(gV)>0
    norm=np.linalg.norm(gV);gV=gV/max(1.,norm)
    expected_V=-.01*gV/(np.abs(gV)+1e-8)
    events=[];state=core.train_residual_head(p,log_callback=events.append)
    assert events[0]['post_U_anchor_distance']==0.
    assert events[0]['post_V_anchor_distance']==pytest.approx(np.linalg.norm(expected_V),rel=1e-13)
    assert events[0]['gradient_norm_pre_clip']==pytest.approx(norm)
    assert len(events)==64 and [x['step'] for x in events]==list(range(1,65))
    audit=state.audit_dict();assert audit['optimizer_steps']==64
    assert audit['loss_total']==pytest.approx(audit['loss_ce']+audit['loss_proximal_U']+audit['loss_proximal_V'])
    assert audit['trace_steps_emitted']==64 and 'steps' not in audit
    assert audit['optimizer_moment_bytes']==2*audit['parameter_bytes']


def test_initial_zero_residual_preserves_original_prediction_and_ties():
    data=inputs();data['classes']=('z','a','m');data['base_scores'][0]=1.
    p=core.prepare_residual_training(**data)
    state=core.ResidualHeadState(core._initial_u(),np.zeros((8,3)),p.q,p.classes,p.support_ids,
        tuple(p.canonical_classes[label] for label in p.labels),{})
    scores=state.score(z_id=data['z_id'],base_scores=data['base_scores'])
    np.testing.assert_array_equal(scores,data['base_scores']/p.q)
    assert state.predict(z_id=data['z_id'][:1],base_scores=data['base_scores'][:1]).tolist()==['a']


def test_scale_large_small_constant_and_class_shift():
    x=np.array([[1.,-2.,.3],[4.,0.,1.]])
    expected=np.sqrt(np.mean(np.ptp(x,axis=1)**2))
    assert core.score_scale(x)==pytest.approx(expected)
    assert core.score_scale(x*1e200)/1e200==pytest.approx(expected)
    assert core.score_scale(x*1e-200)/1e-200==pytest.approx(expected)
    assert core.score_scale(np.ones((4,3)))==1.
    assert core.score_scale(x+np.array([[9.],[3.]]))==pytest.approx(expected)


def test_score_scale_subnormal_nonzero_is_never_zero_range_fallback():
    smallest=np.nextafter(0.,1.)
    scores=np.zeros((4,2));scores[:,1]=smallest
    assert core.score_scale(scores)==smallest
    scores[1:,1]=0.
    with pytest.raises(FloatingPointError,match='representability'):core.score_scale(scores)
    data=inputs(c=2,k=2);data['base_scores']=scores
    with pytest.raises(core.NumericalFailure) as failure:core.prepare_residual_training(**data)
    assert failure.value.audit_dict()['stage']=='prepare_scale'


def test_training_is_class_column_and_physical_row_permutation_equivariant():
    data=inputs(c=4,k=3);a=core.train_residual_head(core.prepare_residual_training(**data))
    order=np.array([2,0,3,1]);rows=np.arange(12)[::-1];permuted=dict(data)
    permuted.update(z_id=data['z_id'][rows],base_scores=data['base_scores'][rows][:,order],
        support_labels=np.array([list(order).index(int(i)) for i in data['support_labels'][rows]]),
        support_ids=tuple(data['support_ids'][i] for i in rows),classes=tuple(data['classes'][i] for i in order))
    b=core.train_residual_head(core.prepare_residual_training(**permuted))
    np.testing.assert_array_equal(a.U,b.U)
    np.testing.assert_array_equal(a.V[:,order],b.V)
    np.testing.assert_array_equal(a.predict(z_id=data['z_id'],base_scores=data['base_scores']),
        b.predict(z_id=data['z_id'],base_scores=data['base_scores'][:,order]))


def test_class_renaming_equivariance():
    data=inputs();a=core.train_residual_head(core.prepare_residual_training(**data))
    renamed=dict(data,classes=('zebra','alpha','middle'))
    b=core.train_residual_head(core.prepare_residual_training(**renamed))
    np.testing.assert_allclose(a.U,b.U,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a.V,b.V,rtol=1e-12,atol=1e-12)


def test_inheritance_class_mapping_new_logits_and_optimizer_reset():
    old=inputs(c=2,k=2);b=core.train_residual_head(core.prepare_residual_training(**old))
    all_data=inputs(c=3,k=2);all_data['z_id'][:4]=old['z_id']
    p=core.prepare_residual_training(**all_data);aU,aV=core._anchors(p,b)
    np.testing.assert_array_equal(aU,b.U);np.testing.assert_array_equal(aV[:,:2],b.V)
    np.testing.assert_array_equal(aV[:,2],np.zeros(8))
    h,_=core._activation(p.z,aU);initial=p.base_logits+h@aV
    np.testing.assert_array_equal(initial[:,2],p.base_logits[:,2])
    stats,gU,gV=core._objective_gradient(p,aU,aV,aU,aV)
    assert np.linalg.norm(gV[:,2])>0 and stats['loss_proximal_V']==0
    events=[];seq=core.train_residual_head(p,inherited=b,log_callback=events.append)
    reset=core.train_residual_head(p)
    norm=math.hypot(np.linalg.norm(gU),np.linalg.norm(gV));clip=min(1.,1./norm)
    expected=.01*(gV*clip)/(np.abs(gV*clip)+1e-8)
    assert events[0]['post_V_anchor_distance']==pytest.approx(np.linalg.norm(expected),rel=1e-12)
    assert events[0]['inherited'] is True and events[0]['optimizer_state_inherited'] is False
    assert seq.q==reset.q==p.q and not np.array_equal(seq.V,reset.V)
    with pytest.raises(ValueError,match='N0'):core.train_residual_head(core.prepare_residual_training(**old),inherited=b)


def test_inheritance_rejects_different_old_ids_or_class_labels():
    old=inputs(c=2,k=2);b=core.train_residual_head(core.prepare_residual_training(**old))
    data=inputs(c=3,k=2);data['support_ids']=('different',)+data['support_ids'][1:]
    with pytest.raises(ValueError,match='physical'):core.train_residual_head(core.prepare_residual_training(**data),inherited=b)


def test_immutable_state_and_query_row_independence():
    data=inputs();p=core.prepare_residual_training(**data);state=core.train_residual_head(p)
    for a in (p.z,p.base_logits,p.labels,state.U,state.V):
        with pytest.raises(ValueError):a.flags.writeable=True
    with pytest.raises(FrozenInstanceError):state.q=2.
    copy=state.audit_dict();copy['optimizer_steps']=-1;assert state.audit_dict()['optimizer_steps']==64
    inputs_one=dict(z_id=data['z_id'][:1],base_scores=data['base_scores'][:1])
    all_scores=state.score(z_id=data['z_id'],base_scores=data['base_scores'])
    np.testing.assert_array_equal(state.score(**inputs_one),all_scores[:1])
    assert state.score(z_id=np.empty((0,160)),base_scores=np.empty((0,3))).shape==(0,3)


@pytest.mark.parametrize('c,k',[(1,1),(3,1),(3,3)])
def test_zero_features_degeneracy(c,k):
    data=inputs(c,k);data['z_id'][:]=0;data['base_scores'][:]=0
    events=[];state=core.train_residual_head(core.prepare_residual_training(**data),log_callback=events.append)
    assert all(e['gradient_zero'] for e in events)
    np.testing.assert_array_equal(state.V,np.zeros((8,c)))
    np.testing.assert_array_equal(state.score(z_id=data['z_id'],base_scores=data['base_scores']),np.zeros((c*k,c)))
    assert state.audit_dict()['loss_ce']==pytest.approx(math.log(c))


def test_failure_contains_last_finite_state_and_context(monkeypatch):
    p=core.prepare_residual_training(**inputs());original=core._objective_gradient;calls=0;events=[]
    def fail(*args):
        nonlocal calls
        calls+=1
        if calls==4:raise FloatingPointError('synthetic NaN')
        return original(*args)
    monkeypatch.setattr(core,'_objective_gradient',fail)
    with pytest.raises(core.NumericalFailure) as caught:
        core.train_residual_head(p,context=dict(scope='proxy',fold=None,trial=7,parent_k=10),log_callback=events.append)
    audit=caught.value.audit_dict();json.dumps(audit,allow_nan=False)
    assert audit['optimizer_steps']==3 and audit['trial']==7 and audit['parent_k']==10
    assert audit['training_physical_ids']==list(p.support_ids) and len(events)==3
    assert np.isfinite(audit['last_finite_U']).all() and np.isfinite(audit['last_finite_V']).all()


def test_nonfinite_inputs_have_serializable_failure():
    data=inputs();data['base_scores'][0,0]=np.nan
    with pytest.raises(core.NumericalFailure) as caught:core.prepare_residual_training(**data)
    assert caught.value.audit_dict()['optimizer_steps']==0
    json.dumps(caught.value.audit_dict(),allow_nan=False)


@pytest.mark.parametrize('n,k',[(26,1),(364,14),(520,20)])
def test_synthetic_supported_sizes_fixed_budget_and_bytes(n,k):
    data=inputs(c=26,k=k,seed=101);assert len(data['z_id'])==n
    state=core.train_residual_head(core.prepare_residual_training(**data))
    audit=state.audit_dict()
    assert audit['optimizer_steps']==64 and audit['trainable_parameter_count']==1488
    assert audit['persistent_state_bytes']==11912 and audit['factorization_count']==0
    assert audit['source_validation'] is None and audit['encoder_backward'] is False
    assert np.isfinite(state.score(z_id=data['z_id'][:2],base_scores=data['base_scores'][:2])).all()


def test_original_base_convenience_wrapper():
    data=inputs(c=2,k=2);rng=np.random.default_rng(13)
    features=dict(z_id=data['z_id'],fft=rng.normal(size=(4,96)),t_emb=rng.normal(size=(4,160)),
        f_emb=rng.normal(size=(4,160)),pa_local=rng.normal(size=(4,160)))
    base,prepared,audit=core.fit_residual_base(**features,support_labels=data['support_labels'],
        support_ids=data['support_ids'],classes=data['classes'],old_classes=data['classes'])
    expected=core.prepare_residual_training(z_id=data['z_id'],base_scores=base.score(**features),
        support_labels=data['support_labels'],support_ids=data['support_ids'],classes=data['classes'])
    np.testing.assert_array_equal(prepared.base_logits,expected.base_logits)
    assert audit['q']==prepared.q and audit['final_fit']['factorization_calls']==1
