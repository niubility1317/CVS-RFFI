"""Deterministic support-only correctness. Direct execution is static only."""
if __name__ == '__main__':
    import ast
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];checked=[]
    for path in (root/'code/cvsrffi/d92_conditional_joint_local_ridge.py',Path(__file__),root/'docs/D92_CONDITIONAL_JOINT_IMPLEMENTATION_20261001.md'):
        raw=path.read_bytes();source=raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source and '\r' not in source
        if path.suffix=='.py':ast.parse(source,filename=str(path))
        checked.append(str(path))
    print(json.dumps(dict(status='STATIC_CHECKS_ONLY',checked=checked),ensure_ascii=False));raise SystemExit(0)

from dataclasses import replace
import json
import numpy as np
import pytest
from cvsrffi import d92_conditional_joint_local_ridge as cj
from cvsrffi import d92_conditional_affine_kernel as ck
from cvsrffi import d92_affine_joint_local_ridge as af


def fixture(k=3,c=3,seed=181):
    rng=np.random.default_rng(seed);labels=np.repeat(np.arange(c),k)
    raw={name:rng.normal(size=(c*k,d))+.4*rng.normal(size=(c,d))[labels]
         for name,d in zip(cj._NAMES,(160,96,160,160,160))}
    names=tuple(f'c{i}' for i in range(c));ids=tuple(f'physical-{i}-{j:02}' for i in range(c) for j in range(k))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=names,old_classes=names)


def scope(**extra):
    return dict(run_id='synthetic-conditional-only',row_id='row-one',split_id='synthetic-split',scope='synthetic_support',**extra)


def old_args(args):
    mask=args['support_labels']<2;old=('c0','c1')
    out={name:args[name][mask] for name in cj._NAMES}
    out.update(support_labels=args['support_labels'][mask],support_ids=tuple(pid for i,pid in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)
    return out


@pytest.fixture(scope='module')
def sequence():
    args=fixture();oa=old_args(args)
    bp=cj.prepare_conditional_joint_training(**oa,context=scope())
    b=cj.fit_conditional_joint_local_ridge(bp)
    args['old_classes']=('c0','c1')
    cp=cj.prepare_conditional_joint_training(**args,inherited=b,context=scope())
    c=cj.fit_conditional_joint_local_ridge(cp,mode='C_seq')
    return bp,b,cp,c,oa,args


def test_independent_schema_readonly_integer_state_and_old_module_unchanged(sequence):
    bp,b,cp,c,_,_=sequence
    metadata=json.loads(json.dumps(cp.audit_dict(),allow_nan=False))
    assert metadata['classes']==list(cp.classes) and metadata['training_physical_ids']==list(cp.ids)
    assert metadata['schema']=='d92_conditional_joint_local_ridge_v1'
    assert cj.FROZEN_CONFIG['schema']=='d92_conditional_joint_local_ridge_v1'
    assert cj.FROZEN_CONFIG['method']=='D92-ConditionalJointLocalRidge-v1'
    assert cj.FROZEN_CONFIG['max_iterations']==4 and cj.FROZEN_CONFIG['max_trials']==12
    assert cj.FROZEN_CONFIG['proximal_coefficient']==0 and cj.FROZEN_CONFIG['coordinate_ball_radius']==.5
    assert af.FROZEN_CONFIG['objective'].endswith('+0.5*Z_frobenius_squared')
    assert b.labels.dtype==c.labels.dtype==np.int64 and cp.problems[0].held_labels.dtype==np.int64
    for value in (b.U,c.Z,c.final_cache['numeric']['s']):
        with pytest.raises(ValueError):value[...] = 1
    assert c.final_cache['numeric']['s'].shape==()


@pytest.mark.parametrize('mode',['B','C_seq'])
def test_real_736_geometry_complete_Z_directional_gradient_no_prox(sequence,mode):
    bp,b,cp,c,_,_=sequence;p=bp if mode=='B' else cp
    rng=np.random.default_rng(14);Z=rng.normal(size=(736,p.r));Z*=.017/np.linalg.norm(Z)
    direction=rng.normal(size=Z.shape);direction/=np.linalg.norm(direction);step=1e-5
    loss,g,audit,cache=cj.evaluate_conditional_joint_objective(p,Z)
    plus=cj.evaluate_conditional_joint_objective(p,Z+step*direction,gradient=False)[0]
    minus=cj.evaluate_conditional_joint_objective(p,Z-step*direction,gradient=False)[0]
    np.testing.assert_allclose(np.sum(g*direction),(plus-minus)/(2*step),rtol=4e-4,atol=3e-8)
    assert audit['loss_proximal']==0 and loss==audit['RMSCE']==audit['loss_total']
    ce=audit['ce_adjoint_solve_count'];assert ce==len(p.problems)
    assert audit['derivative_triangular_solve_count']==(2 if mode=='B' else 4)*ce
    assert audit['derivative_triangular_rhs_count']==len(p.classes)*audit['derivative_triangular_solve_count']
    if mode=='C_seq':
        assert audit['spectral_diagnostic_count']==3*len(p.problems)
        assert all(cf['adjoint_arrays']['g_b'].shape==(len(p.classes),) for cf in cache.folds)
        assert any(np.linalg.norm(cf['adjoint_arrays']['g_b'])>1e-8 for cf in cache.folds)


def test_actual_B_frozen_mapping_all_columns_and_lineage(sequence):
    bp,b,p,c,oa,args=sequence;oldcols=[p.classes.index(x) for x in b.classes]
    expected=b.score(**p.raw)
    np.testing.assert_array_equal(p.full_problem.M_train[:,oldcols],expected)
    assert np.all(p.full_problem.M_train[:,[i for i in range(3) if i not in oldcols]]==0)
    assert np.any(b.U) and np.array_equal(p.anchor_U,b.U)
    for pr,prior in zip(p.problems,p.audit['prior_folds']):
        assert set(pr.audit['old_reference_physical_ids'])==set(prior['training_physical_ids'])
        assert pr.audit['prior_ref']==prior['head_state_ref']
    before=np.array(p.full_problem.M_train,copy=True)
    cj.evaluate_conditional_joint_objective(p,np.full((736,p.r),.0001),gradient=False)
    np.testing.assert_array_equal(p.full_problem.M_train,before)
    for key in ('run_id','row_id','split_id','scope'):
        bad=scope();bad[key]='another'
        with pytest.raises(ValueError,match='crosses'):cj.prepare_conditional_joint_training(**args,inherited=b,context=bad)
    with pytest.raises(ValueError,match='explicit'):cj.prepare_conditional_joint_training(**args,inherited=b)
    # Trial labels are trace coordinates, never an additional B authority gate.
    cj.prepare_conditional_joint_training(**args,inherited=b,context=scope(trial=99))


def test_full_old_constraint_all_registered_columns_and_independent_expansion(sequence):
    _,b,p,c,_,_=sequence;cf=c.final_cache;num=cf['numeric'];pr=p.full_problem
    raw=pr.gamma*cf['radial'];oi=cf['representatives'];ni=pr.new_indices
    residual=raw[:,ni]@num['alpha']-raw[:,oi]@num['beta']+num['v']
    np.testing.assert_allclose(residual[pr.old_indices],0,atol=2e-12,rtol=0)
    np.testing.assert_allclose(cf['train_scores'],pr.M_train+residual,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(c.score(**p.raw),pr.M_train+residual,atol=3e-12,rtol=2e-12)
    # Independent all-variable KKT solves the constrained residual objective.
    m=len(oi);pn=len(ni);C=len(p.classes);e=np.ones((m,1));en=np.ones((pn,1))
    Q=np.block([[num['A'],-num['B'],-e],[-num['B'].T,num['D']+np.eye(pn),en],[-e.T,en.T,np.zeros((1,1))]])
    oracle=np.linalg.solve(Q,np.vstack((np.zeros((m,C)),num['R_N'],np.zeros((1,C)))))
    np.testing.assert_allclose(num['beta'],oracle[:m],atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(num['alpha'],oracle[m:m+pn],atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(num['v'],oracle[-1],atol=2e-12,rtol=2e-12)
    assert np.linalg.eigvalsh(num['K_perp'])[0]>-1e-12
    np.testing.assert_allclose(cf['train_scores'].sum(axis=1),0,atol=2e-12)


def test_B_analytic_head_same_existing_affine_at_fixed_geometry(sequence):
    bp,_,_,_,_,_=sequence;p=bp.problems[0];U=np.full((736,8),.0007)
    cf=cj._forward(p,U);num=cf['numeric'];E=cf['residual_target']
    alpha,intercept,*_=af._solve_affine_head(num['K'],E)
    np.testing.assert_allclose(num['alpha'],alpha,atol=0,rtol=0)
    np.testing.assert_allclose(num['intercept'],intercept,atol=0,rtol=0)


def test_RMSCE_pools_classes_and_forward_cache_exact_billing(sequence):
    _,_,p,_,_,_=sequence;Z=np.zeros((736,p.r))
    loss,_,info,cache=cj.evaluate_conditional_joint_objective(p,Z,gradient=False)
    scores=np.concatenate([x['score'] for x in cache.folds]);labels=np.concatenate([x['problem'].held_labels for x in cache.folds])
    shifted=scores-scores.max(axis=1,keepdims=True);ce=np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)),labels]
    means=np.array([ce[labels==k].mean() for k in range(len(p.classes))])
    np.testing.assert_allclose(loss,np.sqrt(np.mean(means**2)),atol=2e-15)
    _,g,again,_=cj.evaluate_conditional_joint_objective(p,Z,forward_cache=cache)
    assert again['inner_head_fit_count']==again['inner_factorization_count']==again['spectral_diagnostic_count']==0
    assert again['derivative_triangular_solve_count']==4*len(p.problems)
    assert info['inner_factorization_count']==2*len(p.problems)
    with pytest.raises(ValueError,match='cache'):cj.evaluate_conditional_joint_objective(p,Z+.001,forward_cache=cache)


def test_budget_nonincreasing_CE_last_accepted_and_exact_physical_counts(sequence):
    _,b,p,c,_,_=sequence
    for state in (b,c):
        audit=state.audit
        assert audit['optimizer_steps']<=4 and audit['trial_count']<=12*audit['optimizer_iterations']
        assert audit['trial_count']==audit['accepted_trial_count']+audit['rejected_trial_count']
        assert audit['accepted_trial_count']==audit['optimizer_steps']
        assert np.linalg.norm(state.Z)<=.5+1e-14
        for trial in audit['trials']:
            if trial['accepted']:assert trial['loss_after']<=trial['loss_before']+trial['comparison_tolerance']
        if audit['steps']:
            last=audit['steps'][-1]['state_ref']['key']
            np.testing.assert_array_equal(state.Z,state.records[last]['Z'])
        assert audit['head_triangular_rhs_count']>0 and audit['head_triangular_dense_work_unit_count']>0
        assert audit['resident_numeric_state_bytes']>=audit['deployment_numeric_state_bytes']>0
        expected=736*state.W.shape[1] if audit['optimizer_steps'] else 0
        assert audit['trained_parameter_count']==expected
    a=c.audit;heads=a['ajlr_forward_evaluation_count']
    assert a['projection_factorization_count']==a['residual_factorization_count']==heads
    assert a['spectral_diagnostic_count']==3*heads
    assert a['inner_factorization_count']+a['final_factorization_count']==2*heads
    assert a['completed_factorization_count']==2*heads


def test_K1_full_heads_rank0_positive_kernel_and_new0_identity():
    args=fixture(k=1);oa=old_args(args);b=cj.fit_conditional_joint_local_ridge(cj.prepare_conditional_joint_training(**oa,context=scope()))
    args['old_classes']=('c0','c1');p=cj.prepare_conditional_joint_training(**args,inherited=b,context=scope());c=cj.fit_conditional_joint_local_ridge(p,mode='C_seq')
    assert b.audit['optimizer_steps']==c.audit['optimizer_steps']==0
    assert b.audit['final_factorization_count']==1 and c.audit['final_factorization_count']==2
    assert c.audit['spectral_diagnostic_count']==3 and c.audit['trained_parameter_count']==0
    p0=replace(p,H=np.zeros_like(p.H),W=np.empty((8,0)),r=0,audit=dict(p.audit,no_information=True,no_information_reason='ZERO_DICTIONARY_RANK'))
    r=cj.fit_conditional_joint_local_ridge(p0,mode='C_seq');assert r.problem.gamma is not None and r.audit['final_factorization_count']==2
    assert r.audit['trainable_parameter_count']==r.audit['trained_parameter_count']==0
    n0=cj.prepare_conditional_joint_training(**oa,inherited=b,context=scope())
    assert n0.audit['ajlr_preparation_count']==n0.audit['prior_head_fit_count']==0
    assert cj.fit_conditional_joint_local_ridge(n0,mode='C_seq') is b


def test_zero_old_kernel_C_residual_and_intercept_are_zero():
    args=fixture();oa=old_args(args)
    for name in cj._NAMES:oa[name][:]=oa[name][0];args[name][:6]=oa[name]
    b=cj.fit_conditional_joint_local_ridge(cj.prepare_conditional_joint_training(**oa,context=scope()))
    args['old_classes']=('c0','c1');p=cj.prepare_conditional_joint_training(**args,inherited=b,context=scope());c=cj.fit_conditional_joint_local_ridge(p,mode='C_seq')
    assert c.problem.gamma is None and c.audit['final_factorization_count']==0
    np.testing.assert_array_equal(c.final_cache['numeric']['v'],np.zeros(3))
    np.testing.assert_array_equal(c.final_cache['train_scores'],p.full_problem.M_train)
    np.testing.assert_array_equal(c.score(**p.raw),p.full_problem.M_train)


def test_tau0_exact_groups_preserve_physical_ids_and_new_physical_SSE():
    args=fixture();oa=old_args(args)
    for name in cj._NAMES:
        oa[name][3:]=oa[name][:3];args[name][:6]=oa[name];args[name][6]=oa[name][0]
    b=cj.fit_conditional_joint_local_ridge(cj.prepare_conditional_joint_training(**oa,context=scope()))
    args['old_classes']=('c0','c1');p=cj.prepare_conditional_joint_training(**args,inherited=b,context=scope());c=cj.fit_conditional_joint_local_ridge(p,mode='C_seq')
    assert p.full_problem.tau==0 and p.full_problem.gamma is not None
    assert c.audit['optimizer_steps']==0 and c.audit['final_factorization_count']==2
    assert c.audit['final_fit']['old_physical_constraint_count']==6
    assert c.audit['final_fit']['old_projection_representative_count']==3
    assert c.final_cache['numeric']['alpha'].shape==(3,3)
    assert len(c.problem.audit['old_reference_physical_ids'])==6
    np.testing.assert_array_equal(c.final_cache['radial'],(p.full_problem.d0==0).astype(float))
    np.testing.assert_allclose(c.final_cache['numeric']['K_perp'][0],0,atol=1e-12)
    np.testing.assert_allclose(c.final_cache['train_scores'][p.full_problem.old_indices],p.full_problem.M_train[p.full_problem.old_indices],atol=2e-12)


def test_single_sample_policy_empty_score_and_resource_scope(sequence):
    _,_,p,c,_,_=sequence;whole=c.score(**p.raw)
    singles=np.concatenate([c.score(**{name:p.raw[name][i:i+1] for name in cj._NAMES}) for i in range(len(p.ids))])
    np.testing.assert_array_equal(whole,singles)
    measured,audit=c.score_with_audit(**p.raw);np.testing.assert_array_equal(measured,whole)
    assert audit['raw_distance_pair_count']==audit['residual']['raw_distance_pair_count']+audit['prior']['raw_distance_pair_count']
    assert audit['kernel_evaluation_count']==2*len(p.ids)
    assert audit['adapter_physical_evaluation_count']==2*len(p.ids)
    assert c.score(**{name:p.raw[name][:0] for name in cj._NAMES}).shape==(0,3)
    assert list(c.predict(**p.raw))==list(np.array(c.classes)[whole.argmax(axis=1)])


def test_class_row_permutation_and_q_gauge(sequence):
    _,_,p,_,_,_=sequence;Z=np.full((736,p.r),.0001)
    altered=replace(p,problems=tuple(replace(pr,q=np.ones(len(pr.train_labels))/len(pr.train_labels)) for pr in p.problems))
    one=cj.evaluate_conditional_joint_objective(p,Z);two=cj.evaluate_conditional_joint_objective(altered,Z)
    np.testing.assert_array_equal(one[0],two[0]);np.testing.assert_array_equal(one[1],two[1])
    args=fixture(k=1);bargs=old_args(args);b=cj.fit_conditional_joint_local_ridge(cj.prepare_conditional_joint_training(**bargs,context=scope()))
    perm=np.arange(3)[::-1];order=np.arange(3)[::-1]
    changed={name:args[name][order] for name in cj._NAMES}
    changed.update(support_labels=perm[args['support_labels'][order]],support_ids=tuple(args['support_ids'][i] for i in order),
        classes=tuple(args['classes'][i] for i in perm),old_classes=('c1','c0'))
    cp=cj.prepare_conditional_joint_training(**changed,inherited=b,context=scope());s=cj.fit_conditional_joint_local_ridge(cp,mode='C_seq')
    args['old_classes']=('c0','c1');pp=cj.prepare_conditional_joint_training(**args,inherited=b,context=scope());t=cj.fit_conditional_joint_local_ridge(pp,mode='C_seq')
    np.testing.assert_array_equal(s.score(**pp.raw),t.score(**pp.raw))


def test_real_numeric_callback_archive_shapes_refs_and_prior(sequence,tmp_path):
    _,b,p,_,_,_=sequence;records={}
    def save(key,arrays):
        records[key]={name:np.array(value,copy=True) for name,value in arrays.items()}
        np.savez(tmp_path/(key+'.npz'),**arrays)
        return dict(storage='npz',path=str(tmp_path/(key+'.npz')))
    # A K1-sized no-update stage still executes and archives its complete C head.
    quiet=replace(p,problems=(),audit=dict(p.audit,no_information=True,no_information_reason='STATIC_CALLBACK_SYNTHETIC'))
    state=cj.fit_conditional_joint_local_ridge(quiet,mode='C_seq',state_callback=save)
    final=records['final'];external_audit=json.loads(json.dumps(state.audit_dict(),allow_nan=False))
    assert final['s'].shape==() and final['train_labels'].dtype==np.int64
    assert {'A','B','D','F','E','raw_A','raw_D','alpha','beta','v','prior_B_U','prior_old_class_indices'}<=final.keys()
    assert external_audit['final_state_ref']['arrays']['s']['shape']==[]
    for name,value in final.items():
        assert value.dtype.kind in 'fibu' and np.isfinite(value).all()
    with np.load(tmp_path/'final.npz',allow_pickle=False) as restored:
        assert restored['s'].shape==()
        for name,value in final.items():np.testing.assert_array_equal(restored[name],value)
    np.testing.assert_array_equal(final['prior_B_U'],b.U)


def test_exact_group_rule_rejects_nonidentical_inputs_even_with_rounded_zero(sequence):
    bp,_,_,_,_,_=sequence;pr=bp.problems[0];cf=cj._forward(pr,np.zeros((736,8)))
    # Rounded distance alone does not prove whole-input equivalence.
    p=replace(pr,tau=0,d0=np.zeros_like(pr.d0))
    reps,groups=cj._representatives(p,cf['b'],cf['a'],np.ones_like(cf['radial']),np.ones_like(cf['crossrad']))
    assert len(reps)==len(p.old_indices) and len(set(groups))==len(groups)


def test_projected_ball_and_armijo_use_actual_delta():
    z=np.zeros((736,2));z[0,0]=.49;candidate=z.copy();candidate[0,0]=.7
    actual=cj.project_coordinates(candidate)-z
    assert np.linalg.norm(cj.project_coordinates(candidate))==.5
    assert actual[0,0]<candidate[0,0]-z[0,0]
    assert not cj._trial_acceptance(1.,1.01,-.1)['accepted']
    assert cj._trial_acceptance(1.,.98,-.1)['accepted']


def test_synthetic_forced_rejection_exhausts_exact_12_trial_budget(monkeypatch):
    args=fixture(k=2,c=2);p=cj.prepare_conditional_joint_training(**args,context=scope())
    original=cj._trial_acceptance
    def reject(before,after,dot):return dict(original(before,after,dot),accepted=False)
    monkeypatch.setattr(cj,'_trial_acceptance',reject)
    state=cj.fit_conditional_joint_local_ridge(p)
    assert state.audit['stop_reason']=='TRIAL_BUDGET_EXHAUSTED'
    assert state.audit['optimizer_iterations']==1 and state.audit['trial_count']==12
    assert state.audit['rejected_trial_count']==12 and state.audit['optimizer_steps']==0
    assert state.audit['inner_head_fit_count']==13*len(p.problems)
    assert state.audit['final_head_fit_count']==1
    np.testing.assert_array_equal(state.Z,np.zeros_like(state.Z))


def test_strict_raw_projection_failure_has_partial_spectral_cost_and_no_fallback():
    args=fixture(k=1);oa=old_args(args)
    b=cj.fit_conditional_joint_local_ridge(cj.prepare_conditional_joint_training(**oa,context=scope()))
    args['old_classes']=('c0','c1');p=cj.prepare_conditional_joint_training(**args,inherited=b,context=scope())
    # Deliberately inconsistent low-level geometry makes the raw old Gram
    # singular without proving equality of its complete original inputs.
    pr=replace(p.full_problem,d0=np.zeros_like(p.full_problem.d0))
    bad=replace(p,full_problem=pr,problems=(),audit=dict(p.audit,no_information=True,no_information_reason='SYNTHETIC_FAILURE'))
    with pytest.raises(cj.NumericalFailure) as failure:cj.fit_conditional_joint_local_ridge(bad,mode='C_seq')
    audit=failure.value.audit
    assert audit['status']=='TECHNICAL_FAILURE' and audit['spectral_diagnostic_count']==1
    assert audit['final_factorization_count']==0 and audit['final_head_fit_count']==1
    assert audit['partial_final_forward']['kernel_failure_audit']['jitter']==0
    assert audit['failure_state_ref']['key']=='failure'
    assert {'A','B','D','F','E','U','Z'}<=failure.value.records['failure'].keys()
