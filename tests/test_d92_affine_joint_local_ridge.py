"""Deterministic affine-head correctness; no real data or experiment scoring.

Direct execution is a stdlib-only AST/UTF-8 check, never a numerical test.
"""
if __name__ == '__main__':
    import ast
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];checked=[]
    for path in (root/'code/cvsrffi/d92_affine_joint_local_ridge.py',Path(__file__),root/'docs/D92_AFFINE_JOINT_CORE_20261001.md'):
        raw=path.read_bytes();source=raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in source and '\r' not in source
        if path.suffix=='.py':ast.parse(source,filename=str(path))
        checked.append(str(path))
    print(json.dumps({'status':'STATIC_CHECKS_ONLY','checked':checked},ensure_ascii=False))
    raise SystemExit(0)

from dataclasses import replace
import json
import numpy as np
import pytest
from cvsrffi import d92_affine_joint_local_ridge as af
from cvsrffi import d92_anchor_joint_local_ridge as aj


def fixture(k=3,c=3,seed=81):
    rng=np.random.default_rng(seed);labels=np.repeat(np.arange(c),k)
    raw={key:rng.normal(size=(c*k,d))+.35*rng.normal(size=(c,d))[labels]
         for key,d in zip(af._NAMES,(160,96,160,160,160))}
    names=tuple(f'c{i}' for i in range(c));ids=tuple(f'physical-{i}-{j:02}' for i in range(c) for j in range(k))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=names,old_classes=names)


def scope(**extra):
    return dict(run_id='synthetic-affine-only',row_id='row-one',scope='synthetic_support',**extra)


def sequence(k=3):
    args=fixture(k=k,c=3);old=('c0','c1');mask=args['support_labels']<2
    oa={key:args[key][mask] for key in af._NAMES}
    oa.update(support_labels=args['support_labels'][mask],support_ids=tuple(x for i,x in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)
    b=af.fit_affine_joint_local_ridge(af.prepare_affine_joint_training(**oa,context=scope()))
    args['old_classes']=old
    p=af.prepare_affine_joint_training(**args,inherited=b,context=scope())
    return b,p,oa,args


def test_independent_schema_draft_and_integer_state_and_unchanged_old_config():
    assert af.FROZEN_CONFIG['schema']=='d92_affine_joint_local_ridge_v1'
    assert af.FROZEN_CONFIG['method']=='D92-AffineJointLocalRidge-v1'
    assert af.FROZEN_CONFIG['candidate_status']=='STRUCTURE_CANDIDATE_DRAFT_NOT_PUBLISHED'
    assert af.FROZEN_CONFIG['free_intercept'] and af.FROZEN_CONFIG['intercept_penalty']==0
    assert not aj.FROZEN_CONFIG['free_intercept']
    assert af.FROZEN_CONFIG['max_trials']==12 and af.FROZEN_CONFIG['max_iterations']==4
    p=af.prepare_affine_joint_training(**fixture());assert p.labels.dtype==np.int64
    assert p.problems[0].train_labels.dtype==p.problems[0].held_labels.dtype==np.int64
    with pytest.raises(ValueError):af.V0[0,0]=2


def test_schur_solution_against_independent_primal_and_saddle_oracles():
    rng=np.random.default_rng(31);n=8;d=4;c=3;q=np.array([.5,0,.5,0,0,0,0,0])
    phi=rng.normal(size=(n,d));phi-=q@phi;held=rng.normal(size=(5,d))
    y=np.array([0,0,0,0,1,1,2,2]);Y=np.eye(c)[y]-1/c
    M=rng.normal(size=(n,c));M-=M.mean(axis=1,keepdims=True);E=Y-M
    Mh=rng.normal(size=(5,c));Mh-=Mh.mean(axis=1,keepdims=True)
    K=phi@phi.T;L=held@phi.T
    alpha,b,z,s,chol,rhs=af._solve_affine_head(K,E)
    X=np.column_stack((phi,np.ones(n)));penalty=np.diag([1.]*d+[0.])
    beta=np.linalg.solve(X.T@X+penalty,X.T@E)
    expected_train=M+X@beta;expected_held=Mh+np.column_stack((held,np.ones(5)))@beta
    np.testing.assert_allclose(alpha,Y-expected_train,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(b,beta[-1],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(Mh+L@alpha+b,expected_held,rtol=3e-13,atol=3e-13)
    saddle=np.block([[K+np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
    oracle=np.linalg.solve(saddle,np.vstack((E,np.zeros((1,c)))))
    np.testing.assert_allclose(alpha,oracle[:-1],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(b,oracle[-1],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(alpha.sum(axis=0),0,atol=3e-14)
    assert rhs.shape==(n,c+1) and n/(1+np.trace(K))<=s<=n
    np.testing.assert_allclose((K+np.eye(n))@z,np.ones(n),atol=3e-14)
    assert np.linalg.norm(E.mean(axis=0))>1e-2 and np.linalg.norm(b)>1e-2


def test_full_intercept_adjoint_including_nonzero_g_b_finite_difference():
    rng=np.random.default_rng(62);n=7;h=4;c=3
    phi=rng.normal(size=(n,4));K=phi@phi.T;L=rng.normal(size=(h,n));E=rng.normal(size=(n,c))
    alpha,b,z,s,chol,_=af._solve_affine_head(K,E);G=rng.normal(size=(h,c))
    T,eta,g_b,rhs=af._affine_adjoint(chol,L,G,z,s)
    np.testing.assert_allclose(T.sum(axis=0),g_b,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose((K+np.eye(n))@T+eta,L.T@G,rtol=2e-13,atol=3e-13)
    dK=rng.normal(size=(n,n));dK=.5*(dK+dK.T);dL=rng.normal(size=(h,n));step=1e-6
    def value(k,l):
        a,bb,*_=af._solve_affine_head(k,E)
        return np.sum(G*(l@a+bb))
    fd=(value(K+step*dK,L+step*dL)-value(K-step*dK,L-step*dL))/(2*step)
    barK=-.5*(T@alpha.T+alpha@T.T);barL=G@alpha.T
    np.testing.assert_allclose(fd,np.sum(barK*dK)+np.sum(barL*dL),rtol=2e-7,atol=2e-7)
    wrong=np.linalg.solve(K+np.eye(n),L.T@G)
    wrongdot=np.sum((-.5*(wrong@alpha.T+alpha@wrong.T))*dK)+np.sum(barL*dL)
    assert np.linalg.norm(g_b)>1e-2 and abs(wrongdot-fd)>1e-3


@pytest.mark.parametrize('mode',['B','C_seq'])
def test_full_Z_directional_gradient_and_proximal(mode):
    p=af.prepare_affine_joint_training(**fixture(seed=17)) if mode=='B' else sequence()[1]
    rng=np.random.default_rng(52);Z=rng.normal(size=(736,p.r));Z*=.02/np.linalg.norm(Z)
    d=rng.normal(size=Z.shape);d/=np.linalg.norm(d)
    _,g,info,_=af.evaluate_affine_joint_objective(p,Z);step=1e-5
    plus=af.evaluate_affine_joint_objective(p,Z+step*d,gradient=False)
    minus=af.evaluate_affine_joint_objective(p,Z-step*d,gradient=False)
    np.testing.assert_allclose(np.sum(g*d),(plus[0]-minus[0])/(2*step),rtol=3e-4,atol=2e-8)
    np.testing.assert_allclose((plus[2]['loss_proximal']-minus[2]['loss_proximal'])/(2*step),np.sum(Z*d),rtol=2e-9,atol=1e-12)
    assert info['derivative_triangular_rhs_count']==2*len(p.classes)*info['ce_adjoint_solve_count']


def test_q_gauge_alpha_scores_and_complete_raw_VJP_equivalence():
    rng=np.random.default_rng(46);n=7;h=4;c=3;gamma=.8
    x=rng.normal(size=(n,5));hx=rng.normal(size=(h,5))
    R=np.exp(-np.sum((x[:,None]-x[None,:])**2,axis=2)/5)
    Q=np.exp(-np.sum((hx[:,None]-x[None,:])**2,axis=2)/5)
    q=np.array([.5,0,.5,0,0,0,0]);u=np.ones(n)/n
    E=rng.normal(size=(n,c));E-=E.mean(axis=1,keepdims=True)
    G=rng.normal(size=(h,c));results=[]
    for measure in (q,u):
        P=np.eye(n)-np.ones((n,1))*measure[None,:]
        K=gamma*P@R@P.T;L=gamma*(Q-np.ones((h,1))*(measure@R)[None,:])@P.T
        a,b,z,s,chol,_=af._solve_affine_head(K,E)
        T,eta,g_b,_=af._affine_adjoint(chol,L,G,z,s)
        bk=-.5*(T@a.T+a@T.T);bl=G@a.T;br,bq=af._center_vjp(bk,bl,measure,gamma)
        np.testing.assert_allclose(br,-gamma*.5*(T@a.T+a@T.T),rtol=3e-13,atol=3e-13)
        np.testing.assert_allclose(bq,gamma*G@a.T,rtol=3e-13,atol=3e-13)
        results.append((a,b,L@a+b,br,bq,np.sum(a*(K@a))))
    a,b,score,br,bq,ridge=results[0];a2,b2,score2,br2,bq2,ridge2=results[1]
    np.testing.assert_allclose(a,a2,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(score,score2,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(b2,b+gamma*(u-q)@R@a,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(br,br2,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(bq,bq2,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(ridge,ridge2,rtol=3e-13,atol=3e-13)


def test_C_gauge_full_objective_Z_gradient_same_fixed_raw_prior_and_scale():
    _,p,_,_=sequence();Z=np.full((736,p.r),.0002)
    changed=tuple(replace(pr,q=np.ones(len(pr.train_labels))/len(pr.train_labels)) for pr in p.problems)
    p2=replace(p,problems=changed)
    one=af.evaluate_affine_joint_objective(p,Z);two=af.evaluate_affine_joint_objective(p2,Z)
    np.testing.assert_allclose(one[0],two[0],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(one[1],two[1],rtol=5e-10,atol=3e-12)


@pytest.mark.parametrize('nonzero_U',[False,True])
def test_balanced_B_same_as_original_head_and_full_gradient_at_fixed_U(nonzero_U):
    p=af.prepare_affine_joint_training(**fixture());pr=p.problems[0]
    U=np.full((736,8),.001) if nonzero_U else np.zeros((736,8))
    affine=af._forward(pr,U);old=aj._forward(pr,U)
    np.testing.assert_allclose(affine['intercept'],0,atol=3e-14)
    np.testing.assert_allclose(affine['alpha'],old['alpha'],rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(affine['score'],old['score'],rtol=3e-14,atol=3e-14)
    G=np.random.default_rng(12).normal(size=affine['score'].shape)
    ap={'derivative_triangular_solve_count':0,'ce_adjoint_solve_count':0};op={'derivative_triangular_solve_count':0,'ce_adjoint_solve_count':0}
    ga=af._backward(affine,G,ap);go=aj._backward(old,G,op)
    np.testing.assert_allclose(ga,go,rtol=3e-10,atol=3e-12)


def test_unbalanced_low_level_B_changes_intercept_and_scores_without_public_contract_change():
    p=af.prepare_affine_joint_training(**fixture());pr=p.problems[0]
    y=np.zeros(len(pr.train_labels),dtype=np.int64);y[-1]=1;pr=replace(pr,train_labels=y)
    affine=af._forward(pr,np.zeros((736,8)));old=aj._forward(pr,np.zeros((736,8)))
    np.testing.assert_allclose(affine['intercept'],affine['Y'].mean(axis=0),atol=3e-14)
    assert np.linalg.norm(affine['intercept'])>.1 and np.linalg.norm(affine['score']-old['score'])>.1
    args=fixture();args['support_labels']=y
    with pytest.raises(ValueError):af.prepare_affine_joint_training(**args)


def test_actual_prior_same_run_row_scope_fold_and_frozen_values():
    b,p,_,args=sequence();old_indices=[p.classes.index(x) for x in b.classes]
    np.testing.assert_allclose(p.full_problem.M_train[:,old_indices],b.score(**p.raw),rtol=0,atol=0)
    assert np.all(p.full_problem.M_train[:,[i for i in range(3) if i not in old_indices]]==0)
    for pr,pm in zip(p.problems,p.audit['prior_folds']):
        assert set(pr.audit['old_reference_physical_ids'])==set(pm['training_physical_ids'])
        assert not set(pm['training_physical_ids'])&set(pm['held_physical_ids'])
        assert pr.audit['prior_ref']==pm['head_state_ref']
    for key in ('run_id','row_id','scope'):
        bad=scope();bad[key]='different'
        with pytest.raises(ValueError,match='crosses'):af.prepare_affine_joint_training(**args,inherited=b,context=bad)
    with pytest.raises(ValueError,match='explicit'):af.prepare_affine_joint_training(**args,inherited=b)
    with pytest.raises(ValueError,match='fold'):af.prepare_affine_joint_training(**args,inherited=b,context=scope(fold=1))
    # Optimizer/trial labels are not an additional lineage permission mechanism.
    af.prepare_affine_joint_training(**args,inherited=b,context=scope(trial=100))
    old_state=aj.fit_anchor_joint_local_ridge(aj.prepare_anchor_joint_training(**fixture(c=2)))
    with pytest.raises(ValueError):af.prepare_affine_joint_training(**args,inherited=old_state,context=scope())


def test_N0_precise_B_object_reuse_zero_C_solve_and_K1_full_head():
    b,p,oa,_=sequence(k=1);np.testing.assert_allclose(b.final_cache['intercept'],0,atol=3e-14)
    s=af.fit_affine_joint_local_ridge(p,mode='C_seq');assert s.audit['optimizer_steps']==0
    assert s.audit['final_head_fit_count']==1 and s.audit['trained_parameter_count']==0
    n0=af.prepare_affine_joint_training(**oa,inherited=b,context=scope())
    assert n0.audit['ajlr_preparation_count']==n0.audit['prior_head_fit_count']==0
    assert af.fit_affine_joint_local_ridge(n0,mode='C_seq') is b


def test_rank0_complete_affine_head_and_exact_anchor_null_component():
    p=af.prepare_affine_joint_training(**fixture());p=replace(p,H=np.zeros_like(p.H),W=np.empty((8,0)),r=0,
        audit=dict(p.audit,no_information=True,no_information_reason='ZERO_DICTIONARY_RANK'))
    s=af.fit_affine_joint_local_ridge(p);assert s.audit['final_factorization_count']==1
    assert s.audit['trainable_parameter_count']==0
    H=np.zeros((12,8));H[:,0]=np.arange(12)+1;W,_,ci=af.latent_coordinates(H)
    anchor=np.zeros((736,8));anchor[3,7]=13
    np.testing.assert_array_equal(af.reconstruct_U(np.zeros((736,ci['latent_rank'])),anchor,W),anchor)
    assert af.reconstruct_U(np.ones((736,ci['latent_rank']))*.001,anchor,W)[3,7]==13


def test_tau0_nonzero_equivalence_kernel_uses_Schur_not_zero_kernel_mean_shortcut():
    args=fixture(c=2)
    for key in af._NAMES:args[key][3:]=args[key][:3]
    bp=af.prepare_affine_joint_training(**args,context=scope());b=af.fit_affine_joint_local_ridge(bp)
    allargs=fixture(c=3)
    for key in af._NAMES:allargs[key][:6]=args[key];allargs[key][6]=args[key][0]
    allargs['old_classes']=('c0','c1')
    p=af.prepare_affine_joint_training(**allargs,inherited=b,context=scope());s=af.fit_affine_joint_local_ridge(p,mode='C_seq')
    cf=s.final_cache;n=len(p.ids);c=len(p.classes)
    assert p.full_problem.tau==0 and p.full_problem.s0>0 and s.audit['optimizer_steps']==0
    assert s.audit['final_factorization_count']==1 and np.linalg.norm(cf['K'])>0
    saddle=np.block([[cf['K']+np.eye(n),np.ones((n,1))],[np.ones((1,n)),np.zeros((1,1))]])
    oracle=np.linalg.solve(saddle,np.vstack((cf['E'],np.zeros((1,c)))))
    np.testing.assert_allclose(cf['alpha'],oracle[:-1],atol=3e-13)
    np.testing.assert_allclose(cf['intercept'],oracle[-1],atol=3e-13)
    np.testing.assert_array_equal(cf['radial'],(p.full_problem.d0==0).astype(float))
    assert np.linalg.norm(cf['intercept']-cf['E'].mean(axis=0))>1e-6
    prog={'derivative_triangular_solve_count':0,'ce_adjoint_solve_count':0}
    np.testing.assert_array_equal(af._backward(dict(cf,problem=p.full_problem),np.ones((0,c)),prog),np.zeros((736,8)))


@pytest.mark.parametrize('single_class',[False,True])
def test_missing_scale_balanced_exact_zero_tie_and_unbalanced_low_level_mean(single_class):
    args=fixture(c=1 if single_class else 3)
    if not single_class:
        for key in af._NAMES:args[key][:]=args[key][0]
    p=af.prepare_affine_joint_training(**args);s=af.fit_affine_joint_local_ridge(p)
    assert s.problem.gamma is None and s.audit['final_factorization_count']==s.audit['head_triangular_rhs_count']==0
    np.testing.assert_array_equal(s.final_cache['intercept'],np.zeros(len(p.classes)))
    np.testing.assert_array_equal(s.score(**p.raw),np.zeros((len(p.ids),len(p.classes))))
    assert set(s.predict(**p.raw))=={min(p.classes)}
    if not single_class:
        pr=p.full_problem;y=np.zeros(len(pr.train_labels),dtype=np.int64);y[-1]=1
        cf=af._forward(replace(pr,train_labels=y),np.zeros((736,8)))
        np.testing.assert_allclose(cf['intercept'],cf['E'].mean(axis=0),atol=0)
        np.testing.assert_array_equal(cf['alpha'],cf['E']-cf['intercept'])
        assert cf['audit']['head_triangular_solve_count']==0


def score_archive(arrays,raw):
    b0,a0=af.interaction._blocks(**raw,allow_empty=True);out=np.tile(arrays['intercept'],(len(b0),1))
    if arrays['gamma'].size:
        gamma=float(arrays['gamma']);tau=float(arrays['tau']);q=arrays['q'];R=arrays['raw_train_minus_one']
        for i in range(len(b0)):
            d0=af.local._distances(b0[i:i+1],a0[i:i+1],arrays['original_train_b'],arrays['original_train_a'])
            if tau==0:d=d0
            else:
                b,a,_=af.fcr._adapt(af.fcr._context(b0[i:i+1],a0[i:i+1]),arrays['U'])
                d=.5*d0+.5*af.local._distances(b,a,arrays['adapted_train_b'],arrays['adapted_train_a'])
            r=af.local._radial_minus_one(d,tau)[0]
            centered=r-float(r@q)-q@R+float(q@R@q)
            out[i]+=gamma*centered@arrays['alpha']
    if 'prior_B_U' in arrays:
        prior={k[len('prior_B_'):]:v for k,v in arrays.items() if k.startswith('prior_B_')}
        out[:,arrays['prior_old_class_indices']]+=score_archive(prior,raw)
    return out


def test_actual_prior_archive_and_single_sample_all_class_scores_and_resource():
    b,p,_,_=sequence();s=af.fit_affine_joint_local_ridge(p,mode='C_seq');arrays=s.to_arrays()
    assert {'intercept','schur_z','schur_s','combined_rhs','prior_B_intercept','prior_B_U'}<=arrays.keys()
    np.testing.assert_array_equal(arrays['prior_B_intercept'],b.final_cache['intercept'])
    scores=s.score(**p.raw);np.testing.assert_allclose(score_archive(arrays,p.raw),scores,rtol=3e-13,atol=3e-13)
    for size in (1,2,4):
        chunks=[s.score(**{k:v[i:i+size] for k,v in p.raw.items()}) for i in range(0,len(p.ids),size)]
        np.testing.assert_array_equal(np.vstack(chunks),scores)
    np.testing.assert_array_equal(af.predict_affine_joint_local_ridge(s,**p.raw),s.predict(**p.raw))
    with pytest.raises(TypeError):af.predict_affine_joint_local_ridge(b.final_cache,**p.raw)
    score,work=s.score_with_audit(**p.raw);np.testing.assert_array_equal(score,scores)
    assert work['intercept_addition_count']==len(p.ids)*(len(s.classes)+len(b.classes))
    assert s.audit['persistent_state_bytes']==af._unique_bytes(s._resident_values())
    assert s.audit['deployment_numeric_state_bytes']==af._unique_bytes(s._deployment_values())
    assert s.audit['analytic_intercept_parameter_count']==len(s.classes)


def test_real_combined_RHS_adjoint_prior_cost_and_lossless_callback():
    b,p,_,_=sequence();saved={};events=[]
    def writer(key,arrays):
        assert key not in saved and all(v.dtype.kind in 'fibu' and np.isfinite(v).all() for v in arrays.values())
        saved[key]={k:np.array(v,copy=True) for k,v in arrays.items()}
        return {'path':'synthetic/'+key+'.npz'}
    s=af.fit_affine_joint_local_ridge(p,mode='C_seq',state_callback=writer,log_callback=events.append);a=s.audit_dict();c=len(p.classes)
    assert a['head_triangular_rhs_count']==2*(c+1)*(a['inner_factorization_count']+a['final_factorization_count'])
    assert a['head_triangular_rhs_element_count']==sum(2*len(x['train_labels'])*(c+1) for key,x in saved.items() if '_head_' in key)+2*len(p.ids)*(c+1)
    assert a['derivative_triangular_rhs_count']==2*c*a['ce_adjoint_solve_count']
    assert p.audit['prior_triangular_rhs_count']==2*(len(b.classes)+1)*p.audit['prior_factorization_count']
    assert a['intercept_fit_count']==a['inner_head_fit_count']+a['final_head_fit_count']
    assert a['optimizer_steps']==a['accepted_trial_count'] and a['trial_count']==a['accepted_trial_count']+a['rejected_trial_count']
    assert a['inner_objective_evaluation_count']==1+a['trial_count']
    assert a['ajlr_forward_evaluation_count']==a['inner_head_fit_count']+a['final_head_fit_count']
    assert not s.state_records() and a['retained_vector_record_bytes']==0
    assert all('intercept' in x for key,x in saved.items() if '_head_' in key or key=='final')
    assert any('fold_0_adjoint_g_b' in x for key,x in saved.items() if key.startswith('gradient_'))
    json.dumps(a,allow_nan=False);json.dumps(events,allow_nan=False)


def test_twelve_rejected_trials_keep_actual_accepted_cache_and_charge_RHS(monkeypatch):
    p=af.prepare_affine_joint_training(**fixture());real=af.evaluate_affine_joint_objective;calls=[0]
    def wrapped(prepared,Z,anchor_U=None,*,gradient=True,forward_cache=None):
        loss,g,info,cache=real(prepared,Z,anchor_U,gradient=gradient,forward_cache=forward_cache)
        if forward_cache is None:
            calls[0]+=1
            if calls[0]>1:return loss+100,g,dict(info,loss_total=loss+100),cache
        return loss,g,info,cache
    monkeypatch.setattr(af,'evaluate_affine_joint_objective',wrapped)
    s=af.fit_affine_joint_local_ridge(p);a=s.audit_dict()
    assert a['stop_reason']=='TRIAL_BUDGET_EXHAUSTED' and a['rejected_trial_count']==12
    assert a['optimizer_steps']==0 and np.all(s.U==0)
    assert a['final_objective']['loss_total']==a['initial_objective']['loss_total']
    assert a['head_triangular_rhs_count']==2*(len(p.classes)+1)*(13*len(p.problems)+1)


def test_class_row_permutation_CE_aggregation_and_forbidden_inputs():
    args=fixture(k=5);p=af.prepare_affine_joint_training(**args);Z=np.full((736,p.r),.0001)
    out=af.evaluate_affine_joint_objective(p,Z);cache=out[3]
    scores=np.concatenate([f['score'] for f in cache.folds]);labels=np.concatenate([f['problem'].held_labels for f in cache.folds])
    ce,_=af._ce(scores,labels);means=np.array([ce[labels==j].mean() for j in range(3)])
    np.testing.assert_allclose(out[2]['class_ce_means'],means,rtol=3e-15)
    order=np.random.default_rng(7).permutation(len(p.ids));per={key:args[key][order] for key in af._NAMES}
    per.update(support_labels=args['support_labels'][order],support_ids=tuple(args['support_ids'][i] for i in order),classes=('zeta','alpha','mu'),old_classes=('zeta','alpha','mu'))
    p2=af.prepare_affine_joint_training(**per);other=af.evaluate_affine_joint_objective(p2,Z)
    np.testing.assert_allclose(out[0],other[0],rtol=4e-13)
    np.testing.assert_allclose(out[1],other[1],rtol=3e-10,atol=3e-13)
    with pytest.raises(TypeError):af.prepare_affine_joint_training(**args,query_labels=np.zeros(2))
    with pytest.raises(ValueError):af.fit_affine_joint_local_ridge(p,mode='C_reset_init')
