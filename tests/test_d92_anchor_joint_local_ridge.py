"""Deterministic synthetic AJLR correctness tests; no experiment data.

Direct execution performs stdlib-only AST/UTF-8 checks and config generation.
Pytest execution requires the project's verified ssr-gpu environment.
"""
if __name__ == '__main__':
    import ast
    import json
    import sys
    from copy import deepcopy
    from pathlib import Path
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    core = root/'code/cvsrffi/d92_anchor_joint_local_ridge.py'
    base = root/'code/cvsrffi/d92_branch_local_ridge.py'
    def config_expression(path):
        tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        return next(n.value for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='FROZEN_CONFIG' for t in n.targets))
    bc=eval(compile(ast.Expression(config_expression(base)),str(base),'eval'),{'dict':dict})
    cfg=eval(compile(ast.Expression(config_expression(core)),str(core),'eval'),
             {'dict':dict,'deepcopy':deepcopy,'local':SimpleNamespace(FROZEN_CONFIG=bc)})
    config_path=root/'configs/d92_anchor_joint_frozen_20261001.json'
    if '--write-config' in sys.argv:
        config_path.write_text(json.dumps({'algorithm':cfg},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    checked=[]
    for path in (core,Path(__file__),config_path,root/'docs/D92_ANCHOR_JOINT_CORE_20261001.md'):
        raw=path.read_bytes();text=raw.decode('utf-8')
        assert not raw.startswith(b'\xef\xbb\xbf') and '\ufffd' not in text and '\r' not in text
        if path.suffix=='.py':ast.parse(text,filename=str(path))
        checked.append(str(path))
    assert json.loads(config_path.read_text(encoding='utf-8'))=={'algorithm':cfg}
    print(json.dumps({'status':'STATIC_CHECKS_ONLY','checked':checked},ensure_ascii=False))
    sys.exit(0)

from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from cvsrffi import d92_anchor_joint_local_ridge as aj
from cvsrffi import d92_branch_local_ridge as local


def fixture(k=3,c=3,seed=81):
    rng=np.random.default_rng(seed);labels=np.repeat(np.arange(c),k)
    raw={key:rng.normal(size=(c*k,d))+.35*rng.normal(size=(c,d))[labels]
         for key,d in zip(aj._NAMES,(160,96,160,160,160))}
    names=tuple(f'c{i}' for i in range(c));ids=tuple(f'physical-{i}-{j:02}' for i in range(c) for j in range(k))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=names,old_classes=names)


def raw_slice(p,ids):
    ix=[p.ids.index(pid) for pid in ids]
    return {k:np.asarray(p.raw[k])[ix] for k in aj._NAMES}


def sequence(k=3):
    args=fixture(k=k,c=3);old=('c0','c1');mask=args['support_labels']<2
    oa={key:args[key][mask] for key in aj._NAMES}
    oa.update(support_labels=args['support_labels'][mask],support_ids=tuple(x for i,x in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)
    b=aj.fit_anchor_joint_local_ridge(aj.prepare_anchor_joint_training(**oa,context={'row_id':'one'}))
    args['old_classes']=old
    p=aj.prepare_anchor_joint_training(**args,inherited=b,context={'row_id':'one'})
    return b,p,oa,args


def test_frozen_config_and_dictionary():
    p=Path(__file__).parents[1]/'configs/d92_anchor_joint_frozen_20261001.json'
    assert json.loads(p.read_text(encoding='utf-8'))=={'algorithm':aj.FROZEN_CONFIG}
    assert aj.FROZEN_CONFIG['max_trials']==12 and aj.FROZEN_CONFIG['arms']==['local_ridge','ajlr_seq']
    assert not aj.FROZEN_CONFIG['keep'] and not aj.FROZEN_CONFIG['guard']
    np.testing.assert_allclose(aj.V0@aj.V0.T,np.eye(8),atol=3e-15,rtol=0)
    with pytest.raises(ValueError):aj.V0[0,0]=2
    p=aj.prepare_anchor_joint_training(**fixture())
    assert p.labels.dtype==np.int64 and p.problems[0].train_labels.dtype==np.int64
    assert p.problems[0].held_labels.dtype==np.int64


def test_B_fixed_scale_zero_coordinate_matches_R0_and_live_derivative():
    p=aj.prepare_anchor_joint_training(**fixture());Z=np.zeros((736,p.r))
    loss,g,info,cache=aj.evaluate_anchor_joint_objective(p,Z)
    assert np.linalg.norm(g)>1e-9 and info['derivative_triangular_solve_count']==6
    for pr,cf in zip(p.problems,cache.folds):
        base=local.fit_branch_local_ridge(**raw_slice(p,pr.audit['training_physical_ids']),
            support_labels=pr.train_labels,support_ids=pr.audit['training_physical_ids'],classes=p.classes,old_classes=p.old_classes)
        np.testing.assert_allclose(cf['score'],base.score(**raw_slice(p,pr.audit['held_physical_ids'])),rtol=3e-14,atol=2e-14)
    out=aj.evaluate_anchor_joint_objective(p,Z,forward_cache=cache)
    assert out[2]['inner_head_fit_count']==0 and out[2]['inner_objective_evaluation_count']==0
    with pytest.raises(ValueError,match='binding'):
        aj.evaluate_anchor_joint_objective(p,Z+1e-5,forward_cache=cache)
    with pytest.raises(ValueError,match='binding'):
        aj.evaluate_anchor_joint_objective(p,Z,anchor_U=p.anchor_U+1e-5)


@pytest.mark.parametrize('mode',['B','C_seq'])
def test_full_Z_gradient_finite_difference_including_prior_and_reference(mode):
    p=aj.prepare_anchor_joint_training(**fixture(seed=17)) if mode=='B' else sequence()[1]
    rng=np.random.default_rng(52);Z=rng.normal(size=(736,p.r));Z*=.02/np.linalg.norm(Z)
    d=rng.normal(size=Z.shape);d/=np.linalg.norm(d)
    _,g,info,_=aj.evaluate_anchor_joint_objective(p,Z);step=1e-5
    plus=aj.evaluate_anchor_joint_objective(p,Z+step*d,gradient=False)
    minus=aj.evaluate_anchor_joint_objective(p,Z-step*d,gradient=False)
    np.testing.assert_allclose(np.sum(g*d),(plus[0]-minus[0])/(2*step),rtol=3e-4,atol=2e-8)
    np.testing.assert_allclose((plus[2]['loss_proximal']-minus[2]['loss_proximal'])/(2*step),np.sum(Z*d),rtol=2e-9,atol=1e-12)
    assert info['derivative_triangular_solve_count']==2*info['ce_adjoint_solve_count']


def test_reference_subset_center_VJP_explicit_reference_perturbation():
    rng=np.random.default_rng(32);n=7;h=4;q=np.array([.5,0,.5,0,0,0,0])
    R=rng.normal(size=(n,n));R=.5*(R+R.T);Q=rng.normal(size=(h,n));gamma=.7
    bk=rng.normal(size=(n,n));bk=.5*(bk+bk.T);bl=rng.normal(size=(h,n))
    rr,rq=aj._center_vjp(bk,bl,q,gamma)
    # Perturb a reference row/column alone: detaching this part gives a wrong derivative.
    dR=np.zeros((n,n));dR[0,:]=rng.normal(size=n);dR[:,0]=dR[0,:];dQ=np.zeros((h,n));dQ[:,0]=rng.normal(size=h)
    def value(r,z):
        k,ref,rs,m,g=aj._center_train(r,q);l=aj._center_cross(z,q,ref,rs,m,g)
        return gamma*(np.sum(bk*k)+np.sum(bl*l))
    step=1e-6;fd=(value(R+step*dR,Q+step*dQ)-value(R-step*dR,Q-step*dQ))/(2*step)
    np.testing.assert_allclose(fd,np.sum(rr*dR)+np.sum(rq*dQ),rtol=2e-8,atol=2e-8)
    detached=gamma*(np.sum(bk*dR)+np.sum(bl*dQ))
    assert abs(fd-detached)>1e-3


def test_closed_residual_no_intercept_no_second_sample_centering_and_primal_oracle():
    args=fixture(k=3,c=3);p=aj.prepare_anchor_joint_training(**args)
    pr=p.problems[0];n=len(pr.train_labels);c=len(pr.classes)
    # Deliberately nonzero column/sample mean, while every score row sums to zero.
    M=np.tile(np.array([.3,-.1,-.2]),(n,1));pr=replace(pr,M_train=M)
    cf=aj._forward(pr,np.full((736,8),.0003));K=cf['K'];E=cf['Y']-M
    np.testing.assert_allclose(cf['alpha'],np.linalg.solve(K+np.eye(n),E),rtol=3e-14,atol=3e-14)
    assert np.linalg.norm(E.mean(axis=0))>.1
    assert np.linalg.norm(cf['alpha']-np.linalg.solve(K+np.eye(n),E-E.mean(axis=0)))>.1
    # Independent finite-feature RKHS primal identity, valid also for rank-deficient K.
    eig,V=np.linalg.eigh(K);Phi=V*np.sqrt(np.maximum(eig,0))[None,:]
    beta=np.linalg.solve(Phi.T@Phi+np.eye(n),Phi.T@E)
    np.testing.assert_allclose(Phi@beta,K@cf['alpha'],rtol=2e-12,atol=2e-13)
    np.testing.assert_allclose(cf['train_scores'],M+K@cf['alpha'],rtol=0,atol=0)
    assert cf['audit']['old_reference_residual_mean_norm']<1e-12
    assert cf['audit']['class_sum_max_abs']<1e-12


def test_unbalanced_low_level_labels_still_row_class_target():
    args=fixture();p=aj.prepare_anchor_joint_training(**args);pr=p.problems[0]
    y=np.zeros(len(pr.train_labels),dtype=int);y[-1]=1
    cf=aj._forward(replace(pr,train_labels=y),np.zeros((736,8)))
    np.testing.assert_array_equal(cf['Y'],np.eye(3)[y]-1/3)
    assert np.linalg.norm(cf['E'].mean(0))>.1
    np.testing.assert_allclose((cf['K']+np.eye(len(y)))@cf['alpha'],cf['E'],atol=2e-13)


def test_cross_fold_class_CE_aggregation_not_fold_RMS_average():
    p=aj.prepare_anchor_joint_training(**fixture(k=5,c=3));Z=np.full((736,p.r),.0001)
    _,_,info,cache=aj.evaluate_anchor_joint_objective(p,Z,gradient=False)
    scores=np.concatenate([x['score'] for x in cache.folds]);labels=np.concatenate([x['problem'].held_labels for x in cache.folds])
    ce,_=aj._ce(scores,labels);means=np.array([ce[labels==j].mean() for j in range(3)])
    np.testing.assert_allclose(info['class_ce_means'],means,rtol=3e-15)
    np.testing.assert_allclose(info['RMSCE'],np.sqrt(np.mean(means*means)),rtol=3e-15)
    assert info['class_ce_counts']==[5,5,5]


def test_actual_B_prior_fold_binding_fixed_nuisance_and_anchor_null_component():
    b,p,oa,args=sequence();Z=np.zeros((736,p.r));_,_,_,cache=aj.evaluate_anchor_joint_objective(p,Z,gradient=False)
    np.testing.assert_array_equal(aj.reconstruct_U(Z,b.U,p.W),b.U)
    for pr,cf,prior_meta in zip(p.problems,cache.folds,p.audit['prior_folds']):
        assert set(pr.audit['old_reference_physical_ids'])==set(prior_meta['training_physical_ids'])
        assert not set(prior_meta['training_physical_ids'])&set(prior_meta['held_physical_ids'])
        assert pr.audit['prior_ref']==prior_meta['head_state_ref']
        assert pr.tau==prior_meta['bandwidth_tau'] and pr.gamma==prior_meta['trace_scale']
        old=[p.classes.index(name) for name in b.classes]
        assert np.all(pr.M_train[:,[j for j in range(3) if j not in old]]==0)
    assert p.full_problem.tau==b.problem.tau and p.full_problem.gamma==b.problem.gamma
    oldix=np.flatnonzero([p.classes[int(v)] in p.old_classes for v in p.full_problem.train_labels])
    final=aj._forward(p.full_problem,b.U)
    np.testing.assert_allclose(final['K'][np.ix_(oldix,oldix)],b.final_cache['K'],rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(p.full_problem.M_train[:,[p.classes.index(x) for x in b.classes]],b.score(**p.raw),rtol=0,atol=0)
    H=np.zeros((12,8));H[:,0]=np.arange(12)+1;W,_,ci=aj.latent_coordinates(H)
    anchor=np.zeros((736,8));anchor[3,7]=13;U=aj.reconstruct_U(np.ones((736,ci['latent_rank']))*.001,anchor,W)
    assert U[3,7]==13
    with pytest.raises(ValueError,match='crosses'):
        aj.prepare_anchor_joint_training(**args,inherited=b,context={'row_id':'two'})


def test_N0_reuses_actual_B_without_preparation_or_C_head():
    b,_,oa,_=sequence(k=1);p=aj.prepare_anchor_joint_training(**oa,inherited=b)
    assert p.audit['ajlr_preparation_count']==p.audit['latent_svd_count']==p.audit['prior_head_fit_count']==0
    assert aj.fit_anchor_joint_local_ridge(p,mode='C_seq') is b


def test_K1_executes_complete_B_and_nonzero_prior_C_heads_without_adapter():
    b,p,_,_=sequence(k=1);s=aj.fit_anchor_joint_local_ridge(p,mode='C_seq')
    for state in (b,s):
        a=state.audit_dict();assert a['optimizer_steps']==a['inner_head_fit_count']==0
        assert a['final_head_fit_count']==1 and a['trainable_parameter_count']==a['trained_parameter_count']==0
    np.testing.assert_array_equal(s.U,b.U)
    assert s.audit['final_fit']['status']=='CLOSED_FORM_SOLVED'
    assert np.linalg.norm(s.final_cache['alpha'])>0


def test_rank0_preserves_anchor_but_does_not_skip_closed_head():
    p=aj.prepare_anchor_joint_training(**fixture());p=replace(p,H=np.zeros_like(p.H),W=np.empty((8,0)),r=0,
        audit=dict(p.audit,no_information=True,no_information_reason='ZERO_DICTIONARY_RANK'))
    s=aj.fit_anchor_joint_local_ridge(p)
    assert s.audit['final_head_fit_count']==s.audit['final_factorization_count']==1
    assert s.audit['trained_parameter_count']==0 and np.linalg.norm(s.final_cache['alpha'])>0


@pytest.mark.parametrize('single_class',[False,True])
def test_missing_old_kernel_information_canonical_alpha_and_tie(single_class):
    args=fixture(c=1 if single_class else 3)
    if not single_class:
        for key in aj._NAMES:args[key][:]=args[key][0]
    p=aj.prepare_anchor_joint_training(**args);s=aj.fit_anchor_joint_local_ridge(p)
    assert s.problem.gamma is None and s.audit['final_factorization_count']==0
    np.testing.assert_array_equal(s.final_cache['alpha'],s.final_cache['Y']-s.problem.M_train)
    np.testing.assert_array_equal(s.score(**p.raw),np.zeros((len(p.ids),len(p.classes))))
    assert set(s.predict(**p.raw))=={min(p.classes)}


def test_zero_bandwidth_equivalence_nonzero_C_head_and_no_adapter_derivative():
    args=fixture(k=3,c=2)
    for key in aj._NAMES:args[key][3:]=args[key][:3]
    p=aj.prepare_anchor_joint_training(**args);assert p.full_problem.tau==0 and p.full_problem.s0>0
    cf=aj._forward(p.full_problem,np.full((736,8),.01))
    np.testing.assert_array_equal(cf['radial'],(p.full_problem.d0==0).astype(float))
    prog={'derivative_triangular_solve_count':0,'ce_adjoint_solve_count':0}
    np.testing.assert_array_equal(aj._backward(cf,np.ones_like(cf['score']),prog),np.zeros((736,8)))
    # A new registered class shares one old feature, but also has distinct records.
    allargs=fixture(k=3,c=3)
    for key in aj._NAMES:allargs[key][:6]=args[key];allargs[key][6]=args[key][0]
    old=('c0','c1');b=aj.fit_anchor_joint_local_ridge(p);allargs['old_classes']=old
    cp=aj.prepare_anchor_joint_training(**allargs,inherited=b);cs=aj.fit_anchor_joint_local_ridge(cp,mode='C_seq')
    assert cs.problem.tau==0 and cs.audit['optimizer_steps']==0
    assert np.linalg.norm(cs.final_cache['train_scores'])>1e-5


def test_PSD_SPD_reference_kernel_and_single_sample_batch_equivalence():
    b,p,_,_=sequence();s=aj.fit_anchor_joint_local_ridge(p,mode='C_seq');K=s.final_cache['K']
    assert np.linalg.eigvalsh(K).min()>-1e-12
    assert np.linalg.eigvalsh(K+np.eye(len(K))).min()>1-1e-12
    score=s.score(**p.raw)
    for size in (1,2,4):
        chunks=[s.score(**{k:v[i:i+size] for k,v in p.raw.items()}) for i in range(0,len(p.ids),size)]
        np.testing.assert_array_equal(np.vstack(chunks),score)
    out,a=s.score_with_audit(**p.raw);np.testing.assert_array_equal(out,score)
    assert a['score_physical_count']==len(p.ids) and a['prior']['kernel_pair_count']==len(p.ids)*len(b.ids)
    assert a['raw_distance_pair_count']==a['residual']['raw_distance_pair_count']+a['prior']['raw_distance_pair_count']


def test_row_and_class_permutation_and_forbidden_extra_inputs():
    args=fixture();p=aj.prepare_anchor_joint_training(**args);order=np.random.default_rng(7).permutation(len(p.ids))
    per={k:args[k][order] for k in aj._NAMES};per.update(support_labels=args['support_labels'][order],
        support_ids=tuple(args['support_ids'][i] for i in order),classes=('zeta','alpha','mu'),old_classes=('zeta','alpha','mu'))
    p2=aj.prepare_anchor_joint_training(**per);Z=np.full((736,p.r),.0005)
    a=aj.evaluate_anchor_joint_objective(p,Z);b=aj.evaluate_anchor_joint_objective(p2,Z)
    np.testing.assert_allclose(a[0],b[0],rtol=4e-13);np.testing.assert_allclose(a[1],b[1],rtol=3e-10,atol=3e-13)
    with pytest.raises(TypeError):aj.prepare_anchor_joint_training(**args,query_labels=np.zeros(2))
    with pytest.raises(ValueError):aj.fit_anchor_joint_local_ridge(p,mode='C_reset_init')


def test_registration_full_class_argmax_can_override_old_prior():
    # Zero-sum new residual columns can win while old columns stay unchanged.
    old=np.array([[-.2,.2]]);registered=np.column_stack((old,np.array([.4]),np.array([-.4])))
    assert old.argmax(axis=1)[0]==1 and registered.argmax(axis=1)[0]==2
    assert registered.sum()==0
    # Residual registration may alter old scores; no pointwise preservation gate.
    b,p,_,_=sequence();cf=aj._forward(p.full_problem,b.U)
    assert np.linalg.norm(cf['train_scores']-p.full_problem.M_train)>1e-6


def test_Armijo_first_accepted_trial_and_real_workload_and_lossless_records():
    p=aj.prepare_anchor_joint_training(**fixture());events=[];s=aj.fit_anchor_joint_local_ridge(p,log_callback=events.append);a=s.audit_dict()
    assert a['optimizer_steps']==a['accepted_trial_count']<=4
    assert a['trial_count']==a['trial_attempt_count']==a['accepted_trial_count']+a['rejected_trial_count']<=48
    assert a['inner_objective_evaluation_count']==1+a['trial_count']
    assert a['inner_head_fit_count']==len(p.problems)*a['inner_objective_evaluation_count']
    assert a['ajlr_forward_evaluation_count']==a['inner_head_fit_count']+a['final_head_fit_count']
    assert a['head_triangular_solve_count']==2*(a['inner_factorization_count']+a['final_factorization_count'])
    assert a['derivative_triangular_solve_count']==2*a['ce_adjoint_solve_count']<=24
    assert a['coordinate_norm']<=.5+1e-12
    for tr in a['trials']:assert tr['accepted']==(tr['armijo_pass'] and tr['objective_nonincrease_pass'])
    records=s.state_records();np.testing.assert_array_equal(records['final']['U'],s.U)
    assert {'K','L','alpha','E','M_train','q','raw_train','chol'}<=records['final'].keys()
    assert 'final_objective_state_ref' in a
    assert a['persistent_state_bytes']==aj._unique_bytes(s._resident_values())
    assert a['deployment_numeric_state_bytes']==aj._unique_bytes(s._deployment_values())
    assert all(x['event'].startswith('AJLR_') for x in events)
    json.dumps(a,allow_nan=False);json.dumps(events,allow_nan=False)


def test_twelve_rejections_preserve_last_accepted_cache(monkeypatch):
    p=aj.prepare_anchor_joint_training(**fixture());real=aj.evaluate_anchor_joint_objective;calls=[0]
    def wrapped(prepared,Z,anchor_U=None,*,gradient=True,forward_cache=None):
        loss,g,info,cache=real(prepared,Z,anchor_U,gradient=gradient,forward_cache=forward_cache)
        if forward_cache is None:
            calls[0]+=1
            if calls[0]>1:return loss+100,g,dict(info,loss_total=loss+100),cache
        return loss,g,info,cache
    monkeypatch.setattr(aj,'evaluate_anchor_joint_objective',wrapped)
    s=aj.fit_anchor_joint_local_ridge(p);a=s.audit_dict()
    assert a['stop_reason']=='TRIAL_BUDGET_EXHAUSTED' and a['rejected_trial_count']==12
    assert a['optimizer_steps']==0 and np.all(s.U==0)
    assert a['final_objective']['loss_total']==a['initial_objective']['loss_total']
    assert len([k for k in s.state_records() if k.startswith('trial_') and '_head_' not in k])==12


def test_callback_numeric_only_no_retained_vectors_and_prior_state_complete():
    b,p,_,_=sequence();saved={}
    def writer(key,arrays):
        assert key not in saved
        assert all(v.dtype.kind in 'fibu' and np.isfinite(v).all() for v in arrays.values())
        saved[key]={k:np.array(v,copy=True) for k,v in arrays.items()}
        return {'path':'synthetic/'+key+'.npz'}
    s=aj.fit_anchor_joint_local_ridge(p,mode='C_seq',state_callback=writer)
    assert not s.state_records() and s.audit['retained_vector_record_bytes']==0
    final=saved['final'];assert {'prior_B_U','prior_B_alpha','prior_B_q','prior_old_class_indices'}<=final.keys()
    np.testing.assert_array_equal(final['prior_B_U'],b.U)
    np.testing.assert_array_equal(final['prior_B_alpha'],b.final_cache['alpha'])
    np.testing.assert_array_equal(final['prior_old_class_indices'],[s.classes.index(x) for x in b.classes])


def test_zero_blocks_and_near_duplicate_positive_distance_remain_exact_or_fail():
    args=fixture();args['pa_local'][:]=0;p=aj.prepare_anchor_joint_training(**args)
    b,a,_=aj.fcr._adapt(aj.fcr._context(p.background,p.auxiliary),np.full((736,8),.001))
    np.testing.assert_array_equal(a[:,320:],np.zeros((len(a),160)))
    b0=np.zeros((2,256));a0=np.zeros((2,480));b0[:,0]=1;a0[:,0]=1;b0[1,1]=1e-10
    d=local._distances(b0,a0);assert d[0,1]>0
    assert np.isfinite(d).all()
