"""Independent synthetic checks; no target records, source data or weights."""
import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_prototype_transport_local_ridge as pt
from cvsrffi import d92_branch_local_ridge as local


def fixture(k=3,c=3,seed=81):
    rng=np.random.default_rng(seed);labels=np.repeat(np.arange(c),k)
    raw={key:rng.normal(size=(c*k,d))+.35*rng.normal(size=(c,d))[labels]
         for key,d in zip(pt._NAMES,(160,96,160,160,160))}
    ids=tuple(f'physical-{cls}-{i:02}' for cls in range(c) for i in range(k))
    classes=tuple(f'c{cls}' for cls in range(c))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=classes,old_classes=classes)


def prepare(**kw):return pt.prepare_prototype_transport_training(**fixture(**kw))


def theta(seed=7):
    rng=np.random.default_rng(seed);x=rng.normal(size=10)*.08;x[5:]-=x[5:].mean();return x


def raw_slice(p,ids):
    ix=[p.ids.index(pid) for pid in ids]
    return {key:np.asarray(p.raw[key])[ix] for key in pt._NAMES}


def test_frozen_config_exact():
    path=Path(__file__).parents[1]/'configs/d92_prototype_transport_frozen_20260930.json'
    assert json.loads(path.read_text(encoding='utf-8'))=={'algorithm':pt.FROZEN_CONFIG}
    assert pt.FROZEN_CONFIG['trainable_parameter_count']==10
    assert pt.FROZEN_CONFIG['effective_parameter_count']==9


def test_zero_beta_original_forward_live_beta_zero_eta_gradient():
    p=prepare();problem=p.problems[0];r=pt._forward(problem,np.zeros(10))
    o=problem.original
    baseline=local.fit_branch_local_ridge(**raw_slice(p,o.audit['training_physical_ids']),
        support_labels=np.asarray(o.train_labels,dtype=int),support_ids=o.audit['training_physical_ids'],
        classes=p.classes,old_classes=p.old_classes)
    np.testing.assert_array_equal(r['score'],baseline.score(**raw_slice(p,o.audit['held_physical_ids'])))
    loss,g,audit,_=pt.evaluate_transport_objective(p,np.zeros(10),np.zeros(10))
    assert np.linalg.norm(g[:5])>1e-9
    np.testing.assert_array_equal(g[5:],np.zeros(5))
    assert audit['derivative_triangular_solve_count']==6


@pytest.mark.parametrize('seed',[5,17,73])
def test_nonzero_beta_eta_complete_objective_finite_difference(seed):
    p=prepare(seed=seed);x=theta(seed+1);anchor=theta(seed+2)*.1
    loss,g,audit,_=pt.evaluate_transport_objective(p,x,anchor)
    d=theta(seed+3);h=1e-4
    lp=pt.evaluate_transport_objective(p,x+h*d,anchor,gradient=False)[0]
    lm=pt.evaluate_transport_objective(p,x-h*d,anchor,gradient=False)[0]
    np.testing.assert_allclose(g@d,(lp-lm)/(2*h),rtol=3e-5,atol=3e-9)
    assert np.linalg.norm(g[5:])>1e-10
    assert audit['inner_head_fit_count']==3 and audit['inner_factorization_count']==3


def test_rotation_vjp_independent_fd_and_norms():
    p=prepare();f=p.full_problem;x=theta();rng=np.random.default_rng(1)
    gb=rng.normal(size=f.original.train_b.shape);ga=rng.normal(size=f.original.train_a.shape)
    mb,ma,cache=pt._adapt(f.train_context,f.prototypes,x,derivative_cache=True)
    before=np.concatenate((f.original.train_b,f.original.train_a),axis=1);after=np.concatenate((mb,ma),axis=1)
    for sl in pt._SLICES:np.testing.assert_allclose(np.linalg.norm(before[:,sl],axis=1),np.linalg.norm(after[:,sl],axis=1),rtol=3e-14,atol=0)
    g=pt._rotation_vjp(f.train_context,f.prototypes,x,gb,ga,cache);d=theta(52)
    def loss(t):
        b,a,_=pt._adapt(f.train_context,f.prototypes,x+t*d)
        return np.sum(gb*b)+np.sum(ga*a)
    h=1e-5
    np.testing.assert_allclose(g@d,(loss(h)-loss(-h))/(2*h),rtol=2e-6,atol=2e-9)


def test_q_zero_and_zero_blocks_continuous_vjp():
    # All prototype directions parallel to input: tangent is exactly zero.
    b=np.zeros((2,256));a=np.zeros((2,480));b[:,0]=[1.,2.]
    table=pt._make_prototypes(b,a,np.array([0,1]),('a','b'),('i','j'))
    ctx=pt._map_context(b,a,table);x=theta()
    mb,ma,cache=pt._adapt(ctx,table,x,derivative_cache=True)
    np.testing.assert_array_equal(mb,b);np.testing.assert_array_equal(ma,a)
    assert np.all(cache['q']==0)
    g=pt._rotation_vjp(ctx,table,x,np.ones_like(b),np.ones_like(a),cache)
    np.testing.assert_array_equal(g,np.zeros(10))


def test_prototype_ties_include_all_second_and_renaming_equivariance():
    b=np.zeros((3,256));a=np.zeros((3,480));b[:,0]=[-1.,0.,1.]
    table=pt._make_prototypes(b,a,np.arange(3),('a','b','c'),('i','j','k'))
    qb=np.zeros((1,256));qa=np.zeros((1,480));ctx=pt._map_context(qb,qa,table)
    assert ctx['neighbors'].tolist()==[[True,True,True]]
    perm=[2,0,1];tab2=replace(table,p=table.p[perm],m=table.m[perm],classes=('c','a','b'))
    r1=pt._adapt(ctx,table,theta())
    r2=pt._adapt(pt._map_context(qb,qa,tab2),tab2,theta())
    for one,two in zip(r1[:2],r2[:2]):np.testing.assert_array_equal(one,two)


def test_unequal_folds_class_rms_after_all_fold_sums():
    p=prepare(k=4);x=theta();loss,g,audit,_=pt.evaluate_transport_objective(p,x,np.zeros(10))
    sums=np.sum([f['class_margin_loss_sums'] for f in audit['inner_folds']],axis=0)
    counts=np.sum([f['class_physical_counts'] for f in audit['inner_folds']],axis=0)
    np.testing.assert_array_equal(counts,[4,4,4])
    expected=np.sqrt(np.mean((sums/counts)**2))+.5*np.sum(x*x)/len(p.ids)
    np.testing.assert_allclose(loss,expected,rtol=2e-15)


def test_inner_held_features_do_not_build_geometry_or_prototypes():
    args=fixture();p=pt.prepare_prototype_transport_training(**args);f=p.problems[0]
    held=set(f.original.audit['held_physical_ids'])
    for key in pt._NAMES:
        args[key]=args[key].copy()
        for i,pid in enumerate(args['support_ids']):
            if pid in held:args[key][i]*=np.linspace(.7,1.3,args[key].shape[1])
    f2=pt.prepare_prototype_transport_training(**args).problems[0]
    np.testing.assert_array_equal(f.prototypes.p,f2.prototypes.p);np.testing.assert_array_equal(f.prototypes.m,f2.prototypes.m)
    assert f.prototypes.nu==f2.prototypes.nu
    np.testing.assert_array_equal(f.original.d0,f2.original.d0)
    np.testing.assert_array_equal(pt._forward(f,theta())['alpha'],pt._forward(f2,theta())['alpha'])


def test_inner_labels_only_affect_supervised_objective_not_geometry():
    p=prepare();f=p.problems[0]
    changed=replace(f,original=replace(f.original,held_labels=(f.original.held_labels+1)%3))
    r1=pt._forward(f,theta());r2=pt._forward(changed,theta())
    np.testing.assert_array_equal(r1['score'],r2['score']);np.testing.assert_array_equal(r1['alpha'],r2['alpha'])
    p2=replace(p,problems=(changed,)+p.problems[1:])
    assert not np.allclose(pt.evaluate_transport_objective(p,theta(),np.zeros(10))[1],
                           pt.evaluate_transport_objective(p2,theta(),np.zeros(10))[1])


def test_rows_and_class_permutation_equivariance():
    args=fixture(k=2);p=pt.prepare_prototype_transport_training(**args);x=theta();r=pt.evaluate_transport_objective(p,x,np.zeros(10))
    perm=np.random.default_rng(8).permutation(len(args['support_ids']))
    changed={key:args[key][perm] for key in pt._NAMES}
    changed.update(support_labels=args['support_labels'][perm],support_ids=tuple(args['support_ids'][i] for i in perm),
        classes=('z','x','y'),old_classes=('z','x','y'))
    p2=pt.prepare_prototype_transport_training(**changed);r2=pt.evaluate_transport_objective(p2,x,np.zeros(10))
    np.testing.assert_allclose(r[0],r2[0],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(r[1],r2[1],rtol=1e-9,atol=1e-12)


def test_noninjective_adapted_map_allowed_joint_lower_bound():
    b0=np.zeros((2,256));a0=np.zeros((2,480));b0[1,0]=1e-7
    b=np.zeros_like(b0);a=np.zeros_like(a0)
    d0=local._distances(b0,a0)
    joint=pt._joint_distances(b,a,b,a,b0,a0,b0,a0,symmetric=True)
    np.testing.assert_array_equal(joint,.5*d0);assert joint[0,1]>0
    # A deterministic map cannot split genuinely identical originals.
    with pytest.raises(FloatingPointError,match='ORIGINAL_EQUAL'):
        pt._joint_distances(b0,a0,b0,a0,b,a,b,a,symmetric=True)


def test_near_repeat_stable_dual_distance_decimal_oracle():
    b=np.array([[1.1,-1.7]]);a=np.array([[.6,-.9,1.3]])
    tb=b+np.array([[1e-13,-2e-13]]);ta=a+np.array([[-3e-13,2e-13,1e-13]])
    # Independent explicit original and adapted interaction sums, binary64 input
    # converted exactly; no float Gram subtraction in this reference.
    mb=b*np.array([[.8,1.1]]);ma=a*np.array([[.9,1.2,.7]])
    mtb=tb*np.array([[.8,1.1]]);mta=ta*np.array([[.9,1.2,.7]])
    with localcontext() as ctx:
        ctx.prec=100
        def explicit(bb,aa,tt,uu):
            B,A,T,U=[[Decimal.from_float(float(x)) for x in arr[0]] for arr in (bb,aa,tt,uu)]
            phi=B+A+[x*y for x in B for y in A];psi=T+U+[x*y for x in T for y in U]
            return sum((x-y)**2 for x,y in zip(phi,psi))
        expected=float((explicit(b,a,tb,ta)+explicit(mb,ma,mtb,mta))/2)
    actual=pt._joint_distances(mb,ma,mtb,mta,b,a,tb,ta)[0,0]
    assert 0<expected<1e-24
    np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=0)


def test_zero_tau_original_equivalence_and_zero_derivative():
    args=fixture(c=2)
    for key in pt._NAMES:args[key][3:]=args[key][:3]
    p=pt.prepare_prototype_transport_training(**args);f=p.problems[0];assert f.original.tau0==0
    r0=pt._forward(f,np.zeros(10));r1=pt._forward(f,theta())
    np.testing.assert_array_equal(r0['score'],r1['score'])
    progress={'derivative_triangular_solve_count':0}
    np.testing.assert_array_equal(pt._backward(r1,np.ones_like(r1['score']),progress),np.zeros(10))
    assert progress['derivative_triangular_solve_count']==0


def test_projection_kkt_and_fixed_geometric_mean():
    x=np.array([3,-2,1,-4,.1,4,-3,.2,1,-.5]);out=pt.project_transport(x);pt._parameter(out)
    h=pt.FROZEN_CONFIG['eta_bound'];free=np.abs(out[5:])<h-1e-10
    assert abs(out[5:].sum())<=128*np.finfo(float).eps*5*h
    np.testing.assert_allclose((x[5:]-out[5:])[free],(x[5:]-out[5:])[free][0],atol=1e-13)
    np.testing.assert_allclose(np.prod(np.exp(out[5:])),1.,rtol=1e-13)


@pytest.mark.parametrize('k,c',[(1,3),(3,1)])
def test_no_information_bitwise_baseline_reuse(k,c):
    args=fixture(k=k,c=c);p=pt.prepare_prototype_transport_training(**args);base=local.fit_branch_local_ridge(**args)
    state=pt.fit_prototype_transport_local_ridge(p,baseline_state=base);audit=state.audit_dict()
    assert audit['optimizer_steps']==audit['inner_objective_evaluation_count']==audit['final_head_fit_count']==0
    assert audit['no_information'] and audit['identity_forward']
    np.testing.assert_array_equal(state.score(**{k:args[k] for k in pt._NAMES}),base.score(**{k:args[k] for k in pt._NAMES}))


def test_cache_reuse_identical_gradient_zero_additional_heads():
    p=prepare();x=theta();a=np.zeros(10)
    loss,g,au,cache=pt.evaluate_transport_objective(p,x,a)
    l2,g2,au2,cache2=pt.evaluate_transport_objective(p,x,a,forward_cache=cache)
    np.testing.assert_array_equal(g,g2);assert loss==l2
    assert au2['inner_head_fit_count']==au2['inner_factorization_count']==au2['inner_objective_evaluation_count']==0
    assert au2['derivative_triangular_solve_count']==6
    assert all(f['transport_forward_evaluation_count']==0 for f in au2['inner_folds'])
    with pytest.raises(ValueError,match='cache'):
        pt.evaluate_transport_objective(p,x*.5,a,forward_cache=cache)


def test_optimizer_actual_cost_bounds_last_accepted_and_batch_independence():
    p=prepare();events=[];state=pt.fit_prototype_transport_local_ridge(p,log_callback=events.append);a=state.audit_dict()
    assert a['optimizer_steps']==a['accepted_trial_count']==len(a['steps'])<=4
    assert a['trial_count']==a['accepted_trial_count']+a['rejected_trial_count']<=12
    assert a['inner_objective_evaluation_count']==1+a['trial_count']
    assert a['inner_head_fit_count']==3*(1+a['trial_count'])<=39
    assert a['derivative_triangular_solve_count']==6*a['backward_evaluation_count']<=24
    assert a['persistent_state_bytes']==a['head_state_bytes']+a['adapter_state_bytes']+a['prototype_state_bytes']+a['lineage_state_bytes']
    if a['steps']:np.testing.assert_array_equal(state.theta,a['steps'][-1]['u_post'])
    for t in a['trials']:assert t['accepted']==(t['loss_after']<=t['armijo_rhs']+t['armijo_tolerance'])
    raw={k:np.asarray(p.raw[k]) for k in pt._NAMES};batch=state.score(**raw)
    singles=np.vstack([state.score(**{k:v[i:i+1] for k,v in raw.items()}) for i in range(len(p.ids))])
    np.testing.assert_array_equal(batch,singles)
    with pytest.raises(ValueError):state.theta[0]=5
    with pytest.raises(ValueError):state.prototypes.p[0,0]=5


def test_rejected_trials_do_not_replace_accepted_cache(monkeypatch):
    p=prepare();original=pt.evaluate_transport_objective;initial_cache=[];seen=[]
    def fake(prepared,x,anchor,*,gradient=True,forward_cache=None):
        result=original(prepared,x,anchor,gradient=gradient,forward_cache=forward_cache)
        loss,g,audit,cache=result
        if forward_cache is None and not initial_cache:
            initial_cache.append(cache);return result
        if forward_cache is None:
            audit=dict(audit,loss_total=loss+100.);return loss+100.,g,audit,cache
        seen.append(forward_cache)
        return result
    monkeypatch.setattr(pt,'evaluate_transport_objective',fake)
    state=pt.fit_prototype_transport_local_ridge(p);a=state.audit_dict()
    assert a['stop_reason']=='ARMIJO_BUDGET_EXHAUSTED' and a['rejected_trial_count']==3
    assert a['optimizer_steps']==0 and np.all(state.theta==0)
    assert all(cache is initial_cache[0] or np.array_equal(cache.theta,initial_cache[0].theta) for cache in seen)
    assert a['final_objective']['loss_total']==a['initial_objective']['loss_total']


def test_B_C_lineage_prototypes_and_N0():
    args=fixture(c=3,k=2);old=('c0','c1');mask=args['support_labels']<2
    bargs={k:args[k][mask] for k in pt._NAMES}
    bargs.update(support_labels=args['support_labels'][mask],support_ids=tuple(pid for i,pid in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)
    b=pt.fit_prototype_transport_local_ridge(pt.prepare_prototype_transport_training(**bargs,context={'row_id':'r'}))
    args['old_classes']=old;p=pt.prepare_prototype_transport_training(**args,inherited=b,context={'row_id':'r'})
    np.testing.assert_array_equal(p.full_problem.prototypes.p[:2],b.prototypes.p)
    np.testing.assert_array_equal(p.full_problem.prototypes.m[:2],b.prototypes.m)
    seq=pt.fit_prototype_transport_local_ridge(p,mode='C_seq');reset=pt.fit_prototype_transport_local_ridge(p,mode='C_reset')
    np.testing.assert_array_equal(seq.audit_dict()['anchor'],b.theta);np.testing.assert_array_equal(reset.audit_dict()['anchor'],np.zeros(10))
    assert p.audit_dict()['full_prototypes']['old_prototypes_bitwise_inherited']
    n0=pt.prepare_prototype_transport_training(**bargs,inherited=b)
    assert pt.fit_prototype_transport_local_ridge(n0,mode='C_seq') is b
    with pytest.raises(ValueError,match='crosses'):
        pt.prepare_prototype_transport_training(**args,inherited=b,context={'row_id':'different'})
    args['z_id']=args['z_id'].copy();args['z_id'][0]*=2
    with pytest.raises(ValueError,match='raw features'):pt.prepare_prototype_transport_training(**args,inherited=b)


def test_baseline_wrong_labels_rejected():
    args=fixture(k=1);p=pt.prepare_prototype_transport_training(**args)
    base=local.fit_branch_local_ridge(**dict(args,support_labels=np.roll(args['support_labels'],1)))
    with pytest.raises(ValueError,match='binding|geometry'):pt.fit_prototype_transport_local_ridge(p,baseline_state=base)


def test_partial_forward_failure_preserves_actual_work(monkeypatch):
    p=prepare();original=pt._forward;calls=[0]
    def fail(problem,x,progress=None):
        calls[0]+=1
        if calls[0]==2:
            progress.update(head_fit_count=1,factorization_count=1,derivative_triangular_solve_count=0)
            raise FloatingPointError('INJECTED_FOLD2')
        return original(problem,x,progress)
    monkeypatch.setattr(pt,'_forward',fail)
    with pytest.raises(pt.NumericalFailure) as caught:pt.fit_prototype_transport_local_ridge(p)
    a=caught.value.audit_dict();assert a['inner_objective_evaluation_count']==1 and a['inner_head_fit_count']==2
    assert a['inner_factorization_count']==2 and a['derivative_triangular_solve_count']==0
    json.dumps(a,allow_nan=False)


def test_tie_generalized_derivative_not_single_direction():
    score=np.zeros((1,3));label=np.array([0]);loss,g,_=pt.ch._margin_loss(score,label)
    np.testing.assert_array_equal(g,[[-1,.5,.5]])
    d=np.array([[0.,1.,-1.]]);assert np.sum(g*d)==0
    assert (pt.ch._margin_loss(score+1e-7*d,label)[0]-loss)/1e-7>.999


def test_same_prototype_nonzero_map_bitwise_all_batch_partitions():
    p=prepare(k=4,c=5);f=p.full_problem;table=f.prototypes;x=theta(37)
    b,a=f.original.train_b,f.original.train_a
    whole=pt._adapt(f.train_context,table,x,derivative_cache=True)
    for size in (1,2,3,7,len(b)):
        chunks=[];mus=[]
        for start in range(0,len(b),size):
            ctx=pt._map_context(b[start:start+size],a[start:start+size],table)
            mb,ma,cache=pt._adapt(ctx,table,x,derivative_cache=True)
            chunks.append(np.concatenate((mb,ma),axis=1));mus.append(cache['mu'])
        np.testing.assert_array_equal(np.vstack(mus),whole[2]['mu'])
        np.testing.assert_array_equal(np.vstack(chunks),np.concatenate(whole[:2],axis=1))
    # Repetition and an unrelated query must not alter a support's mapping.
    perm=np.array([4,4,0,12,3,19,4]);mixed=pt._adapt(pt._map_context(b[perm],a[perm],table),table,x)
    np.testing.assert_array_equal(mixed[0],whole[0][perm]);np.testing.assert_array_equal(mixed[1],whole[1][perm])


def test_failed_trial_preserves_attempt_context_and_cost(monkeypatch):
    p=prepare();original=pt.evaluate_transport_objective;calls=[0]
    def fail(prepared,x,anchor,*,gradient=True,forward_cache=None):
        if forward_cache is None:
            calls[0]+=1
            if calls[0]==2:
                raise pt.NumericalFailure('FAILED_TRIAL',dict(inner_objective_evaluation_count=1,
                    inner_head_fit_count=1,inner_factorization_count=1,derivative_triangular_solve_count=0,
                    backward_evaluation_count=0,transport_forward_evaluation_count=1))
        return original(prepared,x,anchor,gradient=gradient,forward_cache=forward_cache)
    monkeypatch.setattr(pt,'evaluate_transport_objective',fail)
    with pytest.raises(pt.NumericalFailure) as caught:pt.fit_prototype_transport_local_ridge(p)
    a=caught.value.audit_dict()
    assert a['trial_attempt_count']==1 and a['trial_count']==0
    assert a['current_trial']['iteration']==1 and a['current_trial']['trial']==1
    assert a['current_trial']['step_size']==.125
    assert a['inner_head_fit_count']==4 and a['inner_objective_evaluation_count']==2
    assert np.all(np.asarray(a['u'])==0) and a['optimizer_steps']==0


def test_public_audit_tree_strict_json_native_including_trial_flags():
    # The full-parent writer consumes audit_dict, not the already-normalized
    # callback. Exercise that actual boundary, including NumPy context scalars.
    p=pt.prepare_prototype_transport_training(**fixture(k=3,c=3),
        context={'synthetic_numpy_integer':np.int64(7),'synthetic_numpy_flag':np.bool_(True)})
    state=pt.fit_prototype_transport_local_ridge(p)
    def native(value,path):
        assert not isinstance(value,np.generic),f'NumPy scalar at {path}: {type(value).__name__}'
        if isinstance(value,dict):
            for key,item in value.items():native(item,f'{path}.{key}')
        elif isinstance(value,(tuple,list)):
            for i,item in enumerate(value):native(item,f'{path}[{i}]')
    for name,audit in [('preparation',p.audit_dict()),('state',state.audit_dict())]:
        native(audit,name)
        decoded=json.loads(json.dumps(audit,allow_nan=False))
        assert decoded==audit
    audit=state.audit_dict()
    assert audit['trials'], 'This regression must exercise an evaluated Armijo trial'
    for i,trial in enumerate(audit['trials']):
        assert type(trial['accepted']) is bool,f'trials[{i}].accepted'
        assert type(trial['armijo_tolerance']) is float,f'trials[{i}].armijo_tolerance'
    # The producer itself also stores native types, before audit normalization.
    assert all(type(t['accepted']) is bool for t in state.audit['trials'])
    assert type(pt._armijo_tolerance(.2,.3,.1)) is float
