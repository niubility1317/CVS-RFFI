import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_margin_constrained_residual8_local_ridge as mc
from cvsrffi import d92_branch_local_ridge as local


def fixture(k=3,c=3,seed=81):
    rng=np.random.default_rng(seed);labels=np.repeat(np.arange(c),k)
    raw={key:rng.normal(size=(c*k,d))+.35*rng.normal(size=(c,d))[labels] for key,d in zip(mc._NAMES,(160,96,160,160,160))}
    names=tuple(f'c{i}' for i in range(c));ids=tuple(f'physical-{i}-{j:02}' for i in range(c) for j in range(k))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=names,old_classes=names)


def prepare(**kw):return mc.prepare_mc_residual8_training(**fixture(**kw))


def parameters(seed=1):
    rng=np.random.default_rng(seed);u,v=mc.initial_parameters()
    du=rng.normal(size=u.shape);dv=rng.normal(size=v.shape)
    return u+.15*du/np.linalg.norm(du),v+.15*dv/np.linalg.norm(dv)


def directions(seed=3):
    rng=np.random.default_rng(seed);du=rng.normal(size=(736,8));dv=rng.normal(size=(8,736))
    return du/np.linalg.norm(du),dv/np.linalg.norm(dv)


def raw_slice(p,ids):
    ix=[p.ids.index(pid) for pid in ids]
    return {k:np.asarray(p.raw[k])[ix] for k in mc._NAMES}


def test_frozen_json_and_dct_orthogonality():
    path=Path(__file__).parents[1]/'configs/d92_mc_residual8_frozen_20260930.json'
    assert json.loads(path.read_text(encoding='utf-8'))=={'algorithm':mc.FROZEN_CONFIG}
    u,v=mc.initial_parameters();assert u.size+v.size==11776
    np.testing.assert_allclose(v@v.T,np.eye(8),rtol=0,atol=3e-15)


def test_identity_exact_R0_live_U_zero_V_data_gradient():
    p=prepare();u,v=mc.initial_parameters();f=p.problems[0];r=p.initial_folds[0];o=f.original
    base=local.fit_branch_local_ridge(**raw_slice(p,o.audit['training_physical_ids']),support_labels=np.asarray(o.train_labels,dtype=int),
        support_ids=o.audit['training_physical_ids'],classes=p.classes,old_classes=p.old_classes)
    np.testing.assert_array_equal(r['score'],base.score(**raw_slice(p,o.audit['held_physical_ids'])))
    loss,g,a,audit,_=mc.evaluate_mc_objective(p,u,v,u,v)
    assert np.linalg.norm(g[0])>1e-9;np.testing.assert_array_equal(g[1],np.zeros_like(v))
    assert audit['inner_head_fit_count']==audit['inner_factorization_count']==0
    assert audit['initial_teacher_cache_reused']
    assert audit['task_derivative_triangular_solve_count']==6 and audit['keep_derivative_triangular_solve_count']==6
    assert p.audit_dict()['initial_inner_head_fit_count']==3


@pytest.mark.parametrize('seed',[5,17])
def test_both_full_objectives_finite_difference(seed):
    p=prepare(seed=seed);u,v=parameters(seed+1);au,av=mc.initial_parameters();du,dv=directions(seed+2)
    loss,g,a,audit,_=mc.evaluate_mc_objective(p,u,v,au,av)
    h=3e-4
    lp=mc.evaluate_mc_objective(p,u+h*du,v+h*dv,au,av,gradient=False)
    lm=mc.evaluate_mc_objective(p,u-h*du,v-h*dv,au,av,gradient=False)
    np.testing.assert_allclose(np.sum(g[0]*du)+np.sum(g[1]*dv),(lp[0]-lm[0])/(2*h),rtol=8e-5,atol=2e-9)
    np.testing.assert_allclose(np.sum(a[0]*du)+np.sum(a[1]*dv),(lp[3]['loss_keep']-lm[3]['loss_keep'])/(2*h),rtol=1e-4,atol=2e-9)
    assert audit['derivative_triangular_solve_count']==12


def test_adapter_VJP_and_norm_angle_zero_blocks():
    p=prepare();f=p.full_problem;u,v=parameters();rng=np.random.default_rng(9)
    gb=rng.normal(size=f.original.train_b.shape);ga=rng.normal(size=f.original.train_a.shape);du,dv=directions(12)
    b,a,cache=mc._adapt(f.train_context,u,v,derivative_cache=True)
    gu,gv=mc._adapter_vjp(f.train_context,u,v,gb,ga,cache)
    def loss(t):
        rb,ra,_=mc._adapt(f.train_context,u+t*du,v+t*dv)
        return np.sum(rb*gb)+np.sum(ra*ga)
    h=1e-5
    np.testing.assert_allclose(np.sum(gu*du)+np.sum(gv*dv),(loss(h)-loss(-h))/(2*h),rtol=1e-5,atol=3e-9)
    changed=np.concatenate((b,a),axis=1);original=f.train_context['original']
    for sl in mc._SLICES:
        np.testing.assert_allclose(np.linalg.norm(changed[:,sl],axis=1),np.linalg.norm(original[:,sl],axis=1),rtol=3e-14,atol=0)
        cos=np.sum(changed[:,sl]*original[:,sl],axis=1)/(np.linalg.norm(changed[:,sl],axis=1)*np.linalg.norm(original[:,sl],axis=1))
        assert np.all(cos>=1/np.sqrt(1+.25**2)-1e-14)
    zero=mc._context(np.zeros((2,256)),np.zeros((2,480)));zb,za,zc=mc._adapt(zero,u,v,derivative_cache=True)
    assert np.all(zb==0) and np.all(za==0)
    zg=mc._adapter_vjp(zero,u,v,np.ones_like(zb),np.ones_like(za),zc)
    assert np.all(zg[0]==0) and np.all(zg[1]==0)


def test_mapping_all_batch_shapes_bitwise():
    p=prepare(k=4,c=4);f=p.full_problem;u,v=parameters(20);b,a=f.original.train_b,f.original.train_a
    allb,alla,_=mc._adapt(f.train_context,u,v)
    for size in (1,2,3,7,len(b)):
        chunks=[mc._adapt(mc._context(b[i:i+size],a[i:i+size]),u,v) for i in range(0,len(b),size)]
        np.testing.assert_array_equal(np.vstack([r[0] for r in chunks]),allb)
        np.testing.assert_array_equal(np.vstack([r[1] for r in chunks]),alla)
    ix=[3,1,3,0];bb,aa,_=mc._adapt(mc._context(b[ix],a[ix]),u,v)
    np.testing.assert_array_equal(bb,allb[ix]);np.testing.assert_array_equal(aa,alla[ix])


def test_unequal_fold_risks_aggregate_by_class_before_RMS():
    p=prepare(k=4);u,v=parameters();au,av=mc.initial_parameters();result=mc.evaluate_mc_objective(p,u,v,au,av,gradient=False);a=result[3]
    counts=np.sum([f['class_physical_counts'] for f in a['inner_folds']],axis=0)
    tasks=np.sum([f['class_task_loss_sums'] for f in a['inner_folds']],axis=0)/counts
    keeps=np.sum([f['class_keep_loss_sums'] for f in a['inner_folds']],axis=0)/counts
    np.testing.assert_array_equal(counts,[4,4,4])
    np.testing.assert_allclose(a['loss_task'],np.sqrt(np.mean(tasks**2)),rtol=2e-15)
    np.testing.assert_allclose(a['loss_keep'],np.sqrt(np.mean(keeps**2)),rtol=2e-15)


def test_B_teacher_inner_head_isolation_and_q_definition():
    args=fixture();p=mc.prepare_mc_residual8_training(**args);f=p.problems[0]
    held=set(f.original.audit['held_physical_ids'])
    for key in mc._NAMES:
        args[key]=args[key].copy()
        for i,pid in enumerate(args['support_ids']):
            if pid in held:args[key][i]*=np.linspace(.7,1.3,args[key].shape[1])
    p2=mc.prepare_mc_residual8_training(**args)
    np.testing.assert_array_equal(p.initial_folds[0]['alpha'],p2.initial_folds[0]['alpha'])
    np.testing.assert_array_equal(f.original.d0,p2.problems[0].original.d0)
    _,_,margin=mc.ch._margin_loss(p.initial_folds[0]['score'],np.asarray(f.original.held_labels,dtype=int))
    np.testing.assert_array_equal(f.teacher_q,np.clip(margin,0,1))
    assert not set(f.teacher_audit['training_physical_ids'])&set(f.teacher_audit['held_physical_ids'])


def test_guard_projection_and_physical_slack_zero_nonzero_anchor():
    g=(np.array([[-1.]]),np.array([[0.]]));a=(np.array([[2.]]),np.array([[0.]]))
    d,info=mc.guarded_direction(g,a,.05)
    assert info['guard_active'] and info['direction_norm']<=1
    np.testing.assert_allclose(info['guard_dot_after'],.05/.125,rtol=2e-15)
    _,inactive=mc.guarded_direction(g,(a[0]*0,a[1]),.05);assert not inactive['guard_active']
    for anchor_risk in (0.,.37):
        slack=1/(2*3*np.sqrt(6));limit=anchor_risk+slack
        np.testing.assert_allclose(limit-anchor_risk,np.sqrt(6)/(2*18),rtol=2e-15)


@pytest.mark.parametrize('k,c',[(1,3),(3,1)])
def test_K1_single_class_no_information_exact_R0(k,c):
    args=fixture(k=k,c=c);p=mc.prepare_mc_residual8_training(**args);base=local.fit_branch_local_ridge(**args)
    state=mc.fit_mc_residual8_local_ridge(p,baseline_state=base);a=state.audit_dict()
    assert a['optimizer_steps']==a['inner_head_fit_count']==a['inner_objective_evaluation_count']==a['final_head_fit_count']==0
    assert p.audit_dict()['teacher_head_fit_count']==p.audit_dict()['initial_inner_head_fit_count']==0
    np.testing.assert_array_equal(state.score(**{key:args[key] for key in mc._NAMES}),base.score(**{key:args[key] for key in mc._NAMES}))


def test_actual_fit_counters_dual_acceptance_json_and_state_records():
    p=prepare();events=[];state=mc.fit_mc_residual8_local_ridge(p,log_callback=events.append);a=state.audit_dict()
    assert a['optimizer_steps']==a['accepted_trial_count']<=4
    assert a['inner_objective_evaluation_count']==1+a['trial_count']<=13
    assert a['inner_head_fit_count']==3*a['trial_count']
    assert a['task_derivative_triangular_solve_count']==a['keep_derivative_triangular_solve_count']==6*a['backward_evaluation_count']
    assert a['derivative_triangular_solve_count']<=48
    for t in a['trials']:
        assert type(t['accepted']) is bool
        assert t['accepted']==(t['loss_after']<=min(t['loss_before'],t['armijo_rhs'])+t['armijo_tolerance'] and t['keep_risk_trial']<=a['keep_limit']+t['keep_tolerance'])
    records=state.state_records();final=records[a['final_state_ref']['key']]
    np.testing.assert_array_equal(final['U'],state.U);np.testing.assert_array_equal(final['V'],state.V)
    assert all('U' not in t and 'V' not in t for t in a['trials'])
    json.dumps(a,allow_nan=False);json.dumps(events,allow_nan=False)
    raw={k:np.asarray(p.raw[k]) for k in mc._NAMES};scores=state.score(**raw)
    single=np.vstack([state.score(**{k:v[i:i+1] for k,v in raw.items()}) for i in range(len(p.ids))])
    np.testing.assert_array_equal(scores,single)
    with pytest.raises(ValueError):state.U[0,0]=1


def test_state_callback_exact_arrays_and_no_retained_duplicate_vectors():
    p=prepare(k=2,c=2);saved={}
    def writer(key,arrays):
        assert key not in saved;saved[key]={k:np.array(v,copy=True) for k,v in arrays.items()}
        return {'path':'synthetic/'+key+'.npz'}
    state=mc.fit_mc_residual8_local_ridge(p,state_callback=writer)
    assert state.state_records()=={}
    assert state.audit_dict()['retained_vector_record_bytes']==0
    final=saved[state.audit_dict()['final_state_ref']['key']]
    np.testing.assert_array_equal(final['U'],state.U)
    for rec in state.audit_dict()['gradients']:
        assert {'g_U','g_V','keep_g_U','keep_g_V','d_U','d_V'}<=saved[rec['state_ref']['key']].keys()


def test_B_C_real_inheritance_fold_teacher_and_N0():
    args=fixture(k=2,c=3);old=('c0','c1');mask=args['support_labels']<2
    oldargs={k:args[k][mask] for k in mc._NAMES};oldargs.update(support_labels=args['support_labels'][mask],
        support_ids=tuple(pid for i,pid in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)
    b=mc.fit_mc_residual8_local_ridge(mc.prepare_mc_residual8_training(**oldargs,context={'row_id':'r'}))
    before=b.U.copy();args['old_classes']=old
    p=mc.prepare_mc_residual8_training(**args,inherited=b,context={'row_id':'r'})
    assert p.audit_dict()['teacher_head_fit_count']==2 and p.audit_dict()['initial_inner_head_fit_count']==0
    for f in p.problems:
        teacher=f.teacher_audit
        assert set(teacher['training_physical_ids'])<=set(f.original.audit['training_physical_ids'])
        assert not set(teacher['training_physical_ids'])&set(teacher['held_physical_ids'])
        assert all(pid.startswith(('physical-0','physical-1')) for pid in teacher['training_physical_ids'])
    seq=mc.fit_mc_residual8_local_ridge(p,mode='C_seq');reset=mc.fit_mc_residual8_local_ridge(p,mode='C_reset_init')
    seq_initial=seq.state_records()['initial'];reset_initial=reset.state_records()['initial']
    np.testing.assert_array_equal(seq_initial['U'],b.U);np.testing.assert_array_equal(seq_initial['V'],b.V)
    np.testing.assert_array_equal(reset_initial['U'],np.zeros_like(b.U));np.testing.assert_array_equal(b.U,before)
    n0=mc.prepare_mc_residual8_training(**oldargs,inherited=b)
    assert n0.audit_dict()['teacher_head_fit_count']==0
    assert mc.fit_mc_residual8_local_ridge(n0,mode='C_seq') is b
    with pytest.raises(ValueError,match='crosses'):mc.prepare_mc_residual8_training(**args,inherited=b,context={'row_id':'wrong'})


def test_single_old_teacher_unavailable_for_C():
    args=fixture(k=2,c=2);old=('c0',);oldargs={k:args[k][:2] for k in mc._NAMES}
    oldargs.update(support_labels=np.zeros(2,dtype=int),support_ids=args['support_ids'][:2],classes=old,old_classes=old)
    b=mc.fit_mc_residual8_local_ridge(mc.prepare_mc_residual8_training(**oldargs));args['old_classes']=old
    p=mc.prepare_mc_residual8_training(**args,inherited=b);u,v=parameters();au,av=mc.initial_parameters()
    result=mc.evaluate_mc_objective(p,u,v,au,av)
    assert not result[3]['keep_available'] and result[3]['loss_keep']==0
    assert np.all(result[2][0]==0) and np.all(result[2][1]==0)
    assert result[3]['keep_derivative_triangular_solve_count']==0


def test_rejected_trial_cache_keeps_last_accepted(monkeypatch):
    p=prepare();real=mc.evaluate_mc_objective;calls=[0]
    def wrapper(prepared,U,V,au,av,*,gradient=True,forward_cache=None):
        result=real(prepared,U,V,au,av,gradient=gradient,forward_cache=forward_cache)
        if forward_cache is None:
            calls[0]+=1
            if calls[0]>1:
                l,g,a,info,cache=result;info=dict(info,loss_total=l+100.,loss_keep=100.)
                return l+100.,g,a,info,cache
        return result
    monkeypatch.setattr(mc,'evaluate_mc_objective',wrapper)
    state=mc.fit_mc_residual8_local_ridge(p);a=state.audit_dict()
    assert a['stop_reason']=='ARMIJO_OR_KEEP_BUDGET_EXHAUSTED' and a['rejected_trial_count']==3
    assert a['optimizer_steps']==0 and np.all(state.U==0)
    assert a['final_objective']['loss_total']==a['initial_objective']['loss_total']


def test_zero_bandwidth_keeps_original_kernel_and_zero_both_adjoint():
    args=fixture(c=2)
    for key in mc._NAMES:args[key][3:]=args[key][:3]
    p=mc.prepare_mc_residual8_training(**args);f=p.problems[0];u,v=parameters();z,w=mc.initial_parameters()
    assert f.original.tau0==0
    r=mc._forward(f,u,v);r0=mc._forward(f,z,w)
    np.testing.assert_array_equal(r['score'],r0['score'])
    progress={'derivative_triangular_solve_count':0};g=mc._backward(r,np.ones_like(r['score']),progress)
    assert np.all(g[0]==0) and np.all(g[1]==0) and progress['derivative_triangular_solve_count']==0


def test_product_balls_projection():
    u,v=parameters();u*=30;v=mc._V0+30*(v-mc._V0)
    pu,pv=mc.project_parameters(u,v)
    np.testing.assert_allclose(np.linalg.norm(pu),1.,rtol=3e-15)
    np.testing.assert_allclose(np.linalg.norm(pv-mc._V0),1.,rtol=3e-15)


def test_ball_projection_can_reverse_descent_nonincrease_guard_rejects():
    U,V=mc.initial_parameters();U[0,0]=1.
    gu=np.zeros_like(U);gv=np.zeros_like(V);gu[0,0]=-2.;gu[1,0]=1.
    d0=-gu/np.linalg.norm(gu);wanted=np.zeros_like(U);wanted[0,0]=.5;wanted[1,0]=.1
    a=d0-wanted;bound=float(np.sum(a*wanted));assert bound>0
    direction,guard=mc.guarded_direction((gu,gv),(a,np.zeros_like(V)),bound*.125)
    np.testing.assert_allclose(direction[0],wanted,atol=2e-15)
    assert np.sum(gu*direction[0])<0
    tu,tv=mc.project_parameters(U+.125*direction[0],V+.125*direction[1])
    gd=float(np.sum(gu*(tu-U))+np.sum(gv*(tv-V)));assert gd>0
    increase=.5*1e-4*gd
    accepted=mc._trial_acceptance(1.,1.+increase,gd,0.,.1)
    assert accepted['armijo_pass'] and accepted['keep_pass']
    assert not accepted['objective_nonincrease_pass'] and not accepted['accepted']
    assert accepted['rejection_reason']=='OBJECTIVE_INCREASE'


def test_failed_keep_adjoint_counts_partial_two_channels(monkeypatch):
    p=prepare();u,v=parameters();au,av=mc.initial_parameters();real=mc._backward;calls=[0]
    def fail(cache,gscore,progress):
        calls[0]+=1
        if calls[0]==2:
            progress['derivative_triangular_solve_count']+=1
            raise FloatingPointError('INJECTED_KEEP_ADJOINT')
        return real(cache,gscore,progress)
    monkeypatch.setattr(mc,'_backward',fail)
    with pytest.raises(mc.NumericalFailure) as exc:mc.evaluate_mc_objective(p,u,v,au,av)
    a=exc.value.audit_dict()
    assert a['task_derivative_triangular_solve_count']==2 and a['keep_derivative_triangular_solve_count']==1
    assert a['derivative_triangular_solve_count']==3 and a['inner_head_fit_count']==3
    json.dumps(a,allow_nan=False)


def test_class_renaming_physical_row_permutation_two_risks_and_gradients():
    args=fixture(k=2,c=3,seed=93);p=mc.prepare_mc_residual8_training(**args)
    u,v=parameters(8);au,av=mc.initial_parameters();r=mc.evaluate_mc_objective(p,u,v,au,av)
    perm=np.random.default_rng(4).permutation(len(args['support_ids']))
    changed={k:args[k][perm] for k in mc._NAMES}
    # Renaming preserves each physical record's semantic class while changing
    # the canonical order of class columns; old role travels with that class.
    changed.update(support_labels=args['support_labels'][perm],support_ids=tuple(args['support_ids'][i] for i in perm),
        classes=('z','x','y'),old_classes=('z','x','y'))
    p2=mc.prepare_mc_residual8_training(**changed);r2=mc.evaluate_mc_objective(p2,u,v,au,av)
    np.testing.assert_allclose(r[0],r2[0],rtol=1e-13,atol=2e-14)
    np.testing.assert_allclose(r[3]['loss_keep'],r2[3]['loss_keep'],rtol=1e-12,atol=2e-14)
    for ga,gb in zip(r[1]+r[2],r2[1]+r2[2]):
        np.testing.assert_allclose(ga,gb,rtol=1e-8,atol=2e-12)
    for f,f2 in zip(p.problems,p2.problems):
        assert f.original.audit['training_physical_ids']==f2.original.audit['training_physical_ids']
        np.testing.assert_allclose(f.teacher_q,f2.teacher_q,rtol=1e-12,atol=2e-14)


def test_near_repeat_residual_map_and_joint_distance_decimal_explicit_phi():
    b=np.zeros((2,256));a=np.zeros((2,480))
    b[:,:2]=[.6,-.8];a[:,:3]=[.3,.4,.1]
    b[1,:2]+=[1e-13,-2e-13];a[1,:3]+=[-3e-13,2e-13,1e-13]
    u,v=parameters(72);mb,ma,_=mc._adapt(mc._context(b,a),u,v)
    # A real nonzero learned map is exercised; original and adapted distances
    # are compared against explicit high-precision interaction coordinates.
    assert not np.array_equal(mb,b) and not np.array_equal(ma,a)
    one=mc._adapt(mc._context(b[1:2],a[1:2]),u,v)
    np.testing.assert_array_equal(one[0],mb[1:2]);np.testing.assert_array_equal(one[1],ma[1:2])
    with localcontext() as ctx:
        ctx.prec=90
        def explicit(bb,aa):
            ib=np.flatnonzero(np.any(bb!=0,axis=0));ia=np.flatnonzero(np.any(aa!=0,axis=0))
            B=[[Decimal.from_float(float(x)) for x in row[ib]] for row in bb]
            A=[[Decimal.from_float(float(x)) for x in row[ia]] for row in aa]
            # Explicit phi differences, not the implementation's rank-2
            # contraction or floating Gram subtraction.
            total=sum((x-y)**2 for x,y in zip(B[0]+A[0],B[1]+A[1]))
            for i in range(len(ib)):
                for j in range(len(ia)):
                    total+=(B[0][i]*A[0][j]-B[1][i]*A[1][j])**2
            return total
        original=explicit(b,a);adapted=explicit(mb,ma);expected=float((original+adapted)/2)
    actual=mc.pt._joint_distances(mb,ma,mb,ma,b,a,b,a,symmetric=True)[0,1]
    assert 0<expected<1e-24
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=0)
    assert actual>=.5*float(original)
