import json
from decimal import Decimal, localcontext
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_joint_channel_local_ridge as ch
from cvsrffi import d92_branch_local_ridge as local


def fixture(k=3,c=3,seed=81):
    rng=np.random.default_rng(seed);labels=np.repeat(np.arange(c),k)
    raw={key:rng.normal(size=(c*k,d))+.35*rng.normal(size=(c,d))[labels]
         for key,d in zip(ch._NAMES,(160,96,160,160,160))}
    ids=tuple(f'physical-{cls}-{i:02}' for cls in range(c) for i in range(k))
    classes=tuple(f'c{cls}' for cls in range(c))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=classes,old_classes=classes)


def prepare(**kw):return ch.prepare_channel_training(**fixture(**kw))


def direction(seed=0,scale=.01):
    value=np.random.default_rng(seed).normal(size=736)
    for sl in ch._SLICES:value[sl]-=value[sl].mean()
    return value*scale


def raw_slice(prepared,ids):
    ix=[prepared.ids.index(x) for x in ids]
    return {key:np.asarray(prepared.raw[key])[ix] for key in ch._NAMES}


def test_frozen_json_exact_constant():
    path=Path(__file__).parents[1]/'configs/d92_joint_channel_frozen_20260930.json'
    assert json.loads(path.read_text(encoding='utf-8'))=={'algorithm':ch.FROZEN_CONFIG}
    assert ch.FROZEN_CONFIG['trainable_parameter_count']==736
    assert ch.FROZEN_CONFIG['effective_parameter_count']==731


@pytest.mark.parametrize('n',[26,364,520])
def test_adapter_all_block_norms_and_shape_sizes(n):
    rng=np.random.default_rng(n);b=rng.normal(size=(n,256));a=rng.normal(size=(n,480))
    b/=np.linalg.norm(b,axis=1)[:,None];a/=np.linalg.norm(a,axis=1)[:,None]
    b[0]=0;a[0]=0
    u=direction(51);mb,ma=ch._adapt_blocks(b,a,u)
    x=np.concatenate([b,a],axis=1);mx=np.concatenate([mb,ma],axis=1)
    for sl in ch._SLICES:np.testing.assert_allclose(np.linalg.norm(x[:,sl],axis=1),np.linalg.norm(mx[:,sl],axis=1),rtol=2e-14,atol=0)
    assert np.all(mx[0]==0)
    z_b,z_a=ch._adapt_blocks(b,a,np.zeros(736))
    assert z_b is b and z_a is a


def test_floor_original_ba_once_no_raw_renormalization(monkeypatch):
    args=fixture(k=2,c=2)
    scales=[0.,.5,1.,2.]
    for key,d in zip(ch._NAMES,(160,96,160,160,160)):
        floor=1e-8 if key=='fft' else 1e-12
        args[key][:]=0
        for i,s in enumerate(scales):args[key][i,:2]=floor*s/np.sqrt(2)
    p=ch.prepare_channel_training(**args)
    b,a=p.background.copy(),p.auxiliary.copy()
    monkeypatch.setattr(ch.interaction,'_blocks',lambda *a,**kw:pytest.fail('Second raw normalization'))
    zero=ch._adapt_blocks(b,a,np.zeros(736))
    np.testing.assert_array_equal(zero[0],b);np.testing.assert_array_equal(zero[1],a)
    adapted=ch._adapt_blocks(b,a,direction(3))
    for x,y in zip((b,a),adapted):np.testing.assert_allclose(np.linalg.norm(x,axis=1),np.linalg.norm(y,axis=1),rtol=2e-14,atol=0)


def test_gate_vjp_independent_direction_difference():
    p=prepare();u=direction(1);d=direction(2);rng=np.random.default_rng(3)
    gb=rng.normal(size=p.background.shape);ga=rng.normal(size=p.auxiliary.shape)
    analytic=ch._gate_vjp(p.background,p.auxiliary,u,gb,ga)@d
    def loss(t):
        b,a=ch._adapt_blocks(p.background,p.auxiliary,u+t*d)
        return float(np.sum(b*gb)+np.sum(a*ga))
    h=2e-5
    np.testing.assert_allclose(analytic,(loss(h)-loss(-h))/(2*h),rtol=2e-6,atol=2e-9)


def test_distance_vjp_explicit_small_tensor_oracle():
    rng=np.random.default_rng(9);b=rng.normal(size=(3,2));a=rng.normal(size=(3,3))
    tb=rng.normal(size=(2,2));ta=rng.normal(size=(2,3));weight=rng.normal(size=(3,2))
    grads=ch._distance_vjp(b,a,tb,ta,weight)
    dirs=[rng.normal(size=x.shape) for x in (b,a,tb,ta)]
    def loss(t):
        bb,aa,ttb,tta=[x+t*d for x,d in zip((b,a,tb,ta),dirs)]
        phi=np.concatenate([bb,aa,np.einsum('ij,ik->ijk',bb,aa).reshape(3,-1)],axis=1)
        train=np.concatenate([ttb,tta,np.einsum('ij,ik->ijk',ttb,tta).reshape(2,-1)],axis=1)
        return np.sum(weight*np.sum((phi[:,None,:]-train[None,:,:])**2,axis=2))
    h=1e-6
    np.testing.assert_allclose(sum(np.sum(g*d) for g,d in zip(grads,dirs)),(loss(h)-loss(-h))/(2*h),rtol=2e-8,atol=2e-8)


def test_symmetric_distance_vjp_sums_both_matrix_triangles():
    rng=np.random.default_rng(93);b=rng.normal(size=(4,2));a=rng.normal(size=(4,3));w=rng.normal(size=(4,4))
    gb,ga,_,_=ch._distance_vjp(b,a,b,a,w,symmetric=True)
    xb,xa,yb,ya=ch._distance_vjp(b,a,b,a,w)
    np.testing.assert_allclose(gb,xb+yb,rtol=3e-14,atol=1e-13)
    np.testing.assert_allclose(ga,xa+ya,rtol=3e-14,atol=1e-13)


def test_zero_exact_original_inner_forward_and_nonzero_gradient():
    p=prepare();problem=p.problems[0]
    loss,g,audit,parts,blocks,score=ch._evaluate_kernel(problem,np.zeros(736),return_scores=True)
    baseline=local.fit_branch_local_ridge(**raw_slice(p,problem.audit['training_physical_ids']),
        support_labels=np.asarray(problem.train_labels,dtype=int),support_ids=problem.audit['training_physical_ids'],
        classes=p.classes,old_classes=p.old_classes)
    np.testing.assert_array_equal(score,baseline.score(**raw_slice(p,problem.audit['held_physical_ids'])))
    assert np.linalg.norm(g)>1e-8
    assert audit['derivative_triangular_solve_count']==2
    assert audit['factorization_count']==1


@pytest.mark.parametrize('seed',[5,17,73])
def test_full_adjoint_smooth_direction_finite_difference(seed):
    p=prepare(seed=seed);u=direction(seed+1);anchor=direction(seed+2,scale=.001)
    loss,g,audit=ch.evaluate_channel_objective(p,u,anchor)
    d=direction(seed+3);h=1e-4
    lp=ch.evaluate_channel_objective(p,u+h*d,anchor,gradient=False)[0]
    lm=ch.evaluate_channel_objective(p,u-h*d,anchor,gradient=False)[0]
    np.testing.assert_allclose(g@d,(lp-lm)/(2*h),rtol=2e-5,atol=2e-9)
    assert audit['derivative_triangular_solve_count']==2*len(p.problems)
    assert audit['inner_factorization_count']==len(p.problems)


def test_wrong_class_tie_average_not_all_direction_derivatives():
    score=np.zeros((1,3));labels=np.array([0])
    loss,g,_=ch._margin_loss(score,labels)
    np.testing.assert_array_equal(g,np.array([[-1.,.5,.5]]))
    direction=np.array([[0.,1.,-1.]])
    assert float(np.sum(g*direction))==0
    h=1e-7
    # max(t,-t) has positive directional derivative in either direction.
    assert (ch._margin_loss(score+h*direction,labels)[0]-loss)/h>0.999
    assert (ch._margin_loss(score-h*direction,labels)[0]-loss)/h>0.999


def test_bandwidth_ties_uniform_and_permutation_equivariant():
    d=np.array([[0.,1.,2.,2.],[1.,0.,2.,2.],[2.,2.,0.,1.],[2.,2.,1.,0.]])
    y=np.array([0,0,1,1]);tau,w=ch._bandwidth_weights(d,y)
    assert tau==2;np.testing.assert_allclose(w[y[:,None]!=y[None,:]],1/8)
    perm=[3,0,2,1];_,wp=ch._bandwidth_weights(d[np.ix_(perm,perm)],y[perm])
    np.testing.assert_array_equal(wp,w[np.ix_(perm,perm)])


def test_inner_held_features_do_not_change_that_folds_head_state():
    args=fixture();p=ch.prepare_channel_training(**args);f=p.problems[0]
    held=set(f.audit['held_physical_ids'])
    for key in ch._NAMES:
        args[key]=args[key].copy()
        for i,pid in enumerate(args['support_ids']):
            if pid in held:args[key][i]*=np.linspace(.7,1.3,args[key].shape[1])
    p2=ch.prepare_channel_training(**args);f2=p2.problems[0]
    np.testing.assert_array_equal(f.train_b,f2.train_b);np.testing.assert_array_equal(f.d0,f2.d0)
    assert f.s0==f2.s0 and f.tau0==f2.tau0
    assert not np.array_equal(f.cross_d0,f2.cross_d0)
    u=direction(15)
    r=ch._evaluate_kernel(f,u,gradient=False);r2=ch._evaluate_kernel(f2,u,gradient=False)
    np.testing.assert_array_equal(r[3][0],r2[3][0])


def test_held_labels_only_change_loss_and_gradient_not_fitted_head():
    p=prepare();f=p.problems[0];other=replace(f,held_labels=(f.held_labels+1)%len(p.classes))
    a=ch._evaluate_kernel(f,direction(3),return_scores=True)
    b=ch._evaluate_kernel(other,direction(3),return_scores=True)
    np.testing.assert_array_equal(a[-1],b[-1]);np.testing.assert_array_equal(a[3][0],b[3][0])
    assert not np.allclose(a[1],b[1])


def test_unequal_inner_folds_physical_sum_weighting():
    p=prepare(k=4);u=direction(4);anchor=np.zeros(736)
    results=[ch._evaluate_kernel(f,u) for f in p.problems]
    loss,g,audit=ch.evaluate_channel_objective(p,u,anchor)
    np.testing.assert_allclose(loss,(sum(r[0] for r in results)+.5*np.sum(u*u))/len(p.ids),rtol=1e-14)
    np.testing.assert_allclose(g,(sum((r[1] for r in results),np.zeros(736))+u)/len(p.ids),rtol=1e-13,atol=1e-15)
    held=[pid for f in p.problems for pid in f.audit['held_physical_ids']]
    assert sorted(held)==sorted(p.ids) and len(set(held))==len(p.ids)


def test_zero_bandwidth_reuses_original_equivalence_with_nonzero_u():
    args=fixture(k=3,c=2)
    for key in ch._NAMES:args[key][3:]=args[key][:3]
    p=ch.prepare_channel_training(**args);f=p.problems[0]
    assert f.tau0==0
    a=ch._evaluate_kernel(f,np.zeros(736),return_scores=True)
    b=ch._evaluate_kernel(f,direction(5),return_scores=True)
    np.testing.assert_array_equal(a[-1],b[-1]);assert np.all(b[1]==0)
    assert b[2]['derivative_triangular_solve_count']==0


def test_mapping_collision_is_technical_failure():
    b0=np.zeros((2,256));a0=np.zeros((2,480));b0[1,0]=1e-7
    b=np.zeros_like(b0);a=np.zeros_like(a0)
    with pytest.raises(FloatingPointError,match='EQUIVALENCE_CHANGED'):
        ch._checked_distances(b,a,b,a,b0,a0,b0,a0,symmetric=True)


def test_projection_is_intersection_not_mean_then_clip():
    value=np.linspace(-4,3,736);out=ch.project_channels(value)
    ch._parameter(out)
    for sl in ch._SLICES:
        v=value[sl];o=out[sl];free=np.abs(o)<ch.FROZEN_CONFIG['bound']-1e-10
        assert abs(o.sum())<128*np.finfo(float).eps*len(o)*ch.FROZEN_CONFIG['bound']
        if np.any(free):np.testing.assert_allclose((v-o)[free],(v-o)[free][0],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('k,c',[(1,3),(3,1)])
def test_no_information_and_baseline_reuse(k,c):
    args=fixture(k=k,c=c);p=ch.prepare_channel_training(**args)
    baseline=local.fit_branch_local_ridge(**args);events=[]
    state=ch.fit_channel_local_ridge(p,baseline_state=baseline,log_callback=events.append)
    a=state.audit_dict()
    assert a['optimizer_steps']==a['inner_objective_evaluation_count']==a['final_head_fit_count']==0
    assert a['no_information'] and a['identity_forward']
    np.testing.assert_array_equal(state.score(**{x:args[x] for x in ch._NAMES}),baseline.score(**{x:args[x] for x in ch._NAMES}))
    assert [e['event'] for e in events]==['JOINT_CHANNEL_FIT']
    if c==1:assert ch._margin_loss(np.ones((3,1)),np.zeros(3,dtype=int))[0]==0


def test_eight_updates_nine_evals_bytes_events_and_batch_independence():
    p=prepare();events=[];state=ch.fit_channel_local_ridge(p,log_callback=events.append);a=state.audit_dict()
    assert a['optimizer_steps']==8 and a['inner_objective_evaluation_count']==9
    assert a['inner_head_fit_count']==a['inner_factorization_count']==27
    assert a['derivative_triangular_solve_count']==48
    assert a['final_head_fit_count']==a['final_factorization_count']==1
    assert a['optimizer_state_bytes']==11776
    assert a['persistent_state_bytes']==a['head_state_bytes']+a['lineage_state_bytes']+5888
    assert [e['step'] for e in events if 'step' in e]==list(range(1,9))
    assert events[-1]['event']=='JOINT_CHANNEL_FIT' and 'step' not in events[-1]
    raw={key:np.asarray(p.raw[key]) for key in ch._NAMES};batch=state.score(**raw)
    single=np.vstack([state.score(**{key:v[i:i+1] for key,v in raw.items()}) for i in range(len(p.ids))])
    np.testing.assert_array_equal(batch,single)
    with pytest.raises(ValueError):state.u[0]=1
    audit=state.audit_dict();audit['u'][0]=999
    assert state.u[0]!=999


def test_zero_instant_gradient_does_not_discard_adam_history(monkeypatch):
    p=prepare();calls=[]
    def objective(prepared,u,anchor,gradient=True):
        calls.append(gradient);g=np.zeros(736)
        if len(calls)==1:g[:2]=[.1,-.1]
        return 0.,g,dict(loss_data=0.,loss_proximal=0.,loss_total=0.,inner_folds=[],
            inner_head_fit_count=0,inner_factorization_count=0,derivative_triangular_solve_count=0)
    monkeypatch.setattr(ch,'evaluate_channel_objective',objective)
    state=ch.fit_channel_local_ridge(p);a=state.audit_dict()
    assert calls==[True]*8+[False]
    assert a['steps'][1]['gradient_norm']==0 and a['steps'][1]['update_norm']>0


def test_actual_B_C_inheritance_and_raw_binding():
    args=fixture(c=3);old=('c0','c1');mask=args['support_labels']<2
    bargs={key:args[key][mask] for key in ch._NAMES}
    bargs.update(support_labels=args['support_labels'][mask],support_ids=tuple(x for i,x in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)
    b=ch.fit_channel_local_ridge(ch.prepare_channel_training(**bargs,context={'row_id':'r'}))
    args['old_classes']=old;p=ch.prepare_channel_training(**args,inherited=b,context={'row_id':'r'})
    seq=ch.fit_channel_local_ridge(p,mode='C_seq');reset=ch.fit_channel_local_ridge(p,mode='C_reset')
    np.testing.assert_array_equal(seq.audit_dict()['anchor'],b.u)
    np.testing.assert_array_equal(reset.audit_dict()['anchor'],np.zeros(736))
    assert seq.audit_dict()['optimizer_state_reset'] and reset.audit_dict()['optimizer_state_reset']
    args['z_id']=args['z_id'].copy();args['z_id'][0]*=2
    with pytest.raises(ValueError,match='raw features'):ch.prepare_channel_training(**args,inherited=b)
    n0=ch.prepare_channel_training(**bargs,inherited=b)
    assert ch.fit_channel_local_ridge(n0,mode='C_seq') is b


def test_rows_and_class_renaming_equivariance():
    args=fixture(c=3,k=2);p=ch.prepare_channel_training(**args);u=direction(25)
    loss,g,_=ch.evaluate_channel_objective(p,u,np.zeros(736))
    perm=np.random.default_rng(8).permutation(len(args['support_ids']))
    changed={key:args[key][perm] for key in ch._NAMES}
    changed.update(support_labels=args['support_labels'][perm],support_ids=tuple(args['support_ids'][i] for i in perm),
        classes=('z','x','y'),old_classes=('z','x','y'))
    p2=ch.prepare_channel_training(**changed);l2,g2,_=ch.evaluate_channel_objective(p2,u,np.zeros(736))
    np.testing.assert_allclose(loss,l2,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(g,g2,rtol=1e-10,atol=1e-13)


def test_partial_inner_failure_keeps_actual_counts_and_context(monkeypatch):
    p=prepare();original=ch._evaluate_kernel;count=[0]
    def fail(problem,u,**kw):
        count[0]+=1
        if count[0]==2:
            kw['progress'].update(head_fit_count=1,factorization_count=1,derivative_triangular_solve_count=0)
            raise FloatingPointError('INJECTED_SECOND_FOLD')
        return original(problem,u,**kw)
    monkeypatch.setattr(ch,'_evaluate_kernel',fail)
    with pytest.raises(ch.NumericalFailure) as exc:ch.fit_channel_local_ridge(p)
    audit=exc.value.audit_dict()
    assert audit['optimizer_steps']==0
    assert audit['inner_objective_evaluation_count']==1 and audit['inner_head_fit_count']==2
    assert audit['inner_factorization_count']==2 and audit['derivative_triangular_solve_count']==2
    assert len(audit['failed_stage']['completed_inner_folds'])==1
    assert audit['failed_stage']['current_inner_fold']['inner_fold']==1
    json.dumps(audit,allow_nan=False)


def test_failure_nonfinite_payload_is_strict_json():
    err=ch.NumericalFailure('bad',{'u':np.array([np.nan,np.inf,-np.inf])})
    assert err.audit_dict()['u']==['NaN','Infinity','-Infinity']
    json.dumps(err.audit_dict(),allow_nan=False)


def test_reject_baseline_with_same_features_ids_but_wrong_physical_labels():
    args=fixture(k=1,c=3);p=ch.prepare_channel_training(**args)
    wrong=dict(args,support_labels=np.roll(args['support_labels'],1))
    baseline=local.fit_branch_local_ridge(**wrong)
    with pytest.raises(ValueError,match='Baseline.*binding|Baseline.*geometry'):
        ch.fit_channel_local_ridge(p,baseline_state=baseline)


def test_zero_feature_baseline_label_invariance_is_explicit():
    args=fixture(k=1,c=3)
    for key in ch._NAMES:args[key][:]=0
    p=ch.prepare_channel_training(**args)
    baseline=local.fit_branch_local_ridge(**dict(args,support_labels=np.roll(args['support_labels'],1)))
    state=ch.fit_channel_local_ridge(p,baseline_state=baseline)
    assert state.audit_dict()['baseline_label_binding']=='EXACT_ZERO_HEAD_LABEL_INVARIANT'
    assert np.all(state.score(**{key:args[key] for key in ch._NAMES})==0)


def test_near_repeat_distance_and_vjp_match_explicit_decimal_interaction():
    # Exact decimal conversion of the actual binary64 inputs: a naive Gram
    # subtraction loses this ~1e-26 distance, whereas direct differences retain it.
    bi=np.array([[1.1,-1.7]]);ai=np.array([[.6,-.9,1.3]])
    bj=bi+np.array([[1e-13,-2e-13]]);aj=ai+np.array([[-3e-13,2e-13,1e-13]])
    with localcontext() as ctx:
        ctx.prec=100
        B,A,T,U=[[Decimal.from_float(float(x)) for x in v[0]] for v in (bi,ai,bj,aj)]
        # Independent explicit phi=(b,a,vec(b outer a)), including its separate
        # four endpoint Jacobians; do not reuse the stable contracted formula.
        delta_b=[x-y for x,y in zip(B,T)];delta_a=[x-y for x,y in zip(A,U)]
        outer=[[B[i]*A[j]-T[i]*U[j] for j in range(3)] for i in range(2)]
        delta_phi=delta_b+delta_a+[x for row in outer for x in row]
        distance=float(sum(x*x for x in delta_phi))
        gbi=[2*(delta_b[i]+sum(outer[i][j]*A[j] for j in range(3))) for i in range(2)]
        gai=[2*(delta_a[j]+sum(outer[i][j]*B[i] for i in range(2))) for j in range(3)]
        gbj=[-2*(delta_b[i]+sum(outer[i][j]*U[j] for j in range(3))) for i in range(2)]
        gaj=[-2*(delta_a[j]+sum(outer[i][j]*T[i] for i in range(2))) for j in range(3)]
        expected=[np.array([[float(x) for x in g]]) for g in (gbi,gai,gbj,gaj)]
    assert 0<distance<1e-24
    np.testing.assert_allclose(local._pair_distances(bi,ai,bj,aj)[0],distance,rtol=8e-15,atol=0)
    actual=ch._distance_vjp(bi,ai,bj,aj,np.ones((1,1)))
    for got,want in zip(actual,expected):
        np.testing.assert_allclose(got,want,rtol=8e-15,atol=0)
