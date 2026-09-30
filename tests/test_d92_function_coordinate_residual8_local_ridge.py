import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_function_coordinate_residual8_local_ridge as fcr
from cvsrffi import d92_branch_local_ridge as local


def fixture(k=3,c=3,seed=81):
    rng=np.random.default_rng(seed);labels=np.repeat(np.arange(c),k)
    raw={key:rng.normal(size=(c*k,d))+.35*rng.normal(size=(c,d))[labels]
         for key,d in zip(fcr._NAMES,(160,96,160,160,160))}
    names=tuple(f'c{i}' for i in range(c));ids=tuple(f'physical-{i}-{j:02}' for i in range(c) for j in range(k))
    return dict(**raw,support_labels=labels,support_ids=ids,classes=names,old_classes=names)


def prepare(**kw):return fcr.prepare_function_coordinate_training(**fixture(**kw))


def raw_slice(p,ids):
    ix=[p.ids.index(pid) for pid in ids]
    return {k:np.asarray(p.raw[k])[ix] for k in fcr._NAMES}


def test_config_fixed_dictionary_and_no_V_training():
    path=Path(__file__).parents[1]/'configs/d92_fcr8_frozen_20260930.json'
    assert json.loads(path.read_text(encoding='utf-8'))=={'algorithm':fcr.FROZEN_CONFIG}
    np.testing.assert_allclose(fcr.V0@fcr.V0.T,np.eye(8),atol=3e-15,rtol=0)
    assert fcr.FROZEN_CONFIG['trainable_parameter_count']==5888
    assert fcr.FROZEN_CONFIG['learn_V'] is False
    with pytest.raises(ValueError):fcr.V0[0,0]=2


def test_latent_whitening_function_identity_and_rank_policy():
    rng=np.random.default_rng(63);H=rng.normal(size=(29,8))*np.geomspace(.001,2,8)
    W,s,a=fcr.latent_coordinates(H);assert a['latent_rank']==8 and a['latent_svd_count']==1
    Z=rng.normal(size=(736,8))*.003;U=fcr.reconstruct_U(Z,np.zeros((736,8)),W)
    direct=np.mean(np.sum((H@U.T)**2,axis=1))
    np.testing.assert_allclose(direct,np.sum(Z*Z),rtol=2e-12)
    Q=np.linalg.qr(rng.normal(size=(20,8)))[0]
    eta=128*np.finfo(float).eps*20
    H=Q@np.diag([1,.5,.2,.1,.01,.001,2*np.sqrt(eta),.5*np.sqrt(eta)])
    W,s,a=fcr.latent_coordinates(H)
    assert a['latent_rank']==7
    assert a['whitening_residual']<=a['whitening_tolerance']


def test_zero_H_rank_and_scaled_subnormal_safe_or_explicit_failure():
    W,s,a=fcr.latent_coordinates(np.zeros((6,8)))
    assert W.shape==(8,0) and a['latent_svd_count']==0 and a['latent_rank']==0
    tiny=np.nextafter(0.,1.)
    with pytest.raises((fcr.NumericalFailure,FloatingPointError)):
        fcr.latent_coordinates(np.eye(8)*tiny)


def test_full_rank_latent_change_of_basis_preserves_function_gradient_step():
    rng=np.random.default_rng(45);H=rng.normal(size=(21,8));W,_,_=fcr.latent_coordinates(H)
    A=np.linalg.qr(rng.normal(size=(8,8)))[0]@np.diag(np.linspace(.5,2,8))
    H2=H@A.T;W2,_,_=fcr.latent_coordinates(H2)
    # Arbitrary function-space gradient, no reuse of the implementation's loss.
    G=rng.normal(size=(21,736));g=G.T@H@W;g2=G.T@H2@W2
    Z=-.125*g/np.linalg.norm(g);Z2=-.125*g2/np.linalg.norm(g2)
    U=Z@W.T;U2=Z2@W2.T
    query=rng.normal(size=(7,8))
    np.testing.assert_allclose(query@U.T,(query@A.T)@U2.T,rtol=2e-11,atol=2e-13)
    np.testing.assert_allclose(np.linalg.norm(Z),np.linalg.norm(Z2),rtol=2e-14)


def test_rank_deficient_anchor_null_component_is_never_projected():
    H=np.zeros((12,8));H[:,0]=np.arange(12)+1;H[:,1]=np.arange(12)[::-1]
    W,_,a=fcr.latent_coordinates(H);assert a['latent_rank']==2
    anchor=np.zeros((736,8));anchor[3,7]=13.
    zero=np.zeros((736,2))
    np.testing.assert_array_equal(fcr.reconstruct_U(zero,anchor,W),anchor)
    Z=np.ones((736,2))*.001;U=fcr.reconstruct_U(Z,anchor,W)
    assert U[3,7]==13.
    np.testing.assert_array_equal(U[:,2:],anchor[:,2:])


def test_zero_Z_exact_R0_with_live_gradient_and_cached_initial():
    p=prepare();Z=np.zeros((736,p.r));anchor=np.zeros((736,8));r=p.initial_folds[0];o=p.problems[0].original
    baseline=local.fit_branch_local_ridge(**raw_slice(p,o.audit['training_physical_ids']),
        support_labels=np.asarray(o.train_labels,dtype=int),support_ids=o.audit['training_physical_ids'],
        classes=p.classes,old_classes=p.old_classes)
    np.testing.assert_array_equal(r['score'],baseline.score(**raw_slice(p,o.audit['held_physical_ids'])))
    loss,g,a,info,cache=fcr.evaluate_function_coordinate_objective(p,Z,anchor)
    assert np.linalg.norm(g)>1e-8 and g.shape==a.shape==Z.shape
    assert info['inner_head_fit_count']==0 and info['initial_teacher_cache_reused']
    assert info['derivative_triangular_solve_count']==12
    assert info['pre_tangent_displacement_mean_squared']==0
    with pytest.raises(ValueError,match='binding'):
        fcr.evaluate_function_coordinate_objective(p,Z,anchor+1e-5,forward_cache=cache)


@pytest.mark.parametrize('seed',[5,17])
def test_both_risks_and_function_proximal_full_finite_difference(seed):
    p=prepare(seed=seed);rng=np.random.default_rng(seed+1)
    Z=rng.normal(size=(736,p.r));Z*=.02/np.linalg.norm(Z)
    d=rng.normal(size=Z.shape);d/=np.linalg.norm(d);anchor=np.zeros((736,8))
    loss,g,a,info,_=fcr.evaluate_function_coordinate_objective(p,Z,anchor)
    step=1e-5
    plus=fcr.evaluate_function_coordinate_objective(p,Z+step*d,anchor,gradient=False)
    minus=fcr.evaluate_function_coordinate_objective(p,Z-step*d,anchor,gradient=False)
    np.testing.assert_allclose(np.sum(g*d),(plus[0]-minus[0])/(2*step),rtol=3e-4,atol=2e-8)
    np.testing.assert_allclose(np.sum(a*d),(plus[3]['loss_keep']-minus[3]['loss_keep'])/(2*step),rtol=4e-4,atol=2e-8)
    proxder=(plus[3]['loss_proximal']-minus[3]['loss_proximal'])/(2*step)
    np.testing.assert_allclose(proxder,np.sum(Z*d),rtol=2e-9,atol=1e-12)
    assert not np.isclose(proxder,np.sum(Z*d)/len(p.ids),rtol=1e-3,atol=1e-12)
    np.testing.assert_allclose(info['pre_tangent_displacement_mean_squared'],np.sum(Z*Z),rtol=2e-11)


def test_fixed_U_head_independent_of_optimizer_W_and_inner_held_inputs():
    args=fixture();p=fcr.prepare_function_coordinate_training(**args);o=p.problems[0].original
    held=set(o.audit['held_physical_ids'])
    for key in fcr._NAMES:
        args[key]=args[key].copy()
        for i,pid in enumerate(args['support_ids']):
            if pid in held:args[key][i]*=np.linspace(.5,1.5,args[key].shape[1])
    p2=fcr.prepare_function_coordinate_training(**args)
    np.testing.assert_array_equal(p.initial_folds[0]['alpha'],p2.initial_folds[0]['alpha'])
    np.testing.assert_array_equal(p.problems[0].original.d0,p2.problems[0].original.d0)
    assert not np.array_equal(p.W,p2.W)
    U=np.zeros((736,8));U[1,0]=.1
    first=fcr._forward(p.problems[0],U);changed=fcr._forward(replace(p,W=p.W*2).problems[0],U)
    np.testing.assert_array_equal(first['score'],changed['score'])
    for t in p.audit_dict()['teacher_folds']:
        assert not set(t['training_physical_ids'])&set(t['held_physical_ids'])


@pytest.mark.parametrize('k,c',[(1,3),(3,1)])
def test_no_information_zero_actual_objectives_and_exact_baseline(k,c):
    args=fixture(k=k,c=c);p=fcr.prepare_function_coordinate_training(**args)
    base=local.fit_branch_local_ridge(**args);s=fcr.fit_function_coordinate_local_ridge(p,baseline_state=base)
    a=s.audit_dict();pa=p.audit_dict()
    assert a['optimizer_steps']==a['inner_objective_evaluation_count']==a['inner_head_fit_count']==0
    assert pa['latent_svd_count']==0 and not pa['rank_estimated']
    assert pa['dictionary_physical_evaluation_count']==k*c
    np.testing.assert_array_equal(s.score(**p.raw),base.score(**p.raw))


def test_all_inner_zero_bandwidth_skip_retains_original_geometry_audit():
    args=fixture(c=2)
    for key in fcr._NAMES:args[key][3:]=args[key][:3]
    p=fcr.prepare_function_coordinate_training(**args)
    assert p.audit['no_information_reason']=='ALL_INNER_GEOMETRY_DEGENERATE'
    assert len(p.audit['original_inner_geometry'])==3 and not p.problems
    assert p.audit['prepared_distance_evaluation_count']==8
    s=fcr.fit_function_coordinate_local_ridge(p)
    assert s.audit['optimizer_steps']==0 and s.audit['identity_forward']


def test_real_B_C_inheritance_new_coordinates_and_reset_teacher():
    args=fixture(k=3,c=3);old=('c0','c1');mask=args['support_labels']<2
    oa={key:args[key][mask] for key in fcr._NAMES}
    oa.update(support_labels=args['support_labels'][mask],support_ids=tuple(x for i,x in enumerate(args['support_ids']) if mask[i]),classes=old,old_classes=old)
    b=fcr.fit_function_coordinate_local_ridge(fcr.prepare_function_coordinate_training(**oa,context={'row_id':'one'}))
    before=b.U.copy();args['old_classes']=old
    p=fcr.prepare_function_coordinate_training(**args,inherited=b,context={'row_id':'one'})
    seq=fcr.fit_function_coordinate_local_ridge(p,mode='C_seq');reset=fcr.fit_function_coordinate_local_ridge(p,mode='C_reset_init')
    np.testing.assert_array_equal(seq.state_records()['initial']['U'],b.U)
    np.testing.assert_array_equal(reset.state_records()['initial']['U'],np.zeros_like(b.U))
    np.testing.assert_array_equal(before,b.U)
    assert seq.audit['preparation']['teacher_folds']==reset.audit['preparation']['teacher_folds']
    assert p.audit['teacher_head_fit_count']==3 and p.audit['initial_inner_head_fit_count']==0
    n0=fcr.prepare_function_coordinate_training(**oa,inherited=b)
    assert fcr.fit_function_coordinate_local_ridge(n0,mode='C_seq') is b
    with pytest.raises(ValueError,match='crosses'):
        fcr.prepare_function_coordinate_training(**args,inherited=b,context={'row_id':'two'})


def test_real_fit_counts_mechanism_JSON_and_batch_independence():
    p=prepare();events=[];state=fcr.fit_function_coordinate_local_ridge(p,log_callback=events.append);a=state.audit_dict()
    assert a['optimizer_steps']==a['accepted_trial_count']<=4
    assert a['inner_objective_evaluation_count']==1+a['trial_count']<=13
    assert a['inner_head_fit_count']==3*a['trial_count']
    assert a['derivative_triangular_solve_count']==12*a['backward_evaluation_count']<=48
    assert a['coordinate_norm']<=.5+1e-12
    for trial in a['trials']:
        assert trial['accepted']==(trial['armijo_pass'] and trial['objective_nonincrease_pass'] and trial['keep_pass'])
    records=state.state_records();final=records[a['final_state_ref']['key']]
    np.testing.assert_array_equal(final['U'],state.U)
    assert 'g_V' not in next(r for r in records.values() if 'g_Z' in r)
    assert a['final_objective']['inner_folds'][0]['kernel_change_from_initial'] is not None
    assert a['final_objective']['inner_folds'][0]['block_angle_radians']['count']>0
    json.dumps(a,allow_nan=False);json.dumps(events,allow_nan=False)
    allscore=state.score(**p.raw)
    for size in (1,2,4):
        chunks=[state.score(**{key:v[i:i+size] for key,v in p.raw.items()}) for i in range(0,len(p.ids),size)]
        np.testing.assert_array_equal(np.vstack(chunks),allscore)


def test_rejected_trials_do_not_overwrite_accepted_cache(monkeypatch):
    p=prepare();real=fcr.evaluate_function_coordinate_objective;calls=[0]
    def wrapped(prepared,Z,anchor_U,*,gradient=True,forward_cache=None):
        out=real(prepared,Z,anchor_U,gradient=gradient,forward_cache=forward_cache)
        if forward_cache is None:
            calls[0]+=1
            if calls[0]>1:
                loss,g,a,info,cache=out
                return loss+100,g,a,dict(info,loss_total=loss+100,loss_keep=100),cache
        return out
    monkeypatch.setattr(fcr,'evaluate_function_coordinate_objective',wrapped)
    s=fcr.fit_function_coordinate_local_ridge(p);a=s.audit_dict()
    assert a['optimizer_steps']==0 and a['rejected_trial_count']==3 and np.all(s.U==0)
    assert a['final_objective']['loss_total']==a['initial_objective']['loss_total']
    assert len([k for k in s.state_records() if k.startswith('trial_')])==3


def test_state_callback_no_duplicate_records_and_fixed_V0():
    p=prepare(k=2,c=2);saved={}
    def writer(key,arrays):
        assert key not in saved;saved[key]={k:np.array(v,copy=True) for k,v in arrays.items()}
        return {'path':'synthetic/'+key+'.npz'}
    s=fcr.fit_function_coordinate_local_ridge(p,state_callback=writer)
    assert not s.state_records() and s.audit['retained_vector_record_bytes']==0
    assert {'Z','U','anchor_U','W','singular_values'}<=saved['final'].keys()
    np.testing.assert_array_equal(s.V,s.V0)
    for r in s.audit['gradients']:
        assert {'g_Z','keep_g_Z','d_Z'}<=saved[r['state_ref']['key']].keys()


def test_class_rename_and_physical_row_permutation():
    args=fixture();p=fcr.prepare_function_coordinate_training(**args)
    order=np.random.default_rng(7).permutation(len(p.ids));per={k:args[k][order] for k in fcr._NAMES}
    per.update(support_labels=args['support_labels'][order],support_ids=tuple(args['support_ids'][i] for i in order),
        classes=('zeta','alpha','mu'),old_classes=('zeta','alpha','mu'))
    p2=fcr.prepare_function_coordinate_training(**per)
    np.testing.assert_array_equal(p.H,p2.H);np.testing.assert_array_equal(p.W,p2.W)
    Z=np.full((736,p.r),.0005);anchor=np.zeros((736,8))
    a=fcr.evaluate_function_coordinate_objective(p,Z,anchor);b=fcr.evaluate_function_coordinate_objective(p2,Z,anchor)
    np.testing.assert_allclose(a[0],b[0],rtol=3e-13);np.testing.assert_allclose(a[1],b[1],rtol=2e-11,atol=2e-13)


def test_near_repeat_full_interaction_distance_VJP_decimal():
    # Sparse originals become dense after a nonzero FCR adapter. Oracle explicitly
    # sums all b, a and outer(b,a) coordinates, not just the tensor component.
    rng=np.random.default_rng(218);b0=rng.normal(size=(2,256));a0=rng.normal(size=(2,480))
    b0[1]=b0[0]+rng.normal(size=256)*1e-10;a0[1]=a0[0]+rng.normal(size=480)*1e-10
    U=rng.normal(size=(736,8))*.003
    b,a,cache=fcr._adapt(fcr._context(b0,a0),U,derivative_cache=True)
    distance=local._pair_distances(b[:1],a[:1],b[1:],a[1:])[0]
    gb,ga,_,_=fcr.ch._distance_vjp(b[:1],a[:1],b[1:],a[1:],np.ones((1,1)))
    with localcontext() as ctx:
        ctx.prec=70;B=[[Decimal.from_float(float(x)) for x in row] for row in b]
        A=[[Decimal.from_float(float(x)) for x in row] for row in a]
        d=sum((x-y)**2 for x,y in zip(B[0],B[1]))+sum((x-y)**2 for x,y in zip(A[0],A[1]))
        db=[2*(x-y) for x,y in zip(B[0],B[1])];da=[2*(x-y) for x,y in zip(A[0],A[1])]
        for i in range(256):
            for j in range(480):
                delta=B[0][i]*A[0][j]-B[1][i]*A[1][j]
                d+=delta*delta;db[i]+=2*delta*A[0][j];da[j]+=2*delta*B[0][i]
    np.testing.assert_allclose(distance,float(d),rtol=4e-13,atol=0)
    np.testing.assert_allclose(gb[0],np.array([float(x) for x in db]),rtol=2e-10,atol=3e-23)
    np.testing.assert_allclose(ga[0],np.array([float(x) for x in da]),rtol=2e-10,atol=3e-23)


def test_common_sign_flip_is_not_equal_in_full_geometry():
    b=np.zeros((2,256));a=np.zeros((2,480));b[0,0]=1.;a[0,0]=1.;b[1]=-b[0];a[1]=-a[0]
    d=local._pair_distances(b[:1],a[:1],b[1:],a[1:])[0]
    assert d==8.  # Tensor part is zero but the two direct blocks contribute 8.
