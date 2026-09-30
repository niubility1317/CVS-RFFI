"""FCR8: fixed dictionary, support function coordinates, two-risk ridge adjoints.

No I/O or query fitting. SVD is an optimizer coordinate, never head geometry.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import time
import numpy as np
from scipy.linalg import solve_triangular
from . import d92_margin_constrained_residual8_local_ridge as mc
from . import d92_joint_channel_local_ridge as ch
from . import d92_prototype_transport_local_ridge as pt
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import _freeze, _plain, _readonly

_EPS=float(np.finfo(np.float64).eps)
_NAMES=ch._NAMES
_SLICES=ch._SLICES
_norm=ch._norm
_finite=ch._finite
NumericalFailure=ch.NumericalFailure
_Recorder=mc._Recorder
_hinge=mc._hinge
_cache_bytes=mc._cache_bytes
_trial_acceptance=mc._trial_acceptance
_V0=_readonly(mc.initial_parameters()[1])
V0=_V0
FROZEN_CONFIG=dict(
    schema='d92_function_coordinate_residual8_local_ridge_v1',method='D92-FCR8LocalRidge-v1',
    base_algorithm=deepcopy(local.FROZEN_CONFIG),
    channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000),
    allowed_scenarios=['practical_high','practical_mid','practical_low_urban'],
    arms=['local_ridge','fcr_seq','fcr_reset_init'],rank=8,input_dim=736,
    trainable_parameter_count=5888,active_parameter_count='736*retained_latent_rank',
    adapter='fixed_DCT_exact_GELU_function_coordinate_tangent_norm_preserving_residual',
    input='concat_original_block_unit_directions_over_sqrt5',block_dimensions=[160,96,160,160,160],
    kappa=.25,fixed_dictionary='first8_orthonormal_DCT_rows',learn_V=False,
    coordinate='U=anchor_U+Z@W.T;H/sqrtN=P*Sigma*R.T;W=R_retained/Sigma_retained',
    coordinate_data='all_current_outer_train_unlabelled_dictionary_not_inner_head_statistics',
    rank_rule='sigma/sigma_max>sqrt(128*eps64*max(N,8))',
    whitening_tolerance='128*eps64*max(N,8)*max(1,sigma_max/sigma_min_retained)',
    initialization_B='U_zero',initialization_C_seq='exact_copy_current_B_U',
    initialization_C_reset_init='U_zero_same_B_teacher',zero_Z='exact_anchor_U_copy',
    zero_U='bitwise_R0_forward_live_U_Z_derivative',joint_original_distance_weight=.5,
    trace_target='original_inner_train_interaction_centered_trace',
    interaction_geometry='concat_b_a_outer_b_a_dimension_123616',
    objective='RMS_class_task+RMS_old_class_teacher_keep+0.5*Z_frobenius_squared',
    function_proximal='mean_physical_pre_tangent_residual_displacement_squared_over2',
    task_weight=1.,keep_weight=1.,function_proximal_coefficient=1.,
    teacher_B='first_R0_inner_forward_cache',teacher_C='frozen_B_adapter_old_inner_train_only_head',
    teacher_margin='clip_true_minus_max_wrong_teacher_old_score_0_1',
    keep_slack='1_over_2k_sqrt_Cprev',keep_limit='initial_keep_plus_fixed_physical_slack',
    optimizer='normalized_function_gradient_keep_halfspace_Armijo_nonincrease_keep',
    max_iterations=4,max_trials=3,initial_step_size=.125,backtrack_factor=.5,armijo_coefficient=1e-4,
    parameter_ball=None,function_coordinate_path_bound=.5,objective_nonincrease_required=True,
    comparison_tolerance_multiplier=128,comparison_tolerance='128*eps64*max(1,abs(compared_scalars))',
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3',
    zero_bandwidth='original_equivalence_kernel_zero_derivative',
    tie_gradient='uniform_exact_wrong_max;uniform_exact_nearest_and_tied_median_groups',
    no_information='physical_K1_single_class_zero_dictionary_all_inner_geometry_degenerate',
    n0='reuse_corresponding_B',query_decision_policy='per_sample_all_registered_classes',
    source_inputs=False,query_fit=False,phase1_frozen=True,encoder_backward=False,
    parameter_search=False,dtype='float64',vector_logging='lossless_state_callback_or_records')
_CONFIG=deepcopy(FROZEN_CONFIG)

def initial_parameters():
    return np.zeros((736,8)),np.array(_V0,copy=True)

def _parameter(U):
    U=np.asarray(U,dtype=np.float64)
    if U.shape!=(736,8):raise ValueError('FCR U shape mismatch')
    _finite(U)
    return U

def _dictionary(x):
    h=np.empty((len(x),8))
    for i in range(len(x)):h[i]=mc._gelu(np.sum(_V0*x[i,None,:],axis=1))
    _finite(h)
    return h

def _context(b,a,H=None):
    ctx=mc._context(b,a)
    ctx['h']=_dictionary(ctx['x']) if H is None else np.asarray(H,dtype=float)
    if ctx['h'].shape!=(len(b),8):raise ValueError('Dictionary context shape mismatch')
    return ctx

def latent_coordinates(H):
    H=np.asarray(H,dtype=float)
    if H.ndim!=2 or H.shape[1]!=8 or len(H)==0:raise ValueError('H must have shape N x 8')
    _finite(H);n=len(H);eta=128*_EPS*max(n,8);scale=float(np.max(np.abs(H)))
    info=dict(latent_svd_count=0,latent_rank=0,rank_estimated=True,rank_energy_threshold=eta,
        singular_values=[0.]*8,whitening_residual=0.,whitening_tolerance=0.,dictionary_rms=float(_norm(H)/np.sqrt(n)))
    if scale==0:return np.empty((8,0)),np.zeros(8),info
    info['latent_svd_count']=1
    try:
        _,s,vt=np.linalg.svd((H/scale)/np.sqrt(n),full_matrices=False)
        keep=(s/s[0])>np.sqrt(eta);r=int(keep.sum())
        W=(vt[keep].T/s[keep])/scale
        values=np.zeros(8);values[:len(s)]=s*scale
        _finite(W,values)
        white=(H/scale)@(vt[keep].T/s[keep])/np.sqrt(n)
        error=float(np.linalg.norm(white.T@white-np.eye(r),ord=2)) if r else 0.
        tolerance=float(eta*max(1.,s[0]/s[keep][-1])) if r else 0.
        if error>tolerance:raise FloatingPointError('FCR_WHITENING_RESIDUAL_EXCEEDED')
        info.update(latent_rank=r,singular_values=values.tolist(),whitening_residual=error,whitening_tolerance=tolerance)
        return W,values,info
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        info.update(status='TECHNICAL_FAILURE',failure_reason=str(exc))
        raise NumericalFailure(str(exc),info) from exc

def reconstruct_U(Z,anchor_U,W):
    Z=np.asarray(Z,dtype=float);anchor_U=_parameter(anchor_U);W=np.asarray(W,dtype=float)
    if W.ndim!=2 or W.shape[0]!=8 or Z.shape!=(736,W.shape[1]):raise ValueError('FCR Z/W shape mismatch')
    _finite(Z,W)
    if not np.any(Z):return anchor_U.copy()
    U=anchor_U+Z@W.T
    _finite(U)
    return U

def _statistics(values):
    v=np.asarray(values,dtype=float).ravel();_finite(v)
    if not len(v):return dict(count=0,minimum=None,mean=None,maximum=None)
    return dict(count=int(len(v)),minimum=float(v.min()),mean=float(v.mean()),maximum=float(v.max()))

def _functional_measurement(prepared,U,anchor_U,Z):
    delta=U-anchor_U
    values=np.stack([np.sum(delta*h[None,:],axis=1) for h in prepared.H])
    actual=float((_norm(values)/np.sqrt(len(values)))**2);expected=float(_norm(Z)**2);_finite(actual,expected)
    return dict(pre_tangent_displacement_mean_squared=actual,pre_tangent_displacement_rms=float(np.sqrt(actual)),
        coordinate_squared_norm=expected,function_coordinate_reconstruction_error=abs(actual-expected))


def _adapt(context,U,*,derivative_cache=False):
    U=_parameter(U);x=context['x'];w=context['w'];rho=context['rho'];n=len(x)
    output=np.empty_like(context['original']);zall=np.empty((n,8));hall=np.empty((n,8))
    tall=np.empty_like(output);dall=np.empty((n,5));yall=np.empty_like(output);normall=np.empty((n,5))
    kappa=.25
    # Identical sample-wise sums in train, teacher and score: no batch GEMM.
    for i in range(n):
        z=np.zeros(8);h=context['h'][i];v=np.sum(U*h[None,:],axis=1)
        zall[i]=z;hall[i]=h
        for l,sl in enumerate(_SLICES):
            wi=w[i,sl];t=v[sl]-wi*float(np.sum(wi*v[sl]));q=_norm(t)
            den=float(np.hypot(kappa,q));delta=kappa*t/den;s=wi+delta;sn=_norm(s)
            if rho[i,l]==0:y=np.zeros_like(wi);sn=1.
            else:
                if sn==0:raise FloatingPointError('RESIDUAL_NORMALIZATION_UNRESOLVED')
                y=s/sn
            output[i,sl]=rho[i,l]*y;tall[i,sl]=t;dall[i,l]=den;yall[i,sl]=y;normall[i,l]=sn
            if rho[i,l]>0:
                err=abs(_norm(output[i,sl])-rho[i,l])
                if err>128*_EPS*len(wi)*rho[i,l]:raise FloatingPointError('RESIDUAL_BLOCK_NORM_FAILED')
                orth=abs(float(np.sum(t*wi)))
                if orth>128*_EPS*len(wi)*max(_norm(v[sl]),_norm(t),np.finfo(float).tiny):
                    raise FloatingPointError('RESIDUAL_TANGENT_CHECK_FAILED')
    _finite(output,zall,hall)
    if np.all(U==0):output=context['original']
    cache=dict(z=zall,h=hall,t=tall,den=dall,y=yall,snorm=normall) if derivative_cache else None
    return output[:,:256],output[:,256:],cache


def _adapter_vjp(context,U,gb,ga,cache):
    g=np.concatenate((gb,ga),axis=1);gU=np.zeros_like(U);kappa=.25
    for i in range(len(g)):
        gv=np.zeros(736)
        for l,sl in enumerate(_SLICES):
            if context['rho'][i,l]==0:continue
            y=cache['y'][i,sl];w=context['w'][i,sl];t=cache['t'][i,sl];den=cache['den'][i,l]
            gd=context['rho'][i,l]/cache['snorm'][i,l]*(g[i,sl]-y*float(np.sum(y*g[i,sl])))
            gt=kappa/den*(gd-t*(float(np.sum(t*gd))/(den*den)))
            gv[sl]=gt-w*float(np.sum(w*gt))
        gU+=gv[:,None]*cache['h'][i,None,:]
    _finite(gU);return gU


@dataclass(frozen=True)
class FCRProblem:
    original: ch.ChannelProblem
    train_context: Mapping
    held_context: Mapping
    teacher_q: np.ndarray
    old_mask: np.ndarray
    teacher_audit: Mapping
    def __post_init__(self):
        for name in ('train_context','held_context'):
            object.__setattr__(self,name,_freeze({k:_readonly(v) for k,v in getattr(self,name).items()}))
        for name in ('teacher_q','old_mask'):object.__setattr__(self,name,_readonly(getattr(self,name)))
        object.__setattr__(self,'teacher_audit',_freeze(self.teacher_audit))
    @property
    def audit(self):
        a=_plain(self.original.audit);a['teacher']=_plain(self.teacher_audit);return a


def _forward(problem,U,progress=None):
    started=time.perf_counter();p=problem.original;n=len(p.train_labels);c=len(p.classes)
    audit={} if progress is None else progress
    audit.update(problem.audit,head_fit_count=1,factorization_count=0,derivative_triangular_solve_count=0,fcr_forward_evaluation_count=1)
    b,a,bc=_adapt(problem.train_context,U,derivative_cache=True)
    hb,ha,hc=_adapt(problem.held_context,U,derivative_cache=True)
    identity=bool(np.all(U==0)) or p.tau0==0
    if identity:distance,cross=p.d0,p.cross_d0
    else:
        distance=pt._joint_distances(b,a,b,a,p.train_b,p.train_a,p.train_b,p.train_a,symmetric=True,d0=p.d0)
        cross=pt._joint_distances(hb,ha,b,a,p.held_b,p.held_a,p.train_b,p.train_a,d0=p.cross_d0)
    tau,weights=ch._bandwidth_weights(distance,np.asarray(p.train_labels,dtype=int))
    if p.tau0 is not None and p.tau0>0 and tau<=0:raise FloatingPointError('FCR_POSITIVE_BANDWIDTH_UNRESOLVED')
    target=np.eye(c)[np.asarray(p.train_labels,dtype=int)]-1/c;alpha=np.zeros((n,c));reference=mean=np.zeros(n)
    refself=grand=0.;gamma=None;kernel=np.zeros((n,n));score=np.zeros((len(p.held_labels),c));tol=128*_EPS*max(n,c)
    audit.update(bandwidth_tau=tau,original_bandwidth_tau=p.tau0,interaction_centered_trace=p.s0,
        radial_centered_trace=None,trace_scale=None,normal_equation_residual=0.,trace_relative_error=0.,numerical_tolerance=tol,identity_forward=identity)
    saved={}
    if c>1 and p.s0>0:
        raw=local._radial_minus_one(distance,tau);radial=local._radial(distance,tau);sr=float(-2*raw[np.triu_indices(n,1)].sum()/n)
        if sr<=0:raise FloatingPointError('FCR_NONPOSITIVE_RADIAL_TRACE')
        gamma=p.s0/sr;center,reference,refself,mean,grand=interaction._center_kernel(raw)
        kernel=gamma*center;matrix=kernel+np.eye(n);audit['factorization_count']=1;chol=np.linalg.cholesky(matrix)
        alpha=np.ascontiguousarray(solve_triangular(chol.T,solve_triangular(chol,target,lower=True),lower=False))
        cc=ch._cross_center(local._radial_minus_one(cross,tau),reference,refself,mean,grand);crosskernel=gamma*cc
        score=np.stack([(gamma*cc[i])@alpha for i in range(len(p.held_labels))]) if len(p.held_labels) else score
        residual=_norm(matrix@alpha-target)/((1+p.s0)*_norm(alpha)+_norm(target));terr=abs(float(np.trace(kernel))-p.s0)/p.s0
        if residual>tol or terr>tol:raise FloatingPointError('FCR_HEAD_RESIDUAL_EXCEEDED')
        audit.update(normal_equation_residual=residual,trace_relative_error=terr,radial_centered_trace=sr,trace_scale=gamma)
        saved.update(chol=chol,crosskernel=crosskernel,radial=radial,sr=sr)
    # Diagnostics use only arrays from this forward; no diagnostic head fits.
    angles=[];tangent=[]
    for ctx,cache in ((problem.train_context,bc),(problem.held_context,hc)):
        for sl,l in zip(_SLICES,range(5)):
            valid=ctx['rho'][:,l]>0
            chord=np.linalg.norm(cache['y'][valid,sl]-ctx['w'][valid,sl],axis=1)
            angles.extend((np.zeros_like(chord) if np.all(U==0) else 2*np.arcsin(np.minimum(1.,chord/2))).tolist())
            tangent.extend((np.linalg.norm(cache['t'][valid,sl],axis=1)/.25).tolist())
    positive=p.d0>0;relative=(distance[positive]-p.d0[positive])/p.d0[positive]
    audit.update(block_angle_radians=_statistics(angles),tangent_over_kappa=_statistics(tangent),
        joint_distance_relative_change=_statistics(relative),
        adapted_distance_relative_change=_statistics(2*relative) if p.tau0!=0 or np.all(U==0) else None,
        adapted_distance_unmeasured_reason='ZERO_BANDWIDTH_BYPASSES_ADAPTED_DISTANCE' if p.tau0==0 and np.any(U!=0) else None,
        original_zero_distance_pair_count=int(np.sum(~positive)),
        kernel_frobenius_norm=_norm(kernel),kernel_change_from_initial=None,
        kernel_change_unmeasured_reason='INITIAL_CACHE_NOT_ATTACHED_TO_THIS_FORWARD')
    fit=kernel@alpha;err=fit-target
    audit.update(head_training_loss_data=float(.5*np.sum(err*err)),head_training_loss_ridge=float(.5*np.sum(alpha*fit)),
        sample_weight=1.,ridge_coefficient=1.,status='CLOSED_FORM_SOLVED' if c>1 and p.s0>0 else 'EXACT_ZERO_CLASSIFIER',forward_seconds=time.perf_counter()-started)
    audit['head_training_loss_total']=audit['head_training_loss_data']+audit['head_training_loss_ridge'];_finite(score,alpha)
    return dict(problem=problem,U=np.asarray(U).copy(),score=score,b=b,a=a,hb=hb,ha=ha,bc=bc,hc=hc,
        distance=distance,cross=cross,weights=weights,kernel=kernel,alpha=alpha,tau=tau,gamma=gamma,
        parts=(alpha,reference,refself,mean,grand,tau,gamma),audit=audit,**saved)


def _backward(cache,gscore,progress):
    start=time.perf_counter();p=cache['problem']
    if cache['gamma'] is None or cache['tau']==0 or not len(gscore):
        progress['adjoint_seconds']=time.perf_counter()-start;return np.zeros((736,8))
    alpha=cache['alpha'];chol=cache['chol'];crosskernel=cache['crosskernel'];kernel=cache['kernel'];n=len(alpha)
    gamma=cache['gamma'];sr=cache['sr'];tau=cache['tau'];barcross=gscore@alpha.T
    progress['derivative_triangular_solve_count']+=1;low=solve_triangular(chol,crosskernel.T@gscore,lower=True)
    progress['derivative_triangular_solve_count']+=1;z=solve_triangular(chol.T,low,lower=False)
    bark=-z@alpha.T;bark=.5*(bark+bark.T)
    barraw=gamma*(bark-bark.mean(axis=0)[None,:]-bark.mean(axis=1)[:,None]+bark.mean())
    crossrow=barcross-barcross.mean(axis=1)[:,None];barraw-=gamma*np.broadcast_to(crossrow.sum(axis=0)/n,(n,n))
    barsr=-(float(np.sum(bark*kernel))+float(np.sum(barcross*crosskernel)))/sr;barraw[~np.eye(n,dtype=bool)]-=barsr/n
    rd=barraw*cache['radial'];crossrad=local._radial(cache['cross'],tau);rc=gamma*crossrow*crossrad
    dd=np.zeros_like(rd);dc=np.zeros_like(rc);active=cache['radial']>0;activec=crossrad>0
    dd[active]=-rd[active]/tau;dc[activec]=-rc[activec]/tau
    bt=(np.sum(rd[active]*(cache['distance'][active]/tau))+np.sum(rc[activec]*(cache['cross'][activec]/tau)))/tau
    dd+=bt*cache['weights']
    gb,ga,_,_=ch._distance_vjp(cache['b'],cache['a'],cache['b'],cache['a'],.5*dd,symmetric=True)
    ghb,gha,gtb,gta=ch._distance_vjp(cache['hb'],cache['ha'],cache['b'],cache['a'],.5*dc)
    gu=_adapter_vjp(p.train_context,cache['U'],gb+gtb,ga+gta,cache['bc'])
    hu=_adapter_vjp(p.held_context,cache['U'],ghb,gha,cache['hc'])
    progress['adjoint_seconds']=time.perf_counter()-start;return gu+hu


def _problem(original,dictionary):
    train=original.audit['training_physical_ids'];held=original.audit['held_physical_ids']
    th=np.array([dictionary[x] for x in train],dtype=float).reshape((-1,8))
    hh=np.array([dictionary[x] for x in held],dtype=float).reshape((-1,8))
    n=len(original.held_labels)
    return FCRProblem(original,_context(original.train_b,original.train_a,th),
        _context(original.held_b,original.held_a,hh),np.zeros(n),np.zeros(n),{})

@dataclass(frozen=True)
class FunctionCoordinateTraining:
    base: ch.ChannelTraining
    problems: tuple
    full_problem: FCRProblem
    initial_folds: tuple
    H: np.ndarray
    W: np.ndarray
    singular_values: np.ndarray
    r: int
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for key in ('H','W','singular_values'):object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'audit',_freeze(self.audit));object.__setattr__(self,'records',_freeze(self.records))
    def __getattr__(self,name):return getattr(self.base,name)
    def audit_dict(self):return ch._safe(_plain(self.audit))
    def state_records(self):return {k:{name:np.array(v,copy=True) for name,v in rec.items()} for k,rec in self.records.items()}

def _teacher_problem(base,original,dictionary):
    old=base.old_classes;mask=np.array([base.classes[int(y)] in old for y in base.labels])
    ids=tuple(pid for i,pid in enumerate(base.ids) if mask[i]);trainset=set(original.audit['training_physical_ids'])
    y=np.array([old.index(base.classes[int(v)]) for v in base.labels[mask]])
    return _problem(ch._make_problem(base.background[mask],base.auxiliary[mask],y,ids,old,
        np.array([pid in trainset for pid in ids]),original.audit['inner_fold']),dictionary)

def prepare_function_coordinate_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,
        classes,old_classes,inherited=None,context=None,log_callback=None,state_callback=None):
    start=time.perf_counter();rec=_Recorder(state_callback)
    if inherited is not None and not isinstance(inherited,FunctionCoordinateState):
        raise ValueError('FCR C requires a current FCR B state')
    base=ch.prepare_channel_training(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local,
        support_labels=support_labels,support_ids=support_ids,classes=classes,old_classes=old_classes,
        inherited=inherited,context=context)
    audit=base.audit_dict();audit.pop('channel_preparation_count',None)
    audit.update(fcr_preparation_count=1,inner_head_fit_count=0,inner_factorization_count=0,
        initial_inner_head_fit_count=0,initial_inner_factorization_count=0,teacher_head_fit_count=0,
        teacher_factorization_count=0,teacher_score_evaluation_count=0,teacher_score_physical_count=0,
        fcr_forward_evaluation_count=0,prepared_distance_evaluation_count=2*len(base.problems),
        latent_svd_count=0,dictionary_physical_evaluation_count=0,teacher_folds=[],completed_stages=[],
        optimizer_coordinate_ids=list(base.ids),optimizer_coordinate_scope='ALL_CURRENT_OUTER_TRAIN_UNLABELLED',
        fixed_dictionary_trainable=False,original_inner_geometry=[_plain(p.audit) for p in base.problems])
    problems=[];initial=[];progress={};kind=None
    try:
        fullctx=_context(base.background,base.auxiliary);H=fullctx['h'];audit['dictionary_physical_evaluation_count']=len(H)
        dictionary={pid:H[i] for i,pid in enumerate(base.ids)}
        n0=inherited is not None and set(base.classes)==set(base.old_classes)
        all_degenerate=bool(base.problems) and all(p.s0==0 or p.tau0==0 for p in base.problems)
        if audit['no_information'] or n0 or all_degenerate:
            W=np.empty((8,0));singular=np.zeros(8)
            ci=dict(latent_svd_count=0,latent_rank=0,rank_estimated=False,singular_values=None,
                rank_energy_threshold=128*_EPS*max(len(H),8),whitening_residual=None,whitening_tolerance=None,
                dictionary_rms=float(_norm(H)/np.sqrt(len(H))))
        else:W,singular,ci=latent_coordinates(H)
        audit.update(ci)
        if not audit['no_information']:
            reason='N0_REUSE_B' if n0 else ('ALL_INNER_GEOMETRY_DEGENERATE' if all_degenerate else ('ZERO_DICTIONARY' if W.shape[1]==0 else None))
            if reason:audit.update(no_information=True,no_information_reason=reason)
        audit['coordinate_state_ref']=rec.save('coordinates',H=H,W=W,singular_values=singular)
        for original in (() if audit['no_information'] else base.problems):
            p=_problem(original,dictionary);oldmask=np.array([base.classes[int(v)] in base.old_classes for v in original.held_labels])
            q=np.zeros(len(oldmask));ta=dict(available=len(base.old_classes)>1,old_classes=list(base.old_classes))
            if len(base.old_classes)>1:
                progress={};kind='initial' if inherited is None else 'teacher'
                teacherproblem=p if inherited is None else _teacher_problem(base,original,dictionary)
                if inherited is not None:audit['prepared_distance_evaluation_count']+=2
                U=np.zeros((736,8)) if inherited is None else inherited.U
                cached=_forward(teacherproblem,U,progress)
                if inherited is None:
                    initial.append(cached);audit['initial_inner_head_fit_count']+=1
                    audit['initial_inner_factorization_count']+=progress['factorization_count']
                    audit['inner_head_fit_count']+=1;audit['inner_factorization_count']+=progress['factorization_count']
                else:
                    audit['teacher_head_fit_count']+=1;audit['teacher_factorization_count']+=progress['factorization_count']
                audit['fcr_forward_evaluation_count']+=1;audit['teacher_score_evaluation_count']+=1
                audit['teacher_score_physical_count']+=len(teacherproblem.original.held_labels)
                _,_,margin=ch._margin_loss(cached['score'],np.asarray(teacherproblem.original.held_labels,dtype=int))
                teacherq=np.clip(margin,0,1);q[oldmask]=teacherq
                ref=rec.save('teacher_fold_'+str(original.audit['inner_fold']),teacher_scores=cached['score'],teacher_q=teacherq,held_old_mask=oldmask.astype(float))
                ta.update(training_physical_ids=list(teacherproblem.original.audit['training_physical_ids']),
                    held_physical_ids=list(teacherproblem.original.audit['held_physical_ids']),old_inner_train_only=True,
                    source='INITIAL_R0_CACHE' if inherited is None else 'FROZEN_B_ADAPTER_OLD_INNER_HEAD',
                    state_ref=ref,head_audit=deepcopy(progress))
                progress={};kind=None
            else:ta.update(reason='SINGLE_TEACHER_CLASS_NO_WRONG_MARGIN',state_ref=None)
            p=replace(p,teacher_q=q,old_mask=oldmask,teacher_audit=ta);problems.append(p)
            if inherited is None and initial:initial[-1]['problem']=p
            audit['teacher_folds'].append(ta);audit['completed_stages'].append(p.audit)
            payload=dict(context or {});payload.update(p.audit);ch._emit(log_callback,'FCR_INNER_PREPARED',payload)
        original=ch._make_problem(base.background,base.auxiliary,np.asarray(base.labels,dtype=int),base.ids,
            base.classes,np.ones(len(base.ids),dtype=bool),None)
        full=_problem(original,dictionary);audit['prepared_distance_evaluation_count']+=2
        extra=sum(sum(v.nbytes for ctx in (p.train_context,p.held_context) for v in ctx.values())+
            p.teacher_q.nbytes+p.old_mask.nbytes for p in problems+[full])
        extra+=sum(getattr(original,n).nbytes for n in ('train_b','train_a','held_b','held_a','train_labels','held_labels','d0','cross_d0'))
        cachebytes=sum(_cache_bytes(c) for c in initial)
        audit.update(inner_folds=[p.audit for p in problems],keep_available=len(base.old_classes)>1,
            prepare_seconds=time.perf_counter()-start,preparation_seconds=time.perf_counter()-start,
            prepared_numeric_state_bytes=int(audit['prepared_numeric_state_bytes']+extra+cachebytes+H.nbytes+W.nbytes+singular.nbytes+_V0.nbytes),
            initial_forward_cache_numeric_bytes=cachebytes,
            teacher_record_numeric_bytes=sum(sum(v.nbytes for v in r.values()) for r in rec.records.values()))
        return FunctionCoordinateTraining(base,tuple(problems),full,tuple(initial),H,W,singular,W.shape[1],rec.records,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):audit['failed_coordinate_stage']=exc.audit_dict();audit['latent_svd_count']+=exc.audit_dict().get('latent_svd_count',0)
        if progress:
            prefix='initial_inner' if kind=='initial' else 'teacher'
            audit[prefix+'_head_fit_count']+=progress.get('head_fit_count',0);audit[prefix+'_factorization_count']+=progress.get('factorization_count',0)
            if kind=='initial':audit['inner_head_fit_count']+=progress.get('head_fit_count',0);audit['inner_factorization_count']+=progress.get('factorization_count',0)
            audit['fcr_forward_evaluation_count']+=progress.get('fcr_forward_evaluation_count',0)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),current_teacher_progress=progress)
        error=NumericalFailure(str(exc),audit);error.records=rec.records;raise error from exc

@dataclass(frozen=True)
class ObjectiveCache:
    prepared: object
    Z: np.ndarray
    anchor_U: np.ndarray
    U: np.ndarray
    folds: tuple
    def __post_init__(self):
        for key in ('Z','anchor_U','U'):object.__setattr__(self,key,_readonly(getattr(self,key)))


def evaluate_function_coordinate_objective(prepared,Z,anchor_U,*,gradient=True,forward_cache=None):
    U=reconstruct_U(Z,anchor_U,prepared.W);Z=np.asarray(Z,dtype=float);anchor_U=_parameter(anchor_U);n=len(prepared.ids);c=len(prepared.classes)
    use_initial=forward_cache is None and bool(prepared.initial_folds) and np.all(U==0)
    reuse=forward_cache is not None or use_initial
    if forward_cache is not None and (forward_cache.prepared is not prepared or not np.array_equal(forward_cache.U,U) or not np.array_equal(forward_cache.Z,Z) or not np.array_equal(forward_cache.anchor_U,anchor_U)):
        raise ValueError('FCR objective cache binding mismatch')
    prior=prepared.initial_folds if use_initial else (forward_cache.folds if forward_cache is not None else ())
    audit=dict(inner_objective_evaluation_count=int(forward_cache is None),inner_head_fit_count=0,inner_factorization_count=0,
        fcr_forward_evaluation_count=0,derivative_triangular_solve_count=0,task_derivative_triangular_solve_count=0,
        keep_derivative_triangular_solve_count=0,backward_evaluation_count=0,forward_cache_reused=reuse,
        initial_teacher_cache_reused=use_initial,inner_folds=[])
    folds=[];partial={};start=time.perf_counter()
    try:
        for i,p in enumerate(prepared.problems):
            partial={};r=prior[i] if reuse else _forward(p,U,partial);folds.append(r)
            if not reuse:audit['inner_head_fit_count']+=1;audit['inner_factorization_count']+=partial['factorization_count'];audit['fcr_forward_evaluation_count']+=1
            audit['inner_folds'].append(dict(r['audit'],head_fit_count=0 if reuse else 1,factorization_count=0 if reuse else r['audit']['factorization_count'],
                fcr_forward_evaluation_count=0 if reuse else 1,derivative_triangular_solve_count=0,
                task_derivative_triangular_solve_count=0,keep_derivative_triangular_solve_count=0))
            partial={}
        counts=np.zeros(c,dtype=int);task_sum=np.zeros(c);keep_sum=np.zeros(c);taskgs=[];keepgs=[]
        oldix=np.array([i for i,x in enumerate(prepared.classes) if x in prepared.old_classes]);available=bool(prepared.audit['keep_available'])
        for i,(p,r) in enumerate(zip(prepared.problems,folds)):
            labels=np.asarray(p.original.held_labels,dtype=int);tl,tg,margin=_hinge(r['score'],labels,1.)
            kl,kg,_=_hinge(r['score'],labels,p.teacher_q);mask=np.array(p.old_mask,dtype=bool,copy=True)
            if not available:mask[:]=False
            kl=kl*mask;kg=kg*mask[:,None]
            ns=np.bincount(labels,minlength=c);ts=np.bincount(labels,weights=tl,minlength=c);ks=np.bincount(labels,weights=kl,minlength=c)
            counts+=ns;task_sum+=ts;keep_sum+=ks;taskgs.append(tg);keepgs.append(kg)
            audit['inner_folds'][i].update(class_physical_counts=ns.tolist(),class_task_loss_sums=ts.tolist(),class_keep_loss_sums=ks.tolist(),
                held_training_correct_count=int(np.sum(r['score'].argmax(axis=1)==labels)),held_training_margin_mean=float(margin.mean()))
        if np.any(counts==0):raise ValueError('FCR objective requires complete inner-held coverage')
        taskmeans=task_sum/counts;keepmeans=keep_sum/counts;rt=_norm(taskmeans)/np.sqrt(c)
        rk=_norm(keepmeans[oldix])/np.sqrt(len(oldix)) if available else 0.
        taskweights=np.zeros(c) if rt==0 else taskmeans/(c*rt*counts)
        keepweights=np.zeros(c)
        if available and rk>0:keepweights[oldix]=keepmeans[oldix]/(len(oldix)*rk*counts[oldix])
        gU=np.zeros_like(U);aU=np.zeros_like(U);gZ=np.zeros_like(Z);aZ=np.zeros_like(Z)
        if gradient:
            audit['backward_evaluation_count']=1
            for i,(p,r) in enumerate(zip(prepared.problems,folds)):
                labels=np.asarray(p.original.held_labels,dtype=int);fa=audit['inner_folds'][i]
                partial={'derivative_triangular_solve_count':0};fa['current_adjoint_channel']='task'
                tu=_backward(r,taskgs[i]*taskweights[labels,None],partial)
                fa['task_derivative_triangular_solve_count']=partial['derivative_triangular_solve_count'];fa['task_adjoint_seconds']=partial.get('adjoint_seconds',0.)
                partial={};ku=np.zeros_like(U)
                if available:
                    partial={'derivative_triangular_solve_count':0};fa['current_adjoint_channel']='keep'
                    ku=_backward(r,keepgs[i]*keepweights[labels,None],partial)
                    fa['keep_derivative_triangular_solve_count']=partial['derivative_triangular_solve_count'];fa['keep_adjoint_seconds']=partial.get('adjoint_seconds',0.)
                partial={};fa['current_adjoint_channel']=None
                fa['derivative_triangular_solve_count']=fa['task_derivative_triangular_solve_count']+fa['keep_derivative_triangular_solve_count']
                gU+=tu+ku;aU+=ku
            gZ=gU@prepared.W+Z;aZ=aU@prepared.W
        for key in ('derivative_triangular_solve_count','task_derivative_triangular_solve_count','keep_derivative_triangular_solve_count'):
            audit[key]=sum(f[key] for f in audit['inner_folds'])
        prox=float(.5*_norm(Z)**2)
        audit.update(loss_task=float(rt),loss_keep=float(rk),loss_data=float(rt+rk),loss_proximal=prox,loss_total=float(rt+rk+prox),
            class_task_loss_means=taskmeans.tolist(),class_keep_loss_means=keepmeans.tolist(),class_held_counts=counts.tolist(),
            classes=list(prepared.classes),old_class_indices=oldix.tolist(),keep_available=available,
            objective_seconds=time.perf_counter()-start,forward_cache_numeric_bytes=sum(_cache_bytes(r) for r in folds),
            loss_scope='CLASS_RMS_TASK_PLUS_OLD_CLASS_RMS_TEACHER_KEEP_PLUS_PROXIMAL')
        _finite(gZ,aZ,audit['loss_total'])
        audit.update(_functional_measurement(prepared,U,anchor_U,Z))
        return audit['loss_total'],gZ,aZ,ch._safe(audit),ObjectiveCache(prepared,Z,anchor_U,U,tuple(folds))
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if len(folds)<len(prepared.problems):
            audit['inner_head_fit_count']+=partial.get('head_fit_count',0);audit['inner_factorization_count']+=partial.get('factorization_count',0)
            audit['fcr_forward_evaluation_count']+=partial.get('fcr_forward_evaluation_count',0)
        elif partial and audit['inner_folds']:
            fa=next(f for f in audit['inner_folds'] if f.get('current_adjoint_channel'))
            fa[fa['current_adjoint_channel']+'_derivative_triangular_solve_count']=partial.get('derivative_triangular_solve_count',0)
        for f in audit['inner_folds']:f['derivative_triangular_solve_count']=f['task_derivative_triangular_solve_count']+f['keep_derivative_triangular_solve_count']
        for key in ('derivative_triangular_solve_count','task_derivative_triangular_solve_count','keep_derivative_triangular_solve_count'):audit[key]=sum(f[key] for f in audit['inner_folds'])
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),current_progress=partial)
        raise NumericalFailure(str(exc),audit) from exc


@dataclass(frozen=True)
class FunctionCoordinateState:
    U: np.ndarray
    base_state: local.BranchLocalRidgeState
    original_background: np.ndarray
    original_auxiliary: np.ndarray
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for name in ('U','original_background','original_auxiliary','labels'):object.__setattr__(self,name,_readonly(getattr(self,name)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}));object.__setattr__(self,'audit',_freeze(self.audit));object.__setattr__(self,'records',_freeze(self.records))
    @property
    def V(self):return _V0
    @property
    def V0(self):return _V0
    @property
    def classes(self):return self.base_state.classes
    def audit_dict(self):return ch._safe(_plain(self.audit))
    def state_records(self):return {k:{name:np.array(v,copy=True) for name,v in record.items()} for k,record in self.records.items()}
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        b0,a0=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True);s=self.base_state
        if self.audit['identity_forward']:
            return local._score_local(b0,a0,self.original_background,self.original_auxiliary,s.alpha,s.reference_kernel,s.reference_self,s.center_mean,s.center_grand,s.bandwidth_tau,s.trace_scale)
        out=np.empty((len(b0),len(self.classes)))
        for i in range(len(b0)):
            b,a,_=_adapt(_context(b0[i:i+1],a0[i:i+1]),self.U)
            d=pt._joint_distances(b,a,s.support_background,s.support_auxiliary,b0[i:i+1],a0[i:i+1],self.original_background,self.original_auxiliary)
            cc=ch._cross_center(local._radial_minus_one(d,s.bandwidth_tau),s.reference_kernel,s.reference_self,s.center_mean,s.center_grand)
            out[i]=(s.trace_scale*cc[0])@s.alpha
        _finite(out);return out
    def predict(self,**features):return np.asarray(self.classes)[np.argmax(self.score(**features),axis=1)]



def guarded_direction(g,a,remaining):
    gn=_norm(g);an=_norm(a);bound=float(max(0.,remaining)/.125)
    d=np.zeros_like(g) if gn==0 else -g/gn
    dot=float(np.sum(a*d));active=an>0 and dot>bound
    if active:d=d-(a/an)*((dot-bound)/an)
    after=float(np.sum(a*d));dn=_norm(d);_finite(d,gn,an)
    if dn>1+128*_EPS*max(1,g.size):raise FloatingPointError('FCR_GUARD_DIRECTION_NORM_EXCEEDED')
    return d,dict(gradient_norm=gn,keep_gradient_norm=an,direction_norm=dn,guard_active=bool(active),
        guard_dot_before=dot,guard_dot_after=after,guard_bound=bound)

_COUNTERS=('inner_objective_evaluation_count','inner_head_fit_count','inner_factorization_count',
    'fcr_forward_evaluation_count','derivative_triangular_solve_count','task_derivative_triangular_solve_count',
    'keep_derivative_triangular_solve_count','backward_evaluation_count')
COUNTERS=_COUNTERS
def _cost(audit,obj):
    for key in _COUNTERS:audit[key]+=obj.get(key,0)

def _annotate_kernel_changes(obj,cache,initial):
    for info,current,first in zip(obj['inner_folds'],cache.folds,initial.folds):
        info['kernel_change_from_initial']=_norm(current['kernel']-first['kernel'])
        denom=_norm(first['kernel'])
        info['kernel_relative_change_from_initial']=info['kernel_change_from_initial']/denom if denom else None
        info['kernel_change_unmeasured_reason']=None
        delta=current['score']-first['score']
        info['held_score_change_rms']=float(_norm(delta)/np.sqrt(delta.size)) if delta.size else None
        info['held_winner_change_count']=int(np.sum(current['score'].argmax(1)!=first['score'].argmax(1))) if len(delta) else 0


def fit_function_coordinate_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None,state_callback=None):
    if mode not in ('B','C_seq','C_reset_init'):raise ValueError('Unknown FCR mode')
    if (mode=='B')!=(prepared.inherited is None):raise ValueError('B/C FCR lineage mismatch')
    if prepared.inherited is not None and set(prepared.classes)==set(prepared.old_classes):return prepared.inherited
    start=time.perf_counter();binding={};baseline_state=ch._canonical_baseline(prepared,baseline_state,binding);rec=_Recorder(state_callback)
    au=prepared.inherited.U.copy() if mode=='C_seq' else np.zeros((736,8))
    Z=np.zeros((736,prepared.r))
    U=au.copy();audit=dict(mode=mode,classes=list(prepared.classes),old_classes=list(prepared.old_classes),
        training_physical_ids=list(prepared.ids),train_k=prepared.audit['train_k'],fcr_stage_count=1,preparation=prepared.audit_dict(),
        no_information=prepared.audit['no_information'],optimizer_steps=0,optimizer_iterations=0,accepted_trial_count=0,rejected_trial_count=0,
        trial_count=0,trial_attempt_count=0,nonzero_projected_update_count=0,final_head_fit_count=0,final_factorization_count=0,
        steps=[],trials=[],gradients=[],initial_objective=None,final_objective=None,trainable_parameter_count=736*prepared.r,maximum_trainable_parameter_count=5888,latent_rank=prepared.r,
        source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',optimizer_state_reset=True,optimizer_state_bytes=0,
        **{k:0 for k in _COUNTERS},**binding)
    audit['initialization_state_ref']=rec.save('initial',Z=Z,U=U,anchor_U=au,W=prepared.W,singular_values=prepared.singular_values)
    finalprogress={};cache=None
    def event(name,record):
        payload=dict(_plain(prepared.audit));payload.update(record,mode=mode);ch._emit(log_callback,name,payload)
    try:
        if audit['no_information']:
            audit['stop_reason']=prepared.audit['no_information_reason'];audit['no_update_reason']=audit['stop_reason']
            audit.update(keep_available=prepared.audit['keep_available'],keep_anchor_risk=None,keep_slack=None,keep_limit=None)
        else:
            loss,_,_,obj,cache=evaluate_function_coordinate_objective(prepared,Z,au,gradient=False);_cost(audit,obj)
            initial_cache=cache;_annotate_kernel_changes(obj,cache,initial_cache)
            k=prepared.audit['train_k'];cp=len(prepared.old_classes);available=bool(prepared.audit['keep_available'])
            slack=float(1/(2*k*np.sqrt(cp))) if available else None;limit=float(obj['loss_keep']+slack) if available else None
            audit.update(initial_objective=deepcopy(obj),keep_available=available,keep_anchor_risk=obj['loss_keep'],keep_slack=slack,keep_limit=limit)
            event('FCR_INITIAL',dict(objective=obj,state_ref=audit['initialization_state_ref'],keep_anchor_risk=obj['loss_keep'],keep_slack=slack,keep_limit=limit))
            audit['stop_reason']='MAX_ITERATIONS'
            for iteration in range(1,5):
                begin=time.perf_counter();audit['optimizer_iterations']+=1
                loss,g,a,obj,cache=evaluate_function_coordinate_objective(prepared,Z,au,gradient=True,forward_cache=cache);_cost(audit,obj)
                _annotate_kernel_changes(obj,cache,initial_cache)
                direction,guard=guarded_direction(g,a,limit-obj['loss_keep'] if available else 0.)
                ref=rec.save('gradient_'+str(iteration),Z=Z,U=U,anchor_U=au,W=prepared.W,singular_values=prepared.singular_values,g_Z=g,keep_g_Z=a,d_Z=direction)
                gr=dict(iteration=iteration,state_ref=ref,objective=deepcopy(obj),**guard);audit['gradients'].append(gr);event('FCR_GRADIENT',gr)
                if guard['gradient_norm']==0:audit['stop_reason']='ZERO_GRADIENT';break
                if guard['direction_norm']==0:audit['stop_reason']='ZERO_GUARDED_DIRECTION';break
                accepted=False
                for trial in range(1,4):
                    step=.125*.5**(trial-1);tz=Z+step*direction;tu=reconstruct_U(tz,au,prepared.W);dz=tz-Z
                    dn=_norm(dz)
                    if dn==0:audit['stop_reason']='ZERO_PROJECTED_STEP';break
                    audit['trial_attempt_count']+=1;tref=rec.save('trial_'+str(iteration)+'_'+str(trial),Z=tz,U=tu,anchor_U=au,W=prepared.W,singular_values=prepared.singular_values)
                    audit['current_trial']=dict(iteration=iteration,trial=trial,step_size=step,state_ref=tref)
                    tl,_,_,to,tc=evaluate_function_coordinate_objective(prepared,tz,au,gradient=False);_cost(audit,to)
                    _annotate_kernel_changes(to,tc,initial_cache)
                    acceptance=_trial_acceptance(loss,tl,float(np.sum(g*dz)),to['loss_keep'],limit)
                    accepted=acceptance['accepted']
                    tr=dict(iteration=iteration,trial=trial,step_size=step,state_ref=tref,gradient_state_ref=ref,
                        loss_before=loss,loss_after=tl,update_norm=dn,objective=deepcopy(to),**acceptance)
                    audit['trials'].append(tr);audit['trial_count']+=1;audit['current_trial']=None
                    audit['accepted_trial_count' if accepted else 'rejected_trial_count']+=1;event('FCR_TRIAL',tr)
                    if accepted:
                        Z,U=tz,tu;cache=tc;loss=tl;audit['optimizer_steps']+=1;audit['nonzero_projected_update_count']+=1
                        sr=dict(step=audit['optimizer_steps'],iteration=iteration,trial=trial,state_ref=tref,gradient_state_ref=ref,
                            step_size=step,learning_rate=step,loss_before=tr['loss_before'],loss_after=loss,update_norm=dn,
                            step_seconds=time.perf_counter()-begin,objective=deepcopy(to))
                        audit['steps'].append(sr);event('FCR_STEP',sr);break
                if not accepted:
                    if audit['stop_reason']!='ZERO_PROJECTED_STEP':audit['stop_reason']='ARMIJO_OR_KEEP_BUDGET_EXHAUSTED'
                    break
            _,_,_,final,_=evaluate_function_coordinate_objective(prepared,Z,au,gradient=False,forward_cache=cache);_annotate_kernel_changes(final,cache,initial_cache);audit['final_objective']=final
        identity=bool(np.all(U==0)) or prepared.full_problem.original.tau0==0
        if identity and baseline_state is not None:base=baseline_state;ff=base.audit_dict()['final_fit']
        else:
            r=_forward(prepared.full_problem,U,finalprogress);audit['final_head_fit_count']=1;audit['final_factorization_count']=r['audit']['factorization_count']
            audit['fcr_forward_evaluation_count']+=1;ff=r['audit'];alpha,ref,rs,mean,grand,tau,gamma=r['parts']
            base=local.BranchLocalRidgeState(r['b'],r['a'],alpha,ref,rs,mean,grand,tau,gamma,prepared.classes,ff)
        headbytes=sum(getattr(base,k).nbytes for k in ('support_background','support_auxiliary','alpha','reference_kernel','center_mean'))+16+8*(base.bandwidth_tau is not None)+8*(base.trace_scale is not None)
        lineage=prepared.background.nbytes+prepared.auxiliary.nbytes+prepared.labels.nbytes+sum(v.nbytes for v in prepared.raw.values())
        audit.update(final_state_ref=rec.save('final',Z=Z,U=U,anchor_U=au,W=prepared.W,singular_values=prepared.singular_values),
            parameter_U_norm=_norm(U),parameter_V_delta_norm=0.,coordinate_norm=_norm(Z),parameter_changed_from_anchor=bool(np.any(U!=au)),
            u_changed_from_anchor=bool(np.any(U!=au)),u_update_norm=_norm(U-au),
            identity_forward=identity or base.trace_scale is None,final_fit=ff,head_state_bytes=int(headbytes),adapter_state_bytes=int(U.nbytes+_V0.nbytes),
            lineage_state_bytes=int(lineage),persistent_state_bytes=int(headbytes+lineage+U.nbytes+_V0.nbytes),
            retained_vector_record_bytes=sum(sum(v.nbytes for v in r.values()) for r in rec.records.values()),
            state_byte_scope='deployed_head_U_V_original_support_raw_labels_excludes_audit_training_records_Python',
            factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'],
            status='FCR_STAGE_COMPLETE',fit_seconds=time.perf_counter()-start,config=deepcopy(_CONFIG))
        audit.update(_functional_measurement(prepared,U,au,Z))
        event('FCR_FINAL',audit)
        return FunctionCoordinateState(U,base,prepared.background,prepared.auxiliary,prepared.raw,prepared.labels,prepared.ids,rec.records,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):audit['failed_stage']=exc.audit_dict();_cost(audit,audit['failed_stage'])
        if finalprogress and not audit['final_head_fit_count']:
            audit['failed_final_kernel_progress']=finalprogress;audit['final_head_fit_count']=finalprogress.get('head_fit_count',0)
            audit['final_factorization_count']=finalprogress.get('factorization_count',0);audit['fcr_forward_evaluation_count']+=finalprogress.get('fcr_forward_evaluation_count',0)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),accepted_state_ref=rec.save('failure_accepted',Z=Z,U=U,anchor_U=au,W=prepared.W,singular_values=prepared.singular_values),
            fit_seconds=time.perf_counter()-start,factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'])
        error=NumericalFailure(str(exc),audit);error.records=rec.records;raise error from exc
