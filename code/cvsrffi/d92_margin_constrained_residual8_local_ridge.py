"""MC-Residual8: physical-support supervision through a two-geometry ridge head."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import time
import numpy as np
from scipy.special import erf
from scipy.linalg import solve_triangular
from . import d92_joint_channel_local_ridge as ch
from . import d92_prototype_transport_local_ridge as pt
from . import d92_branch_local_ridge as local
from . import d92_branch_interaction as interaction
from .d92_branch_ridge import _freeze, _plain, _readonly

_EPS=np.finfo(np.float64).eps
_NAMES=ch._NAMES
_SLICES=ch._SLICES
_norm=ch._norm
_finite=ch._finite
NumericalFailure=ch.NumericalFailure
FROZEN_CONFIG=dict(
    schema='d92_margin_constrained_residual8_local_ridge_v1',method='D92-MCResidual8LocalRidge-v1',
    base_algorithm=deepcopy(local.FROZEN_CONFIG),
    channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000),
    allowed_scenarios=['practical_high','practical_mid','practical_low_urban'],
    arms=['local_ridge','mc_seq','mc_reset_init'],rank=8,input_dim=736,trainable_parameter_count=11776,
    adapter='cross_branch_exact_GELU_tangent_saturated_norm_preserving_residual',
    input='concat_original_block_unit_directions_over_sqrt5',block_dimensions=[160,96,160,160,160],
    kappa=.25,U_frobenius_bound=1.,V_minus_V0_frobenius_bound=1.,
    initialization='U_zero_V_first8_orthonormal_DCT_rows',joint_original_distance_weight=.5,
    trace_target='original_inner_train_interaction_centered_trace',
    objective='RMS_class_task_half_squared_unit_margin+RMS_old_class_teacher_deficit_half_squared_margin+parameter_anchor_norm2_over_2N',
    task_weight=1.,keep_weight=1.,proximal_physical_sum_coefficient=1.,
    teacher_B='first_R0_inner_forward_cache',teacher_C='frozen_B_adapter_old_inner_train_only_head',
    teacher_margin='clip_true_minus_max_wrong_teacher_old_score_0_1',
    keep_slack='1_over_2k_sqrt_Cprev',keep_limit='keep_risk_at_stage_anchor_plus_fixed_physical_slack',
    optimizer='normalized_gradient_keep_halfspace_product_ball_Armijo_and_keep',
    objective_nonincrease_required=True,
    max_iterations=4,max_trials=3,initial_step_size=.125,backtrack_factor=.5,armijo_coefficient=1e-4,
    comparison_tolerance_multiplier=128,comparison_tolerance='128*eps64*max(1,abs(compared_scalars))',
    parameter_validation_tolerance='128*eps64*11776',
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3',
    initialization_B='zero_U_DCT_V',initialization_C_seq='inherit_B_U_V',initialization_C_reset_init='zero_U_DCT_V',
    reset_teacher='same_B_teacher_as_seq',zero_U='bitwise_R0_forward_live_U_derivative',
    zero_bandwidth='original_equivalence_kernel_zero_derivative',
    tie_gradient='uniform_exact_wrong_max;uniform_exact_nearest_and_tied_median_groups',
    no_information='physical_K1_or_single_class_preserve_anchor',n0='reuse_corresponding_B',
    query_decision_policy='per_sample_all_registered_classes',source_inputs=False,query_fit=False,
    phase1_frozen=True,encoder_backward=False,parameter_search=False,dtype='float64',
    vector_logging='lossless_float64_state_callback_or_in_memory_records_no_JSON_vectors')
_CONFIG=deepcopy(FROZEN_CONFIG)


def initial_parameters():
    j=np.arange(8,dtype=float)[:,None];k=np.arange(736,dtype=float)[None,:]
    V=np.sqrt(2/736)*np.cos(np.pi*j*(k+.5)/736);V[0]=1/np.sqrt(736)
    return np.zeros((736,8)),V


_U0,_V0=initial_parameters()
_V0=_readonly(_V0)


def _parameters(U,V):
    U=np.asarray(U,dtype=float);V=np.asarray(V,dtype=float)
    if U.shape!=(736,8) or V.shape!=(8,736):raise ValueError('U/V shape mismatch')
    _finite(U,V);tol=128*_EPS*11776
    if _norm(U)>1+tol or _norm(V-_V0)>1+tol:raise ValueError('U/V outside product Frobenius balls')
    return U,V


def project_parameters(U,V):
    U=np.asarray(U,dtype=float).copy();V=np.asarray(V,dtype=float).copy();_finite(U,V)
    un=_norm(U);dv=V-_V0;vn=_norm(dv)
    if un>1:U/=un
    if vn>1:V=np.asarray(_V0)+dv/vn
    return _parameters(U,V)


def _context(b,a):
    original=np.concatenate((b,a),axis=1);w,rho=pt._unit_blocks(original)
    return dict(original=original,w=w,rho=rho,x=w/np.sqrt(5.))


def _gelu(z):return .5*z*(1+erf(z/np.sqrt(2.)))


def _gelu_prime(z):return .5*(1+erf(z/np.sqrt(2.)))+z*np.exp(-.5*z*z)/np.sqrt(2*np.pi)


def _adapt(context,U,V,*,derivative_cache=False):
    U,V=_parameters(U,V);x=context['x'];w=context['w'];rho=context['rho'];n=len(x)
    output=np.empty_like(context['original']);zall=np.empty((n,8));hall=np.empty((n,8))
    tall=np.empty_like(output);dall=np.empty((n,5));yall=np.empty_like(output);normall=np.empty((n,5))
    kappa=.25
    # Identical sample-wise sums in train, teacher and score: no batch GEMM.
    for i in range(n):
        z=np.sum(V*x[i,None,:],axis=1);h=_gelu(z);v=np.sum(U*h[None,:],axis=1)
        zall[i]=z;hall[i]=h
        for l,sl in enumerate(_SLICES):
            wi=w[i,sl];t=v[sl]-wi*float(np.sum(wi*v[sl]));q=_norm(t)
            den=float(np.sqrt(kappa*kappa+q*q));delta=kappa*t/den;s=wi+delta;sn=_norm(s)
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


def _adapter_vjp(context,U,V,gb,ga,cache):
    g=np.concatenate((gb,ga),axis=1);gU=np.zeros_like(U);gV=np.zeros_like(V);kappa=.25
    for i in range(len(g)):
        gv=np.zeros(736)
        for l,sl in enumerate(_SLICES):
            if context['rho'][i,l]==0:continue
            y=cache['y'][i,sl];w=context['w'][i,sl];t=cache['t'][i,sl];den=cache['den'][i,l]
            gd=context['rho'][i,l]/cache['snorm'][i,l]*(g[i,sl]-y*float(np.sum(y*g[i,sl])))
            gt=kappa/den*(gd-t*(float(np.sum(t*gd))/(den*den)))
            gv[sl]=gt-w*float(np.sum(w*gt))
        gU+=gv[:,None]*cache['h'][i,None,:]
        gz=np.sum(U*gv[:,None],axis=0)*_gelu_prime(cache['z'][i])
        gV+=gz[:,None]*context['x'][i,None,:]
    _finite(gU,gV);return gU,gV


@dataclass(frozen=True)
class MCProblem:
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


def _problem(original):
    n=len(original.held_labels)
    return MCProblem(original,_context(original.train_b,original.train_a),_context(original.held_b,original.held_a),
        np.zeros(n),np.zeros(n),{})


def _forward(problem,U,V,progress=None):
    started=time.perf_counter();p=problem.original;n=len(p.train_labels);c=len(p.classes)
    audit={} if progress is None else progress
    audit.update(problem.audit,head_fit_count=1,factorization_count=0,derivative_triangular_solve_count=0,mc_forward_evaluation_count=1)
    b,a,bc=_adapt(problem.train_context,U,V,derivative_cache=True)
    hb,ha,hc=_adapt(problem.held_context,U,V,derivative_cache=True)
    identity=bool(np.all(U==0)) or p.tau0==0
    if identity:distance,cross=p.d0,p.cross_d0
    else:
        distance=pt._joint_distances(b,a,b,a,p.train_b,p.train_a,p.train_b,p.train_a,symmetric=True,d0=p.d0)
        cross=pt._joint_distances(hb,ha,b,a,p.held_b,p.held_a,p.train_b,p.train_a,d0=p.cross_d0)
    tau,weights=ch._bandwidth_weights(distance,np.asarray(p.train_labels,dtype=int))
    if p.tau0 is not None and p.tau0>0 and tau<=0:raise FloatingPointError('MC_POSITIVE_BANDWIDTH_UNRESOLVED')
    target=np.eye(c)[np.asarray(p.train_labels,dtype=int)]-1/c;alpha=np.zeros((n,c));reference=mean=np.zeros(n)
    refself=grand=0.;gamma=None;kernel=np.zeros((n,n));score=np.zeros((len(p.held_labels),c));tol=128*_EPS*max(n,c)
    audit.update(bandwidth_tau=tau,original_bandwidth_tau=p.tau0,interaction_centered_trace=p.s0,
        radial_centered_trace=None,trace_scale=None,normal_equation_residual=0.,trace_relative_error=0.,numerical_tolerance=tol,identity_forward=identity)
    saved={}
    if c>1 and p.s0>0:
        raw=local._radial_minus_one(distance,tau);radial=local._radial(distance,tau);sr=float(-2*raw[np.triu_indices(n,1)].sum()/n)
        if sr<=0:raise FloatingPointError('MC_NONPOSITIVE_RADIAL_TRACE')
        gamma=p.s0/sr;center,reference,refself,mean,grand=interaction._center_kernel(raw)
        kernel=gamma*center;matrix=kernel+np.eye(n);audit['factorization_count']=1;chol=np.linalg.cholesky(matrix)
        alpha=np.ascontiguousarray(solve_triangular(chol.T,solve_triangular(chol,target,lower=True),lower=False))
        cc=ch._cross_center(local._radial_minus_one(cross,tau),reference,refself,mean,grand);crosskernel=gamma*cc
        score=np.stack([(gamma*cc[i])@alpha for i in range(len(p.held_labels))]) if len(p.held_labels) else score
        residual=_norm(matrix@alpha-target)/((1+p.s0)*_norm(alpha)+_norm(target));terr=abs(float(np.trace(kernel))-p.s0)/p.s0
        if residual>tol or terr>tol:raise FloatingPointError('MC_HEAD_RESIDUAL_EXCEEDED')
        audit.update(normal_equation_residual=residual,trace_relative_error=terr,radial_centered_trace=sr,trace_scale=gamma)
        saved.update(chol=chol,crosskernel=crosskernel,radial=radial,sr=sr)
    fit=kernel@alpha;err=fit-target
    audit.update(head_training_loss_data=float(.5*np.sum(err*err)),head_training_loss_ridge=float(.5*np.sum(alpha*fit)),
        sample_weight=1.,ridge_coefficient=1.,status='CLOSED_FORM_SOLVED' if c>1 and p.s0>0 else 'EXACT_ZERO_CLASSIFIER',forward_seconds=time.perf_counter()-started)
    audit['head_training_loss_total']=audit['head_training_loss_data']+audit['head_training_loss_ridge'];_finite(score,alpha)
    return dict(problem=problem,U=np.asarray(U).copy(),V=np.asarray(V).copy(),score=score,b=b,a=a,hb=hb,ha=ha,bc=bc,hc=hc,
        distance=distance,cross=cross,weights=weights,kernel=kernel,alpha=alpha,tau=tau,gamma=gamma,
        parts=(alpha,reference,refself,mean,grand,tau,gamma),audit=audit,**saved)


def _backward(cache,gscore,progress):
    start=time.perf_counter();p=cache['problem']
    if cache['gamma'] is None or cache['tau']==0 or not len(gscore):
        progress['adjoint_seconds']=time.perf_counter()-start;return np.zeros((736,8)),np.zeros((8,736))
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
    gu,gv=_adapter_vjp(p.train_context,cache['U'],cache['V'],gb+gtb,ga+gta,cache['bc'])
    hu,hv=_adapter_vjp(p.held_context,cache['U'],cache['V'],ghb,gha,cache['hc'])
    progress['adjoint_seconds']=time.perf_counter()-start;return gu+hu,gv+hv


class _Recorder:
    def __init__(self,callback):self.callback=callback;self.records={};self.keys=set()
    def save(self,key,**arrays):
        if key in self.keys:raise ValueError('Duplicate state record key '+key)
        self.keys.add(key);values={k:_readonly(v) for k,v in arrays.items()}
        metadata={k:dict(shape=list(v.shape),dtype=str(v.dtype),nbytes=int(v.nbytes)) for k,v in values.items()}
        if self.callback is None:
            self.records[key]=_freeze(values);ref=dict(storage='in_memory',key=key,arrays=metadata)
        else:
            ref=ch._safe(self.callback(key,values))
            if not isinstance(ref,dict):raise ValueError('state_callback must return JSON mapping')
            ref=dict(ref,key=key,arrays=metadata)
        return ref


@dataclass(frozen=True)
class MCTraining:
    base: ch.ChannelTraining
    problems: tuple
    full_problem: MCProblem
    initial_folds: tuple
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        object.__setattr__(self,'audit',_freeze(self.audit));object.__setattr__(self,'records',_freeze(self.records))
    def __getattr__(self,name):return getattr(self.base,name)
    def audit_dict(self):return ch._safe(_plain(self.audit))
    def state_records(self):return {k:{name:np.array(v,copy=True) for name,v in record.items()} for k,record in self.records.items()}


def _teacher_problem(base,original):
    old=base.old_classes;mask=np.array([base.classes[int(y)] in old for y in base.labels])
    ids=tuple(pid for i,pid in enumerate(base.ids) if mask[i]);trainset=set(original.audit['training_physical_ids'])
    y=np.array([old.index(base.classes[int(v)]) for v in base.labels[mask]])
    return _problem(ch._make_problem(base.background[mask],base.auxiliary[mask],y,ids,old,np.array([pid in trainset for pid in ids]),original.audit['inner_fold']))


def prepare_mc_residual8_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes,
        inherited=None,context=None,log_callback=None,state_callback=None):
    start=time.perf_counter();rec=_Recorder(state_callback)
    base=ch.prepare_channel_training(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local,
        support_labels=support_labels,support_ids=support_ids,classes=classes,old_classes=old_classes,inherited=inherited,context=context)
    audit=base.audit_dict();audit.pop('channel_preparation_count',None)
    audit.update(mc_preparation_count=1,inner_head_fit_count=0,inner_factorization_count=0,initial_inner_head_fit_count=0,
        initial_inner_factorization_count=0,teacher_head_fit_count=0,teacher_factorization_count=0,
        teacher_score_evaluation_count=0,teacher_score_physical_count=0,mc_forward_evaluation_count=0,
        prepared_distance_evaluation_count=2*len(base.problems),teacher_folds=[],completed_stages=[])
    problems=[];initial=[];progress={};kind=None
    try:
        n0=inherited is not None and set(base.classes)==set(base.old_classes)
        for original in (() if n0 else base.problems):
            p=_problem(original);oldmask=np.array([base.classes[int(v)] in base.old_classes for v in original.held_labels])
            q=np.zeros(len(oldmask));ta=dict(available=len(base.old_classes)>1,old_classes=list(base.old_classes))
            if len(base.old_classes)>1:
                progress={};kind='initial' if inherited is None else 'teacher'
                teacherproblem=p if inherited is None else _teacher_problem(base,original)
                U,V=initial_parameters() if inherited is None else (inherited.U,inherited.V)
                cached=_forward(teacherproblem,U,V,progress)
                if inherited is None:
                    initial.append(cached);audit['initial_inner_head_fit_count']+=1;audit['initial_inner_factorization_count']+=progress['factorization_count']
                    audit['inner_head_fit_count']+=1;audit['inner_factorization_count']+=progress['factorization_count']
                else:
                    audit['teacher_head_fit_count']+=1;audit['teacher_factorization_count']+=progress['factorization_count']
                audit['mc_forward_evaluation_count']+=1;audit['teacher_score_evaluation_count']+=1
                audit['teacher_score_physical_count']+=len(teacherproblem.original.held_labels)
                _,_,margin=ch._margin_loss(cached['score'],np.asarray(teacherproblem.original.held_labels,dtype=int))
                teacherq=np.clip(margin,0,1);q[oldmask]=teacherq
                ref=rec.save('teacher_fold_'+str(original.audit['inner_fold']),teacher_scores=cached['score'],teacher_q=teacherq,held_old_mask=oldmask.astype(float))
                ta.update(training_physical_ids=list(teacherproblem.original.audit['training_physical_ids']),
                    held_physical_ids=list(teacherproblem.original.audit['held_physical_ids']),
                    old_inner_train_only=True,source='INITIAL_R0_CACHE' if inherited is None else 'FROZEN_B_ADAPTER_OLD_INNER_HEAD',
                    state_ref=ref,head_audit=deepcopy(progress))
                progress={};kind=None
            else:ta.update(reason='SINGLE_TEACHER_CLASS_NO_WRONG_MARGIN',state_ref=None)
            p=replace(p,teacher_q=q,old_mask=oldmask,teacher_audit=ta);problems.append(p)
            if inherited is None and initial:initial[-1]['problem']=p
            audit['teacher_folds'].append(ta);audit['completed_stages'].append(p.audit)
            payload=dict(context or {});payload.update(p.audit);ch._emit(log_callback,'MC_INNER_PREPARED',payload)
        original=ch._make_problem(base.background,base.auxiliary,np.asarray(base.labels,dtype=int),base.ids,base.classes,np.ones(len(base.ids),dtype=bool),None)
        full=_problem(original);audit['prepared_distance_evaluation_count']+=2
        audit['prepared_distance_evaluation_count']+=2*len(base.problems) if inherited is not None and len(base.old_classes)>1 else 0
        extra=sum(sum(v.nbytes for ctx in (p.train_context,p.held_context) for v in ctx.values())+p.teacher_q.nbytes+p.old_mask.nbytes for p in problems+[full])
        extra+=sum(getattr(original,n).nbytes for n in ('train_b','train_a','held_b','held_a','train_labels','held_labels','d0','cross_d0'))
        cachebytes=sum(_cache_bytes(c) for c in initial)
        audit.update(inner_folds=[p.audit for p in problems],keep_available=len(base.old_classes)>1,
            prepare_seconds=time.perf_counter()-start,preparation_seconds=time.perf_counter()-start,
            prepared_numeric_state_bytes=int(audit['prepared_numeric_state_bytes']+extra+cachebytes+_V0.nbytes),
            initial_forward_cache_numeric_bytes=cachebytes,teacher_record_numeric_bytes=sum(sum(v.nbytes for v in r.values()) for r in rec.records.values()))
        return MCTraining(base,tuple(problems),full,tuple(initial),rec.records,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if progress:
            prefix='initial_inner' if kind=='initial' else 'teacher'
            audit[prefix+'_head_fit_count']+=progress.get('head_fit_count',0);audit[prefix+'_factorization_count']+=progress.get('factorization_count',0)
            if kind=='initial':audit['inner_head_fit_count']+=progress.get('head_fit_count',0);audit['inner_factorization_count']+=progress.get('factorization_count',0)
            audit['mc_forward_evaluation_count']+=progress.get('mc_forward_evaluation_count',0)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),current_teacher_progress=progress)
        raise NumericalFailure(str(exc),audit) from exc


def _cache_bytes(cache):
    seen=set();total=0
    def visit(x):
        nonlocal total
        if isinstance(x,np.ndarray) and id(x) not in seen:seen.add(id(x));total+=x.nbytes
        elif isinstance(x,dict):
            for k,v in x.items():
                if k not in ('problem','audit'):visit(v)
        elif isinstance(x,tuple):
            for v in x:visit(v)
    visit(cache);return int(total)


@dataclass(frozen=True)
class ObjectiveCache:
    prepared: object
    U: np.ndarray
    V: np.ndarray
    folds: tuple
    def __post_init__(self):
        object.__setattr__(self,'U',_readonly(self.U));object.__setattr__(self,'V',_readonly(self.V))


def _hinge(score,labels,threshold):
    _,_,margin=ch._margin_loss(score,labels);n,c=score.shape;grad=np.zeros_like(score)
    if c==1:return np.zeros(n),grad,margin
    deficit=np.maximum(0.,np.asarray(threshold)-margin)
    wrong=score.copy();wrong[np.arange(n),labels]=-np.inf;maximum=wrong.max(axis=1)
    for i in range(n):
        tied=wrong[i]==maximum[i];grad[i,tied]=deficit[i]/int(tied.sum());grad[i,labels[i]]=-deficit[i]
    return .5*deficit*deficit,grad,margin


def evaluate_mc_objective(prepared,U,V,anchor_U,anchor_V,*,gradient=True,forward_cache=None):
    U,V=_parameters(U,V);anchor_U,anchor_V=_parameters(anchor_U,anchor_V);n=len(prepared.ids);c=len(prepared.classes)
    use_initial=forward_cache is None and bool(prepared.initial_folds) and np.all(U==0) and np.array_equal(V,_V0)
    reuse=forward_cache is not None or use_initial
    if forward_cache is not None and (forward_cache.prepared is not prepared or not np.array_equal(forward_cache.U,U) or not np.array_equal(forward_cache.V,V)):
        raise ValueError('MC objective cache binding mismatch')
    prior=prepared.initial_folds if use_initial else (forward_cache.folds if forward_cache is not None else ())
    audit=dict(inner_objective_evaluation_count=int(forward_cache is None),inner_head_fit_count=0,inner_factorization_count=0,
        mc_forward_evaluation_count=0,derivative_triangular_solve_count=0,task_derivative_triangular_solve_count=0,
        keep_derivative_triangular_solve_count=0,backward_evaluation_count=0,forward_cache_reused=reuse,
        initial_teacher_cache_reused=use_initial,inner_folds=[])
    folds=[];partial={};start=time.perf_counter()
    try:
        for i,p in enumerate(prepared.problems):
            partial={};r=prior[i] if reuse else _forward(p,U,V,partial);folds.append(r)
            if not reuse:audit['inner_head_fit_count']+=1;audit['inner_factorization_count']+=partial['factorization_count'];audit['mc_forward_evaluation_count']+=1
            audit['inner_folds'].append(dict(r['audit'],head_fit_count=0 if reuse else 1,factorization_count=0 if reuse else r['audit']['factorization_count'],
                mc_forward_evaluation_count=0 if reuse else 1,derivative_triangular_solve_count=0,
                task_derivative_triangular_solve_count=0,keep_derivative_triangular_solve_count=0))
            partial={}
        counts=np.zeros(c,dtype=int);task_sum=np.zeros(c);keep_sum=np.zeros(c);taskgs=[];keepgs=[]
        oldix=np.array([i for i,x in enumerate(prepared.classes) if x in prepared.old_classes]);available=bool(prepared.audit['keep_available'])
        for i,(p,r) in enumerate(zip(prepared.problems,folds)):
            labels=np.asarray(p.original.held_labels,dtype=int);tl,tg,margin=_hinge(r['score'],labels,1.)
            kl,kg,_=_hinge(r['score'],labels,p.teacher_q);mask=np.asarray(p.old_mask,dtype=bool)
            if not available:mask[:]=False
            kl=kl*mask;kg=kg*mask[:,None]
            ns=np.bincount(labels,minlength=c);ts=np.bincount(labels,weights=tl,minlength=c);ks=np.bincount(labels,weights=kl,minlength=c)
            counts+=ns;task_sum+=ts;keep_sum+=ks;taskgs.append(tg);keepgs.append(kg)
            audit['inner_folds'][i].update(class_physical_counts=ns.tolist(),class_task_loss_sums=ts.tolist(),class_keep_loss_sums=ks.tolist(),
                held_training_correct_count=int(np.sum(r['score'].argmax(axis=1)==labels)),held_training_margin_mean=float(margin.mean()))
        if np.any(counts==0):raise ValueError('MC objective requires complete inner-held coverage')
        taskmeans=task_sum/counts;keepmeans=keep_sum/counts;rt=_norm(taskmeans)/np.sqrt(c)
        rk=_norm(keepmeans[oldix])/np.sqrt(len(oldix)) if available else 0.
        taskweights=np.zeros(c) if rt==0 else taskmeans/(c*rt*counts)
        keepweights=np.zeros(c)
        if available and rk>0:keepweights[oldix]=keepmeans[oldix]/(len(oldix)*rk*counts[oldix])
        gU=np.zeros_like(U);gV=np.zeros_like(V);aU=np.zeros_like(U);aV=np.zeros_like(V)
        if gradient:
            audit['backward_evaluation_count']=1
            for i,(p,r) in enumerate(zip(prepared.problems,folds)):
                labels=np.asarray(p.original.held_labels,dtype=int);fa=audit['inner_folds'][i]
                partial={'derivative_triangular_solve_count':0};fa['current_adjoint_channel']='task'
                tu,tv=_backward(r,taskgs[i]*taskweights[labels,None],partial)
                fa['task_derivative_triangular_solve_count']=partial['derivative_triangular_solve_count'];fa['task_adjoint_seconds']=partial.get('adjoint_seconds',0.)
                partial={};ku=np.zeros_like(U);kv=np.zeros_like(V)
                if available:
                    partial={'derivative_triangular_solve_count':0};fa['current_adjoint_channel']='keep'
                    ku,kv=_backward(r,keepgs[i]*keepweights[labels,None],partial)
                    fa['keep_derivative_triangular_solve_count']=partial['derivative_triangular_solve_count'];fa['keep_adjoint_seconds']=partial.get('adjoint_seconds',0.)
                partial={};fa['current_adjoint_channel']=None
                fa['derivative_triangular_solve_count']=fa['task_derivative_triangular_solve_count']+fa['keep_derivative_triangular_solve_count']
                gU+=tu+ku;gV+=tv+kv;aU+=ku;aV+=kv
            gU+=(U-anchor_U)/n;gV+=(V-anchor_V)/n
        for key in ('derivative_triangular_solve_count','task_derivative_triangular_solve_count','keep_derivative_triangular_solve_count'):
            audit[key]=sum(f[key] for f in audit['inner_folds'])
        prox=float((_norm(U-anchor_U)**2+_norm(V-anchor_V)**2)/(2*n))
        audit.update(loss_task=float(rt),loss_keep=float(rk),loss_data=float(rt+rk),loss_proximal=prox,loss_total=float(rt+rk+prox),
            class_task_loss_means=taskmeans.tolist(),class_keep_loss_means=keepmeans.tolist(),class_held_counts=counts.tolist(),
            classes=list(prepared.classes),old_class_indices=oldix.tolist(),keep_available=available,
            objective_seconds=time.perf_counter()-start,forward_cache_numeric_bytes=sum(_cache_bytes(r) for r in folds),
            loss_scope='CLASS_RMS_TASK_PLUS_OLD_CLASS_RMS_TEACHER_KEEP_PLUS_PROXIMAL')
        _finite(gU,gV,aU,aV,audit['loss_total'])
        return audit['loss_total'],(gU,gV),(aU,aV),ch._safe(audit),ObjectiveCache(prepared,U,V,tuple(folds))
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if len(folds)<len(prepared.problems):
            audit['inner_head_fit_count']+=partial.get('head_fit_count',0);audit['inner_factorization_count']+=partial.get('factorization_count',0)
            audit['mc_forward_evaluation_count']+=partial.get('mc_forward_evaluation_count',0)
        elif partial and audit['inner_folds']:
            fa=next(f for f in audit['inner_folds'] if f.get('current_adjoint_channel'))
            fa[fa['current_adjoint_channel']+'_derivative_triangular_solve_count']=partial.get('derivative_triangular_solve_count',0)
        for f in audit['inner_folds']:f['derivative_triangular_solve_count']=f['task_derivative_triangular_solve_count']+f['keep_derivative_triangular_solve_count']
        for key in ('derivative_triangular_solve_count','task_derivative_triangular_solve_count','keep_derivative_triangular_solve_count'):audit[key]=sum(f[key] for f in audit['inner_folds'])
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),current_progress=partial)
        raise NumericalFailure(str(exc),audit) from exc


def guarded_direction(g,a,remaining):
    gn=float(np.hypot(_norm(g[0]),_norm(g[1])));an=float(np.hypot(_norm(a[0]),_norm(a[1])))
    if gn==0:return (np.zeros_like(g[0]),np.zeros_like(g[1])),dict(gradient_norm=0.,keep_gradient_norm=an,direction_norm=0.,guard_active=False,guard_dot_before=0.,guard_dot_after=0.,guard_bound=max(0.,remaining)/.125)
    d=(-g[0]/gn,-g[1]/gn);dot=float(np.sum(a[0]*d[0])+np.sum(a[1]*d[1]));bound=max(0.,remaining)/.125
    active=an>0 and dot>bound
    if active:
        coefficient=(dot-bound)/an
        d=(d[0]-(a[0]/an)*coefficient,d[1]-(a[1]/an)*coefficient)
    after=float(np.sum(a[0]*d[0])+np.sum(a[1]*d[1]));dn=float(np.hypot(_norm(d[0]),_norm(d[1])))
    _finite(*d,gn,an,dn)
    if dn>1+128*_EPS*11776:raise FloatingPointError('MC_GUARD_DIRECTION_NORM_EXCEEDED')
    return d,dict(gradient_norm=gn,keep_gradient_norm=an,direction_norm=dn,guard_active=bool(active),guard_dot_before=dot,guard_dot_after=after,guard_bound=float(bound))


@dataclass(frozen=True)
class MCResidual8State:
    U: np.ndarray
    V: np.ndarray
    base_state: local.BranchLocalRidgeState
    original_background: np.ndarray
    original_auxiliary: np.ndarray
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    records: Mapping
    audit: Mapping
    def __post_init__(self):
        for name in ('U','V','original_background','original_auxiliary','labels'):object.__setattr__(self,name,_readonly(getattr(self,name)))
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}));object.__setattr__(self,'audit',_freeze(self.audit));object.__setattr__(self,'records',_freeze(self.records))
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
            b,a,_=_adapt(_context(b0[i:i+1],a0[i:i+1]),self.U,self.V)
            d=pt._joint_distances(b,a,s.support_background,s.support_auxiliary,b0[i:i+1],a0[i:i+1],self.original_background,self.original_auxiliary)
            cc=ch._cross_center(local._radial_minus_one(d,s.bandwidth_tau),s.reference_kernel,s.reference_self,s.center_mean,s.center_grand)
            out[i]=(s.trace_scale*cc[0])@s.alpha
        _finite(out);return out
    def predict(self,**features):return np.asarray(self.classes)[np.argmax(self.score(**features),axis=1)]


_COUNTERS=('inner_objective_evaluation_count','inner_head_fit_count','inner_factorization_count','mc_forward_evaluation_count',
    'derivative_triangular_solve_count','task_derivative_triangular_solve_count','keep_derivative_triangular_solve_count','backward_evaluation_count')


def _cost(audit,obj):
    for k in _COUNTERS:audit[k]+=obj.get(k,0)


def _trial_acceptance(before,trial,gradient_dot_delta,keep_risk,keep_limit):
    rhs=before+1e-4*gradient_dot_delta;tol=pt._armijo_tolerance(before,trial,rhs)
    ktol=float(128*_EPS*max(1.,abs(keep_risk),abs(keep_limit))) if keep_limit is not None else 0.
    armijo=bool(trial<=rhs+tol);nonincrease=bool(trial<=before+tol)
    keep=bool(keep_limit is None or keep_risk<=keep_limit+ktol)
    reasons=[]
    if not armijo:reasons.append('ARMIJO')
    if not nonincrease:reasons.append('OBJECTIVE_INCREASE')
    if not keep:reasons.append('KEEP')
    return dict(armijo_rhs=float(rhs),objective_acceptance_bound=float(min(before,rhs)),armijo_tolerance=tol,
        armijo_pass=armijo,objective_nonincrease_pass=nonincrease,keep_pass=keep,
        armijo_accepted=armijo,keep_accepted=keep,accepted=bool(armijo and nonincrease and keep),
        keep_risk_trial=float(keep_risk),keep_limit=keep_limit,keep_tolerance=ktol,
        keep_violation=float(max(0.,keep_risk-keep_limit)) if keep_limit is not None else 0.,
        rejection_reason='_AND_'.join(reasons) if reasons else None,reject_reason='_AND_'.join(reasons) if reasons else None)


def fit_mc_residual8_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None,state_callback=None):
    if mode not in ('B','C_seq','C_reset_init'):raise ValueError('Unknown MC mode')
    if (mode=='B')!=(prepared.inherited is None):raise ValueError('B/C MC lineage mismatch')
    if prepared.inherited is not None and set(prepared.classes)==set(prepared.old_classes):return prepared.inherited
    start=time.perf_counter();binding={};baseline_state=ch._canonical_baseline(prepared,baseline_state,binding);rec=_Recorder(state_callback)
    au,av=(prepared.inherited.U.copy(),prepared.inherited.V.copy()) if mode=='C_seq' else initial_parameters()
    U,V=au.copy(),av.copy();audit=dict(mode=mode,classes=list(prepared.classes),old_classes=list(prepared.old_classes),
        training_physical_ids=list(prepared.ids),train_k=prepared.audit['train_k'],mc_stage_count=1,preparation=prepared.audit_dict(),
        no_information=prepared.audit['no_information'],optimizer_steps=0,optimizer_iterations=0,accepted_trial_count=0,rejected_trial_count=0,
        trial_count=0,trial_attempt_count=0,nonzero_projected_update_count=0,final_head_fit_count=0,final_factorization_count=0,
        steps=[],trials=[],gradients=[],initial_objective=None,final_objective=None,trainable_parameter_count=11776,
        source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',optimizer_state_reset=True,optimizer_state_bytes=0,
        **{k:0 for k in _COUNTERS},**binding)
    audit['initialization_state_ref']=rec.save('initial',U=U,V=V,anchor_U=au,anchor_V=av)
    finalprogress={};cache=None
    def event(name,record):
        payload=dict(_plain(prepared.audit));payload.update(record,mode=mode);ch._emit(log_callback,name,payload)
    try:
        if audit['no_information']:
            audit['stop_reason']=prepared.audit['no_information_reason'];audit['no_update_reason']=audit['stop_reason']
            audit.update(keep_available=prepared.audit['keep_available'],keep_anchor_risk=None,keep_slack=None,keep_limit=None)
        else:
            loss,_,_,obj,cache=evaluate_mc_objective(prepared,U,V,au,av,gradient=False);_cost(audit,obj)
            k=prepared.audit['train_k'];cp=len(prepared.old_classes);available=bool(prepared.audit['keep_available'])
            slack=float(1/(2*k*np.sqrt(cp))) if available else None;limit=float(obj['loss_keep']+slack) if available else None
            audit.update(initial_objective=deepcopy(obj),keep_available=available,keep_anchor_risk=obj['loss_keep'],keep_slack=slack,keep_limit=limit)
            event('MC_INITIAL',dict(objective=obj,state_ref=audit['initialization_state_ref'],keep_anchor_risk=obj['loss_keep'],keep_slack=slack,keep_limit=limit))
            audit['stop_reason']='MAX_ITERATIONS'
            for iteration in range(1,5):
                begin=time.perf_counter();audit['optimizer_iterations']+=1
                loss,g,a,obj,cache=evaluate_mc_objective(prepared,U,V,au,av,gradient=True,forward_cache=cache);_cost(audit,obj)
                direction,guard=guarded_direction(g,a,limit-obj['loss_keep'] if available else 0.)
                ref=rec.save('gradient_'+str(iteration),U=U,V=V,g_U=g[0],g_V=g[1],keep_g_U=a[0],keep_g_V=a[1],d_U=direction[0],d_V=direction[1])
                gr=dict(iteration=iteration,state_ref=ref,objective=deepcopy(obj),**guard);audit['gradients'].append(gr);event('MC_GRADIENT',gr)
                if guard['gradient_norm']==0:audit['stop_reason']='ZERO_GRADIENT';break
                if guard['direction_norm']==0:audit['stop_reason']='ZERO_GUARDED_DIRECTION';break
                accepted=False
                for trial in range(1,4):
                    step=.125*.5**(trial-1);tu,tv=project_parameters(U+step*direction[0],V+step*direction[1]);du,dv=tu-U,tv-V
                    dn=float(np.hypot(_norm(du),_norm(dv)))
                    if dn==0:audit['stop_reason']='ZERO_PROJECTED_STEP';break
                    audit['trial_attempt_count']+=1;tref=rec.save('trial_'+str(iteration)+'_'+str(trial),U=tu,V=tv)
                    audit['current_trial']=dict(iteration=iteration,trial=trial,step_size=step,state_ref=tref)
                    tl,_,_,to,tc=evaluate_mc_objective(prepared,tu,tv,au,av,gradient=False);_cost(audit,to)
                    acceptance=_trial_acceptance(loss,tl,float(np.sum(g[0]*du)+np.sum(g[1]*dv)),to['loss_keep'],limit)
                    accepted=acceptance['accepted']
                    tr=dict(iteration=iteration,trial=trial,step_size=step,state_ref=tref,gradient_state_ref=ref,
                        loss_before=loss,loss_after=tl,update_norm=dn,objective=deepcopy(to),**acceptance)
                    audit['trials'].append(tr);audit['trial_count']+=1;audit['current_trial']=None
                    audit['accepted_trial_count' if accepted else 'rejected_trial_count']+=1;event('MC_TRIAL',tr)
                    if accepted:
                        U,V=tu,tv;cache=tc;loss=tl;audit['optimizer_steps']+=1;audit['nonzero_projected_update_count']+=1
                        sr=dict(step=audit['optimizer_steps'],iteration=iteration,trial=trial,state_ref=tref,gradient_state_ref=ref,
                            step_size=step,learning_rate=step,loss_before=tr['loss_before'],loss_after=loss,update_norm=dn,
                            step_seconds=time.perf_counter()-begin,objective=deepcopy(to))
                        audit['steps'].append(sr);event('MC_STEP',sr);break
                if not accepted:
                    if audit['stop_reason']!='ZERO_PROJECTED_STEP':audit['stop_reason']='ARMIJO_OR_KEEP_BUDGET_EXHAUSTED'
                    break
            _,_,_,final,_=evaluate_mc_objective(prepared,U,V,au,av,gradient=False,forward_cache=cache);audit['final_objective']=final
        identity=bool(np.all(U==0)) or prepared.full_problem.original.tau0==0
        if identity and baseline_state is not None:base=baseline_state;ff=base.audit_dict()['final_fit']
        else:
            r=_forward(prepared.full_problem,U,V,finalprogress);audit['final_head_fit_count']=1;audit['final_factorization_count']=r['audit']['factorization_count']
            audit['mc_forward_evaluation_count']+=1;ff=r['audit'];alpha,ref,rs,mean,grand,tau,gamma=r['parts']
            base=local.BranchLocalRidgeState(r['b'],r['a'],alpha,ref,rs,mean,grand,tau,gamma,prepared.classes,ff)
        headbytes=sum(getattr(base,k).nbytes for k in ('support_background','support_auxiliary','alpha','reference_kernel','center_mean'))+16+8*(base.bandwidth_tau is not None)+8*(base.trace_scale is not None)
        lineage=prepared.background.nbytes+prepared.auxiliary.nbytes+prepared.labels.nbytes+sum(v.nbytes for v in prepared.raw.values())
        audit.update(final_state_ref=rec.save('final',U=U,V=V,anchor_U=au,anchor_V=av),
            parameter_U_norm=_norm(U),parameter_V_delta_norm=_norm(V-_V0),parameter_changed_from_anchor=bool(np.any(U!=au) or np.any(V!=av)),
            u_changed_from_anchor=bool(np.any(U!=au) or np.any(V!=av)),u_update_norm=float(np.hypot(_norm(U-au),_norm(V-av))),
            identity_forward=identity or base.trace_scale is None,final_fit=ff,head_state_bytes=int(headbytes),adapter_state_bytes=int(U.nbytes+V.nbytes),
            lineage_state_bytes=int(lineage),persistent_state_bytes=int(headbytes+lineage+U.nbytes+V.nbytes),
            retained_vector_record_bytes=sum(sum(v.nbytes for v in r.values()) for r in rec.records.values()),
            state_byte_scope='deployed_head_U_V_original_support_raw_labels_excludes_audit_training_records_Python',
            factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'],
            status='MC_STAGE_COMPLETE',fit_seconds=time.perf_counter()-start,config=deepcopy(_CONFIG))
        event('MC_FIT',audit)
        return MCResidual8State(U,V,base,prepared.background,prepared.auxiliary,prepared.raw,prepared.labels,prepared.ids,rec.records,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):audit['failed_stage']=exc.audit_dict();_cost(audit,audit['failed_stage'])
        if finalprogress and not audit['final_head_fit_count']:
            audit['failed_final_kernel_progress']=finalprogress;audit['final_head_fit_count']=finalprogress.get('head_fit_count',0)
            audit['final_factorization_count']=finalprogress.get('factorization_count',0);audit['mc_forward_evaluation_count']+=finalprogress.get('mc_forward_evaluation_count',0)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),accepted_state_ref=rec.save('failure_accepted',U=U,V=V,anchor_U=au,anchor_V=av),
            fit_seconds=time.perf_counter()-start,factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'])
        error=NumericalFailure(str(exc),audit);error.records=rec.records;raise error from exc
