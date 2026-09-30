"""Support-supervised channel adapter with an adjoint LocalRidge classifier.

Only original b/a subblocks are transformed. There is no second feature
normalization, encoder backward, free classifier logit, or query fitting.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import solve_triangular
from . import d92_branch_interaction as interaction
from . import d92_branch_local_ridge as local
from .d92_branch_ridge import _freeze, _plain, _readonly

FROZEN_CONFIG = dict(
    schema='d92_joint_channel_local_ridge_v1', method='D92-JointChannelLocalRidge-v1',
    base_algorithm=deepcopy(local.FROZEN_CONFIG),
    channel=dict(route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000),
    allowed_scenarios=['practical_high','practical_mid','practical_low_urban'],
    arms=['local_ridge','channel_seq','channel_reset'],
    adapter='norm_preserving_positive_diagonal_on_original_b_a_subblocks',
    block_dimensions=[160,96,160,160,160], trainable_parameter_count=736, effective_parameter_count=731,
    feasible_set='per_block_sum_zero_and_abs_u_le_log2_over_2', bound=float(np.log(2)/2),
    initialization_B='zero', initialization_C_seq='inherit_B_u', initialization_C_reset='zero',
    objective='(sum_physical_inner_held_squared_unit_margin_hinge+u_anchor_squared_distance)/(2*N)',
    proximal_physical_sum_coefficient=1., optimizer='projected_Adam', optimizer_steps=8,
    learning_rate=.02, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, gradient_clip_norm=1.,
    projection_max_bisections=80, projection_tolerance='128*eps64*block_dim*bound',
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3', final_objective_evaluation=True,
    trace_target='original_untransformed_inner_train_interaction_centered_trace',
    bandwidth='current_u_median_nearest_other_class_including_zero',
    tie_gradient='uniform_exact_wrong_max;uniform_exact_nearest_and_tied_median_groups',
    derivative='adjoint_ridge_centering_trace_bandwidth_stable_pair_and_channel_VJP',
    zero_u='bitwise_original_b_a_and_rowwise_R0_forward_with_live_derivative',
    zero_bandwidth='original_b_a_equivalence_kernel_zero_derivative',
    no_information='physical_K1_or_single_registered_class_preserve_anchor',
    n0='reuse_corresponding_B', probe_k1='numerical_only_no_fit_no_holdout',
    query_decision_policy='per_sample_all_registered_classes', tie_break='physical_class_id_lexicographic',
    source_inputs=False, query_fit=False, phase1_frozen=True, encoder_backward=False,
    early_stopping=False, parameter_search=False, dtype='float64')
_CONFIG=deepcopy(FROZEN_CONFIG)
_NAMES=('z_id','fft','t_emb','f_emb','pa_local')
_SLICES=(slice(0,160),slice(160,256),slice(256,416),slice(416,576),slice(576,736))
_EPS=np.finfo(np.float64).eps


def _safe(value):
    if isinstance(value, Mapping): return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, (list,tuple)): return [_safe(v) for v in value]
    if isinstance(value, np.generic): return _safe(value.item())
    if isinstance(value,float) and not np.isfinite(value): return 'NaN' if np.isnan(value) else ('Infinity' if value>0 else '-Infinity')
    return value


class NumericalFailure(FloatingPointError):
    def __init__(self,message,audit):
        super().__init__(message); self.audit=_safe(audit)
    def audit_dict(self): return deepcopy(self.audit)


def _finite(*values):
    for v in values:
        if not np.isfinite(v).all(): raise FloatingPointError('NONFINITE_CHANNEL_ARITHMETIC')


def _norm(value):
    value=np.asarray(value,dtype=np.float64)
    scale=float(np.max(np.abs(value))) if value.size else 0.
    return 0. if scale==0 else scale*float(np.sqrt(np.sum((value/scale)**2)))


def _row_norm(value):
    if not len(value): return np.empty(0)
    scale=np.max(np.abs(value),axis=1)
    out=np.zeros(len(value)); active=scale>0
    out[active]=scale[active]*np.sqrt(np.sum((value[active]/scale[active,None])**2,axis=1))
    _finite(out)
    return out


def _emit(callback,event,payload):
    if callback is not None: callback(dict(_safe(payload),event=event))


def _parameter(value):
    value=np.asarray(value,dtype=np.float64)
    h=_CONFIG['bound']
    if value.shape!=(736,) or not np.isfinite(value).all(): raise ValueError('u must be finite shape (736,)')
    for sl in _SLICES:
        v=value[sl];tol=128*_EPS*len(v)*h
        if np.max(np.abs(v))>h+tol or abs(float(v.sum()))>tol: raise ValueError('u outside per-block zero-sum box')
    return value


def project_channels(value):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(736,): raise ValueError('Channel parameter shape')
    _finite(value);out=np.empty_like(value);h=_CONFIG['bound']
    for sl in _SLICES:
        v=value[sl];tol=128*_EPS*len(v)*h
        if np.max(np.abs(v))<=h and abs(float(v.sum()))<=tol:
            out[sl]=v;continue
        lo=float(np.min(v-h));hi=float(np.max(v+h))
        for _ in range(_CONFIG['projection_max_bisections']):
            mid=lo+(hi-lo)/2;x=np.clip(v-mid,-h,h);total=float(x.sum())
            if abs(total)<=tol: break
            if total>0:lo=mid
            else:hi=mid
        if abs(float(x.sum()))>tol or np.max(np.abs(x))>h+tol:
            raise FloatingPointError('CHANNEL_PROJECTION_RESIDUAL_EXCEEDED')
        out[sl]=x
    return out


def _adapt_blocks(b,a,u):
    """The only adapter entry: b/a are already original-normalized blocks."""
    u=_parameter(u)
    if np.all(u==0): return b,a
    original=np.concatenate((b,a),axis=1);mapped=np.empty_like(original)
    gain=np.exp(u)
    for sl in _SLICES:
        r=original[:,sl];rho=_row_norm(r);dr=r*gain[sl];den=_row_norm(dr)
        active=np.any(r!=0,axis=1)
        if np.any(active&(den==0)):raise FloatingPointError('CHANNEL_NONZERO_NORM_UNDERFLOW')
        result=np.zeros_like(r)
        # Normalize after scaling by max(|Dr|), avoiding subnormal division.
        if np.any(active):
            scale=np.max(np.abs(dr[active]),axis=1)
            z=dr[active]/scale[:,None]
            w=z/np.sqrt(np.sum(z*z,axis=1))[:,None]
            result[active]=rho[active,None]*w
        err=np.abs(_row_norm(result)-rho)
        if np.any(err>128*_EPS*r.shape[1]*rho):raise FloatingPointError('CHANNEL_NORM_PRESERVATION_FAILED')
        mapped[:,sl]=result
    _finite(mapped)
    return mapped[:,:256],mapped[:,256:]


def _gate_vjp(b,a,u,gb,ga):
    original=np.concatenate((b,a),axis=1);adj=np.concatenate((gb,ga),axis=1)
    grad=np.zeros(736);gain=np.exp(u)
    for sl in _SLICES:
        r=original[:,sl];rho=_row_norm(r);active=np.any(r!=0,axis=1)
        if not np.any(active):continue
        dr=r[active]*gain[sl];scale=np.max(np.abs(dr),axis=1)
        z=dr/scale[:,None];w=z/np.sqrt(np.sum(z*z,axis=1))[:,None]
        g=adj[active,sl]
        grad[sl]=np.sum(rho[active,None]*w*(g-np.sum(g*w,axis=1)[:,None]*w),axis=0)
    _finite(grad)
    return grad


def _checked_distances(b,a,tb,ta,b0,a0,tb0,ta0,*,symmetric=False):
    out=local._distances(b,a,None if symmetric else tb,None if symmetric else ta)
    for i in range(len(b)):
        before=np.all(b0[i]==tb0,axis=1)&np.all(a0[i]==ta0,axis=1)
        after=np.all(b[i]==tb,axis=1)&np.all(a[i]==ta,axis=1)
        if np.any(before!=after):raise FloatingPointError('CHANNEL_FLOATING_EQUIVALENCE_CHANGED')
    return out


def _bandwidth_weights(distance,labels):
    n=len(labels);weights=np.zeros_like(distance)
    if len(set(labels.tolist()))==1:return None,weights
    allowed=labels[:,None]!=labels[None,:]
    nearest=np.min(np.where(allowed,distance,np.inf),axis=1)
    tau=float(np.median(nearest));order=np.sort(nearest)
    middle=[order[n//2]] if n%2 else [order[n//2-1],order[n//2]]
    rowweights=np.zeros(n)
    for val in middle:
        tied=nearest==val;rowweights[tied]+=1/(len(middle)*int(tied.sum()))
    for i in range(n):
        tied=allowed[i]&(distance[i]==nearest[i])
        weights[i,tied]=rowweights[i]/int(tied.sum())
    return tau,weights


def _cross_center(raw,reference,reference_self,mean,grand):
    out=np.empty_like(raw)
    for i in range(len(raw)):
        diff=(raw[i]-raw[i,0])-reference+reference_self
        out[i]=diff-float(diff.mean())-mean+grand
    return out


def _margin_loss(score,labels):
    n,c=score.shape;g=np.zeros_like(score)
    if c==1:return 0.,g,np.zeros(n)
    wrong=score.copy();wrong[np.arange(n),labels]=-np.inf
    maximum=wrong.max(axis=1);margin=score[np.arange(n),labels]-maximum
    hinge=np.maximum(0.,1-margin)
    for i in range(n):
        tied=wrong[i]==maximum[i]
        g[i,tied]=hinge[i]/int(tied.sum());g[i,labels[i]]=-hinge[i]
    return float(.5*np.sum(hinge*hinge)),g,margin


def _distance_vjp(b,a,tb,ta,adj,*,symmetric=False):
    gb,ga=np.zeros_like(b),np.zeros_like(a)
    gtb,gta=(gb,ga) if symmetric else (np.zeros_like(tb),np.zeros_like(ta))
    chunk=local.FROZEN_CONFIG['distance_pair_chunk']
    for i in range(len(b)):
        for start in range(i+1 if symmetric else 0,len(tb),chunk):
            end=min(start+chunk,len(tb));w=adj[i,start:end].copy()
            if symmetric:w+=adj[start:end,i]
            db=b[i]-tb[start:end];da=a[i]-ta[start:end]
            abi=float(a[i]@a[i]);bbi=float(b[i]@b[i])
            leftb=2*((1+abi)*db+(da@a[i])[:,None]*tb[start:end])
            lefta=2*((1+bbi)*da+(db@b[i])[:,None]*ta[start:end])
            rightb=2*(-(1+np.sum(ta[start:end]**2,axis=1))[:,None]*db
                -np.sum(ta[start:end]*da,axis=1)[:,None]*b[i])
            righta=2*(-(1+np.sum(tb[start:end]**2,axis=1))[:,None]*da
                -np.sum(tb[start:end]*db,axis=1)[:,None]*a[i])
            gb[i]+=np.sum(w[:,None]*leftb,axis=0);ga[i]+=np.sum(w[:,None]*lefta,axis=0)
            gtb[start:end]+=w[:,None]*rightb;gta[start:end]+=w[:,None]*righta
    _finite(gb,ga,gtb,gta)
    return gb,ga,gtb,gta


@dataclass(frozen=True)
class ChannelProblem:
    train_b: np.ndarray
    train_a: np.ndarray
    held_b: np.ndarray
    held_a: np.ndarray
    train_labels: np.ndarray
    held_labels: np.ndarray
    classes: tuple
    d0: np.ndarray
    cross_d0: np.ndarray
    s0: float
    tau0: float | None
    audit: Mapping
    def __post_init__(self):
        for key in ('train_b','train_a','held_b','held_a','train_labels','held_labels','d0','cross_d0'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'audit',_freeze(self.audit))


def _make_problem(b,a,y,ids,classes,keep,fold):
    keep=np.asarray(keep,dtype=bool);n=int(keep.sum())
    d0,s0,tau0,_,_=local._geometry(b[keep],a[keep],y[keep],len(classes))
    cross=local._distances(b[~keep],a[~keep],b[keep],a[keep])
    audit=dict(inner_fold=fold,training_physical_ids=[pid for i,pid in enumerate(ids) if keep[i]],
        held_physical_ids=[pid for i,pid in enumerate(ids) if not keep[i]],
        train_physical_count=n,held_physical_count=int((~keep).sum()),
        original_interaction_centered_trace=s0,original_bandwidth_tau=tau0,
        all_head_statistics_from_inner_train_only=True)
    return ChannelProblem(b[keep],a[keep],b[~keep],a[~keep],y[keep],y[~keep],classes,d0,cross,s0,tau0,audit)


def _evaluate_kernel(problem,u,*,gradient=True,progress=None,return_scores=False):
    u=_parameter(u);n=len(problem.train_labels);c=len(problem.classes)
    audit={} if progress is None else progress
    audit.update(head_fit_count=1,factorization_count=0,derivative_triangular_solve_count=0,
        **_plain(problem.audit))
    y=np.asarray(problem.train_labels,dtype=int);hy=np.asarray(problem.held_labels,dtype=int)
    s0=problem.s0;tol=128*_EPS*max(n,c)
    original_forward=bool(np.all(u==0)) or problem.tau0==0
    if original_forward:
        b,a=problem.train_b,problem.train_a;hb,ha=problem.held_b,problem.held_a
        distance,cross=problem.d0,problem.cross_d0
    else:
        b,a=_adapt_blocks(problem.train_b,problem.train_a,u)
        hb,ha=_adapt_blocks(problem.held_b,problem.held_a,u)
        distance=_checked_distances(b,a,b,a,problem.train_b,problem.train_a,problem.train_b,problem.train_a,symmetric=True)
        cross=_checked_distances(hb,ha,b,a,problem.held_b,problem.held_a,problem.train_b,problem.train_a)
    tau,weights=_bandwidth_weights(distance,y)
    if problem.tau0 is not None and problem.tau0>0 and tau<=0:
        raise FloatingPointError('CHANNEL_POSITIVE_BANDWIDTH_UNRESOLVED')
    target=np.eye(c)[y]-1/c
    alpha=np.zeros((n,c));reference=mean=np.zeros(n);refself=grand=0.;gamma=None
    score=np.zeros((len(hy),c));grad=np.zeros(736);kernel=np.zeros((n,n))
    audit.update(bandwidth_tau=tau,original_bandwidth_tau=problem.tau0,
        interaction_centered_trace=s0,radial_centered_trace=None,trace_scale=None,
        normal_equation_residual=0.,trace_relative_error=0.,numerical_tolerance=tol,
        zero_bandwidth_original_equivalence=problem.tau0==0,identity_forward=original_forward)
    if c>1 and s0>0:
        raw=local._radial_minus_one(distance,tau);radial=local._radial(distance,tau)
        sr=float(-2*raw[np.triu_indices(n,1)].sum()/n)
        if sr<=0:raise FloatingPointError('NONPOSITIVE_RADIAL_TRACE')
        gamma=s0/sr;centered,reference,refself,mean,grand=interaction._center_kernel(raw)
        kernel=gamma*centered;matrix=kernel+np.eye(n)
        audit['factorization_count']=1;chol=np.linalg.cholesky(matrix)
        alpha=np.ascontiguousarray(solve_triangular(chol.T,solve_triangular(chol,target,lower=True),lower=False))
        rawcross=local._radial_minus_one(cross,tau)
        crosscentered=_cross_center(rawcross,reference,refself,mean,grand);crosskernel=gamma*crosscentered
        score=np.stack([(gamma*crosscentered[i])@alpha for i in range(len(hy))]) if len(hy) else np.zeros((0,c))
        residual=_norm(matrix@alpha-target)/((1+s0)*_norm(alpha)+_norm(target))
        trace_error=abs(float(np.trace(kernel))-s0)/s0
        if residual>tol or trace_error>tol:raise FloatingPointError('CHANNEL_HEAD_RESIDUAL_EXCEEDED')
        audit.update(normal_equation_residual=residual,trace_relative_error=trace_error,
            radial_centered_trace=sr,trace_scale=gamma)
    loss,gscore,margin=_margin_loss(score,hy)
    if gradient and c>1 and s0>0 and tau>0 and len(hy):
        begin=time.perf_counter()
        barcross=gscore@alpha.T
        audit['derivative_triangular_solve_count']+=1
        lower=solve_triangular(chol,crosskernel.T@gscore,lower=True)
        audit['derivative_triangular_solve_count']+=1
        z=solve_triangular(chol.T,lower,lower=False)
        bark=-z@alpha.T;bark=.5*(bark+bark.T)
        # Adjoint of H R H, cross centering E H - 1 (1^T R H)/n,
        # and gamma=s0/sr. No per-parameter kernel derivative tensor.
        barraw=gamma*(bark-bark.mean(axis=0)[None,:]-bark.mean(axis=1)[:,None]+bark.mean())
        crossrow=barcross-barcross.mean(axis=1)[:,None]
        barraw-=gamma*np.broadcast_to(crossrow.sum(axis=0)/n,(n,n))
        barsr=-(float(np.sum(bark*kernel))+float(np.sum(barcross*crosskernel)))/sr
        off=~np.eye(n,dtype=bool);barraw[off]-=barsr/n
        barcrossraw=gamma*crossrow
        rd=barraw*radial;crossrad=local._radial(cross,tau);rc=barcrossraw*crossrad
        dd=np.zeros_like(distance);dc=np.zeros_like(cross)
        active=radial>0;activec=crossrad>0
        dd[active]=-rd[active]/tau;dc[activec]=-rc[activec]/tau
        bartau=(float(np.sum(rd[active]*(distance[active]/tau)))+
            float(np.sum(rc[activec]*(cross[activec]/tau))))/tau
        dd+=bartau*weights
        gb,ga,_,_=_distance_vjp(b,a,b,a,dd,symmetric=True)
        ghb,gha,gtb,gta=_distance_vjp(hb,ha,b,a,dc)
        grad=_gate_vjp(problem.train_b,problem.train_a,u,gb+gtb,ga+gta)
        grad+=_gate_vjp(problem.held_b,problem.held_a,u,ghb,gha)
        audit['adjoint_seconds']=time.perf_counter()-begin
    else:audit['adjoint_seconds']=0.
    fitted=kernel@alpha;fiterr=fitted-target
    audit.update(held_physical_count=len(hy),held_training_correct_count=int(np.sum(score.argmax(axis=1)==hy)) if len(hy) else 0,
        held_training_margin_mean=float(margin.mean()) if len(margin) else None,
        held_training_margin_min=float(margin.min()) if len(margin) else None,
        loss_data_sum=loss,loss_data=float(.5*np.sum(fiterr*fiterr)),
        loss_ridge=float(.5*np.sum(alpha*fitted)),sample_weight=1.,ridge_coefficient=1.,
        status='CLOSED_FORM_SOLVED' if c>1 and s0>0 else 'EXACT_ZERO_CLASSIFIER')
    audit['loss_total']=audit['loss_data']+audit['loss_ridge']
    audit.update(held_margin_loss_sum=loss,head_training_loss_data=audit['loss_data'],
        head_training_loss_ridge=audit['loss_ridge'],head_training_loss_total=audit['loss_total'],
        loss_scope='HEAD_TRAINING_RIDGE_OBJECTIVE;HELD_MARGIN_LOSS_REPORTED_SEPARATELY')
    _finite(score,grad,alpha,loss)
    result=(loss,grad,audit,(alpha,reference,refself,mean,grand,tau,gamma),(b,a))
    return result+(score,) if return_scores else result


@dataclass(frozen=True)
class ChannelTraining:
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    classes: tuple
    old_classes: tuple
    background: np.ndarray
    auxiliary: np.ndarray
    problems: tuple
    inherited: object
    audit: Mapping
    def __post_init__(self):
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}))
        for key in ('labels','background','auxiliary'):object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'audit',_freeze(self.audit))
    def audit_dict(self):return _plain(self.audit)


def prepare_channel_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,
        classes,old_classes,inherited=None,context=None,log_callback=None):
    started=time.perf_counter();audit=dict(context or {})
    audit.update(channel_preparation_count=1,inner_folds=[],completed_stages=[])
    b,a,y,ids,canonical,_,old,k=interaction._prepare(z_id,fft,t_emb,f_emb,pa_local,
        support_labels,support_ids,classes,old_classes)
    old=tuple(sorted(old))
    if not old:raise ValueError('Old support classes required')
    order=np.asarray(sorted(range(len(support_ids)),key=lambda i:support_ids[i]))
    raw={key:np.asarray(value,dtype=np.float64)[order] for key,value in zip(_NAMES,(z_id,fft,t_emb,f_emb,pa_local))}
    if inherited is None:
        if set(canonical)!=set(old):raise ValueError('C requires inherited B state')
    else:
        if inherited.audit['mode']!='B' or set(inherited.classes)!=set(old):raise ValueError('C requires old-only supervised B')
        previous=inherited.audit['preparation']
        for key in ('row_id','split_id','scope','fold','trial'):
            if key in (context or {}) and key in previous and context[key]!=previous[key]:raise ValueError('C lineage crosses '+key)
        oldix=[i for i,v in enumerate(y) if canonical[int(v)] in old]
        if tuple(ids[i] for i in oldix)!=inherited.ids:raise ValueError('C old physical IDs differ from B')
        if any(canonical[int(y[i])]!=inherited.classes[int(inherited.labels[j])] for j,i in enumerate(oldix)):
            raise ValueError('C old labels differ from B')
        if any(not np.array_equal(raw[key][oldix],inherited.raw[key]) for key in _NAMES):
            raise ValueError('C old raw features differ from B')
    audit.update(training_physical_ids=list(ids),classes=list(canonical),old_classes=list(old),train_k=k,
        train_physical_count=len(ids),inherited_state=inherited is not None)
    problems=[]
    try:
        noinfo=k==1 or len(canonical)==1
        if not noinfo:
            folds=min(k,3);assignment=np.empty(len(ids),dtype=int)
            for cls in range(len(canonical)):
                ix=np.flatnonzero(y==cls);assignment[ix]=np.arange(len(ix))%folds
            for f in range(folds):
                problem=_make_problem(b,a,y,ids,canonical,assignment!=f,f);problems.append(problem)
                fa=_plain(problem.audit);audit['inner_folds'].append(fa)
                event=dict(context or {});event.update(fa);_emit(log_callback,'CHANNEL_INNER_PREPARED',event)
        audit.update(no_information=noinfo,no_information_reason='PHYSICAL_K1' if k==1 else ('SINGLE_REGISTERED_CLASS' if len(canonical)==1 else None),
            prepare_seconds=time.perf_counter()-started,preparation_seconds=time.perf_counter()-started,
            prepared_numeric_state_bytes=int(sum(v.nbytes for v in raw.values())+b.nbytes+a.nbytes+y.nbytes+
                sum(sum(getattr(p,name).nbytes for name in ('train_b','train_a','held_b','held_a','train_labels','held_labels','d0','cross_d0')) for p in problems)),
            transient_distance_bytes=sum(p.d0.nbytes+p.cross_d0.nbytes for p in problems))
        return ChannelTraining(raw,y,ids,canonical,old,b,a,tuple(problems),inherited,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),prepare_seconds=time.perf_counter()-started)
        raise NumericalFailure(str(exc),audit) from exc


def evaluate_channel_objective(prepared,u,anchor,*,gradient=True):
    u=_parameter(u);anchor=_parameter(anchor);n=len(prepared.ids)
    total=0.;grad=np.zeros(736);folds=[]
    for problem in prepared.problems:
        partial={}
        try:loss,g,audit,_,_=_evaluate_kernel(problem,u,gradient=gradient,progress=partial)
        except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
            raise NumericalFailure(str(exc),dict(status='TECHNICAL_FAILURE',failure_reason=str(exc),
                current_inner_fold=_plain(problem.audit),current_kernel_progress=partial,completed_inner_folds=folds,
                inner_objective_evaluation_count=1,inner_head_fit_count=len(folds)+partial.get('head_fit_count',0),
                inner_factorization_count=sum(f['factorization_count'] for f in folds)+partial.get('factorization_count',0),
                derivative_triangular_solve_count=sum(f['derivative_triangular_solve_count'] for f in folds)+partial.get('derivative_triangular_solve_count',0))) from exc
        total+=loss;grad+=g;folds.append(audit)
    penalty=float(.5*np.sum((u-anchor)**2)/n)
    grad=grad/n+(u-anchor)/n if gradient else np.zeros(736)
    audit=dict(loss_data=total/n,loss_proximal=penalty,loss_total=total/n+penalty,inner_folds=folds,
        loss_scope='PHYSICAL_INNER_HELD_MARGIN_PLUS_PROXIMAL',
        gradient=grad.tolist() if gradient else None,inner_head_fit_count=len(folds),
        inner_factorization_count=sum(f['factorization_count'] for f in folds),
        derivative_triangular_solve_count=sum(f['derivative_triangular_solve_count'] for f in folds))
    _finite(grad,audit['loss_total'])
    return audit['loss_total'],grad,audit


@dataclass(frozen=True)
class ChannelLocalRidgeState:
    u: np.ndarray
    base_state: local.BranchLocalRidgeState
    original_background: np.ndarray
    original_auxiliary: np.ndarray
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    audit: Mapping
    def __post_init__(self):
        for key in ('u','original_background','original_auxiliary','labels'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'raw',_freeze({key:_readonly(value) for key,value in self.raw.items()}))
        object.__setattr__(self,'audit',_freeze(self.audit))
    @property
    def classes(self):return self.base_state.classes
    def audit_dict(self):return _plain(self.audit)
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        state=self.base_state
        b0,a0=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        # Empty batches and exact zero classifiers retain the original contract.
        if np.all(self.u==0) or state.bandwidth_tau==0 or state.trace_scale is None:
            return local._score_local(b0,a0,self.original_background,self.original_auxiliary,state.alpha,
                state.reference_kernel,state.reference_self,state.center_mean,state.center_grand,
                state.bandwidth_tau,state.trace_scale)
        scores=np.empty((len(b0),len(self.classes)))
        # Per-sample gating and matrix-vector scoring enforce batch independence.
        for i in range(len(b0)):
            b,a=_adapt_blocks(b0[i:i+1],a0[i:i+1],self.u)
            d=_checked_distances(b,a,state.support_background,state.support_auxiliary,
                b0[i:i+1],a0[i:i+1],self.original_background,self.original_auxiliary)
            raw=local._radial_minus_one(d,state.bandwidth_tau)
            centered=_cross_center(raw,state.reference_kernel,state.reference_self,state.center_mean,state.center_grand)
            scores[i]=(state.trace_scale*centered[0])@state.alpha
        _finite(scores)
        return scores
    def predict(self,**features):
        return np.asarray(self.classes)[np.argmax(self.score(**features),axis=1)]


def _canonical_baseline(prepared,baseline,progress=None):
    if baseline is None:return None
    if not isinstance(baseline,local.BranchLocalRidgeState):raise ValueError('Expected original LocalRidge baseline')
    started=time.perf_counter();validation={} if progress is None else progress
    ba=baseline.audit_dict()
    if (set(baseline.classes)!=set(prepared.classes) or
        ba['final_fit']['training_physical_ids']!=list(prepared.ids) or
        not np.array_equal(baseline.support_background,prepared.background) or
        not np.array_equal(baseline.support_auxiliary,prepared.auxiliary)):
        raise ValueError('Channel baseline binding mismatch')
    if tuple(baseline.classes)!=prepared.classes:
        baseline=local.BranchLocalRidgeState(baseline.support_background,baseline.support_auxiliary,
            baseline.alpha[:,[baseline.classes.index(x) for x in prepared.classes]],baseline.reference_kernel,
            baseline.reference_self,baseline.center_mean,baseline.center_grand,baseline.bandwidth_tau,
            baseline.trace_scale,prepared.classes,ba)
    # Original LocalRidge states do not retain their physical-ID labels. Verify
    # the actual coefficients against this preparation's labels, rather than
    # treating feature/ID equality or a caller assertion as a label certificate.
    n,c=len(prepared.ids),len(prepared.classes)
    validation.update(baseline_binding_distance_evaluation_count=1,baseline_binding_factorization_count=0)
    distance,s0,tau,_,_=local._geometry(prepared.background,prepared.auxiliary,np.asarray(prepared.labels,dtype=int),c)
    if c==1 or s0==0:
        if baseline.trace_scale is not None or np.any(baseline.alpha!=0):raise ValueError('Baseline exact-zero head mismatch')
        validation['baseline_label_binding']='EXACT_ZERO_HEAD_LABEL_INVARIANT'
        validation['baseline_label_residual']=0.
    else:
        raw=local._radial_minus_one(distance,tau)
        sr=float(-2*raw[np.triu_indices(n,1)].sum()/n)
        gamma=s0/sr;center,reference,refself,mean,grand=interaction._center_kernel(raw)
        if (baseline.bandwidth_tau!=tau or baseline.trace_scale!=gamma or
            not np.array_equal(baseline.reference_kernel,reference) or baseline.reference_self!=refself or
            not np.array_equal(baseline.center_mean,mean) or baseline.center_grand!=grand):
            raise ValueError('Baseline original head geometry mismatch')
        target=np.eye(c)[np.asarray(prepared.labels,dtype=int)]-1/c
        residual=_norm((gamma*center+np.eye(n))@baseline.alpha-target)
        ratio=residual/((1+s0)*_norm(baseline.alpha)+_norm(target))
        _finite(ratio)
        if ratio>128*_EPS*max(n,c):raise ValueError('Baseline physical label coefficient binding mismatch')
        validation.update(baseline_label_binding='CURRENT_LABEL_NORMAL_EQUATION_VERIFIED',baseline_label_residual=ratio)
    validation['baseline_binding_seconds']=time.perf_counter()-started
    return baseline


def fit_channel_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None):
    if mode not in ('B','C_seq','C_reset'):raise ValueError('Unknown channel mode')
    if mode=='B' and prepared.inherited is not None:raise ValueError('B cannot inherit target state')
    if mode!='B' and prepared.inherited is None:raise ValueError('C requires B lineage')
    if prepared.inherited is not None and set(prepared.classes)==set(prepared.old_classes):return prepared.inherited
    started=time.perf_counter();baseline_validation={}
    baseline_state=_canonical_baseline(prepared,baseline_state,baseline_validation)
    anchor=prepared.inherited.u.copy() if mode=='C_seq' else np.zeros(736)
    u=anchor.copy();m=np.zeros(736);v=np.zeros(736)
    audit=dict(mode=mode,classes=list(prepared.classes),old_classes=list(prepared.old_classes),
        training_physical_ids=list(prepared.ids),train_k=prepared.audit['train_k'],
        optimizer_steps=0,inner_objective_evaluation_count=0,inner_head_fit_count=0,inner_factorization_count=0,
        derivative_triangular_solve_count=0,final_head_fit_count=0,final_factorization_count=0,
        steps=[],completed_stages=[],anchor=anchor.tolist(),u_anchor=anchor.tolist(),
        no_information=prepared.audit['no_information'],preparation=prepared.audit_dict(),
        trainable_parameter_count=736,effective_parameter_count=731,nonzero_projected_update_count=0,
        source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN',
        optimizer_state_reset=True,optimizer_state_bytes=int(m.nbytes+v.nbytes),**baseline_validation)
    finalprogress={}
    try:
        if not audit['no_information']:
            for step in range(1,_CONFIG['optimizer_steps']+1):
                begin=time.perf_counter();before=u.copy()
                _,g,objective=evaluate_channel_objective(prepared,u,anchor)
                # Record completed numerical work before the projection/callback.
                audit['inner_objective_evaluation_count']+=1
                for key in ('inner_head_fit_count','inner_factorization_count','derivative_triangular_solve_count'):audit[key]+=objective[key]
                gn=_norm(g);clip=min(1.,_CONFIG['gradient_clip_norm']/gn) if gn else 1.;used=g*clip
                m=_CONFIG['adam_beta1']*m+(1-_CONFIG['adam_beta1'])*used
                v=_CONFIG['adam_beta2']*v+(1-_CONFIG['adam_beta2'])*used*used
                mh=m/(1-_CONFIG['adam_beta1']**step);vh=v/(1-_CONFIG['adam_beta2']**step)
                proposal=u-_CONFIG['learning_rate']*mh/(np.sqrt(vh)+_CONFIG['adam_epsilon'])
                u=project_channels(proposal)
                record=dict(step=step,u_pre=before.tolist(),u_post=u.tolist(),anchor=anchor.tolist(),
                    learning_rate=_CONFIG['learning_rate'],gradient=g.tolist(),gradient_norm=gn,
                    clipped_gradient_norm=_norm(used),gradient_clip_scale=clip,
                    unprojected_update_norm=_norm(proposal-before),update_norm=_norm(u-before),
                    active_box_count=int(np.sum(np.abs(u)>=_CONFIG['bound']-8*_EPS)),
                    projection_zero_sum_residuals=[float(u[sl].sum()) for sl in _SLICES],
                    u_rms=float(np.sqrt(np.mean(u*u))),u_max_abs=float(np.max(np.abs(u))),
                    step_seconds=time.perf_counter()-begin,**{k:value for k,value in objective.items() if k!='gradient'})
                audit['steps'].append(record);audit['optimizer_steps']+=1
                audit['nonzero_projected_update_count']+=int(np.any(u!=before))
                event=dict(_plain(prepared.audit));event.update(record,mode=mode)
                _emit(log_callback,'JOINT_CHANNEL_STEP',event)
            _,_,final=evaluate_channel_objective(prepared,u,anchor,gradient=False)
            audit['final_objective']=final;audit['inner_objective_evaluation_count']+=1
            for key in ('inner_head_fit_count','inner_factorization_count'):audit[key]+=final[key]
        else:
            audit['final_objective']=None;audit['no_update_reason']=prepared.audit['no_information_reason']
        identity=bool(np.all(u==0));b,a=prepared.background,prepared.auxiliary
        if identity and baseline_state is not None:
            base=baseline_state;finalfit=base.audit_dict()['final_fit']
        else:
            problem=_make_problem(b,a,np.asarray(prepared.labels,dtype=int),prepared.ids,prepared.classes,np.ones(len(b),dtype=bool),None)
            _,_,finalfit,parts,(mb,ma)=_evaluate_kernel(problem,u,gradient=False,progress=finalprogress)
            audit['final_head_fit_count']=1;audit['final_factorization_count']=finalfit['factorization_count']
            alpha,ref,rs,mean,grand,tau,gamma=parts
            base=local.BranchLocalRidgeState(mb,ma,alpha,ref,rs,mean,grand,tau,gamma,prepared.classes,finalfit)
        headbytes=int(sum(getattr(base,key).nbytes for key in ('support_background','support_auxiliary','alpha','reference_kernel','center_mean'))+
            16+8*(base.bandwidth_tau is not None)+8*(base.trace_scale is not None))
        lineagebytes=int(b.nbytes+a.nbytes+prepared.labels.nbytes+sum(x.nbytes for x in prepared.raw.values()))
        audit.update(u=u.tolist(),u_changed_from_anchor=bool(np.any(u!=anchor)),u_update_norm=_norm(u-anchor),
            identity_forward=identity or base.bandwidth_tau==0 or base.trace_scale is None,
            final_fit=finalfit,head_state_bytes=headbytes,adapter_state_bytes=u.nbytes,
            lineage_state_bytes=lineagebytes,persistent_state_bytes=headbytes+u.nbytes+lineagebytes,
            state_byte_scope='head_plus_adapter_original_b_a_raw_support_and_labels_excludes_Python_audit_and_ids',
            factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'],
            status='CHANNEL_STAGE_COMPLETE',fit_seconds=time.perf_counter()-started,config=deepcopy(_CONFIG))
        _emit(log_callback,'JOINT_CHANNEL_FIT',audit)
        return ChannelLocalRidgeState(u,base,b,a,prepared.raw,prepared.labels,prepared.ids,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):
            failed=exc.audit_dict();audit['failed_stage']=failed
            for key in ('inner_objective_evaluation_count','inner_head_fit_count','inner_factorization_count','derivative_triangular_solve_count'):
                audit[key]+=failed.get(key,0)
        if finalprogress and not audit['final_head_fit_count']:
            audit['failed_final_kernel_progress']=finalprogress
            audit['final_head_fit_count']=finalprogress.get('head_fit_count',0)
            audit['final_factorization_count']=finalprogress.get('factorization_count',0)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),u=u.tolist(),
            adam_first_moment=m.tolist(),adam_second_moment=v.tolist(),fit_seconds=time.perf_counter()-started,
            factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'])
        raise NumericalFailure(str(exc),audit) from exc
