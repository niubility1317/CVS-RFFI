"""Train-only, mean-span-protected interaction covariance contraction.

No I/O, source state, query fitting, rank search, or optimizer.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import solve_triangular
from . import d92_branch_interaction as interaction
from . import d92_branch_local_ridge as local
from .d92_branch_ridge import _readonly, _freeze, _plain


FROZEN_CONFIG = dict(
    schema='d92_within_class_metric_v1', method='D92-WithinClassMetric-v1',
    base_algorithm=deepcopy(local.FROZEN_CONFIG),
    channel=dict(route='residual', mode='post_sync', equalization_enabled=False, fs_hz=25000000),
    allowed_scenarios=['practical_high', 'practical_mid', 'practical_low_urban'],
    arms=['local_ridge', 'within_class_metric'],
    formula='R=(I-P_centered_old_means)E;W=(I+R@R.T/trace(R@R.T))^-1',
    estimation='current_path_old_training_support_only',
    coordinates='reference_difference_reduced_QR_no_rank_truncation',
    protected_rank='singular_value>128*eps64*max(1,N,coordinate_dim,C)*largest',
    unresolved_residual='E_nonzero_and_R_norm<=tolerance*E_norm_is_technical_failure',
    identity='train_K1_or_exact_per_class_equal_background_and_auxiliary',
    normalized_covariance='scale_before_Frobenius_normalization',
    contraction='T=chol(I+F@F.T)^-1@F;dW2=d02-norm(T@delta_psi)^2',
    distance='original_rank2_QR_with_bounded_correction_and_stable_direct_difference',
    trace_target='original_untransformed_interaction_centered_trace',
    C_state='inherit_frozen_B_metric_refit_all_registered_LocalRidge_head',
    n0='reuse_B', proxy='exact_baseline_identity_reuse',
    probe_k1='numerical_only_no_fit_no_holdout',
    diagnostics='leave_one_old_class_out_inside_B_trainfold_only',
    diagnostic_K1='skip', optimizer_steps=0, source_inputs=False, query_fit=False,
    phase1_frozen=True, encoder_backward=False, parameter_search=False,
    query_decision_policy='per_sample_all_registered_classes',
    tie_break='physical_class_id_lexicographic', dtype='float64')
_CONFIG = deepcopy(FROZEN_CONFIG)
_EPS = np.finfo(np.float64).eps


def _json_safe(x):
    if isinstance(x, Mapping): return {str(k): _json_safe(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return _json_safe(x.tolist())
    if isinstance(x, np.generic): return _json_safe(x.item())
    if isinstance(x, float) and not np.isfinite(x): return str(x)
    return x


class NumericalFailure(FloatingPointError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = _json_safe(deepcopy(audit))
    def audit_dict(self): return deepcopy(self.audit)


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    scale = float(np.max(np.abs(x))) if x.size else 0.
    return scale * float(np.sqrt(np.sum((x/scale)**2))) if scale else 0.


def _check(x, reason):
    if not np.isfinite(x).all(): raise FloatingPointError(reason)


def _ctx(context, stage):
    result = dict(scope='final', fold=None, trial=None, stage=stage)
    result.update(context or {})
    return result


def _emit(callback, event, audit):
    if callback is not None: callback(_json_safe(dict(event=event, **audit)))


def _delta_psi(b, a, tb, ta, qb, qa):
    """Projected feature difference; preserves small input differences first."""
    db, da = (b-tb)@qb, (a-ta)@qa
    right_a, left_b = a@qa, tb@qb
    outer = db[:, :, None]*right_a[:, None, :] + left_b[:, :, None]*da[:, None, :]
    out = np.concatenate((db, da, outer.reshape(len(db), qb.shape[1]*qa.shape[1])), axis=1)
    _check(out, 'NONFINITE_COORDINATE_DIFFERENCE')
    return out


@dataclass(frozen=True)
class WithinClassMetricState:
    identity: bool
    classes: tuple
    support_ids: tuple
    support_labels: np.ndarray
    support_background: np.ndarray
    support_auxiliary: np.ndarray
    Qb: np.ndarray
    Qa: np.ndarray
    T: np.ndarray
    audit: Mapping

    def __post_init__(self):
        for key in ('support_labels','support_background','support_auxiliary','Qb','Qa','T'):
            value = np.asarray(getattr(self,key))
            _check(value, 'NONFINITE_METRIC_STATE')
            object.__setattr__(self,key,_readonly(value))
        object.__setattr__(self,'classes',tuple(self.classes))
        object.__setattr__(self,'support_ids',tuple(self.support_ids))
        object.__setattr__(self,'audit',_freeze(self.audit))

    def audit_dict(self): return _plain(self.audit)

    def distances(self, *, z_id, fft, t_emb, f_emb, pa_local):
        b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        return _metric_distances(self,b,a)[0]


def _metric_projection(metric, b, a):
    refb,refa=metric.support_background[0],metric.support_auxiliary[0]
    us=[];norms=[]
    for bi,ai in zip(b,a):
        delta=_delta_psi(bi[None],ai[None],refb[None],refa[None],metric.Qb,metric.Qa)[0]
        us.append(metric.T@delta)
        # Use the unprojected norm: projected coordinates can vanish while
        # floating-point Q inner products retain first-order rounding error.
        norms.append(float(np.sqrt(local._pair_distances(bi[None],ai[None],refb[None],refa[None])[0])))
    return np.asarray(us).reshape(len(b),len(metric.T)),np.asarray(norms)


def _metric_distances(metric, b, a, tb=None, ta=None, train_projection=None):
    symmetric = tb is None
    if symmetric: tb,ta=b,a
    d0=local._distances(b,a,None if symmetric else tb,None if symmetric else ta)
    if metric.identity: return d0, dict(direct_difference_pair_count=0,cached_difference_pair_count=0)
    qb,qa,t=metric.Qb,metric.Qa,metric.T
    u,un=_metric_projection(metric,b,a)
    v,vn=(u,un) if symmetric else (train_projection if train_projection is not None else _metric_projection(metric,tb,ta))
    dim=max(480,t.shape[1]);gamma=dim*_EPS/(1-dim*_EPS)
    tolerance=metric.audit['numerical_tolerance']
    out=np.empty_like(d0);direct=cached=0
    for i in range(len(b)):
        for j in range(i if symmetric else 0,len(tb)):
            base=d0[i,j]
            if base==0: distance=0.
            else:
                du=u[i]-v[j];h=_norm(du)
                error=8*gamma*(un[i]+vn[j])
                if 2*h*error+error*error <= tolerance*base:
                    correction=float(du@du);cached+=1
                else:
                    delta=_delta_psi(b[i:i+1],a[i:i+1],tb[j:j+1],ta[j:j+1],qb,qa)[0]
                    du=t@delta;correction=float(du@du);direct+=1
                distance=base-correction
                if not np.isfinite(distance) or distance < (.5-tolerance)*base or distance > (1+tolerance)*base:
                    raise NumericalFailure('METRIC_DISTANCE_BOUND_VIOLATION',dict(
                        **metric.audit_dict(),original_squared_distance=float(base),
                        correction=correction,transformed_squared_distance=float(distance),
                        direct_difference_pair_count=direct,cached_difference_pair_count=cached))
            out[i,j]=distance
            if symmetric: out[j,i]=distance
    return out,dict(direct_difference_pair_count=direct,cached_difference_pair_count=cached)


def fit_within_class_metric(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,
                           support_ids,classes,context=None,log_callback=None):
    started=time.perf_counter()
    b,a,labels,ids,canonical,requested,old,k=interaction._prepare(
        z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,())
    n,c=len(ids),len(canonical)
    audit=dict(_ctx(context,'metric'),training_physical_ids=list(ids),classes=list(canonical),
        train_k=k,train_physical_count=n,metric_fit_count=1,metric_factorization_count=0,
        optimizer_steps=0,completed_stages=[])
    try:
        reason='TRAIN_K1_NO_WITHIN_CLASS_INFORMATION' if k==1 else None
        if reason is None and all(np.all(b[labels==j]==b[labels==j][0]) and
                np.all(a[labels==j]==a[labels==j][0]) for j in range(c)):
            reason='EXACT_ZERO_WITHIN_CLASS_VARIATION'
        qb,qa,t=np.empty((256,0)),np.empty((480,0)),np.empty((0,0))
        within_trace=residual_trace=scale=scaled_trace=contraction=0.
        singular=eigen=np.empty(0);rank=0;threshold=0.;tol=128*_EPS*max(n,c,1)
        protection=chol_residual=orth_error=0.;m=0
        if reason is None:
            qb=np.linalg.qr(np.column_stack((b[0],(b[1:]-b[0]).T)),mode='reduced')[0]
            qa=np.linalg.qr(np.column_stack((a[0],(a[1:]-a[0]).T)),mode='reduced')[0]
            m=qb.shape[1]+qa.shape[1]+qb.shape[1]*qa.shape[1]
            tol=128*_EPS*max(1,n,m,c)
            orth_error=max(_norm(qb.T@qb-np.eye(qb.shape[1])),_norm(qa.T@qa-np.eye(qa.shape[1])))
            if orth_error>tol: raise FloatingPointError('COORDINATE_ORTHOGONALITY_FAILED')
            for values,q in ((b,qb),(a,qa)):
                differences=np.vstack((values[0],values[1:]-values[0]))
                if _norm(differences-(differences@q)@q.T)>tol*max(_norm(differences),np.finfo(float).tiny):
                    raise FloatingPointError('COORDINATE_RECONSTRUCTION_FAILED')
            psi=_delta_psi(b,a,b[0:1],a[0:1],qb,qa)
            means=np.stack([np.mean(psi[labels==j],axis=0) for j in range(c)])
            means-=np.mean(means,axis=0)
            e=np.empty((n,m))
            for j in range(c):
                ix=np.flatnonzero(labels==j)
                differences=_delta_psi(b[ix],a[ix],b[ix[0]:ix[0]+1],a[ix[0]:ix[0]+1],qb,qa)
                e[ix]=differences-np.mean(differences,axis=0)
            _,singular,vh=np.linalg.svd(means,full_matrices=False)
            threshold=tol*float(singular[0]) if len(singular) else 0.
            basis=vh[singular>threshold].T;rank=basis.shape[1]
            r=e-(e@basis)@basis.T
            enorm,rnorm=_norm(e),_norm(r)
            if enorm==0 or rnorm<=tol*enorm:
                raise FloatingPointError('NUMERICALLY_UNRESOLVED_ZERO_RESIDUAL')
            scale=float(np.max(np.abs(r)));scaled=r/scale
            scaled_trace=float(np.sum(scaled*scaled));f=scaled/np.sqrt(scaled_trace)
            within_trace=enorm*enorm;residual_trace=rnorm*rnorm
            if residual_trace==0: residual_trace=None
            gram=f@f.T
            eigen=np.linalg.eigvalsh(gram)
            if (eigen[0]<-tol or abs(float(np.trace(gram))-1)>tol or eigen[-1]>1+tol):
                raise FloatingPointError('NORMALIZED_COVARIANCE_PSD_TRACE_FAILED')
            audit['metric_factorization_count']=1
            matrix=np.eye(n)+gram;chol=np.linalg.cholesky(matrix)
            t=solve_triangular(chol,f,lower=True)
            chol_residual=_norm(chol@chol.T-matrix)/max(_norm(matrix),1.)
            contraction=float(np.linalg.eigvalsh(t@t.T)[-1])
            protection=_norm(means@t.T)/max(_norm(means),np.finfo(float).tiny)
            if chol_residual>tol or contraction>.5+tol or protection>tol:
                raise FloatingPointError('METRIC_FACTORIZATION_OR_PROTECTION_FAILED')
            _check(t,'NONFINITE_CONTRACTION_FACTOR')
        arrays=dict(support_labels=labels,support_background=b,support_auxiliary=a,Qb=qb,Qa=qa,T=t)
        size={key:int(val.nbytes) for key,val in arrays.items()}
        concentration=None if reason else dict(largest=float(eigen[-1]),
            squared_sum=float(eigen@eigen),effective_rank=float(1/(eigen@eigen)))
        audit.update(identity=reason is not None,identity_reason=reason,
            status='EXACT_IDENTITY' if reason else 'METRIC_SOLVED',coordinate_dim=m,
            protected_rank=rank,protected_singular_values=singular.tolist(),
            protected_rank_threshold=threshold,within_trace=within_trace,residual_trace=residual_trace,
            residual_trace_scale=scale,residual_scaled_trace=scaled_trace,
            residual_trace_unavailable_reason='FLOAT64_SQUARED_TRACE_UNDERFLOW' if residual_trace is None else None,
            normalized_residual_eigenvalues=eigen.tolist(),spectral_concentration=concentration,
            contraction_max=contraction,numerical_tolerance=tol,protection_relative_error=protection,
            cholesky_relative_residual=chol_residual,coordinate_orthogonality_error=orth_error,
            state_array_bytes=size,state_scalar_bytes=0,persistent_state_bytes=sum(size.values()),
            state_byte_scope='numeric_arrays_excludes_audit_ids_python_and_serialization',
            fit_seconds=time.perf_counter()-started)
        state=WithinClassMetricState(reason is not None,canonical,ids,labels,b,a,qb,qa,t,audit)
        _emit(log_callback,'WITHIN_CLASS_METRIC_FIT',audit)
        return state
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure): raise
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),fit_seconds=time.perf_counter()-started)
        raise NumericalFailure(str(exc),audit) from exc


def _check_metric_binding(metric,b,a,labels,ids,canonical):
    if {pid for i,pid in enumerate(ids) if canonical[labels[i]] in metric.classes}!=set(metric.support_ids):
        raise ValueError('Metric old physical support set changed')
    positions={pid:i for i,pid in enumerate(ids)}
    for j,pid in enumerate(metric.support_ids):
        if pid not in positions: raise ValueError('Metric old physical support missing')
        i=positions[pid]
        if canonical[labels[i]]!=metric.classes[int(metric.support_labels[j])]:
            raise ValueError('Metric old physical class changed')
        if not np.array_equal(b[i],metric.support_background[j]) or not np.array_equal(a[i],metric.support_auxiliary[j]):
            raise ValueError('Metric old physical features changed')


@dataclass(frozen=True)
class WithinClassLocalRidgeState:
    base_state: local.BranchLocalRidgeState
    metric: WithinClassMetricState
    audit: Mapping
    support_projection: np.ndarray | None = None
    support_reference_norm: np.ndarray | None = None
    def __post_init__(self):
        object.__setattr__(self,'audit',_freeze(self.audit))
        for key in ('support_projection','support_reference_norm'):
            if getattr(self,key) is not None: object.__setattr__(self,key,_readonly(getattr(self,key)))
    @property
    def classes(self): return self.base_state.classes
    def audit_dict(self): return _plain(self.audit)
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        if self.metric.identity:
            return self.base_state.score(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local)
        b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        state=self.base_state
        scores=np.zeros((len(b),len(state.classes)))
        if state.trace_scale is None: return scores
        for i in range(len(b)):
            distance=_metric_distances(self.metric,b[i:i+1],a[i:i+1],state.support_background,state.support_auxiliary,
                train_projection=(self.support_projection,self.support_reference_norm))[0][0]
            raw=local._radial_minus_one(distance,state.bandwidth_tau)
            diff=(raw-raw[0])-state.reference_kernel+state.reference_self
            scores[i]=(state.trace_scale*(diff-diff.mean()-state.center_mean+state.center_grand))@state.alpha
        _check(scores,'NONFINITE_METRIC_HEAD_SCORE')
        return scores
    def predict(self,**features):
        order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(self.score(**features)[:,order],axis=1)]]


def fit_within_class_local_ridge(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,
        support_ids,classes,old_classes,metric,baseline_state=None,context=None,log_callback=None):
    started=time.perf_counter()
    b,a,labels,ids,canonical,requested,old,k=interaction._prepare(
        z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes)
    _check_metric_binding(metric,b,a,labels,ids,canonical)
    if set(old)!=set(metric.classes): raise ValueError('Metric old class set mismatch')
    n,c=len(ids),len(canonical)
    audit=dict(_ctx(context,'metric_head'),training_physical_ids=list(ids),classes=list(requested),
        train_k=k,parent_k=(context or {}).get('parent_k',k),train_physical_count=n,head_factorization_count=0,
        metric_head_fit_count=0 if metric.identity else 1,optimizer_steps=0,completed_stages=[])
    if metric.identity:
        if baseline_state is None: raise ValueError('Identity metric requires corresponding baseline_state')
        ba=baseline_state.audit_dict()
        if (tuple(baseline_state.classes)!=requested or
            ba['final_fit']['training_physical_ids']!=list(ids) or
            not np.array_equal(baseline_state.support_background,b) or
            not np.array_equal(baseline_state.support_auxiliary,a)):
            raise ValueError('Identity baseline binding mismatch')
        audit.update(ba)
        audit.update(dict(_ctx(context,'metric_head'),train_k=k,train_physical_count=n,
            classes=list(requested),training_physical_ids=list(ids),
            metric_head_fit_count=0,head_factorization_count=0,
            metric_identity_reuse=True,shared_metric_state_bytes=metric.audit['persistent_state_bytes'],
            head_state_bytes=ba['persistent_state_bytes'],persistent_state_bytes=ba['persistent_state_bytes']+metric.audit['persistent_state_bytes'],
            transformed_interaction_centered_trace=ba['final_fit']['interaction_centered_trace'],
            direct_difference_pair_count=0,fit_seconds=time.perf_counter()-started))
        _emit(log_callback,'WITHIN_CLASS_HEAD_FIT',audit)
        return WithinClassLocalRidgeState(baseline_state,metric,audit)
    try:
        d0=local._distances(b,a);distance,counts=_metric_distances(metric,b,a)
        s0=float(d0[np.triu_indices(n,1)].sum()/n)
        sw=float(distance[np.triu_indices(n,1)].sum()/n)
        target=np.eye(c)[labels]-1/c;tol=128*_EPS*max(n,c)
        tau=None if c==1 else float(np.median(np.min(np.where(labels[:,None]!=labels[None,:],distance,np.inf),axis=1)))
        factorized=c>1 and s0>0
        alpha=np.zeros((n,c));reference=mean=np.zeros(n);refself=grand=0.
        gamma=None;sradial=None;centered=np.zeros((n,n));trace_error=normal=solve_seconds=0.
        reason='SINGLE_REGISTERED_CLASS' if c==1 else ('IDENTICAL_COMPLETE_FEATURES' if s0==0 else ('ZERO_BANDWIDTH_EQUIVALENCE_KERNEL' if tau==0 else None))
        if factorized:
            rm1=local._radial_minus_one(distance,tau)
            sradial=float(-2*rm1[np.triu_indices(n,1)].sum()/n)
            if sradial<=0: raise FloatingPointError('NONPOSITIVE_RADIAL_CENTERED_TRACE')
            gamma=s0/sradial
            centered,reference,refself,mean,grand=interaction._center_kernel(rm1)
            centered*=gamma
            trace_error=abs(float(np.trace(centered))-s0)/s0
            if trace_error>tol: raise FloatingPointError('CENTERED_TRACE_RESIDUAL_EXCEEDED')
            matrix=centered+np.eye(n);tick=time.perf_counter()
            audit['head_factorization_count']=1
            chol=np.linalg.cholesky(matrix)
            alpha=solve_triangular(chol.T,solve_triangular(chol,target,lower=True),lower=False)
            solve_seconds=time.perf_counter()-tick
            normal=_norm(matrix@alpha-target)/((1+s0)*_norm(alpha)+_norm(target))
            if normal>tol: raise FloatingPointError('NORMAL_EQUATION_RESIDUAL_EXCEEDED')
        fitted=centered@alpha
        loss_data=float(.5*np.sum((fitted-target)**2));loss_ridge=float(.5*np.sum(alpha*fitted))
        _check(alpha,'NONFINITE_METRIC_HEAD');_check(np.asarray([loss_data,loss_ridge]),'NONFINITE_HEAD_LOSS')
        final=dict(_ctx(context,'metric_head'),arm='within_class_metric',training_physical_ids=list(ids),
            train_k=k,train_physical_count=n,class_count=c,bandwidth_tau=tau,
            interaction_centered_trace=s0,transformed_interaction_centered_trace=sw,
            radial_centered_trace=sradial,trace_scale=gamma,trace_relative_error=trace_error,
            normal_equation_residual=normal,numerical_tolerance=tol,degeneracy_reason=reason,
            status='CLOSED_FORM_SOLVED' if factorized else 'EXACT_ZERO_CLASSIFIER',
            factorization_calls=audit['head_factorization_count'],optimizer_steps=0,
            physical_loss_mass=float(n),sample_weight=1.,ridge_coefficient=1.,
            loss_data=loss_data,loss_ridge=loss_ridge,loss_total=loss_data+loss_ridge,
            all_states_estimated_from_trainfold_only=True,solve_seconds=solve_seconds,
            gram_seconds=time.perf_counter()-started-solve_seconds,fit_seconds=time.perf_counter()-started,
            **counts)
        alpha=alpha[:,[canonical.index(cls) for cls in requested]]
        projection,reference_norm=_metric_projection(metric,b,a)
        size=int(b.nbytes+a.nbytes+alpha.nbytes+reference.nbytes+mean.nbytes+projection.nbytes+reference_norm.nbytes+16+8*(tau is not None)+8*(gamma is not None))
        audit.update(config=deepcopy(_CONFIG),final_fit=final,metric_identity_reuse=False,
            factorization_count=audit['head_factorization_count'],transformed_interaction_centered_trace=sw,
            shared_metric_state_bytes=metric.audit['persistent_state_bytes'],head_state_bytes=size,
            persistent_state_bytes=size+metric.audit['persistent_state_bytes'],
            fit_seconds=time.perf_counter()-started,**counts)
        base=local.BranchLocalRidgeState(b,a,alpha,reference,refself,mean,grand,tau,gamma,requested,audit)
        state=WithinClassLocalRidgeState(base,metric,audit,projection,reference_norm)
        _emit(log_callback,'WITHIN_CLASS_HEAD_FIT',audit)
        return state
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):
            exc.audit.update(audit);raise
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),fit_seconds=time.perf_counter()-started)
        raise NumericalFailure(str(exc),audit) from exc


def _contraction_energy(metric,b,a,labels):
    trace=energy=0.
    for cls in sorted(set(labels.tolist())):
        ix=np.flatnonzero(labels==cls)
        distances=local._distances(b[ix],a[ix]);base=float(distances[np.triu_indices(len(ix),1)].sum()/len(ix))
        trace+=base
        if base and not metric.identity:
            transformed=_metric_distances(metric,b[ix],a[ix])[0]
            energy+=float((distances-transformed)[np.triu_indices(len(ix),1)].sum()/len(ix))
    ratio=energy/trace if trace else None
    if ratio is not None and not -metric.audit['numerical_tolerance']<=ratio<=.5+metric.audit['numerical_tolerance']:
        raise NumericalFailure('LOCO_CONTRACTION_ENERGY_BOUND',metric.audit_dict())
    return dict(within_trace=trace,contraction_energy=energy,contraction_fraction=ratio,
        unavailable_reason='EXACT_ZERO_WITHIN_CLASS_VARIATION' if trace==0 else None)


def diagnose_leave_one_class_out(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,
        support_ids,classes,context=None,log_callback=None):
    b,a,labels,ids,canonical,requested,old,k=interaction._prepare(
        z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,())
    result=dict(status='SKIPPED_TRAIN_K1' if k==1 else 'COMPLETE',classes=list(canonical),folds=[],
        diagnostic_fit_count=0,diagnostic_factorization_count=0,diagnostic_fit_seconds=0.,
        diagnostic_score_seconds=0.,optimizer_steps=0)
    if k==1: return result
    if len(canonical)<2: raise ValueError('LOCO requires at least two old classes')
    # Slice original input rows, preserving their original per-class label mapping.
    rawids=list(support_ids);rawlabels=np.asarray(support_labels)
    original={name:np.asarray(value) for name,value in dict(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local).items()}
    for held in canonical:
        keep=rawlabels!=requested.index(held)
        subset=tuple(cls for cls in requested if cls!=held)
        remap={requested.index(cls):subset.index(cls) for cls in subset}
        ctx=dict(context or {});ctx.update(stage='loco_metric',held_class=held)
        try:
            fitted=fit_within_class_metric(**{name:value[keep] for name,value in original.items()},
                support_labels=np.array([remap[int(x)] for x in rawlabels[keep]]),
                support_ids=[pid for i,pid in enumerate(rawids) if keep[i]],classes=subset,context=ctx)
            fa=fitted.audit_dict();result['diagnostic_fit_count']+=1
            result['diagnostic_factorization_count']+=fa['metric_factorization_count']
            result['diagnostic_fit_seconds']+=fa['fit_seconds']
            tick=time.perf_counter();ix=labels==canonical.index(held)
            evaluation=_contraction_energy(fitted,b[ix],a[ix],labels[ix])
            training=_contraction_energy(fitted,b[~ix],a[~ix],labels[~ix])
            seconds=time.perf_counter()-tick;result['diagnostic_score_seconds']+=seconds
            row=dict(held_class=held,training_physical_ids=list(fitted.support_ids),
                held_physical_ids=[pid for i,pid in enumerate(ids) if ix[i]],
                held_class_within_trace=evaluation['within_trace'],
                held_class_contraction_fraction=evaluation['contraction_fraction'],
                held_class_unavailable_reason=evaluation['unavailable_reason'],
                training=training,metric_audit=fa,diagnostic_score_seconds=seconds)
            result['folds'].append(row)
            _emit(log_callback,'WITHIN_CLASS_LOCO',dict(ctx,**row))
        except NumericalFailure as exc:
            exc.audit['completed_stages']=deepcopy(result['folds'])
            exc.audit['diagnostic_progress']=deepcopy(result);raise
    return result
