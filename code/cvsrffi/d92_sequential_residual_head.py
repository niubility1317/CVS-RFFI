"""Support-only, inherited rank-eight residual classifier on fixed LocalRidge scores."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
import time
import numpy as np
from scipy.special import erf

from .d92_branch_local_ridge import FROZEN_CONFIG as BASE_CONFIG, fit_branch_local_ridge
from .d92_branch_ridge import _freeze, _plain, _readonly
from .d92_branch_support_probe import _strings

_CONFIG = dict(schema='d92_sequential_residual_head_v1',method='D92-LocalRidge-SequentialResidualHead8-v1',
    base_algorithm=deepcopy(BASE_CONFIG),
    channel=dict(route='residual',mode='post_sync',equalization_enabled=False,fs_hz=25000000),
    allowed_scenarios=['practical_high','practical_mid','practical_low_urban'],
    arms=['local_ridge','residual_reset','residual_seq'],selected='residual_seq',
    identity_dim=160,rank=8,norm_floor=1e-12,dtype='float64',
    initialization='nonzero_DCT_columns_1_to_8_U_and_zero_V',initialization_seed=None,
    residual='GELU(sqrt(160)*unit(z_id)@U)@V/sqrt(8)',activation='exact_erf_GELU',bias=False,
    score='frozen_current_stage_LocalRidge_scores/q_plus_residual',
    score_scale='train_RMS_class_range_stable_norm_exact_zero_to_1',
    objective='mean_CE+(squared_U_anchor_distance+squared_V_anchor_distance)/(2*N)',
    physical_sum_proximal_coefficient=1.0,sample_weight=1.0,
    optimizer='Adam_full_batch',steps=64,learning_rate=0.01,beta1=0.9,beta2=0.999,
    adam_epsilon=1e-8,global_gradient_norm_clip=1.0,weight_decay=0.0,
    initialization_B='DCT_U_zero_V',initialization_C_reset='DCT_U_zero_V',
    initialization_C_seq='inherit_B_U_and_class_bound_old_V_append_zero_new_V',
    proximal_anchor='stage_initial_parameters',optimizer_state_inheritance=False,
    C_base='refit_original_LocalRidge_all_registered_training_support',
    sharing='same_base_and_q_for_C_reset_and_seq_same_B_residual_for_both',
    n0='reuse_B_model_scores_and_optimizer_step_count_no_C_training',
    probe_k1='numerical_only_no_fit_no_holdout',full_support_k1='same_fixed_supervised_training_no_generalization_claim',
    physical_folds='per_class_physical_id_sort_position_mod_min_K_3',
    proxy='all_per_class_sorted_anchor_positions_parent_equal_mean',
    query_decision_policy='per_sample_all_registered_classes',tie_break='physical_class_id_lexicographic',
    source_inputs=False,summary_inputs=False,query_fit=False,phase1_frozen=True,encoder_backward=False,
    trace='64_post_update_events_with_pre_update_loss_and_gradient_final_loss_separate',
    early_stopping=False,parameter_search=False)
FROZEN_CONFIG=deepcopy(_CONFIG)


def _json_safe(value):
    if isinstance(value, Mapping):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return _json_safe(value.tolist())
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value,np.generic):return _json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return 'NaN' if math.isnan(value) else ('Infinity' if value>0 else '-Infinity')
    return value


class NumericalFailure(FloatingPointError):
    def __init__(self,message,audit):
        super().__init__(message);self.audit=_json_safe(audit)
    def audit_dict(self):return deepcopy(self.audit)


def _finite(*values):
    if any(not np.isfinite(v).all() for v in values):raise FloatingPointError('Nonfinite residual computation')


def _matrix(value,columns,name,allow_empty=False):
    x=np.asarray(value)
    if x.ndim!=2 or x.shape[1]!=columns or x.dtype.kind not in 'fiu' or (not allow_empty and not len(x)):
        raise ValueError('Invalid '+name+' shape/type')
    x=np.asarray(x,dtype=np.float64);_finite(x);return x


def _norm(x):
    """Stable Euclidean norm, including exact zero; never square unscaled values."""
    x=np.asarray(x,dtype=np.float64)
    scale=float(np.max(np.abs(x))) if x.size else 0.
    if scale==0:return 0.
    result=scale*float(np.sqrt(np.sum(np.square(x/scale))))
    _finite(result);return result


def _unit_z(z_id,allow_empty=False):
    z=_matrix(z_id,160,'z_id',allow_empty)
    if not len(z):return z.copy()
    scale=np.max(np.abs(z),axis=1,keepdims=True)
    reduced=z/np.where(scale>0,scale,1.)
    reduced_norm=np.sqrt(np.sum(reduced*reduced,axis=1,keepdims=True))
    # Compare safely with the original norm floor without overflowing large norms.
    safe_norm=np.where(reduced_norm>0,reduced_norm,1.)
    use_norm=scale>=_CONFIG['norm_floor']/safe_norm
    out=reduced/safe_norm;small=~use_norm[:,0]
    out[small]=z[small]/_CONFIG['norm_floor']
    return out


def score_scale(base_scores):
    scores=np.asarray(base_scores,dtype=np.float64)
    if scores.ndim!=2 or not scores.shape[0] or not scores.shape[1]:raise ValueError('Nonempty training score matrix required')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        _finite(scores);ranges=np.max(scores,axis=1)-np.min(scores,axis=1)
        scale=float(np.max(ranges))
        if scale==0:return 1.
        q=scale*float(np.sqrt(np.mean(np.square(ranges/scale))))
        if q==0:raise FloatingPointError('Nonzero score range RMS is below float64 representability')
        _finite(q)
    return q


@dataclass(frozen=True)
class PreparedResidualTraining:
    z: np.ndarray
    base_logits: np.ndarray
    labels: np.ndarray
    support_ids: tuple
    canonical_classes: tuple
    classes: tuple
    q: float
    train_k: int
    prepare_seconds: float

    def __post_init__(self):
        n=len(self.z);c=len(self.classes)
        if (self.z.shape!=(n,160) or self.base_logits.shape!=(n,c) or self.labels.shape!=(n,)
                or not n or self.canonical_classes!=tuple(sorted(self.classes))
                or len(set(self.classes))!=c or len(self.support_ids)!=n
                or len(set(self.support_ids))!=n or not math.isfinite(self.q) or self.q<=0
                or self.labels.dtype.kind not in 'iu' or set(self.labels.tolist())!=set(range(c))
                or type(self.train_k) is not int or self.train_k<=0
                or any(np.count_nonzero(self.labels==i)!=self.train_k for i in range(c))):
            raise ValueError('Invalid prepared residual training state')
        _finite(self.z,self.base_logits)
        for key in ('z','base_logits'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        labels=np.ascontiguousarray(self.labels,dtype=np.int64)
        object.__setattr__(self,'labels',np.frombuffer(labels.tobytes(),dtype=np.int64))

    @property
    def numerical_bytes(self):return int(self.z.nbytes+self.base_logits.nbytes+self.labels.nbytes+8)


def prepare_residual_training(*,z_id,base_scores,support_labels,support_ids,classes):
    started=time.perf_counter();ids=_strings(support_ids,'support_ids');requested=_strings(classes,'classes')
    try:
        z=_unit_z(z_id);scores=_matrix(base_scores,len(requested),'base_scores')
    except (FloatingPointError,OverflowError) as exc:
        raise NumericalFailure(str(exc),dict(status='TECHNICAL_FAILURE',stage='prepare_inputs',
            training_physical_ids=list(ids),classes=list(requested),optimizer_steps=0,completed_stages=[])) from exc
    labels=np.asarray(support_labels);n=len(z);c=len(requested)
    if (len(scores)!=n or len(ids)!=n or labels.shape!=(n,) or labels.dtype.kind not in 'iu'
            or set(labels.tolist())!=set(range(c))):raise ValueError('Invalid physical support labels/counts')
    counts=np.bincount(labels.astype(np.int64),minlength=c)
    if np.any(counts!=counts[0]):raise ValueError('Equal positive physical K required')
    canonical=tuple(sorted(requested));columns=[requested.index(cls) for cls in canonical]
    remap=np.array([canonical.index(cls) for cls in requested]);order=np.array(sorted(range(n),key=lambda i:ids[i]))
    scores=scores[order][:,columns]
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            q=score_scale(scores);base=scores/q;_finite(base)
    except (FloatingPointError,OverflowError) as exc:
        raise NumericalFailure(str(exc),dict(status='TECHNICAL_FAILURE',stage='prepare_scale',
            training_physical_ids=list(ids),classes=list(requested),optimizer_steps=0,completed_stages=[])) from exc
    return PreparedResidualTraining(z[order],base,remap[labels.astype(np.int64)][order],
        tuple(ids[i] for i in order),canonical,requested,q,int(counts[0]),time.perf_counter()-started)


def fit_residual_base(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes=()):
    """Optional convenience wrapper: one original fit, one training score call."""
    features=dict(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local)
    state=fit_branch_local_ridge(**features,support_labels=support_labels,support_ids=support_ids,
        classes=classes,old_classes=old_classes,arm='local_ridge')
    started=time.perf_counter();scores=state.score(**features);seconds=time.perf_counter()-started
    prepared=prepare_residual_training(z_id=z_id,base_scores=scores,support_labels=support_labels,
        support_ids=support_ids,classes=classes)
    audit=state.audit_dict();audit.update(training_score_seconds=seconds,prepare_seconds=prepared.prepare_seconds,
        q=prepared.q,prepared_numerical_bytes=prepared.numerical_bytes)
    return state,prepared,audit


def _initial_u():
    a=np.arange(160,dtype=np.float64)[:,None];b=np.arange(1,9,dtype=np.float64)[None,:]
    return math.sqrt(2./160)*np.cos(np.pi*(a+.5)*b/160)


def _activation(z,U):
    a=math.sqrt(160.)*(z@U);gaussian=np.exp(-.5*a*a)/math.sqrt(2.*math.pi)
    cdf=.5*(1.+erf(a/math.sqrt(2.)))
    return a*cdf/math.sqrt(8.),(cdf+a*gaussian)/math.sqrt(8.)


def _objective_gradient(prepared,U,V,anchor_U,anchor_V):
    """Analytic mean objective; inputs already canonical and normalized."""
    n=len(prepared.z);h,dh=_activation(prepared.z,U);residual=h@V
    logits=prepared.base_logits+residual
    maximum=np.max(logits,axis=1,keepdims=True);shifted=logits-maximum
    exponents=np.exp(shifted);denominator=exponents.sum(axis=1,keepdims=True)
    ce=float(np.mean(np.log(denominator[:,0])-shifted[np.arange(n),prepared.labels]))
    delta=exponents/denominator;delta[np.arange(n),prepared.labels]-=1.;delta/=n
    difference_U=U-anchor_U;difference_V=V-anchor_V
    reg_U=float(np.sum(difference_U*difference_U)/(2*n));reg_V=float(np.sum(difference_V*difference_V)/(2*n))
    grad_V=h.T@delta+difference_V/n
    grad_U=math.sqrt(160.)*prepared.z.T@((delta@V.T)*dh)+difference_U/n
    total=ce+reg_U+reg_V
    _finite(total,grad_U,grad_V,logits)
    stats=dict(loss_ce=ce,loss_proximal_U=reg_U,loss_proximal_V=reg_V,loss_total=total,
        training_accuracy=float(np.mean(logits.argmax(axis=1)==prepared.labels)),
        residual_rms=_norm(residual)/math.sqrt(residual.size),base_logits_rms=_norm(prepared.base_logits)/math.sqrt(logits.size))
    return stats,grad_U,grad_V


@dataclass(frozen=True)
class ResidualHeadState:
    U: np.ndarray
    V: np.ndarray
    q: float
    classes: tuple
    training_ids: tuple
    training_class_ids: tuple
    audit: Mapping

    def __post_init__(self):
        classes=_strings(self.classes,'classes');ids=_strings(self.training_ids,'training_ids')
        if self.U.shape!=(160,8) or self.V.shape!=(8,len(classes)) or not math.isfinite(self.q) or self.q<=0:
            raise ValueError('Invalid residual state')
        _finite(self.U,self.V)
        if len(self.training_class_ids)!=len(ids) or not set(self.training_class_ids)<=set(classes):raise ValueError('Invalid training identity mapping')
        for key in ('U','V'):object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'classes',classes);object.__setattr__(self,'training_ids',ids)
        object.__setattr__(self,'training_class_ids',tuple(self.training_class_ids))
        object.__setattr__(self,'audit',_freeze(self.audit))

    def score(self,*,z_id,base_scores):
        try:
            z=_unit_z(z_id,allow_empty=True);base=_matrix(base_scores,len(self.classes),'base_scores',allow_empty=True)
            if len(base)!=len(z):raise ValueError('Prediction row count mismatch')
            # Independent vector kernels preserve the per-physical-sample decision rule.
            output=np.empty_like(base)
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                for i in range(len(z)):
                    h,_=_activation(z[i:i+1],self.U);output[i]=base[i]/self.q+(h@self.V)[0]
            _finite(output);return output
        except (FloatingPointError,OverflowError) as exc:
            raise NumericalFailure(str(exc),dict(self.audit_dict(),status='TECHNICAL_FAILURE',
                failure_stage='score',failure_reason=str(exc),last_finite_U=self.U,last_finite_V=self.V)) from exc

    def predict(self,**inputs):
        scores=self.score(**inputs);order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:,order],axis=1)]]

    def audit_dict(self):return _plain(self.audit)


def _anchors(prepared,inherited):
    c=len(prepared.canonical_classes)
    if inherited is None:return _initial_u(),np.zeros((8,c),dtype=np.float64)
    if not isinstance(inherited,ResidualHeadState):raise ValueError('Inherited state must be a ResidualHeadState')
    if not set(inherited.classes)<set(prepared.classes):raise ValueError('C inheritance requires added classes; N0 must directly reuse B')
    old=set(inherited.classes)
    physical={pid:prepared.canonical_classes[label] for pid,label in zip(prepared.support_ids,prepared.labels) if prepared.canonical_classes[label] in old}
    expected=dict(zip(inherited.training_ids,inherited.training_class_ids))
    if physical!=expected:raise ValueError('C old physical support/class binding differs from B')
    V=np.zeros((8,c),dtype=np.float64)
    for j,cls in enumerate(inherited.classes):V[:,prepared.canonical_classes.index(cls)]=inherited.V[:,j]
    return inherited.U.copy(),V


def train_residual_head(prepared,*,inherited=None,context=None,log_callback=None):
    if not isinstance(prepared,PreparedResidualTraining):raise ValueError('Use prepare_residual_training first')
    started=time.perf_counter();anchor_U,anchor_V=_anchors(prepared,inherited)
    U=anchor_U.copy();V=anchor_V.copy();n=len(prepared.z);c=len(prepared.classes)
    mU=np.zeros_like(U);mV=np.zeros_like(V);vU=np.zeros_like(U);vV=np.zeros_like(V)
    context=deepcopy(context or {});steps=0;forward_backward=0.;optimizer_seconds=0.;callback_seconds=0.
    context.update(training_physical_ids=list(prepared.support_ids),classes=list(prepared.classes),
        train_physical_count=n,class_count=c,train_k=prepared.train_k,q=prepared.q,
        rank=8,max_steps=64,learning_rate=.01,beta1=.9,beta2=.999,adam_epsilon=1e-8,
        global_gradient_norm_clip=1.,proximal_coefficient_physical_sum=1.,
        inherited=inherited is not None,optimizer_state_inherited=False,new_column_count=c-(len(inherited.classes) if inherited is not None else 0),
        inheritance_binding_checked='physical_ids_and_class_mapping_only' if inherited is not None else None,
        inheritance_old_feature_equality_checked=False if inherited is not None else None,
        feature_binding_responsibility='caller_same_immutable_cache_physical_rows',
        initialization_seed=None,optimizer='Adam_full_batch',dtype='float64')
    last_stats=None;last_event=None
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for step in range(1,65):
                begin=time.perf_counter();stats,gU,gV=_objective_gradient(prepared,U,V,anchor_U,anchor_V)
                forward_backward+=time.perf_counter()-begin
                norm=math.hypot(_norm(gU),_norm(gV));_finite(norm)
                clip=1./norm if norm>1. else 1.;gU*=clip;gV*=clip
                begin=time.perf_counter()
                mU=.9*mU+.1*gU;mV=.9*mV+.1*gV;vU=.999*vU+.001*gU*gU;vV=.999*vV+.001*gV*gV
                update_U=.01*(mU/(1-.9**step))/(np.sqrt(vU/(1-.999**step))+1e-8)
                update_V=.01*(mV/(1-.9**step))/(np.sqrt(vV/(1-.999**step))+1e-8)
                next_U=U-update_U;next_V=V-update_V;_finite(next_U,next_V,mU,mV,vU,vV)
                U=next_U;V=next_V;steps=step;optimizer_seconds+=time.perf_counter()-begin
                event=dict(context,step=step,optimizer_steps=step,state_timing='loss_gradient_pre_update_parameters_post_update',
                    **stats,gradient_norm_pre_clip=norm,gradient_norm_post_clip=norm*clip,gradient_clip_scale=clip,
                    gradient_zero=norm==0.,post_U_norm=_norm(U),post_V_norm=_norm(V),
                    post_U_anchor_distance=_norm(U-anchor_U),post_V_anchor_distance=_norm(V-anchor_V),
                    update_norm=math.hypot(_norm(update_U),_norm(update_V)),elapsed_seconds=time.perf_counter()-started)
                last_stats=stats;last_event=event
                if log_callback is not None:
                    begin=time.perf_counter();log_callback(deepcopy(event));callback_seconds+=time.perf_counter()-begin
            begin=time.perf_counter();final,_,_=_objective_gradient(prepared,U,V,anchor_U,anchor_V)
            final_evaluation_seconds=time.perf_counter()-begin
    except Exception as exc:
        raise NumericalFailure(str(exc),dict(context,status='TECHNICAL_FAILURE',failure_reason=str(exc),
            error_type=type(exc).__name__,optimizer_steps=steps,completed_stages=[],last_pre_update_stats=last_stats,
            last_step_event=last_event,last_finite_U=U,last_finite_V=V,
            anchor_U=anchor_U,anchor_V=anchor_V,fit_seconds=time.perf_counter()-started)) from exc
    requested_columns=[prepared.canonical_classes.index(cls) for cls in prepared.classes]
    audit=dict(context,status='RESIDUAL_HEAD_COMPLETE',optimizer_steps=steps,training_stages=1,
        factorization_count=0,factorization_calls=0,config=deepcopy(_CONFIG),
        **final,final_state_timing='after_64_updates',fit_seconds=time.perf_counter()-started,
        forward_backward_seconds=forward_backward,optimizer_seconds=optimizer_seconds,
        callback_seconds=callback_seconds,final_evaluation_seconds=final_evaluation_seconds,
        prepare_seconds=prepared.prepare_seconds,prepared_numerical_bytes=prepared.numerical_bytes,
        trainable_parameter_count=int(U.size+V.size),parameter_bytes=int(U.nbytes+V.nbytes),
        deployment_numerical_bytes=int(U.nbytes+V.nbytes+8),persistent_state_bytes=int(U.nbytes+V.nbytes+8),
        optimizer_moment_bytes=int(mU.nbytes+mV.nbytes+vU.nbytes+vV.nbytes),
        anchor_bytes=int(anchor_U.nbytes+anchor_V.nbytes),gradient_bytes=int(gU.nbytes+gV.nbytes),
        post_U_anchor_distance=_norm(U-anchor_U),post_V_anchor_distance=_norm(V-anchor_V),
        query_rows_used=0,source_rows_used=0,encoder_backward=False,held_used_for_training=False,
        trace_steps_emitted=steps if log_callback is not None else 0,
        all_states_estimated_from_trainfold_only=True,source_validation=None,
        source_validation_unavailable_reason='SOURCE_ACCESS_FORBIDDEN',peak_rss_bytes=None,
        peak_rss_unavailable_reason='Measured by outer process; array accounting is not RSS')
    return ResidualHeadState(U,V[:,requested_columns],prepared.q,prepared.classes,prepared.support_ids,
        tuple(prepared.canonical_classes[label] for label in prepared.labels),audit)
