"""Independent group-balanced gate / support-metric GGN1 candidate.

Only supplied legal support records are used. Gate weights use only each head train labels. Frozen prototypes define geometry,
never teacher targets. No checkpoint, query label, source sample or experiment I/O.
Five-direction analytic head helpers are reused; coordinates here belong to U,
and cannot be lifted from another method's Q-coordinate state.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace
import math
import time

import numpy as np

from . import d92_proto_frame_joint_local_ridge as geometry
from . import d92_proto_frame_primitives as primitive
from . import d92_support_metric_basis as basis_module
from . import d92_support_metric_step as metric_module
from . import d92_group_balanced_barrier_gate as balanced_gate

BRANCHES=geometry.BRANCHES
SCHEMA='d92_group_balanced_support_metric_joint_local_ridge_v1'
METHOD='D92-GroupBalancedSupportMetric-GGN1-LocalRidge'
FROZEN_CONFIG=dict(schema=SCHEMA,method=METHOD,nominal_coordinates=5,
    coordinates='EXACT_STORED_Q_RANK_PHYSICAL_U',kappa=.25,ridge_coefficient=1.,
    objective='RMS_class_mean_OOF_CE',damping='ACTUAL_U_GRAM_PLUS_CLASS_BALANCED_PREDICTION_FISHER',
    radius=.5,max_updates=1,max_trials=12,initial_step=1.,backtrack=.5,armijo=1e-4,
    float_comparison_multiplier=128,folds='physical_ID_class_rank_mod_min_K_3',
    prototype_role='FROZEN_GEOMETRY_ONLY',C_anchor='ACTUAL_CURRENT_B_U_THETA',
    old_conditional='FROZEN_ACTUAL_B_FUNCTION',
    kernel=geometry.FROZEN_CONFIG['kernel'],barrier_average_objective_gap=1e-4,
    phase1_frozen=True,source_examples=False,query_fit=False,parameter_search=False,
    encoder_backward=False,coordinate_lift_from_prior_methods=False,
    gate_supervision='GROUP_BALANCED_CLASS_BALANCED_CURRENT_TRAIN',
    gate_weight_rule='N/(2*C_group*n_class)',gate_weight_scope='CURRENT_FOLD_LEGAL_TRAIN_LABELS_ONLY',
    evidence_level='FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE')
EPS=np.finfo(np.float64).eps
_BASIS_SUM=('fraction_operations_attempted','fraction_operations_completed',
    'integer_operations_attempted','integer_operations_completed','integer_guard_checks',
    'binary64_decodes','float_conversions','nextafter_steps','square_root_enclosures',
    'columns_started','columns_completed','numeric_array_payload_bytes',
    'logical_rational_integer_payload_bytes','certificate_utf8_bytes','wall_seconds')
_BASIS_MAX=('max_observed_integer_bits','max_intermediate_bit_bound')
AUDIT_COUNTERS=geometry.AUDIT_COUNTERS+('basis_calls','basis_binding_calls',
    'basis_binding_wall_seconds','basis_binding_physical_gram_evaluation_count','support_metric_step_calls')+tuple(
    'basis_'+k for k in _BASIS_SUM)+tuple('support_metric_step_'+k for k in metric_module.AUDIT_SUM_KEYS)+tuple(
    operation+'_'+key for operation in ('gate_forward','gate_jvp') for key in balanced_gate.WEIGHT_SUM_KEYS)
PEAK_COUNTERS=geometry.PEAK_COUNTERS+tuple('basis_'+k for k in _BASIS_MAX)+tuple(
    'support_metric_step_'+k for k in metric_module.AUDIT_MAX_KEYS)
PREPARATION_COUNTERS=STAGE_COUNTERS=WORK_SUM_KEYS=AUDIT_COUNTERS
WORK_MAX_KEYS=PEAK_COUNTERS
_seal=geometry._seal
_freeze=geometry._freeze
_plain=geometry._plain
_finite=geometry._finite
_raw=geometry._raw
_subset=geometry._subset
_stack=geometry._stack
_Recorder=geometry._Recorder
_invoke=geometry._invoke


class GroupBalancedSupportMetricFailure(geometry.ProtoFrameFailure):
    """Preserved actual work and partial numeric state; no soft fallback."""


class _Ledger(geometry._Ledger):
    def __init__(self):self.work=dict.fromkeys(AUDIT_COUNTERS+PEAK_COUNTERS,0);self.calls=[]
    def add(self,operation,audit):
        audit=_plain(audit);self.calls.append(dict(operation=operation,audit=audit))
        self.work[operation+'_calls']=self.work.get(operation+'_calls',0)+1
        for key,value in audit.items():
            if type(value) not in (int,float) or not math.isfinite(value):continue
            name=operation+'_'+key
            if name in PEAK_COUNTERS:self.work[name]=max(self.work[name],value)
            elif name in AUDIT_COUNTERS:self.work[name]+=value
    def merge(self,other):
        for key,value in other['actual_work'].items():
            if key not in self.work:raise ValueError('Unknown actual work key '+key)
            self.work[key]=max(self.work[key],value) if key in PEAK_COUNTERS else self.work[key]+value
        self.calls.extend(other['operation_audits'])


def _emit(callback,name,context,**payload):
    obj=payload.get('objective',payload.get('audit',{}))
    record=dict(context or {},schema=SCHEMA,method=METHOD,event='GROUP_BALANCED_SUPPORT_METRIC_'+name,
        text='GROUP_BALANCED_SUPPORT_METRIC '+name+' RMSCE='+str(obj.get('RMSCE'))+' eta='+str(payload.get('step_size'))+
            ' gradient='+str(payload.get('gradient_norm'))+' source_validation=N/A',
        source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN',**payload)
    if callback:callback(_plain(record))


compact_training_record=geometry.compact_training_record


def _pad(theta):
    result=np.zeros(5,dtype=np.float64);result[:len(theta)]=theta;return result


def _basis_arrays(basis):
    return dict(original_dictionary_Q=basis.Q,U=basis.basis,U_lower=basis.basis_lower,
        U_upper=basis.basis_upper,U_error_bounds=basis.error_bounds,
        exact_column_indices=np.asarray(basis.column_indices,dtype=np.int64))


def _frame(basis):
    Q=np.zeros((160,5));Q[:,:basis.rank]=basis.basis
    return SimpleNamespace(Q=_seal(Q))


def _theta(value,rank):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or raw.shape!=(rank,):raise ValueError('theta must have exact U rank coordinates')
    out=np.array(raw,dtype=np.float64,copy=True);_finite(out);return out


@dataclass(frozen=True)
class GroupBalancedSupportMetricTraining:
    raw: object
    labels: np.ndarray
    ids: tuple
    classes: tuple
    old_classes: tuple
    prototype_frame: object
    basis: object
    physical_gram: np.ndarray
    anchor: np.ndarray
    inherited: object
    folds: tuple
    full_problem: object
    limits: object
    context: object
    audit: object
    records: object
    new0: bool=False
    def audit_dict(self):return _plain(self.audit)


@dataclass(frozen=True)
class ObjectiveCache:
    theta: np.ndarray
    scores: np.ndarray
    score_jacobian: object
    heads: tuple
    arrays: object


_geometry=geometry._geometry
_kernel=geometry._kernel
_ridge=geometry._ridge
_ridge_head=geometry._ridge_head
_logsoftmax=geometry._logsoftmax


def _gate_jvp(state,K_jacobian,L,L_jacobian,bounds_jacobian,ledger):
    result=_invoke(ledger,'gate_jvp',balanced_gate.group_balanced_barrier_gate_jvp,
        state=state,K_jacobian=K_jacobian,L=L,L_jacobian=L_jacobian,bounds_jacobian=bounds_jacobian)
    return result.score_jacobian,dict(result.arrays)


def _geometry_head(p,prepared,theta,jacobian,ledger,partial):
    if prepared.inherited is None:
        head=_ridge_head(p.train,p.labels,p.held,prepared.prototype_frame,theta,p.tau,p.gamma,p.classes,jacobian,ledger)
        partial.update(head['arrays']);return head
    frame=prepared.prototype_frame;old=prepared.old_classes
    oldcols=np.asarray([p.classes.index(c) for c in old],dtype=int)
    newcols=np.asarray([i for i,c in enumerate(p.classes) if c not in old],dtype=int)
    oi=np.flatnonzero(np.isin(p.labels,oldcols));ni=np.flatnonzero(~np.isin(p.labels,oldcols))
    if not len(oi) or not len(ni):raise ValueError('Both physical train groups required')
    original=_geometry(p.train,ledger);oh=_geometry(p.held,ledger)
    kr=_kernel(p.train,None,original,None,frame,theta,p.tau,p.gamma,jacobian,ledger)
    lr=_kernel(p.held,p.train,oh,original,frame,theta,p.tau,p.gamma,jacobian,ledger)
    K,L=kr.kernel,lr.kernel;partial.update(K=K,L=L,theta=theta,old_indices=oi,new_indices=ni,
        prior_train=p.prior_train,prior_held=p.prior_held)
    yl=np.asarray([int(np.flatnonzero(newcols==p.labels[i])[0]) for i in ni])
    Y=np.eye(len(newcols))[yl]-1/len(newcols)
    combined=np.concatenate((K[:,ni],L[:,ni]))
    dK=None if not jacobian else kr.jacobian[np.ix_(ni,ni)]
    dL=None if not jacobian else np.concatenate((kr.jacobian[:,ni],lr.jacobian[:,ni]))
    fitted=_ridge(K[np.ix_(ni,ni)],Y,combined,dK,dL,ledger)
    partial.update({'new_'+k:v for k,v in fitted.arrays.items()})
    htrain,hheld=fitted.scores[:len(K)],fitted.scores[len(K):]
    logtrain=_logsoftmax(htrain);logheld=_logsoftmax(hheld)
    own=np.asarray([old.index(p.classes[int(p.labels[i])]) for i in oi])
    prior=p.prior_train[oi];competitors=np.array(prior,copy=True);competitors[np.arange(len(oi)),own]=-np.inf
    wrong=np.maximum(0.,competitors.max(axis=1));rawmargin=prior[np.arange(len(oi)),own]-wrong
    top=prior.max(axis=1);lse=top+np.log(np.exp(prior-top[:,None]).sum(axis=1))
    bounds=(lse-wrong)[:,None]+logtrain[oi];_finite(bounds,rawmargin)
    partial.update(lower_bounds=bounds,raw_old_margin=rawmargin,new_train_scores=htrain,new_held_scores=hheld)
    gs=_invoke(ledger,'gate_forward',balanced_gate.fit_group_balanced_barrier_gate,K=K,targets=np.isin(p.labels,oldcols).astype(float),labels=p.labels,
        old_indices=oi,lower_bounds=bounds,**prepared.limits)
    partial.update({'gate_'+k:v for k,v in gs.arrays.items()})
    partial.update(gate_b=np.asarray(gs.b),gate_zeta=np.asarray(gs.zeta))
    gh=L@gs.alpha+gs.b;lo=_logsoftmax(p.prior_held)
    scores=np.empty((len(L),len(p.classes)));scores[:,oldcols]=-np.logaddexp(0.,-gh)[:,None]+lo
    scores[:,newcols]=-np.logaddexp(0.,gh)[:,None]+logheld
    J=None
    if jacobian:
        jtrain,jheld=fitted.score_jacobian[:len(K)],fitted.score_jacobian[len(K):]
        dlogtrain=jtrain-np.einsum('ij,ijp->ip',np.exp(logtrain),jtrain)[:,None,:]
        dlogheld=jheld-np.einsum('ij,ijp->ip',np.exp(logheld),jheld)[:,None,:]
        da=dlogtrain[oi];dg,jarrays=_gate_jvp(gs,kr.jacobian,L,lr.jacobian,da,ledger)
        partial.update(jarrays,lower_bounds_jacobian=da,K_jacobian=kr.jacobian,L_jacobian=lr.jacobian,
            new_score_jacobian=fitted.score_jacobian)
        e=np.exp(-np.abs(gh));positive=np.where(gh>=0,1/(1+e),e/(1+e));negative=np.where(gh>=0,e/(1+e),1/(1+e))
        J=np.empty((len(L),len(p.classes),5));J[:,oldcols]=negative[:,None,None]*dg[:,None,:]
        J[:,newcols]=-positive[:,None,None]*dg[:,None,:]+dlogheld
        partial['score_jacobian']=J
    _finite(scores);partial['scores']=scores
    return dict(raw=p.train,original=original,tau=p.tau,gamma=p.gamma,theta=theta,new_indices=ni,
        old_columns=oldcols,new_columns=newcols,new_alpha=fitted.alpha,new_intercept=fitted.intercept,
        gate=gs,classes=p.classes,scores=scores,J=J,arrays=dict(partial))


def _head(p,prepared,theta,jacobian,ledger,partial):
    shim=SimpleNamespace(inherited=prepared.inherited,prototype_frame=_frame(prepared.basis),
        old_classes=prepared.old_classes,limits={k:prepared.limits[k] for k in
        ('max_newton_iterations','max_line_search_trials','max_factor_buffer_bytes')})
    result=_geometry_head(p,shim,_pad(theta),jacobian,ledger,partial)
    # Explicit ABI: public theta is r-dimensional; helper arrays retain five padded
    # derivative directions, which are actually solved and charged once together.
    partial['head_theta_padded']=partial.pop('theta',_pad(theta))
    partial['theta']=theta
    partial['physical_gram']=prepared.physical_gram
    result['arrays']=dict(partial)
    result['theta']=theta
    result['head_theta_padded']=_pad(theta)
    if jacobian:result['J']=result['J'][...,:prepared.basis.rank]
    return result


def prepare_group_balanced_support_metric_joint_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,
        classes,old_classes,prototype_frame,max_integer_bits,max_fraction_operations,
        max_newton_iterations,max_line_search_trials,max_factor_buffer_bytes,max_secular_iterations,
        support_metric_basis=None,inherited=None,context=None,log_callback=None,state_callback=None):
    started=time.perf_counter();rec=_Recorder(state_callback);ledger=_Ledger();ctx=dict(context or {})
    limits=dict(max_integer_bits=max_integer_bits,max_fraction_operations=max_fraction_operations,
        max_newton_iterations=max_newton_iterations,max_line_search_trials=max_line_search_trials,
        max_factor_buffer_bytes=max_factor_buffer_bytes,max_secular_iterations=max_secular_iterations)
    for key,value in limits.items():
        if type(value) is not int or value<=0:raise ValueError(key+' must be an explicit positive integer')
    if max_secular_iterations>128:raise ValueError('max_secular_iterations must be at most 128')
    original_classes=tuple(classes);old=tuple(sorted(old_classes));ids=tuple(support_ids)
    if not isinstance(prototype_frame,primitive.ProtoFrame) or len(old)!=6 or len(set(old))!=6 or tuple(prototype_frame.classes)!=old:
        raise ValueError('Frozen compatible six-class ProtoFrame required')
    if not all(isinstance(c,str) and c for c in original_classes) or len(set(original_classes))!=len(original_classes):
        raise ValueError('Unique declared classes required')
    canonical=tuple(sorted(original_classes))
    if not set(old)<=set(canonical):raise ValueError('Missing registered old class')
    raw=_raw(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local)
    labels=np.asarray(support_labels);n=len(raw['z_id'])
    if labels.shape!=(n,) or labels.dtype.kind not in 'iu' or len(ids)!=n or len(set(ids))!=n or not all(isinstance(v,str) and v for v in ids):
        raise ValueError('Physical support/label binding differs')
    if np.any(labels<0) or np.any(labels>=len(canonical)):raise ValueError('Support label outside registry')
    labels=np.asarray([canonical.index(original_classes[int(v)]) for v in labels],dtype=np.int64)
    counts=np.bincount(labels,minlength=len(canonical))
    if not np.all(counts==counts[0]) or counts[0]<=0:raise ValueError('Every registered class requires equal positive K')
    order=np.argsort(np.asarray(ids));ids=tuple(ids[i] for i in order);labels=_seal(labels[order]);raw=_subset(raw,order)
    oi=np.asarray([i for i,y in enumerate(labels) if canonical[int(y)] in old],dtype=int)
    if inherited is not None:
        if not isinstance(inherited,GroupBalancedSupportMetricState) or inherited.mode!='B' or inherited.classes!=old:
            raise ValueError('C requires this method actual current U-coordinate B')
        previous=inherited.audit['preparation']['context']
        for key in ('run_id','row_id','split_id','scope','fold','trial','parent_k','train_k'):
            if previous.get(key)!=ctx.get(key):raise ValueError('Actual B lineage crosses '+key)
        if dict(inherited.audit['limits'])!=limits:raise ValueError('Actual B resource contract changed')
        if inherited.ids!=tuple(ids[i] for i in oi) or not np.array_equal(inherited.prototype_frame.Q,prototype_frame.Q):
            raise ValueError('Actual B physical IDs/frozen dictionary mismatch')
        expected=np.asarray([old.index(canonical[int(labels[i])]) for i in oi])
        if not np.array_equal(inherited.labels,expected) or any(not np.array_equal(inherited.raw[k],raw[k][oi]) for k in BRANCHES):
            raise ValueError('Actual B support records/labels changed')
    elif canonical!=old:raise ValueError('Registered C requires actual B inheritance')
    new0=inherited is not None and canonical==old
    if inherited is not None and support_metric_basis is not None and support_metric_basis is not inherited.basis:
        raise ValueError('C must use its actual B basis object')
    audit=dict(schema=SCHEMA,method=METHOD,classes=list(canonical),old_classes=list(old),
        training_physical_ids=list(ids),K=int(counts[0]),nominal_parameter_count=5,
        prototype_reference_only=True,source_validation=None,context=ctx,limits=limits,new0_exact_reuse=new0,
        evidence_level=FROZEN_CONFIG['evidence_level'],complete_head_jvp_error_bound=None,
        unknown_device_cost=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN')
    basis=None;anchor=np.empty(0)
    try:
        basis=inherited.basis if inherited is not None else support_metric_basis
        if basis is None:
            basis=_invoke(ledger,'basis',basis_module.build_support_metric_basis,
                Q=prototype_frame.Q,max_integer_bits=max_integer_bits,max_fraction_operations=max_fraction_operations)
        binding_started=time.perf_counter()
        gram_evaluations=0
        try:
            if not isinstance(basis,basis_module.SupportMetricBasis) or basis.Q.shape!=(160,5) or not np.array_equal(basis.Q,prototype_frame.Q):
                raise ValueError('Prebuilt basis must bind the exact frozen dictionary')
            ba=basis.audit_dict()
            if ba.get('max_integer_bits')!=max_integer_bits or ba.get('max_fraction_operations')!=max_fraction_operations:
                raise ValueError('Prebuilt basis construction resource contract changed')
            if type(basis.rank) is not int or not 0<=basis.rank<=5 or ba.get('exact_rank')!=basis.rank or ba.get('status')!='COMPLETE':
                raise ValueError('Prebuilt basis has an incompatible exact rank/status')
            for value in (basis.basis,basis.basis_lower,basis.basis_upper,basis.error_bounds):
                if value.shape!=(160,basis.rank) or value.dtype!=np.dtype('float64'):
                    raise ValueError('Prebuilt basis has an incompatible numeric shape/dtype')
                _finite(value)
            anchor=_seal(np.zeros(basis.rank) if inherited is None else inherited.theta)
            if new0:gram=inherited.head['arrays']['physical_gram']
            else:
                gram_evaluations=1;gram=_seal(basis.basis.T@basis.basis)
        finally:
            if not new0:ledger.add('basis_binding',dict(wall_seconds=time.perf_counter()-binding_started,
                physical_gram_evaluation_count=gram_evaluations))
        audit.update(effective_parameter_rank=basis.rank,basis_audit=basis.audit_dict(),
            basis_certificate=basis.certificate_json(),coordinate_scope='CURRENT_METHOD_PHYSICAL_U_ONLY',
            basis_construction_charged_here=inherited is None and support_metric_basis is None,
            basis_construction_scope='ONCE_AT_EXPLICIT_FACTORY_OWNER; PREBUILT_CONSTRUCTION_AUDIT_NOT_RECHARGED')
        if new0:
            audit.update(status='PREPARED_EXACT_B_REUSE',preparation_seconds=time.perf_counter()-started,**ledger.audit())
            return GroupBalancedSupportMetricTraining(raw,labels,ids,canonical,old,prototype_frame,basis,gram,anchor,inherited,(),None,
                _freeze(limits),_freeze(ctx),_freeze(audit),_freeze(rec.records),True)
        frame=_frame(basis)
        audit['prepared_state_ref']=rec.save('prepared',dict(theta=anchor,labels=labels,physical_gram=gram,**_basis_arrays(basis)))
        def problem(train_indices,held_indices,name):
            train=_subset(raw,train_indices);held=_subset(raw,held_indices);tl=labels[train_indices];hl=labels[held_indices]
            oldlocal=np.asarray([i for i,y in enumerate(tl) if canonical[int(y)] in old],dtype=int)
            oldraw=_subset(train,oldlocal);ol=np.asarray([old.index(canonical[int(tl[i])]) for i in oldlocal],dtype=int)
            tau,gamma,ga=geometry._fixed_geometry(oldraw,ol,old,ledger)
            priortrain=priorheld=priorref=None
            if inherited is not None:
                evalraw=_stack(train,held)
                if name=='full':
                    predicted=geometry._score_old(inherited.head,evalraw,frame,_pad(inherited.theta),ledger)
                    priorref=inherited.audit_dict()['final_state_ref']
                else:
                    teacher=geometry._ridge_head(oldraw,ol,evalraw,frame,_pad(anchor),tau,gamma,old,False,ledger)
                    teacher['arrays'].update(theta=anchor,head_theta_padded=_pad(anchor),physical_gram=gram,**_basis_arrays(basis))
                    predicted=teacher['scores'];priorref=rec.save(name+'_old_inner_prior',teacher['arrays'])
                priortrain=_seal(predicted[:len(tl)]);priorheld=_seal(predicted[len(tl):])
            return geometry._Problem(train,held,_seal(tl),_seal(hl),tuple(ids[i] for i in train_indices),
                tuple(ids[i] for i in held_indices),canonical,tau,gamma,_freeze(ga),priortrain,priorheld,_freeze(priorref))
        folds=[];fold_count=min(int(counts[0]),3)
        if counts[0]>1:
            membership=np.empty(n,dtype=int)
            for c in range(len(canonical)):
                ix=np.flatnonzero(labels==c);membership[ix]=np.arange(len(ix))%fold_count
            for f in range(fold_count):folds.append(problem(np.flatnonzero(membership!=f),np.flatnonzero(membership==f),'fold_'+str(f)))
        full=problem(np.arange(n),np.empty(0,dtype=int),'full')
        audit.update(status='PREPARED',preparation_seconds=time.perf_counter()-started,
            folds=[dict(fold=i,train_ids=list(p.ids),held_ids=list(p.held_ids),
                old_inner_train_ids=[pid for pid,y in zip(p.ids,p.labels) if canonical[int(y)] in old],
                prior_ref=p.prior_ref,geometry=p.geometry) for i,p in enumerate(folds)],
            full_prior_ref=full.prior_ref,**ledger.audit())
        _emit(log_callback,'PREPARED',ctx,audit=audit,actual_parameters=FROZEN_CONFIG)
        return GroupBalancedSupportMetricTraining(raw,labels,ids,canonical,old,prototype_frame,basis,gram,anchor,inherited,
            tuple(folds),full,_freeze(limits),_freeze(ctx),_freeze(audit),_freeze(rec.records))
    except Exception as exc:
        arrays=dict(theta=anchor,original_dictionary_Q=prototype_frame.Q,**getattr(exc,'arrays',{}))
        if isinstance(basis,basis_module.SupportMetricBasis):arrays.update(_basis_arrays(basis))
        audit.update(status='TECHNICAL_FAILURE',failure_code=getattr(exc,'code',str(exc)),
            preparation_seconds=time.perf_counter()-started,**ledger.audit())
        if hasattr(exc,'exact_state'):
            audit['basis_failure_exact_state']=basis_module._exact_encode(_plain(exc.exact_state))
        try:audit['failure_state_ref']=rec.save('preparation_failure',arrays,failed=True)
        except Exception as err:audit['failure_archive_error']=str(err)
        error=GroupBalancedSupportMetricFailure(getattr(exc,'code',str(exc)),audit,arrays);error.records=_freeze(rec.records);raise error from exc


def evaluate_group_balanced_support_metric_joint_objective(prepared,theta,*,jacobian=True):
    if not isinstance(prepared,GroupBalancedSupportMetricTraining) or prepared.new0:raise ValueError('No objective for exact B reuse')
    if type(jacobian) is not bool:raise ValueError('jacobian must be bool')
    theta=_theta(theta,prepared.basis.rank)
    if not prepared.folds:raise ValueError('K1 has no legal OOF objective')
    tick=time.perf_counter();ledger=_Ledger();heads=[];arrays=dict(theta=theta,physical_gram=prepared.physical_gram,
        **_basis_arrays(prepared.basis));partial={}
    try:
        scores=np.empty((len(prepared.ids),len(prepared.classes)))
        Jr=np.empty(scores.shape+(prepared.basis.rank,)) if jacobian else None
        index={pid:i for i,pid in enumerate(prepared.ids)}
        for fold,p in enumerate(prepared.folds):
            partial={};head=_head(p,prepared,theta,jacobian,ledger,partial);heads.append(head)
            where=[index[pid] for pid in p.held_ids];scores[where]=head['scores']
            if jacobian:Jr[where]=head['J']
            arrays.update({'fold_'+str(fold)+'_'+k:v for k,v in partial.items()})
        J5=None
        if jacobian:
            J5=np.zeros(scores.shape+(5,));J5[...,:prepared.basis.rank]=Jr
        rms=_invoke(ledger,'rmsce',primitive.class_balanced_rmsce,scores=scores,labels=prepared.labels,score_jacobian=J5)
        arrays.update(scores=scores,labels=prepared.labels,probabilities=rms.arrays['probabilities'])
        arrays.update({'rms_'+k:v for k,v in rms.arrays.items() if v is not None})
        gradient=None
        if jacobian:
            gradient=rms.gradient[:prepared.basis.rank]
            arrays.update(score_jacobian=Jr,gradient=gradient,curvature=rms.curvature[:prepared.basis.rank,:prepared.basis.rank])
        info=dict(schema=SCHEMA,method=METHOD,RMSCE=float(rms.loss),loss_total=float(rms.loss),loss_proximal=0.,
            objective_seconds=time.perf_counter()-tick,objective_scope='CLASS_MEANS_OVER_ALL_PHYSICAL_OOF_RECORDS',
            jacobian_available=jacobian,rms_audit=_plain(rms.audit),**ledger.audit())
        cache=ObjectiveCache(_seal(theta),_seal(scores),None if Jr is None else _seal(Jr),tuple(_freeze(h) for h in heads),_freeze(arrays))
        return float(rms.loss),None if gradient is None else _seal(gradient),info,cache
    except Exception as exc:
        arrays.update({'failed_'+k:v for k,v in partial.items()})
        arrays.update({'component_failure_'+k:v for k,v in getattr(exc,'arrays',{}).items()})
        arrays.update(getattr(exc,'proto_partial_arrays',{}))
        raise GroupBalancedSupportMetricFailure(getattr(exc,'code',str(exc)),dict(status='TECHNICAL_FAILURE',
            completed_fold_count=len(heads),objective_seconds=time.perf_counter()-tick,**ledger.audit()),arrays) from exc


@dataclass(frozen=True)
class GroupBalancedSupportMetricState:
    theta: np.ndarray
    classes: tuple
    ids: tuple
    labels: np.ndarray
    raw: object
    prototype_frame: object
    basis: object
    head: object
    prior: object
    mode: str
    audit: object
    records: object
    def audit_dict(self):return _plain(self.audit)
    def state_records(self):return self.records
    def to_arrays(self):
        out=dict(self.head['arrays']);out.update(theta=self.theta,labels=self.labels,**_basis_arrays(self.basis))
        out.update({'support_'+k:v for k,v in self.raw.items()})
        if self.prior is not None:out.update({'actual_B_'+k:v for k,v in self.prior.to_arrays().items()})
        return _freeze(out)
    def _proxy(self):
        prior=None if self.prior is None else self.prior._proxy()
        return geometry.ProtoFrameState(_seal(_pad(self.theta)),self.classes,self.ids,self.labels,self.raw,
            _frame(self.basis),self.head,prior,self.mode,self.audit,self.records)
    def score_with_audit(self,*,z_id,fft,t_emb,f_emb,pa_local):
        raw=_raw(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local)
        if len(raw['z_id'])!=1:raise ValueError('Query inference requires exactly one physical record')
        ledger=_Ledger();tick=time.perf_counter();scores,_=self._proxy()._score(raw,ledger)
        return _seal(scores),dict(score_seconds=time.perf_counter()-tick,single_record_all_registered_classes=True,**ledger.audit())
    def score(self,**features):return self.score_with_audit(**features)[0]
    def predict(self,**features):return self._proxy().predict(**features)


def fit_group_balanced_support_metric_joint_local_ridge(prepared,*,mode='B',log_callback=None,state_callback=None):
    if not isinstance(prepared,GroupBalancedSupportMetricTraining):raise TypeError('Prepared SupportMetric training required')
    if mode not in ('B','C_seq') or (mode=='B')!=(prepared.inherited is None):raise ValueError('Stage/actual B mismatch')
    if prepared.new0:return prepared.inherited
    rec=_Recorder(state_callback);ledger=_Ledger();theta=np.array(prepared.anchor,copy=True);started=time.perf_counter()
    eligible=bool(prepared.folds) and prepared.basis.rank>0
    audit=dict(schema=SCHEMA,method=METHOD,mode=mode,classes=list(prepared.classes),training_physical_ids=list(prepared.ids),
        anchor_theta=prepared.anchor.tolist(),optimizer_steps=0,trial_count=0,nominal_parameter_count=5,
        effective_parameter_rank=prepared.basis.rank,trainable_parameter_count=prepared.basis.rank if eligible else 0,
        preparation=prepared.audit_dict(),trials=[],no_held=not bool(prepared.folds),initial_objective=None,final_objective=None,
        source_validation=None,limits=_plain(prepared.limits),config=FROZEN_CONFIG,context=_plain(prepared.context),
        evidence_level=FROZEN_CONFIG['evidence_level'],complete_head_jvp_error_bound=None,device_cost=None)
    partial={};info=None
    try:
        if eligible:
            try:loss,g,info,cache=evaluate_group_balanced_support_metric_joint_objective(prepared,theta,jacobian=True)
            except GroupBalancedSupportMetricFailure as exc:ledger.merge(exc.audit);raise
            ledger.merge(info);audit['initial_objective']=info;audit['initial_state_ref']=rec.save('initial',cache.arrays)
            _emit(log_callback,'INITIAL',prepared.context,objective=info,state_ref=audit['initial_state_ref'])
            partial=dict(theta=theta,gradient=g,curvature=cache.arrays['curvature'],score_jvp=cache.score_jacobian,
                probabilities=cache.arrays['probabilities'],labels=prepared.labels,physical_gram=prepared.physical_gram)
            step=_invoke(ledger,'support_metric_step',metric_module.solve_support_metric_step,
                gradient=g,ggn=cache.arrays['curvature'],physical_gram=prepared.physical_gram,
                score_jvp=cache.score_jacobian,probabilities=cache.arrays['probabilities'],labels=prepared.labels,
                max_secular_iterations=prepared.limits['max_secular_iterations'])
            partial.update(dict(step.arrays));direction=step.direction;slope=float(g@direction)
            audit.update(gradient_norm=float(np.linalg.norm(g)),direction_norm=float(np.linalg.norm(direction)),
                support_metric_step_audit=_plain(step.audit),quadratic_multiplier=float(step.multiplier),
                metric_scope='FROZEN_AT_STAGE_ANCHOR_USING_ALL_LEGAL_OOF_SUPPORT')
            direction_ref=rec.save('metric_direction',partial)
            _emit(log_callback,'GRADIENT',prepared.context,objective=info,state_ref=direction_ref,
                gradient_norm=audit['gradient_norm'],direction_norm=audit['direction_norm'],step_audit=_plain(step.audit))
            if np.any(direction):
                if not math.isfinite(slope) or slope>=0:raise ArithmeticError('METRIC_DIRECTION_NOT_DESCENT')
                for trial in range(12):
                    eta=.5**trial;attempt=prepared.anchor+eta*direction;delta=attempt-prepared.anchor
                    partial=dict(attempt_theta=attempt,last_accepted_theta=theta,direction=direction,M=step.M)
                    try:tl,_,ti,tc=evaluate_group_balanced_support_metric_joint_objective(prepared,attempt,jacobian=False)
                    except GroupBalancedSupportMetricFailure as exc:ledger.merge(exc.audit);raise
                    ledger.merge(ti);audit['trial_count']+=1
                    rhs=loss+1e-4*float(g@delta);tol=128*EPS*max(1.,abs(loss),abs(tl),abs(rhs))
                    accepted=bool(tl<=rhs+tol);ref=rec.save('trial_'+str(trial),tc.arrays)
                    record=dict(trial=trial,step_size=eta,loss_before=loss,loss_after=tl,armijo_rhs=rhs,
                        metric_ball_value=float(delta@step.M@delta),real_inequality_holds=bool(tl<=rhs),
                        comparison_tolerance=tol,observed_objective_increase=bool(tl>loss),accepted=accepted,state_ref=ref)
                    audit['trials'].append(record);_emit(log_callback,'TRIAL',prepared.context,objective=ti,**record)
                    if accepted:
                        theta=np.array(attempt,copy=True);loss,info,cache=tl,ti,tc;audit['optimizer_steps']=1
                        audit['stop_reason']='ONE_SUPPORT_METRIC_UPDATE_ACCEPTED'
                        _emit(log_callback,'STEP',prepared.context,objective=info,state_ref=ref,step=0,step_size=eta,
                            update_norm=float(np.linalg.norm(delta)),accepted=True);break
                else:audit['stop_reason']='TRIAL_BUDGET_EXHAUSTED_RETAIN_INITIAL'
            else:audit['stop_reason']='ZERO_METRIC_DIRECTION'
            audit['final_objective']=info
        else:audit['stop_reason']='EXACT_RANK_ZERO_COMPLETE_FINAL_HEAD' if prepared.basis.rank==0 else 'K1_NO_OOF_COMPLETE_FINAL_HEAD'
        partial={};head=_head(prepared.full_problem,prepared,theta,False,ledger,partial)
        arrays=dict(head['arrays'],theta=theta,labels=prepared.labels,**_basis_arrays(prepared.basis))
        arrays.update({'support_'+k:v for k,v in prepared.raw.items()})
        if prepared.inherited is not None:arrays.update({'actual_B_'+k:v for k,v in prepared.inherited.to_arrays().items()})
        audit['final_state_ref']=rec.save('final',arrays)
        audit.update(status='COMPLETED',fit_seconds=time.perf_counter()-started,theta=theta.tolist(),
            final_head_complete=True,actual_updated_coordinate_count=int(np.count_nonzero(theta-prepared.anchor)),
            parameter_buffer_bytes=int(theta.nbytes),final_prior_ref=prepared.full_problem.prior_ref,
            parameter_changed_from_anchor=bool(np.any(theta!=prepared.anchor)),**ledger.audit())
        records=dict(rec.records);records.update({'preparation_'+key:value for key,value in prepared.records.items()})
        audit.update(resident_numeric_state_bytes=None,deployment_numeric_state_bytes=None,
            deployment_state_scope='FULL_RETAINED_NUMERIC_STATE; NO_COMPACT_DEPLOYMENT_SERIALIZER',
            archive_file_bytes=None,incremental_wire_bytes=None,process_peak_memory_bytes=None)
        state=GroupBalancedSupportMetricState(_seal(theta),prepared.classes,prepared.ids,prepared.labels,prepared.raw,prepared.prototype_frame,
            prepared.basis,_freeze(head),prepared.inherited,mode,_freeze(audit),_freeze(records))
        # Basis arrays/exact certificate are outside generic array walker objects.
        resident=geometry._numeric_bytes(state)+sum(a.nbytes for a in (prepared.basis.Q,prepared.basis.basis,
            prepared.basis.basis_lower,prepared.basis.basis_upper,prepared.basis.error_bounds))
        audit.update(resident_numeric_state_bytes=resident,deployment_numeric_state_bytes=resident,
            basis_certificate_utf8_bytes=len(prepared.basis.certificate_json().encode('utf-8')),
            rational_object_memory_bytes=None,numeric_bytes_scope='NDARRAY_OBJECT_PAYLOAD_EXCLUDING_PYTHON_RATIONAL_OBJECTS_AND_WORKSPACES')
        state=replace(state,audit=_freeze(audit));_emit(log_callback,'FINAL',prepared.context,audit=state.audit_dict(),state_ref=audit['final_state_ref'])
        return state
    except Exception as exc:
        failed=exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else _plain(getattr(exc,'audit',{}))
        arrays=dict(partial);arrays.update({'failed_'+k:v for k,v in getattr(exc,'arrays',{}).items()})
        arrays.update(getattr(exc,'proto_partial_arrays',{}));arrays.update(last_accepted_theta=theta,
            anchor_theta=prepared.anchor,**_basis_arrays(prepared.basis))
        audit.update(status='TECHNICAL_FAILURE',failure_code=getattr(exc,'code',str(exc)),failed_work_audit=failed,
            fit_seconds=time.perf_counter()-started,**ledger.audit())
        try:audit['failure_state_ref']=rec.save('failure',arrays,failed=True)
        except Exception as err:audit['failure_archive_error']=str(err)
        try:_emit(log_callback,'FAILURE',prepared.context,audit=audit)
        except Exception as err:audit['failure_log_error']=str(err)
        error=GroupBalancedSupportMetricFailure(getattr(exc,'code',str(exc)),audit,arrays);error.records=_freeze(rec.records);raise error from exc


def predict_group_balanced_support_metric_joint_local_ridge(state,**features):
    if not isinstance(state,GroupBalancedSupportMetricState):raise TypeError('SupportMetric state required')
    return state.predict(**features)
