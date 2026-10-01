"""Five-coordinate prototype-frame GGN1 candidate; pure support-only mathematics.

No bundle, checkpoint, source example, query label, or experiment I/O exists here.
The prototype frame is fixed input geometry, never a supervision target.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
import math
import time

import numpy as np

from . import d92_proto_frame_primitives as primitive
from . import d92_group_barrier_gate as gate
from . import d92_branch_local_ridge as local

BRANCHES=('z_id','fft','t_emb','f_emb','pa_local')
SCHEMA='d92_proto_frame_joint_local_ridge_v1'
METHOD='D92-ProtoFrameTangent-GGN1-LocalRidge'
FROZEN_CONFIG=dict(schema=SCHEMA,method=METHOD,coordinates=5,kappa=.25,ridge_coefficient=1.,
    objective='RMS_class_mean_OOF_CE',damping='I5',radius=.5,max_updates=1,max_trials=12,
    initial_step=.125,backtrack=.5,armijo=1e-4,float_comparison_multiplier=128,
    folds='physical_ID_class_rank_mod_min_K_3',prototype_role='FROZEN_GEOMETRY_ONLY',
    C_anchor='ACTUAL_CURRENT_B_THETA',old_conditional='FROZEN_ACTUAL_B_FUNCTION',
    kernel='HALF_ORIGINAL_HALF_TANGENT_DISTANCE_FIXED_ORIGINAL_OLD_TRAIN_TAU_GAMMA',
    barrier_average_objective_gap=1e-4,phase1_frozen=True,source_examples=False,query_fit=False,
    parameter_search=False,encoder_backward=False)
EPS=np.finfo(np.float64).eps

# These names describe actual invocations, not a nominal five-times-factor model.
_PRIMITIVE_SUM=primitive.AUDIT_SUM_KEYS
_GATE_SUM=tuple(k for k in gate._ledger() if not k.startswith('peak_'))+('wall_seconds',)
_PRIMITIVE_OPERATIONS=('original_geometry','tangent_transform','adapted_geometry',
    'raw_kernel','fixed_old_distances','ridge','rmsce','ggn_step')
_OPERATIONS=_PRIMITIVE_OPERATIONS+('fixed_old_geometry','gate_forward','gate_jvp')
AUDIT_COUNTERS=tuple(op+'_calls' for op in _OPERATIONS)+tuple(
    op+'_'+key for op in _PRIMITIVE_OPERATIONS for key in _PRIMITIVE_SUM)+tuple(
    op+'_'+key for op in ('gate_forward','gate_jvp') for key in _GATE_SUM)+(
    'fixed_old_geometry_wall_seconds',)
PEAK_COUNTERS=tuple(op+'_'+key for op in ('gate_forward','gate_jvp') for key in
    ('peak_factor_buffer_bytes','peak_explicit_temporary_bytes'))
# Preparation and fit are separate ledgers; never add a nested preparation twice.
PREPARATION_COUNTERS=AUDIT_COUNTERS
STAGE_COUNTERS=AUDIT_COUNTERS
WORK_SUM_KEYS=AUDIT_COUNTERS
WORK_MAX_KEYS=PEAK_COUNTERS


def _seal(value):
    a=np.asarray(value)
    if a.dtype.kind not in 'fiub':raise ValueError('Only numeric state arrays are supported')
    shape=a.shape;a=np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(shape)


def _freeze(value):
    if isinstance(value,np.ndarray):return _seal(value)
    if isinstance(value,Mapping):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,(list,tuple)):return tuple(_freeze(v) for v in value)
    if isinstance(value,np.generic):return value.item()
    return value


def _plain(value):
    if isinstance(value,Mapping):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return _plain(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def _finite(*values):
    if any(not np.isfinite(v).all() for v in values):raise FloatingPointError('NONFINITE_PROTO_FRAME_STATE')


class ProtoFrameFailure(RuntimeError):
    def __init__(self,code,audit,arrays=None):
        super().__init__(code);self.code=code;self.audit=_freeze(audit)
        self.arrays=_freeze(arrays or {});self.records=MappingProxyType({})
    def audit_dict(self):return _plain(self.audit)


class _Ledger:
    def __init__(self):self.work=dict.fromkeys(AUDIT_COUNTERS+PEAK_COUNTERS,0);self.calls=[]
    def add(self,operation,audit):
        audit=_plain(audit);self.calls.append(dict(operation=operation,audit=audit))
        self.work[operation+'_calls']=self.work.get(operation+'_calls',0)+1
        for key,value in audit.items():
            if type(value) not in (int,float) or not math.isfinite(value):continue
            name=operation+'_'+key
            if name in PEAK_COUNTERS:self.work[name]=max(self.work.get(name,0),value)
            elif name in AUDIT_COUNTERS:
                self.work[name]=self.work.get(name,0)+value
    def merge(self,other):
        for key,value in other['actual_work'].items():
            self.work[key]=max(self.work.get(key,0),value) if 'peak' in key else self.work.get(key,0)+value
        self.calls.extend(other['operation_audits'])
    def audit(self):
        return dict(actual_work=dict(self.work),operation_audits=list(self.calls),
            work_scope='ACTUAL_INVOKED_OPERATIONS_INCLUDING_FAILED_ATTEMPTS; PEAK_MAX_OTHER_WORK_SUM',
            seconds_scope='PER_NAMED_OPERATION; SUBTIMERS_OVERLAP_WALL_TIMERS_NOT_ADDITIONAL_WALL_TIME',
            process_peak_bytes=None,gpu_peak_bytes=None,energy=None,wire_bytes=None)


def _invoke(ledger,operation,function,**kwargs):
    try:
        result=function(**kwargs)
    except Exception as exc:
        audit=exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else getattr(exc,'audit',{})
        ledger.add(operation,audit)
        raise
    ledger.add(operation,result.audit)
    return result


class _Recorder:
    def __init__(self,callback):self.callback=callback;self.records={};self.keys=set()
    def save(self,key,arrays,*,failed=False):
        if key in self.keys:raise ValueError('Duplicate state key '+key)
        self.keys.add(key);values={k:_seal(v) for k,v in arrays.items()}
        if not failed:_finite(*values.values())
        self.records[key]=_freeze(values)
        if self.callback is None:return dict(storage='in_memory',key=key,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),nbytes=v.nbytes) for k,v in values.items()})
        result=self.callback(key,values)
        if not isinstance(result,Mapping):raise ValueError('state_callback must return a mapping')
        return dict(result)


def _emit(callback,name,context,**payload):
    obj=payload.get('objective',payload.get('audit',{}))
    text=('PROTO_FRAME '+name+' RMSCE='+str(obj.get('RMSCE'))+' lr='+str(payload.get('step_size'))+
          ' gradient='+str(payload.get('gradient_norm'))+' theta_update='+str(payload.get('update_norm'))+
          ' source_validation=N/A')
    record=dict(context or {},schema=SCHEMA,method=METHOD,event='PROTO_FRAME_'+name,text=text,
        source_validation=None,source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN',**payload)
    if callback:callback(_plain(record))


def compact_training_record(record):
    out={k:v for k,v in record.items() if v is None or type(v) in (str,bool,int,float)}
    obj=record.get('objective',record.get('audit',{}))
    out.update({k:v for k,v in obj.items() if v is None or type(v) in (str,bool,int,float)})
    out['counter_scope']='INCREMENTAL_OBJECTIVE_OR_CUMULATIVE_STAGE_AS_EVENT_DECLARES'
    return _plain(out)


def _raw(**features):
    if any(np.asarray(features[k]).dtype.kind not in 'fiu' for k in BRANCHES):
        raise ValueError('Features must be real numeric arrays')
    arrays={k:np.array(features[k],dtype=np.float64,copy=True) for k in BRANCHES}
    n=len(arrays['z_id'])
    for k,a in arrays.items():
        if a.shape!=(n,96 if k=='fft' else 160):raise ValueError('Invalid branch shape '+k)
        _finite(a)
    return _freeze(arrays)


def _subset(raw,indices):return _freeze({k:v[indices] for k,v in raw.items()})
def _stack(a,b):return _freeze({k:np.concatenate((a[k],b[k]),axis=0) for k in BRANCHES})


def _geometry(raw,ledger):
    return _invoke(ledger,'original_geometry',primitive.branch_geometry,**raw)


def _adapt(raw,frame,theta,jacobian,ledger):
    transformed=_invoke(ledger,'tangent_transform',primitive.transform_z_id,z_id=raw['z_id'],Q=frame.Q,
        theta=theta,jacobian=jacobian)
    return _invoke(ledger,'adapted_geometry',primitive.branch_geometry,z_id=transformed.values,
        **{k:raw[k] for k in BRANCHES if k!='z_id'},z_id_jacobian=transformed.jacobian)


def _kernel(left,right,original_left,original_right,frame,theta,tau,gamma,jacobian,ledger):
    a=_adapt(left,frame,theta,jacobian,ledger)
    b=a if right is None else _adapt(right,frame,theta,jacobian,ledger)
    return _invoke(ledger,'raw_kernel',primitive.raw_kernel,original_left=original_left,adapted_left=a,
        original_right=None if right is None else original_right,adapted_right=None if right is None else b,
        tau=tau,gamma=gamma,jacobian=jacobian)


def _fixed_geometry(raw,labels,classes,ledger):
    geometry=_geometry(raw,ledger)
    result=_invoke(ledger,'fixed_old_distances',primitive.interaction_distances,left=geometry,jacobian=False)
    tick=time.perf_counter()
    try:
        d=result.distance;n=len(d);upper=np.triu_indices(n,1)
        # Exactly the existing old-train trace/nearest-other-class median rule.
        s0=float(np.sum(d[upper])/n)
        delta=np.min(np.where(labels[:,None]!=labels[None,:],d,np.inf),axis=1)
        tau=float(np.median(delta));reason='IDENTICAL_COMPLETE_FEATURES' if s0==0 else (
            'ZERO_BANDWIDTH_EQUIVALENCE_KERNEL' if tau==0 else None)
        gamma=None;sradial=0.
        if s0>0:
            minus=local._radial_minus_one(d,tau);sradial=float(-2*np.sum(minus[upper])/n)
            if sradial<=0:raise FloatingPointError('NONPOSITIVE_FIXED_RADIAL_TRACE')
            gamma=s0/sradial;_finite(gamma)
        return tau,gamma,dict(tau=tau,gamma=gamma,interaction_trace=s0,radial_trace=sradial,reason=reason)
    finally:ledger.add('fixed_old_geometry',dict(wall_seconds=time.perf_counter()-tick))


def _ridge(K,Y,L,dK,dL,ledger):
    return _invoke(ledger,'ridge',primitive.fit_free_intercept_ridge,K=K,Y=Y,L=L,K_jacobian=dK,L_jacobian=dL)


def _logsoftmax(x):
    shift=x-x.max(axis=1,keepdims=True);result=shift-np.log(np.exp(shift).sum(axis=1,keepdims=True));_finite(result);return result


@dataclass(frozen=True)
class _Problem:
    train: object
    held: object
    labels: np.ndarray
    held_labels: np.ndarray
    ids: tuple
    held_ids: tuple
    classes: tuple
    tau: object
    gamma: object
    geometry: object
    prior_train: object=None
    prior_held: object=None
    prior_ref: object=None


@dataclass(frozen=True)
class ProtoFrameTraining:
    raw: object
    labels: np.ndarray
    ids: tuple
    classes: tuple
    old_classes: tuple
    prototype_frame: object
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


def _ridge_head(raw,labels,eval_raw,frame,theta,tau,gamma,classes,jacobian,ledger):
    original=_geometry(raw,ledger);oe=_geometry(eval_raw,ledger)
    kr=_kernel(raw,None,original,None,frame,theta,tau,gamma,jacobian,ledger)
    lr=_kernel(eval_raw,raw,oe,original,frame,theta,tau,gamma,jacobian,ledger)
    Y=np.eye(len(classes))[labels]-1/len(classes)
    fitted=_ridge(kr.kernel,Y,lr.kernel,kr.jacobian,lr.jacobian,ledger)
    arrays=dict(K=kr.kernel,L=lr.kernel,Y=Y,alpha=fitted.alpha,intercept=fitted.intercept,theta=theta)
    arrays.update({str(k):v for k,v in fitted.arrays.items()})
    if jacobian:arrays.update(K_jacobian=kr.jacobian,L_jacobian=lr.jacobian,score_jacobian=fitted.score_jacobian)
    return dict(raw=raw,original=original,tau=tau,gamma=gamma,alpha=fitted.alpha,intercept=fitted.intercept,
        scores=fitted.scores,J=fitted.score_jacobian,arrays=arrays,classes=classes,theta=theta)


def _score_old(head,raw,frame,theta,ledger):
    oe=_geometry(raw,ledger)
    lr=_kernel(raw,head['raw'],oe,head['original'],frame,theta,head['tau'],head['gamma'],False,ledger)
    return lr.kernel@head['alpha']+head['intercept']


def prepare_proto_frame_joint_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,
        classes,old_classes,prototype_frame,max_newton_iterations,max_line_search_trials,max_factor_buffer_bytes,
        inherited=None,context=None,log_callback=None,state_callback=None):
    started=time.perf_counter()
    limits=dict(max_newton_iterations=max_newton_iterations,max_line_search_trials=max_line_search_trials,
        max_factor_buffer_bytes=max_factor_buffer_bytes)
    for key,value in limits.items():gate._positive_int(value,key)
    original_classes=tuple(classes);old=tuple(sorted(old_classes));ids=tuple(support_ids)
    if not all(isinstance(v,str) and v for v in ids):raise ValueError('Opaque physical IDs must be nonempty strings')
    if not isinstance(prototype_frame,primitive.ProtoFrame):raise ValueError('Frozen ProtoFrame factory result required')
    if len(old)!=6 or len(set(old))!=6 or tuple(prototype_frame.classes)!=old or np.asarray(prototype_frame.Q).shape!=(160,5):
        raise ValueError('Frozen compatible six-class prototype frame required')
    if not all(isinstance(v,str) and v for v in original_classes) or len(set(original_classes))!=len(original_classes):
        raise ValueError('Unique declared class IDs required')
    canonical=tuple(sorted(original_classes))
    if not set(old)<=set(canonical):raise ValueError('Missing registered old class')
    labels=np.asarray(support_labels)
    raw=_raw(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local)
    n=len(raw['z_id'])
    if labels.shape!=(n,) or labels.dtype.kind not in 'iu' or len(ids)!=n or len(set(ids))!=n or not all(ids):
        raise ValueError('Physical support/label binding differs')
    if np.any(labels<0) or np.any(labels>=len(canonical)):raise ValueError('Support label outside registry')
    labels=np.asarray([canonical.index(original_classes[int(v)]) for v in labels],dtype=np.int64)
    counts=np.bincount(labels,minlength=len(canonical))
    if not np.all(counts==counts[0]) or counts[0]<=0:raise ValueError('Every registered class must have the same positive K')
    order=np.argsort(np.asarray(ids));ids=tuple(ids[i] for i in order);labels=_seal(labels[order]);raw=_subset(raw,order)
    oi=np.asarray([i for i,y in enumerate(labels) if canonical[int(y)] in old],dtype=int)
    if inherited is not None:
        if not isinstance(inherited,ProtoFrameState) or inherited.mode!='B' or inherited.classes!=old:
            raise ValueError('C requires the actual candidate B state')
        previous=inherited.audit['preparation']['context'];current=dict(context or {})
        for key in ('run_id','row_id','split_id','scope','fold','trial','parent_k','train_k'):
            if previous.get(key)!=current.get(key):raise ValueError('Actual B lineage crosses '+key)
        if dict(inherited.audit['limits'])!=limits:raise ValueError('Actual B resource contract changed')
        if inherited.ids!=tuple(ids[i] for i in oi) or not np.array_equal(inherited.prototype_frame.Q,prototype_frame.Q):
            raise ValueError('Actual B physical IDs/frame mismatch')
        expected=np.asarray([old.index(canonical[int(labels[i])]) for i in oi])
        if not np.array_equal(inherited.labels,expected) or any(not np.array_equal(inherited.raw[k],raw[k][oi]) for k in BRANCHES):
            raise ValueError('Actual B support records/labels changed')
    elif canonical!=old:raise ValueError('Registered C requires actual B inheritance')
    anchor=_seal(np.zeros(5) if inherited is None else inherited.theta)
    rec=_Recorder(state_callback);ledger=_Ledger();ctx=dict(context or {});new0=inherited is not None and canonical==old
    audit=dict(schema=SCHEMA,method=METHOD,classes=list(canonical),old_classes=list(old),training_physical_ids=list(ids),
        K=int(counts[0]),nominal_parameter_count=5,prototype_reference_only=True,source_validation=None,
        source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN',context=ctx,limits=limits,new0_exact_reuse=new0)
    if new0:
        audit.update(status='PREPARED_EXACT_B_REUSE',preparation_seconds=time.perf_counter()-started,**ledger.audit())
        return ProtoFrameTraining(raw,labels,ids,canonical,old,prototype_frame,anchor,inherited,(),None,
            _freeze(limits),_freeze(ctx),_freeze(audit),_freeze(rec.records),True)
    try:
        def problem(train_indices,held_indices,name):
            train=_subset(raw,train_indices);held=_subset(raw,held_indices);tl=labels[train_indices];hl=labels[held_indices]
            oldlocal=np.asarray([i for i,v in enumerate(tl) if canonical[int(v)] in old],dtype=int)
            oldraw=_subset(train,oldlocal);ol=np.asarray([old.index(canonical[int(tl[i])]) for i in oldlocal],dtype=int)
            tau,gamma,ga=_fixed_geometry(oldraw,ol,old,ledger)
            priortrain=priorheld=priorref=None
            if inherited is not None:
                evalraw=_stack(train,held)
                if name=='full':
                    predicted=_score_old(inherited.head,evalraw,prototype_frame,inherited.theta,ledger)
                    priorref=inherited.audit_dict()['final_state_ref']
                else:
                    teacher=_ridge_head(oldraw,ol,evalraw,prototype_frame,anchor,tau,gamma,old,False,ledger)
                    predicted=teacher['scores'];priorref=rec.save(name+'_old_inner_prior',teacher['arrays'])
                priortrain=_seal(predicted[:len(tl)]);priorheld=_seal(predicted[len(tl):])
            return _Problem(train,held,_seal(tl),_seal(hl),tuple(ids[i] for i in train_indices),tuple(ids[i] for i in held_indices),
                canonical,tau,gamma,_freeze(ga),priortrain,priorheld,_freeze(priorref))
        folds=[];fold_count=min(int(counts[0]),3)
        if counts[0]>1:
            membership=np.empty(n,dtype=int)
            for c in range(len(canonical)):
                ix=np.flatnonzero(labels==c);membership[ix]=np.arange(len(ix))%fold_count
            for f in range(fold_count):folds.append(problem(np.flatnonzero(membership!=f),np.flatnonzero(membership==f),'fold_'+str(f)))
        full=problem(np.arange(n),np.empty(0,dtype=int),'full')
        audit.update(status='PREPARED',preparation_seconds=time.perf_counter()-started,folds=[dict(fold=i,train_ids=list(p.ids),held_ids=list(p.held_ids),
            old_inner_train_ids=[pid for pid,y in zip(p.ids,p.labels) if canonical[int(y)] in old],prior_ref=p.prior_ref,
            geometry=p.geometry) for i,p in enumerate(folds)],full_prior_ref=full.prior_ref,**ledger.audit())
        _emit(log_callback,'PREPARED',ctx,audit=audit,actual_parameters=FROZEN_CONFIG)
        return ProtoFrameTraining(raw,labels,ids,canonical,old,prototype_frame,anchor,inherited,tuple(folds),full,
            _freeze(limits),_freeze(ctx),_freeze(audit),_freeze(rec.records))
    except Exception as exc:
        arrays=dict(theta=anchor,Q=prototype_frame.Q,**getattr(exc,'arrays',{}))
        audit.update(status='TECHNICAL_FAILURE',preparation_seconds=time.perf_counter()-started,
            failure_code=getattr(exc,'code',str(exc)),**ledger.audit())
        try:audit['failure_state_ref']=rec.save('preparation_failure',arrays,failed=True)
        except Exception as err:audit['failure_archive_error']=str(err)
        error=ProtoFrameFailure(getattr(exc,'code',str(exc)),audit,arrays);error.records=_freeze(rec.records);raise error from exc


def _gate_jvp(state,K_jacobian,L,L_jacobian,bounds_jacobian,ledger):
    """Five directions through canonical equations, one sqrt(D) SPD system.

    Scaled right sides avoid forming large D times a cancellation-prone vector.
    No inverse or pseudoinverse of K (or dense inverse of D) is constructed.
    """
    audit=gate._ledger();started=time.perf_counter();partial={}
    try:
        root,H,chol=gate._factor(state.K,state.D_eff,state.max_factor_buffer_bytes,audit)
        partial.update(gate_jvp_H=H,gate_jvp_chol=chol)
        dkalpha=np.einsum('ijp,j->ip',K_jacobian,state.alpha)
        ratio=((state.zeta/state.slacks)/state.slacks)/state.D_eff[state.old_indices,None]
        if np.any(ratio<0) or np.any(ratio>1+16*EPS):raise ArithmeticError('INVALID_GATE_JVP_CURVATURE_RATIO')
        # r / sqrt(D) = sqrt(D) * sum_j[(barrier_D / D) da_ij].
        scaled_r=np.zeros((len(root),5))
        scaled_r[state.old_indices]=root[state.old_indices,None]*np.einsum('ij,ijp->ip',ratio,bounds_jacobian)
        rhs=-root[:,None]*dkalpha+scaled_r
        partial['gate_jvp_rhs']=np.column_stack((rhs,root))
        solved=gate._solve(H,chol,partial['gate_jvp_rhs'],audit)
        partial['gate_jvp_solved']=solved
        u=solved[:,-1];s=float(root@u)
        if not math.isfinite(s) or s<=0:raise ArithmeticError('GATE_JVP_INTERCEPT_SCHUR_NONPOSITIVE')
        db=(root@solved[:,:5])/s
        x=solved[:,:5]-u[:,None]*db
        da=root[:,None]*x
        residual=float(np.max(np.abs(H@x+root[:,None]*db-rhs)))
        equality=float(np.max(np.abs(da.sum(axis=0))))
        scale=float(np.max(np.abs(H)@np.abs(x)+root[:,None]*np.abs(db)+np.abs(rhs)))
        tolerance=128*EPS*max(1,len(root))*max(1.,scale)
        equality_tolerance=128*EPS*max(1,len(root))*max(1.,float(np.max(np.sum(np.abs(da),axis=0))))
        dg=np.einsum('ijp,j->ip',L_jacobian,state.alpha)+L@da+db
        _finite(da,db,dg)
        audit.update(jvp_scaled_residual=residual,jvp_scaled_tolerance=tolerance,
            jvp_intercept_residual=equality,jvp_intercept_tolerance=equality_tolerance)
        if residual>tolerance or equality>equality_tolerance:raise ArithmeticError('GATE_JVP_READBACK_FAILED')
        partial.update(gate_alpha_jacobian=da,gate_b_jacobian=db,gate_score_jacobian=dg)
        return dg,partial
    except Exception as exc:
        exc.proto_partial_arrays=partial
        raise
    finally:
        audit['wall_seconds']=time.perf_counter()-started
        ledger.add('gate_jvp',audit)


def _head(p,prepared,theta,jacobian,ledger,partial):
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
    gs=_invoke(ledger,'gate_forward',gate.fit_group_barrier_gate,K=K,targets=np.isin(p.labels,oldcols).astype(float),
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


def evaluate_proto_frame_joint_objective(prepared,theta,*,jacobian=True):
    if not isinstance(prepared,ProtoFrameTraining) or prepared.new0:raise ValueError('No new objective for exact B reuse')
    if type(jacobian) is not bool:raise ValueError('jacobian must be bool')
    if np.asarray(theta).dtype.kind not in 'fiu':raise ValueError('theta must be real numeric coordinates')
    theta=np.array(theta,dtype=np.float64,copy=True)
    if theta.shape!=(5,):raise ValueError('Exactly five theta coordinates required')
    _finite(theta)
    if not prepared.folds:raise ValueError('K1 has no OOF objective; fit the final head without updating')
    tick=time.perf_counter();ledger=_Ledger();heads=[];arrays=dict(theta=theta);partial={}
    try:
        scores=np.empty((len(prepared.ids),len(prepared.classes)))
        J=np.empty(scores.shape+(5,)) if jacobian else None
        index={pid:i for i,pid in enumerate(prepared.ids)}
        for fold,p in enumerate(prepared.folds):
            partial={};head=_head(p,prepared,theta,jacobian,ledger,partial);heads.append(head)
            where=[index[pid] for pid in p.held_ids];scores[where]=head['scores']
            if jacobian:J[where]=head['J']
            arrays.update({'fold_'+str(fold)+'_'+k:v for k,v in partial.items()})
        rms=_invoke(ledger,'rmsce',primitive.class_balanced_rmsce,scores=scores,labels=prepared.labels,score_jacobian=J)
        arrays.update(scores=scores,labels=prepared.labels)
        if jacobian:arrays.update(score_jacobian=J,gradient=rms.gradient,curvature=rms.curvature,damped_hessian=rms.damped_hessian)
        info=dict(schema=SCHEMA,method=METHOD,RMSCE=float(rms.loss),loss_total=float(rms.loss),loss_proximal=0.,
            objective_seconds=time.perf_counter()-tick,objective_scope='CLASS_MEANS_OVER_ALL_PHYSICAL_OOF_RECORDS',
            jacobian_available=jacobian,rms_audit=_plain(rms.audit),**ledger.audit())
        cache=ObjectiveCache(_seal(theta),_seal(scores),None if J is None else _seal(J),tuple(_freeze(h) for h in heads),_freeze(arrays))
        return float(rms.loss),None if not jacobian else _seal(rms.gradient),info,cache
    except Exception as exc:
        arrays.update({'failed_'+k:v for k,v in partial.items()})
        arrays.update({'primitive_failure_'+k:v for k,v in getattr(exc,'arrays',{}).items()})
        arrays.update(getattr(exc,'proto_partial_arrays',{}))
        error=ProtoFrameFailure(getattr(exc,'code',str(exc)),dict(status='TECHNICAL_FAILURE',failed_fold=len(heads),
            completed_fold_count=len(heads),objective_seconds=time.perf_counter()-tick,**ledger.audit()),arrays)
        raise error from exc


def _numeric_bytes(value,seen=None):
    if seen is None:seen=set()
    if id(value) in seen:return 0
    seen.add(id(value))
    if isinstance(value,np.ndarray):return value.nbytes
    if isinstance(value,Mapping):return sum(_numeric_bytes(v,seen) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(_numeric_bytes(v,seen) for v in value)
    if hasattr(value,'__dataclass_fields__'):return sum(_numeric_bytes(getattr(value,k),seen) for k in value.__dataclass_fields__)
    if isinstance(getattr(value,'arrays',None),Mapping):return _numeric_bytes(value.arrays,seen)
    return 0


@dataclass(frozen=True)
class ProtoFrameState:
    theta: np.ndarray
    classes: tuple
    ids: tuple
    labels: np.ndarray
    raw: object
    prototype_frame: object
    head: object
    prior: object
    mode: str
    audit: object
    records: object
    def audit_dict(self):return _plain(self.audit)
    def state_records(self):return self.records
    def to_arrays(self):
        out=dict(self.head['arrays']);out.update(theta=self.theta,Q=self.prototype_frame.Q,labels=self.labels)
        out.update({'support_'+k:v for k,v in self.raw.items()})
        if self.prior is not None:out.update({'actual_B_'+k:v for k,v in self.prior.to_arrays().items()})
        return _freeze(out)
    def _score(self,raw,ledger):
        if self.mode=='B':return _score_old(self.head,raw,self.prototype_frame,self.theta,ledger),None
        h=self.head;original=_geometry(raw,ledger)
        lr=_kernel(raw,h['raw'],original,h['original'],self.prototype_frame,self.theta,h['tau'],h['gamma'],False,ledger)
        L=lr.kernel;prior=_score_old(self.prior.head,raw,self.prototype_frame,self.prior.theta,ledger)
        new=L[:,h['new_indices']]@h['new_alpha']+h['new_intercept'];g=L@h['gate'].alpha+h['gate'].b
        lo,ln=_logsoftmax(prior),_logsoftmax(new);scores=np.empty((len(L),len(self.classes)))
        scores[:,h['old_columns']]=-np.logaddexp(0.,-g)[:,None]+lo
        scores[:,h['new_columns']]=-np.logaddexp(0.,g)[:,None]+ln;_finite(scores)
        return scores,(prior,new,g,lo,ln)
    def score_with_audit(self,*,z_id,fft,t_emb,f_emb,pa_local):
        raw=_raw(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local)
        if len(raw['z_id'])!=1:raise ValueError('Query inference requires exactly one physical record')
        ledger=_Ledger();tick=time.perf_counter();scores,_=self._score(raw,ledger)
        return _seal(scores),dict(score_seconds=time.perf_counter()-tick,single_record_all_registered_classes=True,**ledger.audit())
    def score(self,**features):return self.score_with_audit(**features)[0]
    def predict(self,**features):
        raw=_raw(**features)
        if len(raw['z_id'])!=1:raise ValueError('Query inference requires exactly one physical record')
        values,parts=self._score(raw,_Ledger())
        if parts is None:return np.asarray([self.classes[int(np.argmax(values[0]))]],dtype=str)
        B,h,g,lo,ln=parts;bo,bn=int(np.argmax(B[0])),int(np.argmax(h[0]));gap=g[0]+lo[0,bo]-ln[0,bn]
        oo=self.classes[int(self.head['old_columns'][bo])];nn=self.classes[int(self.head['new_columns'][bn])]
        return np.asarray([oo if gap>0 else nn if gap<0 else min(oo,nn)],dtype=str)


def fit_proto_frame_joint_local_ridge(prepared,*,mode='B',log_callback=None,state_callback=None):
    if not isinstance(prepared,ProtoFrameTraining):raise TypeError('Prepared ProtoFrame training required')
    if mode not in ('B','C_seq') or (mode=='B')!=(prepared.inherited is None):raise ValueError('Stage/actual B lineage mismatch')
    if prepared.new0:return prepared.inherited
    rec=_Recorder(state_callback);ledger=_Ledger();theta=np.array(prepared.anchor,copy=True);started=time.perf_counter()
    audit=dict(schema=SCHEMA,method=METHOD,mode=mode,classes=list(prepared.classes),training_physical_ids=list(prepared.ids),
        anchor_theta=prepared.anchor.tolist(),optimizer_steps=0,trial_count=0,nominal_parameter_count=5,
        trainable_parameter_count=5 if prepared.folds else 0,preparation=prepared.audit_dict(),trials=[],
        no_held=not bool(prepared.folds),initial_objective=None,final_objective=None,source_validation=None,
        source_validation_reason='PHASE2_SOURCE_ACCESS_FORBIDDEN',limits=dict(prepared.limits),
        config=FROZEN_CONFIG,context=_plain(prepared.context))
    partial={};failed_work=None;cache=None
    try:
        if prepared.folds:
            try:loss,g,info,cache=evaluate_proto_frame_joint_objective(prepared,theta,jacobian=True)
            except ProtoFrameFailure as exc:ledger.merge(exc.audit);raise
            ledger.merge(info);audit['initial_objective']=info;audit['initial_state_ref']=rec.save('initial',cache.arrays)
            _emit(log_callback,'INITIAL',prepared.context,objective=info,state_ref=audit['initial_state_ref'])
            step=_invoke(ledger,'ggn_step',primitive.solve_hard_ball_step,gradient=g,hessian=cache.arrays['damped_hessian'],radius=.5)
            direction=step.direction;slope=float(g@direction)
            audit.update(gradient_norm=float(np.linalg.norm(g)),direction_norm=float(np.linalg.norm(direction)),
                quadratic_step_audit=_plain(step.audit),quadratic_multiplier=float(step.multiplier))
            direction_ref=rec.save('ggn_direction',dict(theta=theta,gradient=g,direction=direction,hessian=cache.arrays['damped_hessian']))
            _emit(log_callback,'GRADIENT',prepared.context,objective=info,state_ref=direction_ref,
                gradient_norm=audit['gradient_norm'],direction_norm=audit['direction_norm'],quadratic_step_audit=_plain(step.audit))
            if np.any(direction):
                if not math.isfinite(slope) or slope>=0:raise ArithmeticError('GGN_DIRECTION_NOT_DESCENT')
                for trial in range(12):
                    eta=.125*(.5**trial);attempt=prepared.anchor+eta*direction
                    partial=dict(attempt_theta=attempt,last_accepted_theta=theta,direction=direction)
                    try:tl,_,ti,tc=evaluate_proto_frame_joint_objective(prepared,attempt,jacobian=False)
                    except ProtoFrameFailure as exc:ledger.merge(exc.audit);raise
                    ledger.merge(ti);audit['trial_count']+=1
                    rhs=loss+1e-4*eta*slope;tol=128*EPS*max(1.,abs(loss),abs(tl),abs(rhs))
                    accepted=bool(tl<=rhs+tol);ref=rec.save('trial_'+str(trial),tc.arrays)
                    record=dict(trial=trial,step_size=eta,loss_before=loss,loss_after=tl,armijo_rhs=rhs,
                        real_inequality_holds=bool(tl<=rhs),comparison_tolerance=tol,
                        observed_objective_increase=bool(tl>loss),accepted=accepted,state_ref=ref)
                    audit['trials'].append(record);_emit(log_callback,'TRIAL',prepared.context,objective=ti,**record)
                    if accepted:
                        theta=np.array(attempt,copy=True);loss,info,cache=tl,ti,tc;audit['optimizer_steps']=1
                        audit['stop_reason']='ONE_GGN_UPDATE_ACCEPTED'
                        _emit(log_callback,'STEP',prepared.context,objective=info,state_ref=ref,step=0,step_size=eta,
                            update_norm=float(np.linalg.norm(theta-prepared.anchor)),accepted=True)
                        break
                else:audit['stop_reason']='TRIAL_BUDGET_EXHAUSTED_RETAIN_INITIAL'
            else:audit['stop_reason']='ZERO_GGN_DIRECTION'
            audit['final_objective']=info
        else:audit['stop_reason']='K1_NO_OOF_COMPLETE_FINAL_HEAD_REQUIRED'
        partial={};head=_head(prepared.full_problem,prepared,theta,False,ledger,partial)
        arrays=dict(head['arrays'],theta=theta,Q=prepared.prototype_frame.Q)
        arrays.update({'support_'+k:v for k,v in prepared.raw.items()})
        if prepared.inherited is not None:arrays.update({'actual_B_'+k:v for k,v in prepared.inherited.to_arrays().items()})
        audit['final_state_ref']=rec.save('final',arrays)
        audit.update(status='COMPLETED',fit_seconds=time.perf_counter()-started,theta=theta.tolist(),
            final_head_complete=True,actual_updated_coordinate_count=int(np.count_nonzero(theta-prepared.anchor)),
            parameter_buffer_bytes=40,final_prior_ref=prepared.full_problem.prior_ref,
            parameter_changed_from_anchor=bool(np.any(theta!=prepared.anchor)),**ledger.audit())
        sealedhead=_freeze(head)
        records=dict(rec.records)
        records.update({'preparation_'+key:value for key,value in prepared.records.items()})
        audit.update(resident_numeric_state_bytes=None,deployment_numeric_state_bytes=None,
            deployment_state_scope='CURRENT_FULL_RETAINED_STATE; NO_COMPACT_DEPLOYMENT_SERIALIZER',
            archive_file_bytes=None,incremental_wire_bytes=None)
        state=ProtoFrameState(_seal(theta),prepared.classes,prepared.ids,prepared.labels,prepared.raw,prepared.prototype_frame,
            sealedhead,prepared.inherited,mode,_freeze(audit),_freeze(records))
        resident=_numeric_bytes(state)
        audit.update(resident_numeric_state_bytes=resident,deployment_numeric_state_bytes=resident)
        state=replace(state,audit=_freeze(audit))
        _emit(log_callback,'FINAL',prepared.context,audit=state.audit_dict(),state_ref=audit['final_state_ref'])
        return state
    except Exception as exc:
        failed_work=exc.audit_dict() if callable(getattr(exc,'audit_dict',None)) else _plain(getattr(exc,'audit',{}))
        arrays=dict(partial);arrays.update({'failed_'+k:v for k,v in getattr(exc,'arrays',{}).items()})
        arrays.update(getattr(exc,'proto_partial_arrays',{}));arrays.update(last_accepted_theta=theta,anchor_theta=prepared.anchor,Q=prepared.prototype_frame.Q)
        audit.update(status='TECHNICAL_FAILURE',failure_code=getattr(exc,'code',str(exc)),failed_work_audit=failed_work,
            fit_seconds=time.perf_counter()-started,**ledger.audit())
        try:audit['failure_state_ref']=rec.save('failure',arrays,failed=True)
        except Exception as report_exc:audit['failure_archive_error']=str(report_exc)
        try:_emit(log_callback,'FAILURE',prepared.context,audit=audit)
        except Exception as report_exc:audit['failure_log_error']=str(report_exc)
        error=ProtoFrameFailure(getattr(exc,'code',str(exc)),audit,arrays);error.records=_freeze(rec.records);raise error from exc


def predict_proto_frame_joint_local_ridge(state,**features):
    if not isinstance(state,ProtoFrameState):raise TypeError('ProtoFrame state required')
    return state.predict(**features)
