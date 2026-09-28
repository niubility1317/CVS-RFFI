"""OSC: support-only orbit alignment, common covariance, immutable inference.

Pure NumPy; no file I/O, source inputs, query fitting or persistent support bank.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import time
import numpy as np

FROZEN_CONFIG = dict(
    method='D92-OSC-v1', input_shape=[2,256], phases_quarter_turns=[0,1,2,3],
    identity_dim=160, fft_dim=96, fft='historical_spectral_logmag_sketch',
    fft_norm_floor=1e-8, norm_floor=1e-12, block_weights='equal_1_over_sqrt2',
    medoid='min_sum_pair_min_cyclic_squared_distance_then_physical_id',
    alignment='medoid_shift_zero_other_min_distance_then_smallest_shift',
    covariance='equal_class_equal_view_physical_residual', physical_df='C*(train_k-1)',
    shrinkage='256/(physical_df+256)', covariance_shape='(1-lambda)*256*S/trace(S)+lambda*I',
    zero_residual='trace(S)<=64*float64_eps*mean_squared_feature_norm',
    reference_covariance='Q/256', class_prior='equal',
    score='logmeanexp_four_cyclic_average_gaussian_experts_drop_query_common_term',
    k1='fixed_identity_covariance_no_cv', max_folds=3,
    folds='per_class_physical_id_sort_then_position_mod_min_K_3',
    selection='none_cv_diagnostic_only', optimizer_steps=0, precision='float64',
    query_fit=False, source_inputs=False, summary_inputs=False,
)

def _freeze(x):
    if isinstance(x, Mapping): return MappingProxyType({k: _freeze(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)): return tuple(_freeze(v) for v in x)
    return x


def _plain(x):
    if isinstance(x, Mapping): return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, tuple): return [_plain(v) for v in x]
    return x


_CONFIG = _freeze(FROZEN_CONFIG)
EPS = 1e-12


def _readonly(value):
    x = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(x.tobytes(), dtype=np.float64).reshape(x.shape)


def _numeric(value, shape, name, *, nonempty=False):
    x = np.asarray(value)
    if (x.ndim != len(shape)+1 or x.shape[1:] != shape or x.dtype.kind not in 'fiu'
            or not np.isfinite(x).all() or (nonempty and len(x)==0)):
        raise ValueError(name+' has invalid shape, type, or finite values')
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError(name+' is not representable in float64')
    return x


def _strings(value, name, *, nonempty=True):
    if isinstance(value, (str, bytes)): raise ValueError(name+' must be a sequence')
    try: value=tuple(value)
    except TypeError as exc: raise ValueError(name+' must be a sequence') from exc
    if ((nonempty and not value) or any(not isinstance(v,str) or not v for v in value)
            or len(set(value))!=len(value)):
        raise ValueError(name+' must contain unique physical identifiers')
    return value


def _finite(*values):
    if any(not np.isfinite(v).all() for v in values):
        raise FloatingPointError('Nonfinite OSC computation')


def _unit(value):
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        norm=np.sqrt(np.sum(value*value,axis=-1,keepdims=True))
        result=value/np.maximum(norm,EPS)
    _finite(result)
    return result


def make_received_views(received_iq):
    """Four phases of one received observation; shape [N,4,2,256]."""
    iq=_numeric(received_iq,(2,256),'received IQ')
    views=np.empty((len(iq),4,2,256),dtype=np.float64)
    views[:,0]=iq;views[:,2]=-iq
    views[:,1,0]=-iq[:,1];views[:,1,1]=iq[:,0]
    views[:,3,0]=iq[:,1];views[:,3,1]=-iq[:,0]
    return _readonly(views)


def received_fft96(received_iq):
    """Exact fixed-dim historical FFT recipe, without importing native code."""
    iq=_numeric(received_iq,(2,256),'received FFT IQ')
    with np.errstate(over='raise',invalid='raise'): raw=iq.astype(np.float32)
    result=np.empty((len(raw),96),dtype=np.float32)
    source=np.linspace(0.,1.,256,dtype=np.float64);target=np.linspace(0.,1.,96,dtype=np.float64)
    for index,row in enumerate(raw):
        z=row[0].astype(np.float64)+1j*row[1].astype(np.float64);z-=np.mean(z)
        rms=float(np.sqrt(np.mean(np.abs(z)**2)))
        if rms>1e-8: z/=rms
        spectrum=np.log1p(np.abs(np.fft.fftshift(np.fft.fft(z*np.hanning(256)))))
        sketch=np.interp(target,source,spectrum).astype(np.float32)
        sketch-=np.mean(sketch,dtype=np.float64).astype(np.float32)
        sketch/=max(float(np.linalg.norm(sketch)),1e-8)
        result[index]=sketch
    return _readonly(result)


def _features(identity_views,fft,*,nonempty=False):
    z=_unit(_numeric(identity_views,(4,160),'identity views',nonempty=nonempty))
    f=_unit(_numeric(fft,(96,),'FFT',nonempty=nonempty))
    if len(z)!=len(f): raise ValueError('Identity/FFT count mismatch')
    return np.concatenate((z,np.broadcast_to(f[:,None,:],(len(f),4,96))),axis=2)/np.sqrt(2.)


def _align(x,labels,ids,classes):
    aligned=np.empty_like(x);means=[];medoids=[];shifts=np.zeros(len(x),dtype=int)
    for cls,class_id in enumerate(classes):
        indices=np.flatnonzero(labels==cls);rows=x[indices];n=len(rows)
        distance=np.empty((n,n,4))
        for i in range(n):
            for shift in range(4):
                difference=rows[i][None,:,:]-np.roll(rows,-shift,axis=1)
                distance[i,:,shift]=np.sum(difference*difference,axis=(1,2))/4.
        medoid=int(np.argmin(distance.min(axis=2).sum(axis=1)))
        chosen=np.argmin(distance[medoid],axis=1);chosen[medoid]=0
        for local,index in enumerate(indices):
            aligned[index]=np.roll(rows[local],-int(chosen[local]),axis=0)
        shifts[indices]=chosen;means.append(aligned[indices].mean(axis=0))
        medoids.append(dict(class_id=class_id,physical_id=ids[indices[medoid]]))
    _finite(aligned,means)
    return aligned,np.asarray(means),medoids,shifts


def _covariance(aligned,labels,means,n):
    c=len(means);d=256;nu=c*(n-1);shrinkage=d/(nu+d)
    residual=aligned-means[labels]
    matrix=residual.reshape(-1,d)
    S=matrix.T@matrix/(4*nu) if nu else np.zeros((d,d))
    S=(S+S.T)/2.;trace=float(np.trace(S))
    energy=float(np.mean(np.sum(aligned*aligned,axis=2)))
    tolerance=64*np.finfo(np.float64).eps*energy
    zero=(n==1 or trace<=tolerance or energy==0.)
    Q=np.eye(d) if zero else (1.-shrinkage)*d*S/trace+shrinkage*np.eye(d)
    _finite(S,Q,trace,energy)
    return Q,dict(physical_df=nu,shrinkage=shrinkage,covariance_trace=trace,
        feature_energy=energy,zero_residual_tolerance=tolerance,
        status='ZERO_PHYSICAL_RESIDUAL' if zero else 'SHARED_COVARIANCE_FIT',
        condition_bound=1. if zero else float(nu+1),covariance_matrix_bytes=int(S.nbytes),
        covariance_shape_bytes=int(Q.nbytes))


def _fit_train(x,labels,ids,classes,*,scope,fold):
    started=time.perf_counter();c=len(classes);n=len(x)//c
    aligned,means,medoids,shifts=_align(x,labels,ids,classes)
    medoid_seconds=time.perf_counter()-started;begin=time.perf_counter()
    Q,detail=_covariance(aligned,labels,means,n)
    covariance_seconds=time.perf_counter()-begin;begin=time.perf_counter()
    L=np.linalg.cholesky(Q)
    rhs=means.reshape(-1,256).T
    W=(256*np.linalg.solve(L.T,np.linalg.solve(L,rhs))).T.reshape(c,4,256)
    b=-np.sum(means*W,axis=(1,2))/8.
    _finite(W,b)
    solve_seconds=time.perf_counter()-begin
    detail.update(scope=scope,fold=fold,train_k=n,train_physical_count=len(x),
        training_physical_ids=list(ids),medoid_physical_ids=medoids,
        alignment_shifts=[dict(physical_id=id_,shift=int(s)) for id_,s in zip(ids,shifts)],
        medoid_seconds=medoid_seconds,covariance_seconds=covariance_seconds,
        solve_seconds=solve_seconds,fit_seconds=time.perf_counter()-started,
        optimizer_steps=0,all_states_estimated_from_trainfold_only=True,
        training_feature_bytes=int(x.nbytes),aligned_feature_bytes=int(aligned.nbytes),
        template_bytes=int(means.nbytes),cholesky_bytes=int(L.nbytes),
        coefficient_bytes=int(W.nbytes),intercept_bytes=int(b.nbytes),
        persistent_state_bytes=int(W.nbytes+b.nbytes))
    return W,b,detail


def _score(x,W,b):
    result=np.empty((len(x),len(W)),dtype=np.float64)
    # One physical query per operation fixes numerical reduction across batches.
    for i,row in enumerate(x):
        pair=(row@W.reshape(-1,256).T).reshape(4,len(W),4)
        components=np.stack([sum(pair[v,:, (v+s)%4] for v in range(4))/4.+b for s in range(4)],axis=1)
        maximum=components.max(axis=1)
        result[i]=maximum+np.log(np.exp(components-maximum[:,None]).sum(axis=1))-np.log(4.)
    _finite(result)
    return result


def _diagnostics(scores,labels,classes,old):
    maximum=scores.max(axis=1)
    nll=maximum+np.log(np.exp(scores-maximum[:,None]).sum(axis=1))-scores[np.arange(len(scores)),labels]
    by_class=[dict(class_id=cls,nll=float(np.mean(nll[labels==i]))) for i,cls in enumerate(classes)]
    old_values=[v['nll'] for v in by_class if v['class_id'] in old]
    new_values=[v['nll'] for v in by_class if v['class_id'] not in old]
    _finite(nll)
    return nll,dict(macro_nll=float(np.mean([v['nll'] for v in by_class])),
        old_nll=float(np.mean(old_values)) if old_values else None,
        new_nll=float(np.mean(new_values)) if new_values else None,classwise_nll=by_class)


@dataclass(frozen=True)
class OrbitSharedState:
    W: np.ndarray
    b: np.ndarray
    classes: tuple
    audit: Mapping

    def __post_init__(self):
        classes=_strings(self.classes,'classes');W=np.asarray(self.W);b=np.asarray(self.b)
        if (W.shape!=(len(classes),4,256) or b.shape!=(len(classes),)
                or any(v.dtype.kind not in 'fiu' or not np.isfinite(v).all() for v in (W,b))):
            raise ValueError('Invalid immutable OSC state')
        object.__setattr__(self,'classes',classes)
        object.__setattr__(self,'W',_readonly(W));object.__setattr__(self,'b',_readonly(b))
        object.__setattr__(self,'audit',_freeze(self.audit))

    def score(self,identity_views,fft): return _score(_features(identity_views,fft),self.W,self.b)

    def predict(self,identity_views,fft):
        scores=self.score(identity_views,fft)
        order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:,order],axis=1)]]

    def audit_dict(self): return _plain(self.audit)


def fit_orbit_shared(*,support_identity_views,support_fft,support_labels,support_ids,classes,old_classes=()):
    started=time.perf_counter();x=_features(support_identity_views,support_fft,nonempty=True)
    labels=np.asarray(support_labels);ids=_strings(support_ids,'support IDs');requested=_strings(classes,'classes')
    old=_strings(old_classes,'old classes',nonempty=False)
    if (len(ids)!=len(x) or labels.shape!=(len(x),) or labels.dtype.kind not in 'iu'
            or set(labels.tolist())!=set(range(len(requested))) or not set(old).issubset(requested)):
        raise ValueError('Support labels, identifiers or old membership mismatch')
    canonical=tuple(sorted(requested));remap=np.asarray([canonical.index(v) for v in requested])
    labels=remap[labels.astype(np.int64)]
    counts=np.bincount(labels,minlength=len(canonical))
    if np.any(counts!=counts[0]): raise ValueError('Equal positive physical K required')
    k,c=int(counts[0]),len(canonical);order=np.asarray(sorted(range(len(ids)),key=lambda i:ids[i]))
    x,labels=x[order],labels[order];ids=tuple(ids[i] for i in order)
    folds=[];assignment=np.full(len(x),-1,dtype=int);fcount=0 if k==1 else min(k,3)
    oof=None
    if fcount:
        for cls in range(c): assignment[np.flatnonzero(labels==cls)]=np.arange(k)%fcount
        oof_scores=np.empty((len(x),c))
        for fold in range(fcount):
            keep=assignment!=fold;hold=~keep;train_ids=tuple(v for v,flag in zip(ids,keep) if flag)
            W,b,training=_fit_train(x[keep],labels[keep],train_ids,canonical,scope='fold',fold=fold)
            begin=time.perf_counter();scores=_score(x[hold],W,b);held_seconds=time.perf_counter()-begin
            oof_scores[hold]=scores;_,risk=_diagnostics(scores,labels[hold],canonical,old)
            folds.append(dict(fold=fold,train_k=int(keep.sum()//c),heldout_k=int(hold.sum()//c),
                train_ids=list(train_ids),heldout_ids=[v for v,flag in zip(ids,hold) if flag],
                training=training,diagnostics=risk,heldout_score_seconds=held_seconds))
        nll,risk=_diagnostics(oof_scores,labels,canonical,old)
        oof=dict(**risk,physical_records=[dict(physical_id=id_,class_id=canonical[int(y)],fold=int(f),
            nll=float(loss),predicted_class=canonical[int(np.argmax(score))])
            for id_,y,f,loss,score in zip(ids,labels,assignment,nll,oof_scores)])
    W,b,final=_fit_train(x,labels,ids,canonical,scope='final',fold=None)
    requested_indices=[canonical.index(v) for v in requested];W=W[requested_indices];b=b[requested_indices]
    state_bytes=int(W.nbytes+b.nbytes)
    audit=dict(method='D92-OSC-v1',config=_plain(_CONFIG),k=k,classes=c,fold_count=fcount,
        selected='fixed_orbit_shared',selection='none_cv_diagnostic_only',candidate_count=1,
        optimizer_steps=0,final_optimizer_steps=0,steps=[],folds=folds,final_fit=final,oof=oof,
        physical_fold_assignment=[dict(physical_id=id_,class_id=canonical[int(y)],fold=int(f)) for id_,y,f in zip(ids,labels,assignment)],
        head_bytes=state_bytes,persistent_state_bytes=state_bytes,
        persistent_state_bytes_scope='float64 W+b only; class IDs/audit/container separate',
        support_feature_bytes=int(x.nbytes),fit_seconds=time.perf_counter()-started,
        source_fit_rows=0,query_fit_rows=0,source_rows_used_for_fit=0,query_rows_used_for_fit=0,
        new_source_payload_bytes=0,ground_summary_used=False,encoder_updated=False,
        cross_row_adapted_state_reuse=False,source_validation=None,
        source_validation_reason='No source samples/features are permitted or consumed')
    return OrbitSharedState(W,b,requested,audit)
