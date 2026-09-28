"""Physical-balanced multiview ridge; source/query-free closed-form fit.

Every physical sample contributes total loss weight one across four views.
All native feature extraction remains frozen. No file I/O or mutable scoring.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import time
import numpy as np

FROZEN_CONFIG = dict(
    method='D92-MVRidge-v1',input_shape=[2,256],phases_quarter_turns=[0,1,2,3],
    identity_dim=160,fft_dim=96,fft='historical_spectral_logmag_sketch',fft_norm_floor=1e-8,
    norm_floor=1e-12,features='unit(concat(unit(identity160),4*unit(FFT96)))',
    target='onehot_minus_1_over_C',objective='0.5*sum_physical(mean_4_view_squared_error)+0.5*||W||F^2',
    view_weight=0.25,ridge_coefficient=1.0,intercept_regularized=False,
    solver='float64_Cholesky',query_representation='mean_four_features',
    k1='same_closed_form_no_cv',max_folds=3,
    folds='per_class_physical_id_sort_then_position_mod_min_K_3',
    selection='none_cv_diagnostic_only',optimizer_steps=0,precision='float64',
    query_fit=False,source_inputs=False,summary_inputs=False,
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
        raise FloatingPointError('Nonfinite MVRidge computation')


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
    return _unit(np.concatenate((z,4.*np.broadcast_to(f[:,None,:],(len(f),4,96))),axis=2))


def _fit_train(x,labels,ids,classes,*,scope,fold):
    started=time.perf_counter();n=len(x);c=len(classes);k=n//c
    means=x.mean(axis=1);center=means.mean(axis=0);centered=means-center
    targets=np.eye(c)[labels]-1./c
    center_seconds=time.perf_counter()-started;begin=time.perf_counter()
    differences=(x-means[:,None,:]).reshape(-1,256)
    view_scatter=differences.T@differences/4.
    gram=np.eye(256)+centered.T@centered+view_scatter
    cross=centered.T@targets
    _finite(gram,cross)
    gram_seconds=time.perf_counter()-begin;begin=time.perf_counter()
    chol=np.linalg.cholesky(gram)
    W=np.linalg.solve(chol.T,np.linalg.solve(chol,cross));b=-center@W
    _finite(W,b)
    solve_seconds=time.perf_counter()-begin;begin=time.perf_counter()
    residual=x.reshape(-1,256)@W+b-np.repeat(targets,4,axis=0)
    mean_residual=means@W+b-targets
    view_output=differences@W
    loss_mean=float(.5*np.sum(mean_residual*mean_residual))
    loss_view=float(np.sum(view_output*view_output)/8.)
    loss_ridge=float(.5*np.sum(W*W))
    loss_expanded=float(np.sum(residual*residual)/8.)
    gradW=x.reshape(-1,256).T@residual/4.+W;gradb=residual.sum(axis=0)/4.
    gradnorm=float(np.sqrt(np.sum(gradW*gradW)+np.sum(gradb*gradb)))
    normal_residual=float(np.linalg.norm(gram@W-cross))
    _finite(loss_mean,loss_view,loss_ridge,loss_expanded,gradnorm,normal_residual)
    objective_seconds=time.perf_counter()-begin
    audit=dict(scope=scope,fold=fold,status='CLOSED_FORM_SOLVED',train_k=k,train_physical_count=n,
        training_physical_ids=list(ids),physical_loss_mass=float(n),view_weight=.25,ridge_coefficient=1.,
        target_norm_squared=float(np.sum(targets*targets)),loss_data_mean=loss_mean,loss_view=loss_view,
        loss_ridge=loss_ridge,loss_total=loss_mean+loss_view+loss_ridge,loss_expanded_data=loss_expanded,
        gradient_norm=gradnorm,normal_equation_residual=normal_residual,
        intercept_gradient_norm=float(np.linalg.norm(gradb)),condition_bound=float(n+1),
        gram_min_eigenvalue_lower_bound=1.,center_seconds=center_seconds,gram_seconds=gram_seconds,
        solve_seconds=solve_seconds,objective_seconds=objective_seconds,fit_seconds=time.perf_counter()-started,
        coefficient_bytes=int(W.nbytes),intercept_bytes=int(b.nbytes),persistent_state_bytes=int(W.nbytes+b.nbytes),
        training_feature_bytes=int(x.nbytes),gram_bytes=int(gram.nbytes),view_scatter_bytes=int(view_scatter.nbytes),
        cross_moment_bytes=int(cross.nbytes),optimizer_steps=0,all_states_estimated_from_trainfold_only=True)
    return W,b,audit


def _score(x,W,b):
    result=np.empty((len(x),len(b)),dtype=np.float64)
    for i,row in enumerate(x): result[i]=row.mean(axis=0)@W+b
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
class MultiviewRidgeState:
    W: np.ndarray
    b: np.ndarray
    classes: tuple
    audit: Mapping

    def __post_init__(self):
        classes=_strings(self.classes,'classes');W=np.asarray(self.W);b=np.asarray(self.b)
        if (W.shape!=(256,len(classes)) or b.shape!=(len(classes),)
                or any(v.dtype.kind not in 'fiu' or not np.isfinite(v).all() for v in (W,b))):
            raise ValueError('Invalid immutable MVRidge state')
        object.__setattr__(self,'classes',classes)
        object.__setattr__(self,'W',_readonly(W));object.__setattr__(self,'b',_readonly(b))
        object.__setattr__(self,'audit',_freeze(self.audit))

    def score(self,identity_views,fft): return _score(_features(identity_views,fft),self.W,self.b)

    def predict(self,identity_views,fft):
        scores=self.score(identity_views,fft)
        order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:,order],axis=1)]]

    def audit_dict(self): return _plain(self.audit)


def fit_multiview_ridge(*,support_identity_views,support_fft,support_labels,support_ids,classes,old_classes=()):
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
    requested_indices=[canonical.index(v) for v in requested];W=W[:,requested_indices];b=b[requested_indices]
    state_bytes=int(W.nbytes+b.nbytes)
    audit=dict(method='D92-MVRidge-v1',config=_plain(_CONFIG),k=k,classes=c,fold_count=fcount,
        selected='fixed_multiview_ridge',selection='none_cv_diagnostic_only',candidate_count=1,
        optimizer_steps=0,final_optimizer_steps=0,steps=[],folds=folds,final_fit=final,oof=oof,
        oof_score_semantics='softmax_of_fixed_linear_scores_diagnostic_not_calibrated_posterior',
        physical_fold_assignment=[dict(physical_id=id_,class_id=canonical[int(y)],fold=int(f)) for id_,y,f in zip(ids,labels,assignment)],
        head_bytes=state_bytes,persistent_state_bytes=state_bytes,
        persistent_state_bytes_scope='float64 W+b only; class IDs/audit/container separate',
        support_feature_bytes=int(x.nbytes),fit_seconds=time.perf_counter()-started,
        source_fit_rows=0,query_fit_rows=0,source_rows_used_for_fit=0,query_rows_used_for_fit=0,
        new_source_payload_bytes=0,ground_summary_used=False,encoder_updated=False,
        cross_row_adapted_state_reuse=False,source_validation=None,
        source_validation_reason='No source samples/features are permitted or consumed')
    return MultiviewRidgeState(W,b,requested,audit)
