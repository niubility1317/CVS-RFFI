"""BNNA: frozen received views and bounded, physical-support-only adaptation.

Pure NumPy, analytical gradients, no file I/O or torch/NumPy bridge. No query
fit interface, ground/source input, mutable inference state, or warm start.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
import time

import numpy as np


FROZEN_CONFIG = dict(
    method='D92-BNNA-v1', input_shape=[2, 256], phases_quarter_turns=[0, 1, 2, 3],
    identity_dim=160, fft_dim=96, fft='historical_spectral_logmag_sketch', fft_norm_floor=1e-8,
    norm_floor=1e-12, block_weights='equal_1_over_sqrt2', max_rank=8, trace_floor=1e-24,
    eigen_tolerance='64*float64_eps*trace', basis='within_physical_view_covariance_trainfold_only',
    repeated_eigen_basis='project_standard_axes_in_order_then_reorthogonalize',
    gate_bounds=[0.0, 0.5], gate_initialization=0.0, nonlinearity_scale='sqrt160',
    adapter='unit(z-B.T@(a*tanh(sqrt160*B@z)/sqrt160))',
    physical_representation='unit(mean_four_adapted_identity_fft_views)',
    prototype='unit(sum_physical_representations_per_class)', score_scale=10.0,
    loss_cls='physical_leave_one_out_if_train_k>=2_else_cross_view_training_only',
    loss_view='mean_squared_deviation_from_physical_mean', loss_gate='mean_squared_gate',
    loss_view_weight='1/(train_k+1)', loss_gate_weight='1/(train_k+1)',
    optimizer='projected_Adam_float64_analytic_gradient', steps=64, learning_rate=0.03,
    adam_betas=[0.9, 0.999], adam_epsilon=1e-8, final_state='after_step64_not_best_step',
    max_folds=3, folds='per_class_physical_id_sort_then_position_mod_min_K_3',
    candidates=['identity', 'trained'], k1='fixed_trained_no_holdout',
    selection='min_max_old_new_oof_nll_then_macro_nll_then_identity',
    selection_round_decimals=10, refit='from_zero_all_current_row_support',
    precision='float64', query_fit=False, source_inputs=False, summary_inputs=False,
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
TAU = np.sqrt(160.0)


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
        raise FloatingPointError('Nonfinite BNNA computation')


def _unit(value):
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        norm=np.sqrt(np.sum(value*value,axis=-1,keepdims=True))
        result=value/np.maximum(norm,EPS)
    _finite(result)
    return result


def _unit_backward(value,gradient):
    norm=np.sqrt(np.sum(value*value,axis=-1,keepdims=True))
    unit=value/np.maximum(norm,EPS)
    tangent=gradient-unit*np.sum(unit*gradient,axis=-1,keepdims=True)
    return np.where(norm>EPS,tangent/np.maximum(norm,EPS),gradient/EPS)


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


def _canonical_subspace(columns):
    """Coordinate projection makes a repeated-eigenvalue subspace deterministic."""
    projection=columns@columns.T;basis=[]
    threshold=64*np.finfo(float).eps*160
    for axis in range(160):
        v=projection[:,axis].copy()
        for _ in range(2):
            for q in basis: v-=np.dot(q,v)*q
        norm=float(np.linalg.norm(v))
        if norm>threshold:
            v/=norm
            nonzero=np.flatnonzero(np.abs(v)>threshold)
            if len(nonzero) and v[nonzero[0]]<0: v=-v
            basis.append(v)
            if len(basis)==columns.shape[1]: break
    if len(basis)!=columns.shape[1]: raise FloatingPointError('Degenerate canonical nuisance basis')
    return basis


def _nuisance_basis(z):
    residual=z-z.mean(axis=1,keepdims=True)
    flat=residual.reshape(-1,160)
    covariance=(flat.T@flat)/len(flat)
    trace=float(np.trace(covariance));_finite(covariance)
    audit=dict(covariance_trace=trace,covariance_bytes=int(covariance.nbytes),eigen_tolerance=64*np.finfo(float).eps*trace)
    if trace<=1e-24:
        return np.empty((0,160)),dict(audit,active_rank=0,eigenvalues=[],basis_orthogonality_error=0.0,status='NO_VIEW_VARIATION')
    values,vectors=np.linalg.eigh(covariance);values=values[::-1];vectors=vectors[:,::-1]
    tolerance=audit['eigen_tolerance'];positive=int(np.count_nonzero(values>tolerance))
    rows=[];index=0
    while index<positive and len(rows)<8:
        end=index+1
        while end<positive and abs(values[end]-values[index])<=tolerance: end+=1
        rows.extend(_canonical_subspace(vectors[:,index:end]));index=end
    basis=np.asarray(rows[:8],dtype=np.float64).reshape(-1,160)
    error=float(np.max(np.abs(basis@basis.T-np.eye(len(basis))))) if len(basis) else 0.0
    if error>1e-10: raise FloatingPointError('Nuisance basis lost orthogonality')
    return basis,dict(audit,active_rank=len(basis),eigenvalues=[float(v) for v in values[:len(basis)]],
        basis_orthogonality_error=error,status='BASIS_READY' if len(basis) else 'NO_VIEW_VARIATION')


def _adapt(z,fft,basis,gate,terms=None):
    terms=np.tanh(TAU*(z@basis.T))/TAU if terms is None else terms
    residual=(terms*gate)@basis
    w=z-residual;q=_unit(w)
    h=np.concatenate((q,np.repeat(fft[:,None,:],4,axis=1)),axis=-1)/np.sqrt(2.)
    mean=h.mean(axis=1);physical=_unit(mean)
    return h,physical,(terms,w,q,mean,residual)


def _class_prototypes(physical,labels,count):
    return _unit(np.stack([physical[labels==c].sum(axis=0) for c in range(count)]))


def _leave_one_ce(examples,labels,count):
    """Loss and analytic gradient, with both sides of all prototypes trained.

    All negative classes share a prototype matrix. Only the positive class is
    leave-one-out. Memory is O(ED+EC+CD), never an ExCxD prototype tensor.
    """
    total=len(examples)
    sums=np.stack([examples[labels==c].sum(axis=0) for c in range(count)])
    common=_unit(sums)
    positive_raw=sums[labels]-examples
    positive=_unit(positive_raw)
    indices=np.arange(total)
    logits=10.*(examples@common.T)
    logits[indices,labels]=10.*np.sum(examples*positive,axis=1)
    maximum=logits.max(axis=1)
    delta=np.exp(logits-maximum[:,None]);mass=delta.sum(axis=1)
    loss=float(np.mean(maximum+np.log(mass)-logits[indices,labels]))
    delta/=mass[:,None];delta[indices,labels]-=1.;delta/=total
    positive_delta=delta[indices,labels].copy()
    # Remove the common positive prototype before aggregating negative-class
    # contributions; then add the actual leave-one-out positive contribution.
    delta[indices,labels]=0.
    gradient=10.*(delta@common)+10.*positive_delta[:,None]*positive
    common_gradient=_unit_backward(sums,10.*(delta.T@examples))
    common_zero=np.linalg.norm(sums,axis=1)<=EPS
    # The epsilon branch multiplies cancellation error by 1/EPS. Preserve the
    # original contribution order there, rather than amplify GEMM/FMA residue
    # when opposing examples give an exactly zero class sum. Ordinary class
    # sums keep the shared GEMM path; this branch uses only one D-vector.
    for cls in np.flatnonzero(common_zero):
        accumulated=np.zeros(examples.shape[1],dtype=np.float64)
        for i in range(total):
            accumulated+=(10.*delta[i,cls]*examples[i])/EPS
        common_gradient[cls]=accumulated
    positive_gradient=_unit_backward(positive_raw,10.*positive_delta[:,None]*examples)
    positive_sum_gradient=np.stack([positive_gradient[labels==c].sum(axis=0) for c in range(count)])
    gradient+=common_gradient[labels]+positive_sum_gradient[labels]-positive_gradient
    near_zero=(total*int(np.count_nonzero(common_zero))-int(np.count_nonzero(common_zero[labels]))
               +int(np.count_nonzero(np.linalg.norm(positive_raw,axis=1)<=EPS)))
    _finite(loss,gradient)
    return loss,gradient,near_zero


def _objective_gradient(gate,z,fft,labels,basis,train_k,terms=None):
    h,physical,cache=_adapt(z,fft,basis,gate,terms)
    terms,w,q,mean,residual=cache;count=int(labels.max())+1
    if train_k>=2:
        loss_cls,gphysical,near_zero=_leave_one_ce(physical,labels,count)
        gh=np.repeat((_unit_backward(mean,gphysical)/4.)[:,None,:],4,axis=1)
    else:
        loss_cls,gviews,near_zero=_leave_one_ce(h.reshape(-1,256),np.repeat(labels,4),count)
        gh=gviews.reshape(h.shape)
    deviation=h-mean[:,None,:]
    loss_view=float(np.sum(deviation**2)/(len(z)*4))
    loss_gate=float(np.mean(gate**2)) if len(gate) else 0.
    weight=1./(train_k+1)
    gh+=weight*2.*deviation/(len(z)*4)
    gw=_unit_backward(w,gh[:,:,:160]/np.sqrt(2.))
    gradient=-np.sum(terms*(gw@basis.T),axis=(0,1))
    if len(gate): gradient+=weight*2.*gate/len(gate)
    loss=loss_cls+weight*(loss_view+loss_gate)
    _finite(loss,gradient)
    metrics=dict(loss_cls=loss_cls,loss_view=loss_view,loss_gate=loss_gate,loss_cls_weight=1.,
        loss_view_weight=weight,loss_gate_weight=weight,loss_total=float(loss),
        near_zero_prototype_count=near_zero,near_zero_physical_mean_count=int(np.count_nonzero(np.linalg.norm(mean,axis=1)<=EPS)))
    return float(loss),gradient,metrics


def _fit_train(z,fft,labels,ids,count,*,scope,fold,started):
    begin=time.perf_counter();train_k=len(z)//count
    basis,basis_audit=_nuisance_basis(z);rank=len(basis);gate=np.zeros(rank)
    terms=np.tanh(TAU*(z@basis.T))/TAU
    first=np.zeros(rank);second=np.zeros(rank);steps=[]
    for index in range(64 if rank else 0):
        step_start=time.perf_counter()
        _,gradient,metrics=_objective_gradient(gate,z,fft,labels,basis,train_k,terms)
        before=gate.copy();first=.9*first+.1*gradient;second=.999*second+.001*gradient**2
        update=.03*(first/(1.-.9**(index+1)))/(np.sqrt(second/(1.-.999**(index+1)))+1e-8)
        proposed=gate-update;gate=np.clip(proposed,0.,.5);_finite(gate,gradient)
        steps.append(dict(scope=scope,fold=fold,step=index+1,train_k=train_k,train_physical_count=len(z),
            active_rank=rank,**metrics,learning_rate=.03,gradient_norm=float(np.linalg.norm(gradient)),
            gate_before_min=float(before.min()),gate_before_max=float(before.max()),
            gate_after_min=float(gate.min()),gate_after_max=float(gate.max()),
            clipped_coordinates=int(np.count_nonzero(proposed!=gate)),step_seconds=time.perf_counter()-step_start,
            elapsed_seconds=time.perf_counter()-started,loss_evaluation='before_update'))
    _,gradient,metrics=_objective_gradient(gate,z,fft,labels,basis,train_k,terms)
    h,physical,cache=_adapt(z,fft,basis,gate,terms)
    _,w,q,mean,residual=cache
    prototypes=_class_prototypes(physical,labels,count)
    z_norm=np.linalg.norm(z,axis=-1);q_norm=np.linalg.norm(q,axis=-1)
    valid=(z_norm>EPS)&(q_norm>EPS)
    cos=np.clip(np.sum(q*z,axis=-1)/np.maximum(z_norm*q_norm,EPS**2),-1.,1.)
    max_angle=float(np.degrees(np.arccos(cos[valid])).max()) if np.any(valid) else 0.
    bound=.5*np.sqrt(rank/160.)
    relative_bound=np.minimum(.5,bound/np.maximum(z_norm,EPS))
    angle_bound=float(np.degrees(np.arcsin(relative_bound[valid])).max()) if np.any(valid) else 0.
    observed=float(np.linalg.norm(residual,axis=-1).max())
    if observed>bound+1e-10: raise FloatingPointError('BNNA residual bound violated')
    if np.any(np.degrees(np.arccos(cos[valid]))>np.degrees(np.arcsin(relative_bound[valid]))+1e-5):
        raise FloatingPointError('BNNA angular bound violated')
    audit=dict(basis_audit,scope=scope,fold=fold,train_k=train_k,train_physical_count=len(z),training_physical_ids=list(ids),
        status='TRAINING_BUDGET_COMPLETE' if rank else 'NO_VIEW_VARIATION',steps=len(steps),
        final_losses=metrics,final_gradient_norm=float(np.linalg.norm(gradient)),final_gate=[float(v) for v in gate],
        measured_max_residual_norm=observed,residual_norm_upper_bound=float(bound),measured_max_angle_degrees=max_angle,
        angular_upper_bound_degrees=angle_bound,subunit_identity_count=int(np.count_nonzero((z_norm>0)&(z_norm<1-1e-10))),
        fit_seconds=time.perf_counter()-begin,
        training_feature_bytes=int(z.nbytes+fft.nbytes),cached_nonlinearity_bytes=int(terms.nbytes),
        optimizer_state_bytes=int(first.nbytes+second.nbytes),identity_initialization=True,
        basis_estimated_from_trainfold_only=True)
    return basis,gate,prototypes,audit,steps


def _score_inputs(z,fft,basis,gate,prototypes):
    result=np.empty((len(z),len(prototypes)),dtype=np.float64)
    # Each physical query uses the exact same reduction layout, regardless of batch.
    for i in range(len(z)):
        _,physical,_=_adapt(z[i:i+1],fft[i:i+1],basis,gate)
        result[i]=10.*np.sum(prototypes*physical[0][None,:],axis=1)
    _finite(result)
    return result


def _risk(scores,labels,classes,old_classes):
    maximum=scores.max(axis=1);loss=maximum+np.log(np.exp(scores-maximum[:,None]).sum(axis=1))-scores[np.arange(len(labels)),labels]
    by_class=[float(loss[labels==c].mean()) for c in range(len(classes))]
    old=[value for cls,value in zip(classes,by_class) if cls in old_classes]
    new=[value for cls,value in zip(classes,by_class) if cls not in old_classes]
    groups=[float(np.mean(group)) for group in (old,new) if group]
    _finite(loss)
    return dict(objective=max(groups),macro_nll=float(np.mean(by_class)),old_nll=float(np.mean(old)) if old else None,
        new_nll=float(np.mean(new)) if new else None)


@dataclass(frozen=True)
class BNNAState:
    basis: np.ndarray
    gate: np.ndarray
    prototypes: np.ndarray
    classes: tuple
    audit: Mapping

    def __post_init__(self):
        classes=_strings(self.classes,'classes');basis=np.asarray(self.basis);gate=np.asarray(self.gate);proto=np.asarray(self.prototypes)
        if (basis.ndim!=2 or basis.shape[1]!=160 or len(basis)>8 or gate.shape!=(len(basis),)
                or proto.shape!=(len(classes),256) or any(v.dtype.kind not in 'fiu' for v in (basis,gate,proto))
                or any(not np.isfinite(v).all() for v in (basis,gate,proto)) or np.any((gate<0)|(gate>.5))
                or not np.allclose(basis@basis.T,np.eye(len(basis)),rtol=0,atol=1e-10)):
            raise ValueError('Invalid immutable BNNA state')
        object.__setattr__(self,'classes',classes)
        for name,value in (('basis',basis),('gate',gate),('prototypes',proto)): object.__setattr__(self,name,_readonly(value))
        object.__setattr__(self,'audit',_freeze(self.audit))

    def score(self,identity_views,fft):
        z=_unit(_numeric(identity_views,(4,160),'query identity views'))
        f=_numeric(fft,(96,),'query FFT')
        if len(z)!=len(f): raise ValueError('Query identity/FFT count mismatch')
        return _score_inputs(z,f,self.basis,self.gate,self.prototypes)

    def predict(self,identity_views,fft):
        scores=self.score(identity_views,fft)
        order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:,order],axis=1)]]

    def audit_dict(self): return _plain(self.audit)


def fit_bnna(*,support_identity_views,support_fft,support_labels,support_ids,classes,old_classes):
    started=time.perf_counter()
    z=_unit(_numeric(support_identity_views,(4,160),'support identity',nonempty=True))
    fft=_numeric(support_fft,(96,),'support FFT',nonempty=True);labels=np.asarray(support_labels)
    ids=_strings(support_ids,'support IDs');classes=_strings(classes,'classes');old=_strings(old_classes,'old classes',nonempty=False)
    if (len(z)!=len(fft) or len(z)!=len(ids) or labels.shape!=(len(z),) or labels.dtype.kind not in 'iu'
            or set(labels.tolist())!=set(range(len(classes))) or not set(old).issubset(classes)):
        raise ValueError('Support labels, identifiers, or old class membership mismatch')
    labels=labels.astype(np.int64);counts=np.bincount(labels,minlength=len(classes))
    if np.any(counts!=counts[0]): raise ValueError('Equal positive physical K required')
    k,c=int(counts[0]),len(classes);order=np.asarray(sorted(range(len(ids)),key=lambda i:ids[i]))
    z,fft,labels=z[order],fft[order],labels[order];ids=tuple(ids[i] for i in order)
    fcount=0 if k==1 else min(k,3);fold=np.full(len(z),-1,dtype=int);folds=[];steps=[];candidates=[]
    zero_basis=np.empty((0,160));zero_gate=np.empty(0)
    if fcount:
        for cls in range(c):
            positions=np.flatnonzero(labels==cls);fold[positions]=np.arange(k)%fcount
        oof={arm:np.empty((len(z),c)) for arm in ('identity','trained')}
        for f in range(fcount):
            keep,hold=fold!=f,fold==f;train_ids=tuple(id_ for id_,yes in zip(ids,keep) if yes)
            basis,gate,proto,details,updates=_fit_train(z[keep],fft[keep],labels[keep],train_ids,c,scope='fold',fold=f,started=started)
            steps.extend(updates)
            _,base_physical,_=_adapt(z[keep],fft[keep],zero_basis,zero_gate)
            base_proto=_class_prototypes(base_physical,labels[keep],c)
            trained=_score_inputs(z[hold],fft[hold],basis,gate,proto)
            identity=_score_inputs(z[hold],fft[hold],zero_basis,zero_gate,base_proto)
            for arm,value in (('identity',identity),('trained',trained)): oof[arm][hold]=value
            folds.append(dict(fold=f,train_k=int(np.count_nonzero(keep)//c),heldout_k=int(np.count_nonzero(hold)//c),
                train_ids=list(train_ids),heldout_ids=[id_ for id_,yes in zip(ids,hold) if yes],training=details,
                identity_risk=_risk(identity,labels[hold],classes,old),trained_risk=_risk(trained,labels[hold],classes,old)))
        candidates=[dict(candidate=arm,**_risk(scores,labels,classes,old)) for arm,scores in oof.items()]
        selected=min(candidates,key=lambda v:(round(v['objective'],10),round(v['macro_nll'],10),v['candidate']!='identity'))['candidate']
        selection='physical_support_oof';chosen=next(v for v in candidates if v['candidate']==selected)
    else:
        selected='trained';selection='k1_fixed_trained';chosen=None
    if selected=='trained':
        basis,gate,proto,final,updates=_fit_train(z,fft,labels,ids,c,scope='final',fold=None,started=started);steps.extend(updates)
    else:
        begin=time.perf_counter();basis,gate=zero_basis,zero_gate
        _,physical,_=_adapt(z,fft,basis,gate);proto=_class_prototypes(physical,labels,c)
        final=dict(scope='final',fold=None,status='IDENTITY_SELECTED',steps=0,train_k=k,train_physical_count=len(z),
            active_rank=0,training_physical_ids=list(ids),fit_seconds=time.perf_counter()-begin,
            final_losses=None,final_gradient_norm=None,identity_initialization=True)
    head_bytes=int(proto.nbytes);basis_bytes=int(basis.nbytes);gate_bytes=int(gate.nbytes)
    audit=dict(method='D92-BNNA-v1',config=_plain(_CONFIG),k=k,classes=c,selected=selected,selection=selection,
        fold_count=fcount,candidate_count=len(candidates),active_rank=len(basis),candidates=candidates,folds=folds,steps=steps,final_fit=final,
        selected_objective=None if chosen is None else chosen['objective'],selected_macro_nll=None if chosen is None else chosen['macro_nll'],
        selected_old_nll=None if chosen is None else chosen['old_nll'],selected_new_nll=None if chosen is None else chosen['new_nll'],
        physical_fold_assignment=[dict(physical_id=id_,class_id=classes[int(label)],fold=int(f)) for id_,label,f in zip(ids,labels,fold)],
        head_bytes=head_bytes,basis_bytes=basis_bytes,gate_bytes=gate_bytes,persistent_state_bytes=head_bytes+basis_bytes+gate_bytes,
        persistent_state_bytes_scope='float64 prototypes+basis+gate; excludes identifiers/config/logs and working memory',
        support_feature_bytes=int(z.nbytes+fft.nbytes),fit_seconds=time.perf_counter()-started,
        optimizer_status=final['status'],optimizer_steps=len(steps),final_optimizer_steps=final['steps'],
        source_validation=None,source_validation_reason='No source samples/features are permitted or consumed',
        query_rows_used_for_fit=0,source_rows_used_for_fit=0,new_source_payload_bytes=0,ground_summary_used=False,
        encoder_updated=False,cross_row_adapted_state_reuse=False)
    return BNNAState(basis,gate,proto,classes,audit)
