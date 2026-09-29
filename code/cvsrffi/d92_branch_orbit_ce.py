"""Physical-weighted, frozen-view orbit kernel classifiers; no I/O/query fit."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from . import d92_branch_interaction as base
from .d92_branch_support_probe import _classification, _finite, _strings
from .d92_branch_ridge import _freeze, _plain, _readonly

FROZEN_CONFIG = dict(method='D92-BranchOrbitCE-v1', schema='d92_branch_orbit_ce_v1',
    views='C4_original_received_phase_0_pi_over_2_pi_3pi_over_2', view_count=4,
    view_transform='(I,Q),(-Q,I),(-I,-Q),(Q,-I)', branches=['t_emb','f_emb','pa_local'],
    identity_dim=160, fft_dim=96, norm_floor=1e-12,
    fft='historical_spectral_logmag_sketch_original_received_once', fft_norm_floor=1e-8,
    background='unit(concat(unit(z_id),4*unit(fft96)))',
    auxiliary='concat(unit(t_emb),unit(f_emb),unit(pa_local))/sqrt(3)',
    base_kernel='KB+KA+KB*KA', orbit_pooling='mean_over_all_16_cross_view_kernel_pairs',
    orbit_normalization='3*kbar/max(sqrt(kbar_xx),1e-12)/max(sqrt(kbar_yy),1e-12)',
    centering='train_support_only_reference_difference_then_mean',
    intercept='fixed_by_train_feature_centering_no_extra_learned_intercept',
    arms=['single_ridge','single_ce','orbit_ridge','orbit_ce'], selected='orbit_ce', selection='fixed_no_selection',
    objective_ce='sum_physical_multiclass_cross_entropy+0.5*RKHS_norm_squared',
    objective_ridge='0.5*sum_physical_squared_error+0.5*RKHS_norm_squared',
    ridge_coefficient=1., physical_sample_weight=1., view_loss_weight='one_orbit_embedding_per_physical_record',
    ce_solver='strongly_convex_accelerated_RKHS_gradient', ce_initialization='zero',
    ce_lipschitz_bound='1+0.5*trace(centered_kernel)', ce_step='1/L',
    ce_momentum='(sqrt(L)-1)/(sqrt(L)+1)', ce_gradient_rtol=1e-7,
    ce_gradient_tolerance='1e-7*(1+sqrt(train_physical_count))', ce_max_iterations=2000,
    ce_stop='current_train_RKHS_gradient_only', ce_failure='technical_failure_preserve_trace_no_fallback',
    max_folds=3, physical_folds='per_class_physical_id_sort_position_mod_min_K_3',
    oneshot_proxy_anchors='all_per_class_sorted_positions_0_to_parent_K_minus_1',
    k1='same_physical_objective_views_not_independent_shots',
    query_decision_policy='per_sample_all_registered_classes', tie_break='physical_class_id_lexicographic',
    source_inputs=False, summary_inputs=False, query_fit=False, phase1_frozen=True)
_CONFIG=deepcopy(FROZEN_CONFIG)
_ARMS=tuple(_CONFIG['arms'])
_PAIRS=(('orbit_ce','single_ridge'),('orbit_ce','single_ce'),('orbit_ce','orbit_ridge'),
        ('single_ce','single_ridge'),('orbit_ridge','single_ridge'))


class OrbitCEConvergenceError(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit=_plain(audit)


def _blocks(z_id, fft, t_emb, f_emb, pa_local, *, allow_empty=False):
    views=[np.asarray(value) for value in (z_id,t_emb,f_emb,pa_local)]
    f=np.asarray(fft)
    if (any(value.ndim!=3 or value.shape[1:]!=(4,160) or value.dtype.kind not in 'fiu' for value in views)
            or f.ndim!=2 or f.shape[1:]!=(96,) or f.dtype.kind not in 'fiu'
            or len({len(value) for value in views+[f]})!=1):
        raise ValueError('Expected four [N,4,160] branch arrays and original [N,96] FFT')
    b,a=[],[]
    for v in range(4):
        bv,av=base._blocks(views[0][:,v],f,views[1][:,v],views[2][:,v],views[3][:,v],allow_empty=allow_empty)
        b.append(bv);a.append(av)
    return np.stack(b,axis=1),np.stack(a,axis=1)


def _prepare(z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes):
    b,a=_blocks(z_id,fft,t_emb,f_emb,pa_local)
    _,_,labels,ids,canonical,requested,old,k=base._prepare(
        np.asarray(z_id)[:,0],fft,np.asarray(t_emb)[:,0],np.asarray(f_emb)[:,0],np.asarray(pa_local)[:,0],
        support_labels,support_ids,classes,old_classes)
    original_ids=tuple(support_ids)
    order=np.asarray(sorted(range(len(original_ids)),key=lambda i:original_ids[i]))
    return b[order],a[order],labels,ids,canonical,requested,old,k


def _orbit_parts(b,a):
    # mean(B outer A) = Q T with B.T = Q R and T = R A / 4.
    # Direct means and orthonormal Q avoid cancellation of large self-kernel terms.
    q,r=np.linalg.qr(np.swapaxes(b,1,2),mode='reduced')
    t=(r@a)/4.
    return b.mean(axis=1),a.mean(axis=1),np.ascontiguousarray(np.swapaxes(q,1,2)),t


def _parts_kernel(left,right):
    mb,ma,q,t=left;rb,ra,rq,rt=right
    value=mb@rb.T+ma@ra.T
    for v in range(q.shape[1]):
        for w in range(rq.shape[1]):
            value+=(q[:,v]@rq[:,w].T)*(t[:,v]@rt[:,w].T)
    _finite(value)
    return value


def _pooled_kernel(b,a,tb,ta):
    left=_orbit_parts(b,a)
    right=left if b is tb and a is ta else _orbit_parts(tb,ta)
    return _parts_kernel(left,right)


def _scale(b,a):
    mb,ma,q,t=_orbit_parts(b,a)
    diagonal=np.sum(mb*mb,axis=1)+np.sum(ma*ma,axis=1)+np.sum(t*t,axis=(1,2))
    scale=np.sqrt(3.)/np.maximum(np.sqrt(diagonal),_CONFIG['norm_floor'])
    _finite(scale,diagonal)
    return scale,diagonal


def _kernel(b,a,tb,ta,scale,tscale,orbit):
    if orbit: return _pooled_kernel(b,a,tb,ta)*scale[:,None]*tscale[None,:]
    return base._combine(b[:,0]@tb[:,0].T,a[:,0]@ta[:,0].T,'interaction')


def _ce_stats(alpha,g,labels):
    logits=g@alpha
    maxima=logits.max(axis=1,keepdims=True)
    exp=np.exp(logits-maxima);prob=exp/exp.sum(axis=1,keepdims=True)
    loss=float(np.sum(maxima[:,0]+np.log(exp.sum(axis=1))-logits[np.arange(len(labels)),labels]))
    penalty=float(.5*np.sum(alpha*logits))
    residual=prob.copy();residual[np.arange(len(labels)),labels]-=1.
    gradient=alpha+residual
    gradient_norm=float(np.sqrt(max(0.,float(np.sum(gradient*(g@gradient))))))
    _finite(logits,prob,loss,penalty,gradient_norm)
    return logits,loss,penalty,gradient_norm


def _ce_fit(g,labels,c):
    started=time.perf_counter();n=len(labels)
    lipschitz=1.+.5*max(0.,float(np.trace(g)))
    step=1./lipschitz
    momentum=float((np.sqrt(lipschitz)-1.)/(np.sqrt(lipschitz)+1.))
    tolerance=float(_CONFIG['ce_gradient_rtol']*(1.+np.sqrt(n)))
    alpha=np.zeros((n,c));look=alpha.copy();look_logits=np.zeros_like(alpha)
    trace=[]
    def checked_stats(value,attempted_iteration):
        try:
            return _ce_stats(value,g,labels)
        except (FloatingPointError,OverflowError) as exc:
            last=trace[-1] if trace else {}
            audit=dict(status='TECHNICAL_FAILURE',converged=False,steps=trace,
                optimizer_steps=attempted_iteration,iterations=attempted_iteration,
                attempted_iteration=attempted_iteration,failure_reason=str(exc),
                loss_data=last.get('loss_data'),loss_ridge=last.get('loss_ridge'),
                loss_total=last.get('loss_total'),gradient_norm=last.get('gradient_norm'),
                gradient_coordinate='true_RKHS_parameters',stationarity_residual=None,
                gradient_tolerance=tolerance,learning_rate=step,lipschitz_bound=lipschitz,
                momentum=momentum,solve_seconds=time.perf_counter()-started,
                factorization_calls=0,factorization_dim=0,solver=_CONFIG['ce_solver'],epoch=None)
            raise OrbitCEConvergenceError('Nonfinite CE optimization state; no fallback',audit) from exc
    logits,data,penalty,gradient=checked_stats(alpha,0)
    trace=[dict(iteration=0,loss_data=data,loss_ridge=penalty,loss_total=data+penalty,
                gradient_norm=gradient,learning_rate=step,lipschitz_bound=lipschitz,momentum=momentum,
                elapsed_seconds=time.perf_counter()-started)]
    count=0
    while gradient>tolerance and count<_CONFIG['ce_max_iterations']:
        old_alpha,old_logits=alpha,logits
        e=np.exp(look_logits-look_logits.max(axis=1,keepdims=True));residual=e/e.sum(axis=1,keepdims=True)
        residual[np.arange(n),labels]-=1.
        alpha=(1.-step)*look-step*residual
        logits,data,penalty,gradient=checked_stats(alpha,count+1)
        count+=1
        trace.append(dict(iteration=count,loss_data=data,loss_ridge=penalty,loss_total=data+penalty,
            gradient_norm=gradient,learning_rate=step,lipschitz_bound=lipschitz,momentum=momentum,
            elapsed_seconds=time.perf_counter()-started))
        look=alpha+momentum*(alpha-old_alpha)
        look_logits=logits+momentum*(logits-old_logits)
    audit=dict(status='CONVERGED' if gradient<=tolerance else 'TECHNICAL_FAILURE',converged=bool(gradient<=tolerance),
        optimizer_steps=count,iterations=count,steps=trace,loss_data=data,loss_ridge=penalty,loss_total=data+penalty,
        gradient_norm=gradient,gradient_coordinate='true_RKHS_parameters',stationarity_residual=gradient,
        gradient_tolerance=tolerance,learning_rate=step,lipschitz_bound=lipschitz,momentum=momentum,
        solve_seconds=time.perf_counter()-started,factorization_calls=0,factorization_dim=0,
        solver=_CONFIG['ce_solver'],normal_equation_residual=None,normal_equation_reason='Nonlinear CE first-order stationarity',epoch=None)
    if not audit['converged']:
        raise OrbitCEConvergenceError('CE did not reach train-gradient tolerance within fixed iteration cap',audit)
    return alpha,audit


def _fit(b,a,labels,ids,c,arm):
    if arm not in _ARMS: raise ValueError('Unknown fixed orbit CE arm')
    started=time.perf_counter();orbit=arm.startswith('orbit');n=len(labels)
    if arm=='single_ridge':
        parts,audit=base._fit(b[:,0],a[:,0],labels,ids,c,'interaction')
        scale=np.ones(n)
        audit.update(arm=arm,converged=True,iterations=0,steps=[],stationarity_residual=audit['normal_equation_residual'])
    else:
        scale=_scale(b,a)[0] if orbit else np.ones(n)
        kernel=_kernel(b,a,b,a,scale,scale,orbit)
        g,reference,reference_self,mean,grand=base._center_kernel(kernel)
        constant=bool(np.all(b==b[0]) and np.all(a==a[0]))
        if constant:g=np.zeros_like(g)
        gram_seconds=time.perf_counter()-started
        if arm.endswith('_ce'):
            try:
                alpha,audit=_ce_fit(g,labels,c)
            except OrbitCEConvergenceError as exc:
                exc.audit.update(arm=arm,training_physical_ids=list(ids),train_k=n//c,
                    train_physical_count=n,physical_loss_mass=float(n),physical_sample_weight=1.,
                    ridge_coefficient=1.,view_count=4 if orbit else 1,input_view_count=4,
                    output_dim=c,classification_output_dim=c,gram_seconds=gram_seconds,
                    fit_seconds=time.perf_counter()-started)
                raise
        else:
            begin=time.perf_counter();target=np.eye(c)[labels]-1./c
            chol=np.linalg.cholesky(g+np.eye(n))
            alpha=np.linalg.solve(chol.T,np.linalg.solve(chol,target))
            residual=g@alpha-target;normal=(g+np.eye(n))@alpha-target
            data=float(.5*np.sum(residual*residual));penalty=float(.5*np.sum(alpha*(g@alpha)))
            audit=dict(status='CLOSED_FORM_SOLVED',converged=True,optimizer_steps=0,iterations=0,steps=[],
                loss_data=data,loss_ridge=penalty,loss_total=data+penalty,
                gradient_norm=float(np.linalg.norm(g@normal)),gradient_coordinate='dual_coefficients',
                normal_equation_residual=float(np.linalg.norm(normal)),stationarity_residual=float(np.linalg.norm(normal)),
                factorization_calls=1,factorization_dim=n,solve_seconds=time.perf_counter()-begin,
                solver='float64_centered_kernel_Cholesky',learning_rate=None,epoch=None)
        audit.update(arm=arm,gram_seconds=gram_seconds,gram_bytes=int(g.nbytes),
            condition_bound=float(1+np.trace(g)),degenerate_constant_features=constant,
            kernel_diagonal_mean=float(np.diag(kernel).mean()))
        parts=(alpha,reference,reference_self,mean,grand,np.zeros(c))
    audit.update(train_k=n//c,train_physical_count=n,training_physical_ids=list(ids),
        physical_loss_mass=float(n),physical_sample_weight=1.,ridge_coefficient=1.,
        view_count=4 if orbit else 1,input_view_count=4,output_dim=c,classification_output_dim=c,
        source_validation=None,source_validation_reason='No source samples or source feature banks',
        all_states_estimated_from_trainfold_only=True,coefficient_bytes=int(parts[0].nbytes),
        training_feature_bytes=int((b.nbytes+a.nbytes) if orbit else (b[:,0].nbytes+a[:,0].nbytes)),
        fit_seconds=time.perf_counter()-started)
    return parts,scale,audit


def _scores(b,a,tb,ta,parts,tscale,arm):
    alpha,reference,reference_self,mean,grand,bias=parts
    if arm.startswith('single'):
        return base._score_rows(b[:,0],a[:,0],tb[:,0],ta[:,0],*parts,'interaction')
    scores=np.empty((len(b),alpha.shape[1]),dtype=np.float64)
    if np.all(tb==tb[0]) and np.all(ta==ta[0]):
        scores[:]=bias;return scores
    train_parts=_orbit_parts(tb,ta)
    for i in range(len(b)):
        scale=_scale(b[i:i+1],a[i:i+1])[0]
        raw=(_parts_kernel(_orbit_parts(b[i:i+1],a[i:i+1]),train_parts)*scale[:,None]*tscale[None,:])[0]
        diff=(raw-raw[0])-reference+reference_self
        centered=diff-float(diff.mean())-mean+grand
        scores[i]=centered@alpha+bias
    _finite(scores)
    return scores


@dataclass(frozen=True)
class BranchOrbitCEState:
    support_background:np.ndarray
    support_auxiliary:np.ndarray
    alpha:np.ndarray
    reference_kernel:np.ndarray
    reference_self:float
    center_mean:np.ndarray
    center_grand:float
    target_mean:np.ndarray
    support_scale:np.ndarray
    classes:tuple
    arm:str
    audit:Mapping

    def __post_init__(self):
        classes=_strings(self.classes,'classes');b=np.asarray(self.support_background)
        if b.ndim!=3 or not len(b) or self.arm not in _ARMS:raise ValueError('Invalid orbit CE state')
        n,c=len(b),len(classes);views=4 if self.arm.startswith('orbit') else 1
        shapes=dict(support_background=(n,views,256),support_auxiliary=(n,views,480),alpha=(n,c),
            reference_kernel=(n,),center_mean=(n,),target_mean=(c,),support_scale=(n,))
        for key,shape in shapes.items():
            value=np.asarray(getattr(self,key))
            if value.shape!=shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():raise ValueError('Invalid state '+key)
            object.__setattr__(self,key,_readonly(value))
        _finite(self.reference_self,self.center_grand)
        object.__setattr__(self,'classes',classes);object.__setattr__(self,'audit',_freeze(self.audit))

    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        b,a=_blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        parts=(self.alpha,self.reference_kernel,self.reference_self,self.center_mean,self.center_grand,self.target_mean)
        return _scores(b,a,self.support_background,self.support_auxiliary,parts,self.support_scale,self.arm)

    def predict(self,**features):
        scores=self.score(**features);order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(scores[:,order],axis=1)]]

    def audit_dict(self):return _plain(self.audit)


def _numerical(b,a):
    scale,raw=_scale(b,a);normalized=raw*scale*scale
    stats=lambda x:dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()))
    return dict(orbit_pooled_diagonal=stats(raw),orbit_normalized_diagonal=stats(normalized),
        orbit_norm_below_floor_count=int(np.sum(raw<_CONFIG['norm_floor']**2)),
        interpretation='Same-physical deterministic orbit variation; not independent class covariance')


def fit_branch_orbit_ce(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes=(),arm='orbit_ce'):
    started=time.perf_counter()
    b,a,labels,ids,canonical,requested,old,k=_prepare(z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes)
    try:
        parts,scale,audit=_fit(b,a,labels,ids,len(canonical),arm)
    except OrbitCEConvergenceError as exc:
        exc.audit.update(scope='final',parent_k=k,fold=None,trial=None,
            classes=list(requested),completed_stages=[])
        raise
    alpha,reference,reference_self,mean,grand,bias=parts
    order=[canonical.index(cls) for cls in requested];alpha,bias=alpha[:,order],bias[order]
    sb,sa=(b,a) if arm.startswith('orbit') else (b[:,:1],a[:,:1])
    size=int(sum(x.nbytes for x in (sb,sa,alpha,reference,mean,bias,scale))+16)
    audit.update(scope='final',fold=None,trial=None,all_states_estimated_from_current_support_only=True)
    top=dict(config=deepcopy(_CONFIG),classes=list(requested),old_classes=sorted(old),k=k,support_count=len(ids),
        selected=arm,selection='fixed_no_selection',numerical=_numerical(b,a),final_fit=audit,
        optimizer_steps=audit['optimizer_steps'],factorization_count=audit['factorization_calls'],
        fold_count=0,folds=[],oof=None,oneshot_proxy=None,head_bytes=size,persistent_state_bytes=size,
        current_support_feature_bytes=int(sb.nbytes+sa.nbytes),state_byte_scope='seven_float64_arrays_and_two_float64_scalars_excludes_registry_audit_Python_overhead',
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k==1 else 'NO_CV_FIXED_CONFIG',
        query_rows_used=0,source_rows_used=0,new_source_payload_bytes=0,new_ground_statistics_bytes=0,
        phase1_frozen=True,source_validation=None,source_validation_reason='No source samples or source feature banks',
        fit_seconds=time.perf_counter()-started)
    return BranchOrbitCEState(sb,sa,alpha,reference,reference_self,mean,grand,bias,scale,requested,arm,top)


def _paired(oof,compact=False):
    result={}
    for left,right in _PAIRS:
        rows=[];counts=dict(both_correct=0,left_only_correct=0,right_only_correct=0,both_wrong=0)
        for l,r in zip(oof[left]['rows'],oof[right]['rows']):
            rows.append(dict(physical_id=l['physical_id'],correct_delta=int(l['correct'])-int(r['correct'])))
            key=('both_correct' if r['correct'] else 'left_only_correct') if l['correct'] else ('right_only_correct' if r['correct'] else 'both_wrong')
            counts[key]+=1
        entry=dict(mean_correct_delta=float(np.mean([r['correct_delta'] for r in rows])))
        entry.update(dict(counts=counts,record_count=len(rows)) if compact else dict(rows=rows))
        result[left+'_minus_'+right]=entry
    return result


def _compact(oof,classes):
    result={};lookup={cls:i for i,cls in enumerate(classes)}
    for arm,value in oof.items():
        matrix=np.zeros((len(classes),len(classes)),dtype=np.int64);nll=np.zeros(len(classes));counts=np.zeros(len(classes),dtype=np.int64)
        for row in value['rows']:
            cls=lookup[row['class_id']];matrix[cls,lookup[row['predicted_class']]]+=1;nll[cls]+=row['nll'];counts[cls]+=1
        result[arm]=dict(metrics=value['metrics'],confusion=matrix.tolist(),class_order=list(classes),
            classwise_nll_sum=nll.tolist(),classwise_count=counts.tolist(),record_count=len(value['rows']))
    return result


def _fold(b,a,labels,ids,c,keep,parent_k,scope,fold=None,trial=None):
    train_ids=tuple(pid for i,pid in enumerate(ids) if keep[i]);held_ids=[pid for i,pid in enumerate(ids) if not keep[i]]
    entry=dict(training_ids=list(train_ids),held_ids=held_ids,train_k=int(keep.sum())//c,held_k=int((~keep).sum())//c,parent_k=parent_k,stages=[])
    scores={}
    for arm in _ARMS:
        try:
            parts,scale,audit=_fit(b[keep],a[keep],labels[keep],train_ids,c,arm)
        except OrbitCEConvergenceError as exc:
            exc.audit.update(scope=scope,parent_k=parent_k,fold=fold,trial=trial,
                held_ids=held_ids,completed_stages=entry['stages'])
            raise
        begin=time.perf_counter();scores[arm]=_scores(b[~keep],a[~keep],b[keep],a[keep],parts,scale,arm)
        audit.update(scope=scope,parent_k=parent_k,fold=fold,trial=trial,score_seconds=time.perf_counter()-begin)
        entry['stages'].append(audit)
    return entry,scores


def probe_branch_orbit_ce(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes=()):
    started=time.perf_counter()
    b,a,labels,ids,canonical,_,old,k=_prepare(z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,classes,old_classes)
    n,c=len(ids),len(canonical);folds=0 if k==1 else min(k,3)
    result=dict(config=deepcopy(_CONFIG),classes=list(canonical),old_classes=sorted(old),k=k,parent_k=k,support_count=n,
        numerical=_numerical(b,a),fold_count=folds,folds=[],oof=None,paired=None,oneshot_proxy=None,physical_fold_assignment=[],
        factorization_count=0,standard_factorization_count=0,optimizer_steps=0,persistent_state_bytes=0,
        claim_scope='SUPPORT_OOF_AND_SUPPORT_ONESHOT_PROXY_NOT_QUERY_EVALUATION',
        heldout_unavailable_reason='K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k==1 else None)
    if k==1:result['fit_seconds']=time.perf_counter()-started;return result
    positions=[np.flatnonzero(labels==cls) for cls in range(c)];assignments=np.full(n,-1,dtype=int)
    for indices in positions:assignments[indices]=np.arange(k)%folds
    result['physical_fold_assignment']=[dict(physical_id=pid,class_id=canonical[int(labels[i])],fold=int(assignments[i])) for i,pid in enumerate(ids)]
    scores={arm:np.empty((n,c)) for arm in _ARMS}
    completed_stages=[]
    def episode_fold(keep,scope,fold=None,trial=None):
        try:
            entry,partial=_fold(b,a,labels,ids,c,keep,k,scope,fold=fold,trial=trial)
        except OrbitCEConvergenceError as exc:
            exc.audit.update(classes=list(canonical),old_classes=sorted(old),
                completed_stages=completed_stages+exc.audit.get('completed_stages',[]))
            raise
        completed_stages.extend(entry['stages'])
        return entry,partial
    for fold in range(folds):
        keep=assignments!=fold;entry,partial=episode_fold(keep,'support_oof',fold=fold);entry['fold']=fold
        for arm in _ARMS:scores[arm][~keep]=partial[arm]
        result['folds'].append(entry)
    result['oof']={arm:_classification(value,labels,canonical,old,ids,assignments) for arm,value in scores.items()}
    result['paired']=_paired(result['oof'])
    result['standard_factorization_count']=sum(s['factorization_calls'] for f in result['folds'] for s in f['stages'])
    proxy=dict(parent_k=k,proxy_train_k=1,trial_count=k,trials=[],factorization_count=0,
        scope='SUPPORT_ONESHOT_PROXY_FROM_PARENT_SUPPORT_NOT_FORMAL_K1',
        coverage=dict(parent_physical_count=n,training_occurrences=n,held_occurrences=n*(k-1),unique_held_physical_count=n))
    for trial in range(k):
        keep=np.zeros(n,dtype=bool)
        for indices in positions:keep[indices[trial]]=True
        entry,partial=episode_fold(keep,'support_oneshot_proxy',trial=trial);entry.update(trial=trial,proxy_train_k=1)
        held_ids=tuple(entry['held_ids'])
        oof={arm:_classification(value,labels[~keep],canonical,old,held_ids,np.full(len(held_ids),trial)) for arm,value in partial.items()}
        entry['oof']=_compact(oof,canonical);entry['paired']=_paired(oof,True)
        proxy['factorization_count']+=sum(s['factorization_calls'] for s in entry['stages']);proxy['trials'].append(entry)
    result['oneshot_proxy']=proxy
    stages=[s for entry in result['folds']+proxy['trials'] for s in entry['stages']]
    result['factorization_count']=result['standard_factorization_count']+proxy['factorization_count']
    result['optimizer_steps']=sum(s['optimizer_steps'] for s in stages)
    result['fit_seconds']=time.perf_counter()-started
    return result
