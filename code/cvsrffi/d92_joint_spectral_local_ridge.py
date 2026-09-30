"""Supervised two-parameter spectral adapter with a closed-form LocalRidge head.

All representation statistics belong to physical inner-training folds. Held
support labels supervise the shared two parameters; there is no query fitting.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import solve_triangular
from . import d92_branch_interaction as interaction
from . import d92_branch_local_ridge as local
from . import d92_within_class_metric as within
from .d92_branch_ridge import _freeze, _plain, _readonly

FROZEN_CONFIG=dict(
    schema='d92_joint_spectral_local_ridge_v1',method='D92-JointSpectralLocalRidge-v1',
    base_algorithm=deepcopy(local.FROZEN_CONFIG),metric_component=deepcopy(within.FROZEN_CONFIG),
    channel=deepcopy(within.FROZEN_CONFIG['channel']),allowed_scenarios=list(within.FROZEN_CONFIG['allowed_scenarios']),
    arms=['local_ridge','fixed_metric','joint_seq','joint_reset'],
    spectral_gain='lambda/(1+lambda)*(theta0+theta1*lambda)',
    feasible_set='theta>=0;sum(theta)<=1',trainable_parameter_count=2,
    initialization_B=[0.,0.],initialization_C_seq='inherit_B_theta',initialization_C_reset=[0.,0.],
    C_geometry='inherit_full_B_old_geometry;reestimate_each_inner_fold_geometry_from_inner_old_train',
    objective='(sum_physical_inner_held_squared_error+theta_anchor_squared_distance)/(2*N)',
    optimizer='full_batch_projected_gradient',optimizer_steps=8,learning_rate=.1,
    proximal_physical_sum_coefficient=1.,final_objective_evaluation=True,
    inner_folds='per_class_physical_id_sort_position_mod_min_K_3',
    bandwidth='current_theta_median_nearest_other_class_including_zero',
    tie_gradient='average_all_exact_minimizers_then_average_tied_median_groups',
    trace_target='original_untransformed_interaction_centered_trace',
    derivative='analytic_distance_kernel_centering_trace_scale_and_closed_form_solve',
    theta_zero='exact_original_forward_with_nonzero_spectral_derivative',
    no_information='skip_updates_preserve_anchor_if_all_inner_geometries_identity',
    n0='reuse_B',probe_k1='numerical_only_no_fit_no_holdout',
    query_decision_policy='per_sample_all_registered_classes',tie_break='physical_class_id_lexicographic',
    source_inputs=False,query_fit=False,phase1_frozen=True,encoder_backward=False,
    early_stopping=False,parameter_search=False,dtype='float64')
_CONFIG=deepcopy(FROZEN_CONFIG)
_NAMES=('z_id','fft','t_emb','f_emb','pa_local')
NumericalFailure=within.NumericalFailure


def _finite(*values):
    for value in values:
        if not np.isfinite(value).all(): raise FloatingPointError('NONFINITE_JOINT_ARITHMETIC')


def project_theta(value):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(2,): raise ValueError('theta must have shape (2,)')
    _finite(value)
    nonnegative=np.maximum(value,0.)
    if float(nonnegative.sum())<=1.: return nonnegative
    ordered=np.sort(value)[::-1]
    threshold=(float(value.sum())-1.)/2
    if ordered[1]<=threshold: threshold=ordered[0]-1.
    return np.maximum(value-threshold,0.)


def _theta(value):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(2,) or not np.isfinite(value).all() or np.any(value<0) or value.sum()>1+8*np.finfo(float).eps:
        raise ValueError('theta outside the frozen simplex')
    return value


@dataclass(frozen=True)
class SpectralGeometry:
    metric: within.WithinClassMetricState
    eigenvalues: np.ndarray
    audit: Mapping
    def __post_init__(self):
        object.__setattr__(self,'eigenvalues',_readonly(self.eigenvalues))
        object.__setattr__(self,'audit',_freeze(self.audit))
    @property
    def identity(self): return self.metric.identity
    def audit_dict(self): return _plain(self.audit)


def _prepare_geometry(raw,labels,ids,classes,context):
    started=time.perf_counter()
    metric=within.fit_within_class_metric(**raw,support_labels=labels,support_ids=ids,classes=classes,context=context)
    ma=metric.audit_dict();eigen=np.empty(0)
    if not metric.identity:
        b,a=metric.support_background,metric.support_auxiliary
        y=np.asarray(metric.support_labels,dtype=int);qb,qa=metric.Qb,metric.Qa
        coordinates=within._delta_psi(b,a,b[:1],a[:1],qb,qa)
        means=np.stack([coordinates[y==j].mean(axis=0) for j in range(len(metric.classes))])
        means-=means.mean(axis=0)
        e=np.empty_like(coordinates)
        for j in range(len(metric.classes)):
            ix=np.flatnonzero(y==j)
            delta=within._delta_psi(b[ix],a[ix],b[ix[:1]],a[ix[:1]],qb,qa)
            e[ix]=delta-delta.mean(axis=0)
        _,s,v=np.linalg.svd(means,full_matrices=False)
        basis=v[s>ma['protected_rank_threshold']].T
        r=e-(e@basis)@basis.T
        scale=np.max(np.abs(r));scaled=r/scale
        f=scaled/np.sqrt(np.sum(scaled*scaled))
        _,singular,right=np.linalg.svd(f,full_matrices=False)
        eigen=singular*singular
        spectral=np.sqrt(eigen/(1+eigen))[:,None]*right
        _finite(eigen,spectral)
        if abs(float(eigen.sum())-1)>ma['numerical_tolerance'] or eigen.max()>1+ma['numerical_tolerance']:
            raise FloatingPointError('SPECTRAL_NORMALIZATION_FAILED')
        # Same fixed-W operator in a diagonal spectral coordinate system;
        # release the old Cholesky-coordinate T rather than keep two factors.
        metric=within.WithinClassMetricState(False,metric.classes,metric.support_ids,metric.support_labels,
            b,a,qb,qa,spectral,ma)
    audit=dict(ma,geometry_fit_count=1,geometry_factorization_count=ma['metric_factorization_count'],
        spectral_svd_count=int(not metric.identity),geometry_seconds=time.perf_counter()-started,
        spectral_eigenvalues=eigen.tolist(),persistent_state_bytes=ma['persistent_state_bytes']+eigen.nbytes)
    return SpectralGeometry(metric,eigen,audit)


def _distance_parts(geometry,b,a,tb=None,ta=None,train_projection=None):
    symmetric=tb is None
    if symmetric: tb,ta=b,a
    d0=local._distances(b,a,None if symmetric else tb,None if symmetric else ta)
    q=np.zeros((2,)+d0.shape)
    counts=dict(cached_difference_pair_count=0,direct_difference_pair_count=0)
    if geometry.identity:return d0,q,counts
    metric=geometry.metric
    u,un=within._metric_projection(metric,b,a)
    v,vn=(u,un) if symmetric else (train_projection if train_projection is not None else within._metric_projection(metric,tb,ta))
    dim=max(480,metric.T.shape[1]);eps=np.finfo(float).eps;gamma=dim*eps/(1-dim*eps)
    tol=metric.audit['numerical_tolerance']
    for i in range(len(b)):
        for j in range(i+1 if symmetric else 0,len(tb)):
            if d0[i,j]==0:continue
            du=u[i]-v[j];h=within._norm(du);err=8*gamma*(un[i]+vn[j])
            if 2*h*err+err*err<=tol*d0[i,j]:counts['cached_difference_pair_count']+=1
            else:
                delta=within._delta_psi(b[i:i+1],a[i:i+1],tb[j:j+1],ta[j:j+1],metric.Qb,metric.Qa)[0]
                du=metric.T@delta;counts['direct_difference_pair_count']+=1
            square=du*du;q[:,i,j]=[float(square.sum()),float(square@geometry.eigenvalues)]
            if q[0,i,j]>(.5+tol)*d0[i,j] or q[1,i,j]>(1+tol)*q[0,i,j]:
                raise FloatingPointError('SPECTRAL_DISTANCE_BOUND_FAILED')
            if symmetric:q[:,j,i]=q[:,i,j]
    _finite(d0,q)
    return d0,q,counts


def _bandwidth(distance,derivative,labels):
    """Symmetric branch-averaged generalized derivative, not all directions."""
    n=len(labels)
    if len(set(labels.tolist()))==1:return None,np.zeros(2)
    nearest=np.empty(n);grad=np.empty((n,2))
    for i in range(n):
        allowed=labels!=labels[i]
        nearest[i]=np.min(distance[i,allowed])
        tie=allowed&(distance[i]==nearest[i])
        grad[i]=derivative[:,i,:][:,tie].mean(axis=1)
    tau=float(np.median(nearest));order=np.sort(nearest)
    middle=[order[n//2]] if n%2 else [order[n//2-1],order[n//2]]
    dt=np.mean([grad[nearest==value].mean(axis=0) for value in middle],axis=0)
    return tau,dt


def _radial_derivative(distance,derivative,tau,dt):
    raw=local._radial_minus_one(distance,tau)
    gradient=np.zeros((2,)+distance.shape)
    if tau==0:return raw,gradient
    radial=local._radial(distance,tau);active=radial>0
    ratio=distance[active]/tau
    for p in range(2):
        gradient[p][active]=radial[active]*(-derivative[p][active]/tau+ratio*(dt[p]/tau))
    _finite(raw,gradient)
    return raw,gradient


def _cross_center(raw,reference,reference_self,mean,grand):
    out=np.empty_like(raw)
    for i in range(len(raw)):
        diff=(raw[i]-raw[i,0])-reference+reference_self
        out[i]=diff-float(diff.mean())-mean+grand
    return out


@dataclass(frozen=True)
class KernelProblem:
    d0: np.ndarray
    q: np.ndarray
    cross_d0: np.ndarray
    cross_q: np.ndarray
    train_labels: np.ndarray
    held_labels: np.ndarray
    classes: tuple
    geometry: SpectralGeometry
    audit: Mapping
    def __post_init__(self):
        for key in ('d0','q','cross_d0','cross_q','train_labels','held_labels'):
            object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'audit',_freeze(self.audit))


def _evaluate_kernel(problem,theta,gradient=True,progress=None,return_scores=False):
    theta=_theta(theta);n=len(problem.train_labels);c=len(problem.classes)
    audit={} if progress is None else progress
    audit.update(factorization_count=0,head_fit_count=1,derivative_triangular_solve_count=0)
    y=np.asarray(problem.train_labels,dtype=int);hy=np.asarray(problem.held_labels,dtype=int)
    distance=problem.d0-np.einsum('p,pij->ij',theta,problem.q)
    cross=problem.cross_d0-np.einsum('p,pij->ij',theta,problem.cross_q)
    tau,dt=_bandwidth(distance,-problem.q,y)
    s0=float(problem.d0[np.triu_indices(n,1)].sum()/n)
    target=np.eye(c)[y]-1/c
    score=np.zeros((len(hy),c));ds=np.zeros((2,len(hy),c))
    alpha=np.zeros((n,c));reference=mean=np.zeros(n);refself=grand=0.;gamma=None
    audit.update(bandwidth_tau=tau,bandwidth_gradient=dt.tolist(),
        interaction_centered_trace=s0,numerical_tolerance=128*np.finfo(float).eps*max(n,c),
        normal_equation_residual=0.,trace_relative_error=0.,radial_centered_trace=None,trace_scale=None,
        derivative_triangular_solve_count=0)
    if c>1 and s0>0:
        raw,dr=_radial_derivative(distance,-problem.q,tau,dt)
        radial_trace=float(-2*raw[np.triu_indices(n,1)].sum()/n)
        if radial_trace<=0:raise FloatingPointError('NONPOSITIVE_RADIAL_TRACE')
        gamma=s0/radial_trace
        centered,reference,refself,mean,grand=interaction._center_kernel(raw)
        kernel=gamma*centered
        matrix=kernel+np.eye(n)
        audit['factorization_count']=1;chol=np.linalg.cholesky(matrix)
        alpha=solve_triangular(chol.T,solve_triangular(chol,target,lower=True),lower=False)
        rawcross,drcross=_radial_derivative(cross,-problem.cross_q,tau,dt)
        crosscentered=_cross_center(rawcross,reference,refself,mean,grand)
        crosskernel=gamma*crosscentered
        # Match the original _score_local row-wise forward arithmetic exactly,
        # including at theta=0, while retaining the active analytic derivative.
        # BranchLocalRidgeState._readonly also fixes C-contiguous coefficient
        # storage; a raw SciPy Fortran-layout result changes BLAS reduction bits.
        forward_alpha=np.ascontiguousarray(alpha)
        score=np.stack([(gamma*crosscentered[i])@forward_alpha for i in range(len(hy))]) if len(hy) else np.zeros((0,c))
        if gradient:
            for p in range(2):
                dtrace=float(-2*dr[p][np.triu_indices(n,1)].sum()/n)
                dgamma=-gamma*dtrace/radial_trace
                dk,rr,rs,mm,gg=interaction._center_kernel(dr[p])
                dk=dgamma*centered+gamma*dk
                dcross=dgamma*crosscentered+gamma*_cross_center(drcross[p],rr,rs,mm,gg)
                audit['derivative_triangular_solve_count']+=1
                lower=solve_triangular(chol,dk@alpha,lower=True)
                audit['derivative_triangular_solve_count']+=1
                da=-solve_triangular(chol.T,lower,lower=False)
                ds[p]=dcross@alpha+crosskernel@da
        normal=within._norm(matrix@alpha-target)/((1+s0)*within._norm(alpha)+within._norm(target))
        trace_error=abs(float(np.trace(kernel))-s0)/s0
        if normal>audit['numerical_tolerance'] or trace_error>audit['numerical_tolerance']:
            raise FloatingPointError('JOINT_HEAD_RESIDUAL_EXCEEDED')
        audit.update(normal_equation_residual=normal,trace_relative_error=trace_error,
            radial_centered_trace=radial_trace,trace_scale=gamma)
    _finite(score,ds,alpha)
    heldtarget=np.eye(c)[hy]-1/c
    error=score-heldtarget
    loss=float(.5*np.sum(error*error))
    grad=np.array([np.sum(error*ds[p]) for p in range(2)]) if gradient else np.zeros(2)
    audit.update(held_physical_count=len(hy),held_training_correct_count=int(np.sum(score.argmax(axis=1)==hy)) if len(hy) else 0,
        loss_data_sum=loss)
    result=(loss,grad,audit,(alpha,reference,refself,mean,grand,tau,gamma))
    return result+(score,) if return_scores else result


@dataclass(frozen=True)
class JointTraining:
    raw: Mapping
    labels: np.ndarray
    ids: tuple
    classes: tuple
    old_classes: tuple
    background: np.ndarray
    auxiliary: np.ndarray
    geometry: SpectralGeometry
    problems: tuple
    inherited: object
    audit: Mapping
    def __post_init__(self):
        object.__setattr__(self,'raw',_freeze({k:_readonly(v) for k,v in self.raw.items()}))
        for key in ('labels','background','auxiliary'):object.__setattr__(self,key,_readonly(getattr(self,key)))
        object.__setattr__(self,'audit',_freeze(self.audit))
    def audit_dict(self):return _plain(self.audit)


def prepare_joint_training(*,z_id,fft,t_emb,f_emb,pa_local,support_labels,support_ids,
        classes,old_classes,inherited=None,context=None,log_callback=None):
    started=time.perf_counter();audit=dict(context or {},joint_preparation_count=1,geometry_fit_count=0,
        geometry_factorization_count=0,geometry_seconds=0.,inner_folds=[],completed_stages=[])
    b,a,y,ids,canonical,requested,old,k=interaction._prepare(z_id,fft,t_emb,f_emb,pa_local,
        support_labels,support_ids,classes,old_classes)
    if not old:raise ValueError('Joint adapter requires identified old support classes')
    order=np.asarray(sorted(range(len(support_ids)),key=lambda i:support_ids[i]))
    raw={key:np.asarray(value)[order] for key,value in dict(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local).items()}
    old=tuple(sorted(old));oldmask=np.array([canonical[v] in old for v in y])
    oldlabels=np.array([old.index(canonical[v]) for v in y[oldmask]])
    audit.update(training_physical_ids=list(ids),classes=list(canonical),old_classes=list(old),train_k=k,
        train_physical_count=len(ids),inherited_geometry=inherited is not None)
    try:
        if inherited is None:
            if set(canonical)!=set(old):raise ValueError('C preparation requires inherited B state')
            geometry=_prepare_geometry({key:value[oldmask] for key,value in raw.items()},oldlabels,
                [pid for i,pid in enumerate(ids) if oldmask[i]],old,dict(context or {},stage='full_old_geometry'))
            ga=geometry.audit_dict();audit['geometry_fit_count']+=1
            audit['geometry_factorization_count']+=ga['geometry_factorization_count'];audit['geometry_seconds']+=ga['geometry_seconds']
        else:
            geometry=inherited.geometry
            if set(inherited.classes)!=set(old):raise ValueError('Inherited state must be the old-only B model')
            if inherited.audit['mode']!='B':raise ValueError('C lineage must inherit the supervised B state')
            previous=inherited.audit['preparation']
            for key in ('row_id','split_id'):
                if key in (context or {}) and key in previous and context[key]!=previous[key]:
                    raise ValueError('C lineage crosses '+key)
            within._check_metric_binding(geometry.metric,b,a,y,ids,canonical)
        problems=[]
        if k>1:
            folds=min(k,3);assignment=np.empty(len(ids),dtype=int)
            for cls in range(len(canonical)):
                ix=np.flatnonzero(y==cls);assignment[ix]=np.arange(len(ix))%folds
            for f in range(folds):
                keep=assignment!=f;oldkeep=keep&oldmask
                iy=np.array([old.index(canonical[v]) for v in y[oldkeep]])
                ctx=dict(context or {},stage='inner_geometry',inner_fold=f)
                geom=_prepare_geometry({key:value[oldkeep] for key,value in raw.items()},iy,
                    [pid for i,pid in enumerate(ids) if oldkeep[i]],old,ctx)
                ga=geom.audit_dict();audit['geometry_fit_count']+=1
                audit['geometry_factorization_count']+=ga['geometry_factorization_count'];audit['geometry_seconds']+=ga['geometry_seconds']
                d,q,counts=_distance_parts(geom,b[keep],a[keep])
                cross,cq,crosscounts=_distance_parts(geom,b[~keep],a[~keep],b[keep],a[keep])
                fa=dict(inner_fold=f,training_physical_ids=[pid for i,pid in enumerate(ids) if keep[i]],
                    held_physical_ids=[pid for i,pid in enumerate(ids) if not keep[i]],
                    geometry_training_physical_ids=list(geom.metric.support_ids),geometry_identity=geom.identity,
                    geometry_audit=ga,train_physical_count=int(keep.sum()),held_physical_count=int((~keep).sum()),
                    direct_difference_pair_count=counts['direct_difference_pair_count']+crosscounts['direct_difference_pair_count'])
                problems.append(KernelProblem(d,q,cross,cq,y[keep],y[~keep],canonical,geom,fa))
                audit['inner_folds'].append(fa)
                within._emit(log_callback,'JOINT_INNER_PREPARED',dict(context or {},**fa))
        audit.update(no_information=not problems or all(p.geometry.identity for p in problems),
            preparation_seconds=time.perf_counter()-started,
            transient_distance_bytes=sum(p.d0.nbytes+p.q.nbytes+p.cross_d0.nbytes+p.cross_q.nbytes for p in problems),
            geometry_state_bytes=geometry.audit['persistent_state_bytes'],
            inner_geometry_state_bytes=sum(p.geometry.audit['persistent_state_bytes'] for p in problems))
        audit['prepare_seconds']=audit['preparation_seconds']
        return JointTraining(raw,y,ids,canonical,old,b,a,geometry,tuple(problems),inherited,audit)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):audit['failed_geometry']=exc.audit_dict()
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),preparation_seconds=time.perf_counter()-started)
        raise NumericalFailure(str(exc),audit) from exc


def evaluate_joint_objective(prepared,theta,anchor,*,gradient=True):
    theta=_theta(theta);anchor=_theta(anchor);n=len(prepared.ids)
    total=0.;grad=np.zeros(2);folds=[]
    for problem in prepared.problems:
        partial={}
        try:
            loss,g,audit,_=_evaluate_kernel(problem,theta,gradient,progress=partial)
        except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
            raise NumericalFailure(str(exc),dict(status='TECHNICAL_FAILURE',
                failure_reason=str(exc),current_inner_fold=_plain(problem.audit),
                current_kernel_progress=partial,completed_inner_folds=folds,
                inner_objective_evaluation_count=1,
                inner_head_fit_count=len(folds)+partial.get('head_fit_count',0),
                inner_factorization_count=sum(f['factorization_count'] for f in folds)+partial.get('factorization_count',0),
                derivative_triangular_solve_count=sum(f['derivative_triangular_solve_count'] for f in folds)+partial.get('derivative_triangular_solve_count',0))) from exc
        total+=loss;grad+=g;folds.append(audit)
    penalty=float(.5*np.sum((theta-anchor)**2)/n)
    grad=grad/n+(theta-anchor)/n if gradient else np.zeros(2)
    audit=dict(loss_data=total/n,loss_proximal=penalty,loss_total=total/n+penalty,
        gradient=grad.tolist() if gradient else None,inner_folds=folds,
        inner_head_fit_count=len(folds),inner_factorization_count=sum(f['factorization_count'] for f in folds),
        derivative_triangular_solve_count=sum(f['derivative_triangular_solve_count'] for f in folds))
    _finite(audit['loss_total'],grad)
    return audit['loss_total'],grad,audit


@dataclass(frozen=True)
class JointSpectralState:
    theta: np.ndarray
    geometry: SpectralGeometry
    base_state: local.BranchLocalRidgeState
    audit: Mapping
    support_projection: np.ndarray | None=None
    support_reference_norm: np.ndarray | None=None
    def __post_init__(self):
        object.__setattr__(self,'theta',_readonly(self.theta));object.__setattr__(self,'audit',_freeze(self.audit))
        for key in ('support_projection','support_reference_norm'):
            if getattr(self,key) is not None:object.__setattr__(self,key,_readonly(getattr(self,key)))
    @property
    def classes(self):return self.base_state.classes
    def audit_dict(self):return _plain(self.audit)
    def score(self,*,z_id,fft,t_emb,f_emb,pa_local):
        if self.geometry.identity or np.all(self.theta==0):
            return self.base_state.score(z_id=z_id,fft=fft,t_emb=t_emb,f_emb=f_emb,pa_local=pa_local)
        b,a=interaction._blocks(z_id,fft,t_emb,f_emb,pa_local,allow_empty=True)
        state=self.base_state;scores=np.zeros((len(b),len(self.classes)))
        if state.trace_scale is None:return scores
        for i in range(len(b)):
            d,q,_=_distance_parts(self.geometry,b[i:i+1],a[i:i+1],state.support_background,state.support_auxiliary,
                train_projection=(self.support_projection,self.support_reference_norm))
            distance=d-np.einsum('p,pij->ij',self.theta,q)
            raw=local._radial_minus_one(distance,state.bandwidth_tau)
            centered=_cross_center(raw,state.reference_kernel,state.reference_self,state.center_mean,state.center_grand)
            scores[i]=(state.trace_scale*centered[0])@state.alpha
        _finite(scores)
        return scores
    def predict(self,**features):
        order=np.asarray(sorted(range(len(self.classes)),key=lambda i:self.classes[i]))
        return np.asarray(self.classes)[order[np.argmax(self.score(**features)[:,order],axis=1)]]


def fit_joint_spectral_local_ridge(prepared,*,mode='B',baseline_state=None,log_callback=None):
    if mode not in ('B','C_seq','C_reset','fixed'):raise ValueError('Unknown joint stage mode')
    if mode=='B' and prepared.inherited is not None:raise ValueError('B cannot inherit a target state')
    if mode in ('C_seq','C_reset') and prepared.inherited is None:raise ValueError('C requires B lineage')
    if prepared.inherited is not None and set(prepared.classes)==set(prepared.old_classes):return prepared.inherited
    started=time.perf_counter()
    anchor=prepared.inherited.theta.copy() if mode=='C_seq' else np.zeros(2)
    theta=np.array([1.,0.]) if mode=='fixed' else anchor.copy()
    noinfo=prepared.audit['no_information']
    audit=dict(mode=mode,classes=list(prepared.classes),old_classes=list(prepared.old_classes),
        training_physical_ids=list(prepared.ids),train_k=prepared.audit['train_k'],
        optimizer_steps=0,inner_objective_evaluation_count=0,inner_head_fit_count=0,
        inner_factorization_count=0,derivative_triangular_solve_count=0,
        final_head_fit_count=0,final_factorization_count=0,steps=[],completed_stages=[],
        theta_anchor=anchor.tolist(),anchor=anchor.tolist(),no_information=bool(noinfo),
        inherited_geometry=prepared.inherited is not None,preparation=prepared.audit_dict(),
        trainable_parameter_count=0 if mode=='fixed' else 2,nonzero_projected_update_count=0,
        source_validation=None,source_validation_reason='SOURCE_ACCESS_FORBIDDEN')
    finalprogress={}
    try:
        if mode!='fixed' and not noinfo:
            for step in range(1,_CONFIG['optimizer_steps']+1):
                begin=time.perf_counter();before=theta.copy()
                _,gradient,objective=evaluate_joint_objective(prepared,theta,anchor)
                proposal=theta-_CONFIG['learning_rate']*gradient;theta=project_theta(proposal)
                record=dict(step=step,theta_pre=before.tolist(),theta_post=theta.tolist(),
                    theta_anchor=anchor.tolist(),learning_rate=_CONFIG['learning_rate'],
                    gradient=gradient.tolist(),gradient_norm=within._norm(gradient),
                    unprojected_update_norm=within._norm(proposal-before),update_norm=within._norm(theta-before),
                    active_nonnegative_constraints=(theta==0).tolist(),active_sum_constraint=bool(theta.sum()==1),
                    step_seconds=time.perf_counter()-begin,**{k:v for k,v in objective.items() if k!='gradient'})
                audit['steps'].append(record);audit['optimizer_steps']+=1
                audit['nonzero_projected_update_count']+=int(np.any(theta!=before))
                audit['inner_objective_evaluation_count']+=1
                for key in ('inner_head_fit_count','inner_factorization_count','derivative_triangular_solve_count'):
                    audit[key]+=objective[key]
                within._emit(log_callback,'JOINT_SPECTRAL_STEP',dict(mode=mode,**record))
            _,_,final=evaluate_joint_objective(prepared,theta,anchor,gradient=False)
            audit['final_objective']=final;audit['inner_objective_evaluation_count']+=1
            for key in ('inner_head_fit_count','inner_factorization_count'):audit[key]+=final[key]
        else:
            audit['final_objective']=None
            audit['no_update_reason']='FIXED_METRIC_ABLATION' if mode=='fixed' else 'NO_IDENTIFIABLE_INNER_GEOMETRY'
        identity=prepared.geometry.identity or bool(np.all(theta==0))
        n,c=len(prepared.ids),len(prepared.classes)
        b,a=prepared.background,prepared.auxiliary
        if identity:
            if baseline_state is None:
                baseline_state=local.fit_branch_local_ridge(**{key:np.asarray(prepared.raw[key]) for key in _NAMES},
                    support_labels=np.asarray(prepared.labels,dtype=int),support_ids=prepared.ids,
                    classes=prepared.classes,old_classes=prepared.old_classes)
                audit['final_head_fit_count']=1
                audit['final_factorization_count']=baseline_state.audit_dict()['factorization_count']
            ba=baseline_state.audit_dict()
            if (set(baseline_state.classes)!=set(prepared.classes) or ba['final_fit']['training_physical_ids']!=list(prepared.ids)
                or not np.array_equal(baseline_state.support_background,b) or not np.array_equal(baseline_state.support_auxiliary,a)):
                raise ValueError('Joint baseline identity binding mismatch')
            if tuple(baseline_state.classes)!=prepared.classes:
                baseline_state=local.BranchLocalRidgeState(b,a,baseline_state.alpha[:,[baseline_state.classes.index(cls) for cls in prepared.classes]],
                    baseline_state.reference_kernel,baseline_state.reference_self,baseline_state.center_mean,
                    baseline_state.center_grand,baseline_state.bandwidth_tau,baseline_state.trace_scale,prepared.classes,ba)
            base=baseline_state;headbytes=ba['persistent_state_bytes'];projection=refnorm=None
            finalfit=ba['final_fit']
        else:
            d,q,_=_distance_parts(prepared.geometry,b,a)
            problem=KernelProblem(d,q,np.empty((0,n)),np.empty((2,0,n)),prepared.labels,np.empty(0),
                prepared.classes,prepared.geometry,{})
            _,_,finalfit,parts=_evaluate_kernel(problem,theta,gradient=False,progress=finalprogress)
            audit['final_head_fit_count']=1;audit['final_factorization_count']=finalfit['factorization_count']
            alpha,reference,refself,mean,grand,tau,gamma=parts
            projection,refnorm=within._metric_projection(prepared.geometry.metric,b,a)
            headbytes=int(b.nbytes+a.nbytes+alpha.nbytes+reference.nbytes+mean.nbytes+projection.nbytes+refnorm.nbytes+16+8*(tau is not None)+8*(gamma is not None))
            base=local.BranchLocalRidgeState(b,a,alpha,reference,refself,mean,grand,tau,gamma,prepared.classes,finalfit)
        audit.update(theta=theta.tolist(),theta_update_norm=within._norm(theta-anchor),identity_forward=identity,
            theta_changed_from_anchor=bool(np.any(theta!=anchor)),
            final_fit=finalfit,shared_geometry_state_bytes=prepared.geometry.audit['persistent_state_bytes'],
            head_state_bytes=headbytes,theta_state_bytes=16,
            persistent_state_bytes=headbytes+prepared.geometry.audit['persistent_state_bytes']+16,
            factorization_count=audit['inner_factorization_count']+audit['final_factorization_count'],
            status='JOINT_STAGE_COMPLETE',fit_seconds=time.perf_counter()-started,config=deepcopy(_CONFIG))
        within._emit(log_callback,'JOINT_SPECTRAL_FIT',audit)
        return JointSpectralState(theta,prepared.geometry,base,audit,projection,refnorm)
    except (FloatingPointError,np.linalg.LinAlgError,OverflowError) as exc:
        if isinstance(exc,NumericalFailure):
            failed=exc.audit_dict();audit['failed_stage']=failed
            for key in ('inner_objective_evaluation_count','inner_head_fit_count','inner_factorization_count','derivative_triangular_solve_count'):
                audit[key]+=failed.get(key,0)
        if finalprogress and not audit['final_head_fit_count']:
            audit['failed_final_kernel_progress']=finalprogress
            audit['final_head_fit_count']=finalprogress.get('head_fit_count',0)
            audit['final_factorization_count']=finalprogress.get('factorization_count',0)
        audit.update(status='TECHNICAL_FAILURE',failure_reason=str(exc),theta=theta.tolist(),fit_seconds=time.perf_counter()-started)
        raise NumericalFailure(str(exc),audit) from exc
