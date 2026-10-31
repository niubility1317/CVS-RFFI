"""One closed-form target moment fit over frozen source Gaussian coordinates."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density
from . import star_class_density_core as original

ARMS=('SHIFT','CORAL')
METHODS=tuple(a+'_'+m for a in ('E0','PCD',*ARMS) for m in ('pooled_ratio','predicted_ratio'))
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=7
ADAPTER_SCHEMA='pamf_phase3_pcd_moment_adapter_v1'
ADAPTER_FIELDS={'schema','kind','matrix','offset'}
DIMENSION=160


def _state(state):
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=density.SCHEMA or state['known_means'].shape!=(90,160):raise ValueError('Frozen class-major source known90/OE D160 state required')
    return state


def _features(value):
    a=np.asarray(value)
    if a.ndim!=2 or not len(a) or a.shape[1]!=160 or a.dtype!=np.float32 or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a.astype(np.float64),axis=1),1.,atol=2e-6,rtol=0):
        raise ValueError('Finite final-PCD FP32 unit160 features required')
    return a


def _closed(value,n):
    c=np.asarray(value)
    if c.shape!=(n,) or c.dtype!=np.int64 or not np.isin(c,range(6)).all():raise ValueError('Aligned original F0 six-class int64 predictions required')
    return c


def _symmetric(value):return .5*(value+value.T)


def source_moments(state):
    _state(state);mu=state['known_means'].reshape(6,15,160);means=mu.mean(1);residual=(mu-means[:,None]).reshape(90,160)
    covariance=_symmetric(state['known_variances'].mean()*np.eye(160)+residual.T@residual/90)
    eigenvalues,vectors=np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues<=0):raise ValueError('Source model-implied within covariance must be positive definite')
    return dict(class_means=means,covariance=covariance,inverse_sqrt=_symmetric((vectors/eigenvalues**.5)@vectors.T),
                sqrt=_symmetric((vectors*eigenvalues**.5)@vectors.T),eigenvalues=eigenvalues)


def validate_adapter(adapter):
    if not isinstance(adapter,dict) or set(adapter)!=ADAPTER_FIELDS:raise ValueError('Exact aggregate-only moment adapter fields required')
    for key in ('schema','kind'):
        value=adapter[key]
        if not isinstance(value,np.ndarray) or value.shape!=() or value.dtype.kind!='U':raise ValueError('Scalar Unicode moment schema/kind required')
    if adapter['schema'].item()!=ADAPTER_SCHEMA or adapter['kind'].item() not in ARMS:raise ValueError('Fixed SHIFT/CORAL adapter schema required')
    for key,shape in (('matrix',(160,160)),('offset',(6,160))):
        value=adapter[key]
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Finite FP64 moment aggregate required: '+key)
    if adapter['kind'].item()=='SHIFT' and (not np.array_equal(adapter['matrix'],np.eye(160)) or not np.array_equal(adapter['offset'],np.tile(adapter['offset'][0],(6,1)))):
        raise ValueError('SHIFT requires identity matrix and one repeated common offset')
    return adapter


def _adapter(kind,matrix,offset):
    state=dict(schema=np.asarray(ADAPTER_SCHEMA),kind=np.asarray(kind),matrix=np.asarray(matrix,np.float64).copy(),offset=np.asarray(offset,np.float64).copy())
    validate_adapter(state)
    for value in state.values():value.setflags(write=False)
    return state


def fit_adapters(state,U_features,U_logits,U_closed,U_base_scores):
    x=_features(U_features);c=_closed(U_closed,len(x));logits=np.asarray(U_logits);scores=np.asarray(U_base_scores)
    if logits.shape!=(len(x),6) or logits.dtype!=np.float32 or not np.isfinite(logits).all() or not np.array_equal(c,logits.argmax(1)):
        raise ValueError('Same-U original F0 FP32 logits and their exact predictions required')
    if scores.shape!=(len(x),2) or scores.dtype!=np.float64 or not np.isfinite(scores).all():raise ValueError('Two frozen original PCD FP64 margins required')
    moments=source_moments(state);values=logits.astype(np.float64);values-=values.max(1,keepdims=True);p=np.exp(values);p/=p.sum(1,keepdims=True)
    confident=p.max(1)>=.95;accepted=scores[:,1]>=0;mask=confident&accepted
    histogram=np.bincount(c[mask],minlength=6);present=np.flatnonzero(histogram);count=int(mask.sum())
    weights=np.zeros(count,np.float64)
    if count:
        for label in present:weights[c[mask]==label]=1./(len(present)*histogram[label])
    effective=float(1./np.square(weights).sum()) if count else 0.
    diagnostic=dict(status='FITTED' if count>=2 and effective>1 else 'NO_CORE',fit_succeeded=bool(count>=2 and effective>1),
        U_count=len(x),selected_count=count,confidence_core_count=int(confident.sum()),PCD_margin_accepted_count=int(accepted.sum()),
        predicted_class_histogram=np.bincount(c,minlength=6).tolist(),selected_class_histogram=histogram.tolist(),
        selected_class_mass=[float(weights[c[mask]==label].sum()) for label in range(6)],effective_sample_size=effective,
        selected_weight_sum=float(weights.sum()),core_confidence_threshold=.95,core_PCD_margin_threshold=0.,
        class_weighting='equal_nonempty_predicted_class_then_equal_selected_record',covariance_denominator='population_class_balanced_weight_sum_one',
        source_covariance='model_implied_within_class_mean_variance_I_plus_mean_mode_residual_outer',source_covariance_eigenvalues=moments['eigenvalues'].tolist(),
        target_covariance_data_weight=.9,target_covariance_identity_weight=.1,inverse_sqrt_response_bounds=[.5,2.],
        fit_calls=1,optimizer_updates=0,neural_updates=0,source_state_refitted=False,thresholds_refitted=False,
        U_only=True,Q_inputs_used=False,true_U_labels_used=False,true_unknown_count_used=False,
        U_bank_deployed=False,source_bank_deployed=False,identity_classification_unchanged=True,
        target_covariance_eigenvalues=None,regularized_covariance_eigenvalues=None,clipped_inverse_sqrt_responses=None,
        response_clipped_below_count=None,response_clipped_above_count=None,nu_norm=None,shift_norm=None,
        regularized_whitening_residual_fro=None,population_whitening_residual_fro=None,recolored_covariance_relative_residual=None)
    if not diagnostic['fit_succeeded']:
        diagnostic['reason']='selected_core_less_than_two' if count<2 else 'effective_sample_size_not_above_one'
        return {kind:_adapter(kind,np.eye(160),np.zeros((6,160))) for kind in ARMS},diagnostic
    white=(density.unit(x[mask])-state['origin'])@state['whitener'];means=moments['class_means'];P,R=moments['inverse_sqrt'],moments['sqrt']
    v=(white-means[c[mask]])@P;nu=weights@v;centered=v-nu
    covariance=_symmetric(centered.T@(weights[:,None]*centered));regularized=.9*covariance+.1*np.eye(160)
    eigenvalues,vectors=np.linalg.eigh(regularized)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues<=0):raise ValueError('Regularized target population covariance must be finite positive definite')
    response=1/np.sqrt(eigenvalues);clipped=np.clip(response,.5,2.);B=_symmetric((vectors*clipped)@vectors.T)
    matrix=P@B@R;delta=nu@R
    adapters=dict(SHIFT=_adapter('SHIFT',np.eye(160),np.tile(-delta,(6,1))),CORAL=_adapter('CORAL',matrix,means-(means+delta)@matrix))
    recolored=R@B@covariance@B@R
    diagnostic.update(reason='closed_form_class_balanced_selected_U_moments',
        target_covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist(),regularized_covariance_eigenvalues=eigenvalues.tolist(),
        clipped_inverse_sqrt_responses=clipped.tolist(),response_clipped_below_count=int((response<.5).sum()),response_clipped_above_count=int((response>2.).sum()),
        nu_norm=float(np.linalg.norm(nu)),shift_norm=float(np.linalg.norm(delta)),
        regularized_whitening_residual_fro=float(np.linalg.norm(B@regularized@B-np.eye(160))),
        population_whitening_residual_fro=float(np.linalg.norm(B@covariance@B-np.eye(160))),
        recolored_covariance_relative_residual=float(np.linalg.norm(recolored-moments['covariance'])/np.linalg.norm(moments['covariance'])))
    return adapters,diagnostic


def raw_scores(state,features,closed,adapter):
    _state(state);x=_features(features);c=_closed(closed,len(x));validate_adapter(adapter)
    if np.array_equal(adapter['matrix'],np.eye(160)) and np.array_equal(adapter['offset'],np.zeros((6,160))):
        return original.raw_scores(state,x,c)[:,[1,3]]
    # No post-transform normalization or query-derived statistic. The same
    # affine Jacobian cancels between the known and background log densities.
    h=((density.unit(x)-state['origin'])@state['whitener'])@adapter['matrix']+adapter['offset'][c]
    pooled=density._log_density(state,h,'known');background=density._log_density(state,h,'background');conditional=np.empty(len(x),np.float64)
    for label in range(6):
        rows=np.flatnonzero(c==label)
        if not len(rows):continue
        selection=slice(label*15,(label+1)*15);variance=state['known_variances'][selection]
        terms=-np.log(15.)-80*np.log(2*np.pi*variance)[None]-density._distances(h[rows],state['known_means'][selection])/(2*variance[None])
        peak=terms.max(1);conditional[rows]=peak+np.log(np.exp(terms-peak[:,None]).sum(1))+float(state['log_abs_det_whitener'])
    result=np.column_stack((pooled-background,conditional-background))
    if not np.isfinite(result).all():raise ValueError('Two finite transformed density ratios required')
    return result
