"""Source-only low-dimensional ViM in exact native classifier coordinates."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F

SCHEMA='pamf_phase3_vim_readout_v1'
METHODS=('residual_only','vim')
DIMENSION=160
PRINCIPAL_DIM=80
RESIDUAL_DIM=80
STATE_FIELDS={'schema','origin','residual_basis','alpha'}


def classifier_coordinates(reader,raw):
    """Return the actual FP32 native cosine-head input, without another forward.

    Classification remains 30*F.linear(x,normalize(W)); scale is deliberately
    not premultiplied into W because that changes FP32 operation ordering.
    Autograd and inference mode are controlled by the caller.
    """
    try:head=reader.core.id_backbone.cls_head;weight=head.weight;scale=head.scale
    except AttributeError as exc:raise ValueError('Complete original native default head required') from exc
    if (not callable(getattr(reader,'classify_features',None)) or not isinstance(weight,torch.nn.Parameter)
            or weight.shape!=(6,160) or weight.dtype!=torch.float32 or not bool(torch.isfinite(weight).all())
            or type(scale) is not float or scale!=30. or getattr(head,'bias',None) is not None):
        raise ValueError('Native FP32 head W[6,160], fixed Python scale30 and no bias required')
    if (not torch.is_tensor(raw) or raw.dtype!=torch.float32 or raw.ndim!=2 or not len(raw) or raw.shape[1]!=160
            or raw.device!=weight.device or not bool(torch.isfinite(raw).all())):
        raise ValueError('Aligned finite raw native FP32[N,160] required')
    with torch.autocast(device_type=raw.device.type,enabled=False):
        return F.normalize(raw,dim=1,eps=1e-4)


def _inputs(x,logits):
    a=np.asarray(x);l=np.asarray(logits)
    if (a.ndim!=2 or not len(a) or a.shape[1]!=160 or a.dtype!=np.float32 or not np.isfinite(a).all()
            or np.any(np.linalg.norm(a.astype(np.float64),axis=1)>1.+2e-6)):
        raise ValueError('Finite native FP32 classifier coordinates with norm at most one required')
    if l.shape!=(len(a),6) or l.dtype!=np.float32 or not np.isfinite(l).all():raise ValueError('Aligned finite native FP32 six logits required')
    return a.astype(np.float64),l.astype(np.float64)


def validate_state(state):
    if not isinstance(state,dict) or set(state)!=STATE_FIELDS:raise ValueError('Exact four-array source ViM aggregate state required')
    schema=state['schema']
    if not isinstance(schema,np.ndarray) or schema.shape!=() or schema.dtype.kind!='U' or schema.item()!=SCHEMA:raise ValueError('Matching scalar Unicode ViM schema required')
    for name,shape in (('origin',(160,)),('residual_basis',(160,80)),('alpha',())):
        value=state[name]
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('Finite FP64 ViM aggregate required: '+name)
    if not np.array_equal(state['origin'],np.zeros(160)):raise ValueError('The bias-free native cosine head has exactly zero ViM origin')
    if float(state['alpha'])<=0:raise ValueError('ViM alpha must be strictly positive')
    basis=state['residual_basis']
    if not np.allclose(basis.T@basis,np.eye(80),rtol=0,atol=1e-10):raise ValueError('Orthonormal 80-dimensional residual basis required')
    return state


def fit(x,logits):
    a,l=_inputs(x,logits)
    if len(a)<=80:raise ValueError('Source-L row count must exceed the fixed principal dimension80')
    # EmpiricalCovariance(assume_centered=True): there is no empirical-mean
    # subtraction, covariance shrinkage, class fit or source normalization fit.
    second=a.T@a/len(a);eigenvalues,vectors=np.linalg.eigh(second)
    if not np.isfinite(eigenvalues).all() or not np.isfinite(vectors).all():raise ValueError('Finite source uncentered eigendecomposition required')
    basis=vectors[:,:80].copy();origin=np.zeros(160,np.float64)
    residual=np.linalg.norm((a-origin)@basis,axis=1);denominator=float(residual.mean());numerator=float(l.max(axis=1).mean())
    # A constant/rank<=80 bank has zero mathematical residual. Distinguish its
    # eigensolver roundoff from a real residual at the input's own scale, while
    # allowing legitimately tiny classifier coordinates below the head epsilon.
    roundoff=64*np.finfo(np.float64).eps*float(np.linalg.norm(a,axis=1).mean())
    if not np.isfinite(denominator) or denominator<=roundoff:raise ValueError('Degenerate source residual; finite nonzero ViM calibration required')
    if not np.isfinite(numerator) or numerator<=0:raise ValueError('Mean source maximum logit must produce a positive ViM alpha')
    alpha=numerator/denominator
    if not np.isfinite(alpha) or alpha<=0:raise ValueError('Finite positive source ViM alpha required')
    state=dict(schema=np.asarray(SCHEMA),origin=origin,residual_basis=basis,alpha=np.asarray(alpha,np.float64));validate_state(state)
    for value in state.values():value.setflags(write=False)
    diagnostic=dict(schema=SCHEMA,source_count=len(a),feature_dimension=160,principal_dimension=80,residual_dimension=80,
        assume_centered=True,empirical_mean_subtracted=False,origin='zero_for_bias_free_native_cosine_head',
        second_moment='X.T@X/N in FP64',eigenvalues_ascending=eigenvalues.tolist(),
        alpha=float(alpha),alpha_numerator_mean_max_logit=numerator,alpha_denominator_mean_residual=denominator,
        residual_minimum=float(residual.min()),residual_maximum=float(residual.max()),residual_roundoff_bound=roundoff,
        coordinate_norm_minimum=float(np.linalg.norm(a,axis=1).min()),coordinate_norm_maximum=float(np.linalg.norm(a,axis=1).max()),
        source_only=True,source_labels_used=False,OE_inputs_used=False,V_inputs_used=False,U_inputs_used=False,Q_inputs_used=False,
        neural_updates=0,source_rows_deployed=False,deployed_numeric_values=12961,deployed_numeric_bytes=103688)
    return state,diagnostic


def raw_scores(state,x,logits):
    validate_state(state);a,l=_inputs(x,logits)
    residual=np.linalg.norm((a-state['origin'])@state['residual_basis'],axis=1)
    peak=l.max(axis=1);energy=peak+np.log(np.exp(l-peak[:,None]).sum(axis=1))
    result=np.column_stack((-residual,energy-float(state['alpha'])*residual))
    if result.dtype!=np.float64 or result.shape!=(len(a),2) or not np.isfinite(result).all():raise ValueError('Two finite row-local ViM scores required')
    return result
