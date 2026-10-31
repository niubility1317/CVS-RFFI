"""Preserve the three actual 112-d signature views before the frozen projection."""
import numpy as np
from . import support_calibrated as base
from .semantic_residual import unit_features

CONFIGS=tuple((geometry,band,mix,scales) for geometry in ('signature','signature_identity','signature_historical')
              for band in ('nearest','all') for mix in (0.,.5) for scales in ('single','multi'))


def normalize(features):
    if set(features)!=set(base.NAMES)|{'sig_views'}:raise ValueError('Exact signature feature schema required')
    result=unit_features({k:features[k] for k in base.NAMES})
    a=np.array(features['sig_views'],dtype=np.float64,copy=True)
    if a.shape!=(len(result['z_id']),336) or not np.isfinite(a).all():raise ValueError('Expected three actual112 signature views')
    blocks=a.reshape(len(a),3,112);norms=np.linalg.norm(blocks,axis=2,keepdims=True)
    if (norms<=1e-12).any():raise ValueError('Zero signature view')
    a=(blocks/norms/np.sqrt(3)).reshape(len(a),336);a.flags.writeable=False;result['sig_views']=a
    return result


def similarity(a,b,geometry):
    sig=a['sig_views']@b['sig_views'].T
    if geometry=='signature':return sig
    if geometry=='signature_identity':return .5*sig+.5*(a['z_id']@b['z_id'].T)
    if geometry=='signature_historical':return .5*sig+.5*base.similarity(a,b,'historical')
    raise ValueError('Unknown signature geometry')


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is not None and tuple(fixed['config']) not in CONFIGS:raise ValueError('Wrong frozen signature rule')
    return base.fit(features,labels,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo,
                    normalizer=normalize,similarity_fn=similarity,configs=CONFIGS)
