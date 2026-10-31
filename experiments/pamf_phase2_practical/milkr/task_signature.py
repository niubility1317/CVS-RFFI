"""Support-frozen coordinate normalization of the three signature views."""
from dataclasses import dataclass
import numpy as np
from . import signature_kernel as sig,support_calibrated as base

FRACTIONS={'std0.01':.01,'std0.1':.1,'std1':1.,'std10':10.}
MODES=('none','center',*FRACTIONS)


@dataclass(frozen=True)
class Normalizer:
    mode:str
    location:np.ndarray
    scale:np.ndarray

    def __call__(self,features):
        result=sig.normalize(features)
        blocks=result['sig_views'].reshape(-1,3,112)
        shifted=(blocks-self.location)/self.scale
        norm=np.linalg.norm(shifted,axis=2,keepdims=True)
        if not np.isfinite(norm).all() or (norm<=1e-12).any():
            raise FloatingPointError('Zero or nonfinite task-normalized signature view')
        value=(shifted/norm/np.sqrt(3)).reshape(-1,336)
        value.flags.writeable=False;result['sig_views']=value
        return result


def make_normalizer(features,mode):
    if mode not in MODES or mode=='none':raise ValueError('A centering or standardization mode is required')
    x=sig.normalize(features)['sig_views'].reshape(-1,3,112)
    location=x.mean(0,keepdims=True);variance=np.mean((x-location)**2,axis=0,keepdims=True)
    if mode=='center':scale=np.ones_like(location)
    else:
        average=variance.mean(2,keepdims=True)
        if not np.isfinite(average).all() or (average<=0).any():
            raise FloatingPointError('No positive support signature variance')
        scale=np.sqrt(variance+FRACTIONS[mode]*average)
    location.flags.writeable=False;scale.flags.writeable=False
    return Normalizer(mode,location,scale)


def fit(features,labels,ids,*,old_count=6,fixed=None,conditional_loo=True):
    if fixed is None or fixed.get('task_mode') not in MODES:raise ValueError('Frozen source task mode required')
    if tuple(fixed['config']) not in sig.CONFIGS:raise ValueError('Wrong signature kernel rule')
    mode=fixed['task_mode']
    if mode=='none':return sig.fit(features,labels,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo)
    normalizer=make_normalizer(features,mode)
    state=base.fit(features,labels,ids,old_count=old_count,fixed=fixed,conditional_loo=conditional_loo,
                   normalizer=normalizer,similarity_fn=sig.similarity,configs=sig.CONFIGS)
    from dataclasses import replace
    audit=dict(mode=mode,location_scale_from='current_support_only',query_updates=0,
               extra_frozen_array_bytes=normalizer.location.nbytes+normalizer.scale.nbytes)
    return replace(state,selection=dict(state.selection,task_normalization=audit))
