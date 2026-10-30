"""Fixed known-class versus other-known/OE likelihood competition, FP64 only."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density

METHODS=('pooled_ratio','predicted_ratio','class_competition','hybrid_competition')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3


def _logmeanexp(values,axis):
    peak=values.max(axis=axis,keepdims=True)
    return np.squeeze(peak,axis=axis)+np.log(np.exp(values-peak).mean(axis=axis))


def raw_scores(state,features,closed):
    """Two new columns, routed only by the unchanged original F0 prediction.

    Each known class is its original equal-mass fifteen-mode conditional
    mixture. The hybrid background is half original OE and half the five
    other known classes. No score temperature, new fitting or identity head.
    """
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=density.SCHEMA or state['known_means'].shape!=(90,160):
        raise ValueError('Only the original six-by-fifteen D160 equal-mass source density is allowed')
    x=density.unit(features);c=np.asarray(closed);n=len(x)
    if c.shape!=(n,) or c.dtype.kind not in 'iu' or not np.isin(c,range(6)).all():
        raise ValueError('Aligned original F0 old-six predictions required; no true-label routing')
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        z=(x-state['origin'])@state['whitener'];variance=state['known_variances']
        terms=-80*np.log(2*np.pi*variance)[None]-density._distances(z,state['known_means'])/(2*variance[None])
        classes=_logmeanexp(terms.reshape(n,6,15),axis=2)+float(state['log_abs_det_whitener'])
        selected=classes[np.arange(n),c]
        others=classes[np.arange(6)[None]!=c[:,None]].reshape(n,5)
        other=_logmeanexp(others,axis=1);background=density._log_density(state,z,'background')
        result=np.column_stack((selected-other,selected-np.logaddexp(background,other)+np.log(2.)))
    if result.dtype!=np.float64 or result.shape!=(n,2) or not np.isfinite(result).all():
        raise ValueError('Two finite FP64 class-competition scores required')
    return result
