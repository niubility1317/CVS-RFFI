"""Frozen multimodal likelihood ratios with one shared source radial variance."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density

SCHEMA=density.SCHEMA
METHODS=('pooled_ratio','predicted_ratio','tied_pooled_ratio','tied_predicted_ratio')
PRIMARY_METHOD=METHODS[-1]
PRIMARY_INDEX=3


def _logsumexp(terms):
    peak=terms.max(axis=1)
    return peak+np.log(np.exp(terms-peak[:,None]).sum(axis=1))


def raw_scores(state,features,closed):
    """Two tied ratios; source means, weights, whitening and F0 identity stay fixed.

    The common query quadratic, Gaussian volume and whitening Jacobian cancel.
    No component-specific variance, query statistic, fitting or new state is used.
    """
    density.validate_state(state,'density_ratio')
    if state['schema'].item()!=SCHEMA or state['known_means'].shape!=(90,160):
        raise ValueError('Original six-by-fifteen D160 source density required')
    s=float(state['pooled_L_variance'])
    if not np.isfinite(s) or s<=0:raise ValueError('The fixed pooled source variance must be finite and strictly positive')
    x=density.unit(features);c=np.asarray(closed);n=len(x)
    if c.shape!=(n,) or c.dtype.kind not in 'iu' or not np.isin(c,range(6)).all():
        raise ValueError('Aligned original F0 old-six predictions required; no true-label routing')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        z=(x-state['origin'])@state['whitener']
        def linear(means):return (z@means.T-.5*np.einsum('kd,kd->k',means,means)[None])/s
        known=linear(state['known_means']);background=linear(state['background_means'])+np.log(state['background_weights'])[None]
    if not np.isfinite(known).all() or not np.isfinite(background).all():raise ValueError('Finite cancelled Gaussian log terms required')
    bg=_logsumexp(background);pooled=_logsumexp(known+np.log(state['known_weights'])[None])
    conditional=_logsumexp(known.reshape(n,6,15)[np.arange(n),c]-np.log(15.))
    result=np.column_stack((pooled-bg,conditional-bg))
    if not np.isfinite(result).all():raise ValueError('Two finite FP64 tied likelihood ratios required')
    return result
