"""Frozen predicted-class likelihood; no fitting or query statistics."""
from __future__ import annotations
import numpy as np
from . import star_density_core as density

METHODS=('pooled_known','pooled_ratio','predicted_known','predicted_ratio')
PRIMARY_METHOD=METHODS[3]
MODE_ORDER='class-major, RX-major, day-major; six classes by five RX by three days'
MODES_PER_CLASS=15


def raw_scores(state,features,closed):
    """Return four FP64 scores using only the original F0 predicted identity.

    A conditional likelihood has fifteen equal-mass modes within that predicted
    class. Background likelihood is the same frozen all-OE mixture in both
    ratio policies. No true query label or public query RX/day is accepted.
    """
    density.validate_state(state,'density_ratio')
    if state['known_means'].shape!=(90,160):raise ValueError('Original D160 class-major ninety-mode state required')
    x=density.unit(features);c=np.asarray(closed)
    if c.shape!=(len(x),) or c.dtype.kind not in 'iu' or not np.isin(c,density.CLASSES).all():
        raise ValueError('Aligned original F0 predicted old-six identities required')
    z=(x-state['origin'])@state['whitener']
    pooled=density._log_density(state,z,'known')
    background=density._log_density(state,z,'background')
    conditional=np.empty(len(x),np.float64)
    jacobian=float(state['log_abs_det_whitener'])
    for label in density.CLASSES:
        rows=np.flatnonzero(c==label)
        if not len(rows):continue
        selection=slice(label*MODES_PER_CLASS,(label+1)*MODES_PER_CLASS)
        means=state['known_means'][selection];variance=state['known_variances'][selection]
        # Conditional weights are 1/15, not the original joint class mass 1/90.
        components=(-np.log(float(MODES_PER_CLASS))-80.*np.log(2*np.pi*variance)[None]
                    -density._distances(z[rows],means)/(2*variance[None]))
        maximum=components.max(axis=1)
        conditional[rows]=maximum+np.log(np.exp(components-maximum[:,None]).sum(axis=1))+jacobian
    result=np.column_stack((pooled,pooled-background,conditional,conditional-background))
    if result.dtype!=np.float64 or result.shape!=(len(x),4) or not np.isfinite(result).all():
        raise ValueError('Four complete finite FP64 density scores required')
    return result
