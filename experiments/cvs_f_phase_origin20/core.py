"""Exact source counterfactuals; not a practical or test-channel replacement."""
from dataclasses import replace
import json
import math
import numpy as np
from experiments.cvs_urban_sigtime.channel import imports

def require_same_practical(current,persisted):
    # asdict retains tuple fields; JSON artifacts necessarily contain lists.
    if json.loads(json.dumps(current,allow_nan=False))!=persisted:
        raise ValueError('Original practical settings changed')


def phase_from_kernel(cfg,seed,receiver):
    ch,_=imports()
    from leo_practical.equalization import propagation_kernel
    latent=ch.ChannelStream(replace(cfg,processing_route='full'),seed,receiver);state=latent.state
    coefficients=latent.ray_coef.sum(axis=1)*np.sqrt(latent.diffuse[state]*latent.pdp)
    los=0. if state==2 else 10**((cfg.los_mean_db[state]+cfg.los_std_db[state]*latent.z)/20)
    coefficients[0]+=los;coefficients/=math.sqrt(latent.reference_power)
    gain=ch.slant_range(cfg.reference_altitude_m,cfg.reference_elevation_deg)/latent.geometry.distance_m*10**(-cfg.atmosphere_loss_db/20)
    h=propagation_kernel(latent,state,coefficients,gain);peak=int(np.argmax(abs(h)));phase=float(np.angle(h[peak]))
    return phase,dict(state=ch.STATES[state],kernel_length=len(h),peak_index=peak,peak_phase_radians=phase,
        numerical_delay=cfg.fractional_delay_half_length,phase_scope='Propagation H peak only; not all oscillator/image phase')


def factor_views(x,phase,delay=24,boundary='reflect'):
    from leo_practical.equalization import linear_filter
    x=np.asarray(x,dtype=np.complex128)
    if x.shape!=(256,) or not np.isfinite(x).all() or not math.isfinite(phase) or delay not in (0,24) or boundary!='reflect':
        raise ValueError('Finite source IQ256 with fixed0/24 reflect factor only')
    h=np.zeros(delay+1,dtype=np.complex128);h[delay]=1.
    delayed,_=linear_filter(x,h,boundary);rotation=np.exp(1j*phase)
    return dict(phase=(x*rotation).astype(np.complex64),delay=delayed.astype(np.complex64),
        phase_delay=(delayed*rotation).astype(np.complex64))
