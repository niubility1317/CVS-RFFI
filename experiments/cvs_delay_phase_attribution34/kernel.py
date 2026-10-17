"""One-tap counterfactuals preserve white-input kernel energy, never invert IQ."""
import numpy as np
def flat_kernel(h,delay_samples,zero_phase):
 h=np.asarray(h,complex)
 if h.ndim!=1 or not len(h) or not np.isfinite(h).all() or not isinstance(delay_samples,int) or not 0<=delay_samples<len(h):raise ValueError('Finite1D kernel and supported nonnegative impulse index required')
 k=np.zeros_like(h);angle=0. if zero_phase else np.angle(h[np.argmax(abs(h))]);k[delay_samples]=np.sqrt(np.sum(abs(h)**2))*np.exp(1j*angle);return k
