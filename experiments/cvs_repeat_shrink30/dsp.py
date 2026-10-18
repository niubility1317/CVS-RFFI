"""Contract only the within-packet nonperiodic component;do not call it pure noise.

P is the orthogonal four-cycle mean projector. D has the same real phase
weights in each cycle, so DP=PD. I-lambda*D*(I-P) preserves P x and has
spectrum in [1-lambda,1]; it is complex linear and never increases energy.
This algebra does not establish TX label preservation or receiver invariance.
"""
import numpy as np
LENGTH=256;START=80;STOP=160;PERIOD=20;CYCLES=4;BOUND=.1

def taper():
 w=np.sin(np.pi*np.arange(PERIOD,dtype=np.float64)/(PERIOD-1))**2
 w[0]=w[-1]=0.
 return w

def apply(x,strength):
 if not isinstance(x,np.ndarray) or x.ndim!=3 or x.shape[1:]!=(2,LENGTH) or x.dtype not in (np.dtype('float32'),np.dtype('float64')) or not np.isfinite(x).all():raise ValueError('Finite N,2,256 float IQ only')
 a=np.asarray(strength,dtype=np.float64)
 if a.ndim==0:a=np.full((len(x),),a.item(),dtype=np.float64)
 if a.shape!=(len(x),) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>BOUND):raise ValueError('Declared scalar/perpacket contraction in[0,.1] only')
 out=x.copy()
 if not np.any(a):return out
 cycles=x[:,:,START:STOP].astype(np.float64).reshape(len(x),2,CYCLES,PERIOD)
 mean=cycles.mean(axis=2,keepdims=True)
 out[:,:,START:STOP]=(cycles-a[:,None,None,None]*taper()[None,None,None,:]*(cycles-mean)).reshape(len(x),2,STOP-START)
 return out

def numerics(x,z,strength):
 original=x.astype(np.float64);changed=z.astype(np.float64);power=np.square(original).sum(axis=(1,2));den=np.maximum(power,1e-24)
 b=original[:,:,START:STOP].reshape(len(x),2,CYCLES,PERIOD);v=changed[:,:,START:STOP].reshape(len(x),2,CYCLES,PERIOD)
 drift=np.sqrt(np.square(changed-original).sum(axis=(1,2))/den)
 mean_drift=np.sqrt(CYCLES*np.square(v.mean(axis=2)-b.mean(axis=2)).sum(axis=(1,2))/den)
 boundary=np.array([START+PERIOD*k+t for k in range(CYCLES) for t in (0,PERIOD-1)])
 return dict(strength=float(strength),max_energy_increase_relative=float(np.max((np.square(changed).sum(axis=(1,2))-power)/den)),max_cycle_mean_change_relative=float(mean_drift.max()),max_relative_change=float(drift.max()),mean_relative_change=float(drift.mean()),outside_unchanged=bool(np.array_equal(z[:,:,:START],x[:,:,:START]) and np.array_equal(z[:,:,STOP:],x[:,:,STOP:])),cycle_boundaries_unchanged=bool(np.array_equal(z[:,:,boundary],x[:,:,boundary])))
