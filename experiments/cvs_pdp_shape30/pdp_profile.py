"""Change only three-ray delay shape;preserve original ensemble RMS and powers."""
import numpy as np

def delays(latent,shape):
 base=np.asarray(shape,dtype=float)
 if base.shape!=(3,) or base.tolist() not in ([0.,1.,3.],[0.,1.,2.]):raise ValueError('Only original or equispaced fixed shape')
 cfg=latent.cfg
 if np.asarray(latent.pdp).shape!=(3,) or np.asarray(latent.diffuse).shape!=(3,) or np.asarray(latent.mean_los_power).shape!=(3,) or not np.isclose(np.sum(latent.pdp),1.):raise ValueError('Original three diffuse rays required')
 result=[];spreads=[]
 for s in range(3):
  w=latent.diffuse[s]*latent.pdp.copy();w[0]+=latent.mean_los_power[s];w/=w.sum()
  spread=np.sqrt(np.sum(w*(base-w@base)**2))
  if not np.isfinite(spread) or spread<=0:raise ValueError('Nondegenerate positive ensemble PDP required')
  ds=base*cfg.rms_delay_ns[s]*1e-9*cfg.fs_hz/spread
  if not np.isfinite(ds).all() or np.any(ds<0) or np.any(np.diff(ds)<=0):raise ValueError('Finite ordered three-ray delays required')
  actual=np.sqrt(np.sum(w*(ds-w@ds)**2))/cfg.fs_hz*1e9
  if not np.isclose(actual,cfg.rms_delay_ns[s],rtol=1e-12,atol=1e-10):raise ValueError('Total ensemble RMS changed')
  result.append(ds);spreads.append(float(actual))
 return np.asarray(result),spreads

def apply(latent,shape):
 expected,spread=delays(latent,shape)
 original=np.asarray(latent.delays).copy()
 # The original branch retains its exact arithmetic and byte representation.
 if list(shape)==[0.,1.,3.]:
  if not np.allclose(original,expected,rtol=1e-12,atol=1e-12):raise ValueError('Original nominal delay construction differs')
 else:
  latent.delays=expected
  latent.history_size=int(np.floor(expected.max()))+2*latent.cfg.fractional_delay_half_length
 return dict(shape=list(shape),ensemble_RMS_ns=spread,delays_samples=np.asarray(latent.delays).tolist(),max_delay_samples=float(np.max(latent.delays)),original_max_delay_samples=float(np.max(original)),amplitude_renormalization=False)
