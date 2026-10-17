"""Approximate affine lattice observability;never modifies IQ or fits a classifier."""
import numpy as np
MIN_REL=2.**-16;MAX_REL=.5;TOL=.02

def estimate(a):
 a=np.asarray(a,dtype=np.float64)
 if a.shape!=(2,256) or not np.isfinite(a).all():raise ValueError('One finite originalIQ256')
 v=a.ravel();rms=float(np.sqrt(np.mean(v*v)))
 if rms<=0:raise ValueError('Nonzero waveform required')
 values=np.unique(v);scale=max(MIN_REL*rms,64*np.finfo(np.float32).eps*float(np.max(np.abs(v))))
 if len(values)<32:return dict(observable=False,reason='insufficient_distinct_values',unique_values=len(values),q=None,q_over_component_RMS=None,fit_p99=None)
 delta=np.diff(values);delta=np.sort(delta[(delta>=scale)&(delta<=MAX_REL*rms*32)])
 if not len(delta):return dict(observable=False,reason='no_resolvable_spacing',unique_values=len(values),q=None,q_over_component_RMS=None,fit_p99=None)
 chosen=np.unique(np.r_[delta[:16],np.quantile(delta,np.linspace(0,1,17))]);q=(chosen[:,None]/np.arange(1,33)[None,:]).ravel();q=np.unique(q[(q>=scale)&(q<=MAX_REL*rms)])
 best=None
 for qq in q:
  codes=np.rint((v-v[0])/qq);cc=codes-codes.mean();den=float(cc@cc)
  if den<=0:continue
  refined=float(cc@(v-v.mean())/den);offset=float(v.mean()-refined*codes.mean())
  if refined<scale or refined>MAX_REL*rms:continue
  residual=np.abs((v-offset)/refined-codes);p99=float(np.quantile(residual,.99))
  if p99<=TOL and (best is None or refined>best[0]):best=(refined,p99,float(np.max(residual)))
 if best is None:return dict(observable=False,reason='no_resolvable_affine_grid',unique_values=len(values),q=None,q_over_component_RMS=None,fit_p99=None)
 qq,p99,maxr=best;relative=qq/rms
 return dict(observable=True,reason='resolved_affine_grid_not_ADC_causality',unique_values=len(values),q=qq,q_over_component_RMS=relative,fit_p99=p99,fit_max=maxr,equivalent_independent_uniform_dither_SNR_db=float(10*np.log10(12/relative**2)),limits='An affine grid can come from storage/preprocessing;not evidence of ADC bitdepth or source-independent dequantization error')
