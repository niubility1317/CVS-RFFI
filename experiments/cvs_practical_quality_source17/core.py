"""Compare two receiver-quality calculations on an identical physical channel."""
import math
from collections import Counter
import numpy as np
from . import design as d

def propagated_quality(x,h,noise_variance,boundary):
 from leo_practical.equalization import linear_filter
 y,_=linear_filter(x,h,boundary)
 return float(10*np.log10(max(float(np.mean(abs(y)**2)),1e-300)/noise_variance))
def compare(x,cfg,seed,receiver):
 from leo_practical.channel import normalize_input
 from leo_practical.residual import ResidualChannel
 x=np.asarray(x,dtype=np.complex128)
 if x.shape!=(256,) or not np.isfinite(x).all():raise ValueError('Finite original IQ256 required')
 x,_=normalize_input(x);saved=x.copy()
 y0,m0,t0=ResidualChannel(cfg,seed,receiver).process(x,return_traces=True)
 q=propagated_quality(x,t0['propagation_kernel'],10**(-cfg.reference_snr_db/10),cfg.boundary)
 y1,m1,t1=ResidualChannel(cfg,seed,receiver).process(x,quality_snr_db=q,return_traces=True)
 if not np.array_equal(x,saved) or not np.array_equal(t0['propagation_kernel'],t1['propagation_kernel']):raise ValueError('Raw IQ/latent propagation must be identical')
 if m0['state_end']!=m1['state_end'] or m0['geometry']!=m1['geometry'] or m0['receiver']!=m1['receiver'] or m0['added_noise_variance_reference']!=m1['added_noise_variance_reference'] or m0['channel_equalization_applied'] or m1['channel_equalization_applied']:raise ValueError('Paired physical quantities changed')
 if not np.isfinite(y0).all() or not np.isfinite(y1).all() or m1['quality_snr_db']!=q:raise ValueError('Nonfinite or incorrect supplied quality')
 a,b=float(m0['residual_scale']),float(m1['residual_scale']);lock=m0['receiver_lock_status']!=m1['receiver_lock_status'];ratio=max(a/b,b/a)
 return dict(white_quality_db=m0['quality_snr_db'],waveform_quality_db=q,delta_quality_db=q-m0['quality_snr_db'],white_lock=m0['receiver_lock_status'],waveform_lock=m1['receiver_lock_status'],white_scale=a,waveform_scale=b,scale_ratio=ratio,lock_changed=lock,effective_change=bool(lock or ratio>=1.10),relative_output_difference=float(np.mean(abs(y1-y0)**2)/np.mean(abs(y0)**2)),state=m0['state_end'],elevation_deg=m0['geometry']['elevation_deg'],noise_variance=m0['added_noise_variance_reference'],same_input_latent=True)
def summarize(rows):
 if len(rows)!=18900 or any(not x['same_input_latent'] for x in rows):raise ValueError('All L/scenes,finite identical physical pairs required')
 for key,values in (('RX',d.FIT),('TX',tuple(range(6))),('day',(1,2,3)),('scene',d.SCENES)):
  if set(x[key] for x in rows)!=set(values):raise ValueError('Missing source stratum')
 def stats(xx):
  delta=np.array([x['delta_quality_db'] for x in xx]);diff=np.array([x['relative_output_difference'] for x in xx]);n=len(xx)
  return dict(count=n,effective_changes=sum(x['effective_change'] for x in xx),effective_fraction=sum(x['effective_change'] for x in xx)/n,lock_changes=sum(x['lock_changed'] for x in xx),delta_quality_quantiles_db=np.quantile(delta,[0,.05,.5,.95,1]).tolist(),absolute_delta_median_db=float(np.median(abs(delta))),output_relative_difference_quantiles=np.quantile(diff,[0,.5,.95,1]).tolist(),lock_pairs={str(k):v for k,v in Counter((x['white_lock'],x['waveform_lock']) for x in xx).items()})
 by={key:{str(v):stats([x for x in rows if x[key]==v]) for v in sorted(set(x[key] for x in rows))} for key in ('RX','TX','day','scene','state')};total=stats(rows)
 qualified=total['effective_fraction']>=.05 and all(x['effective_fraction']>=.03 for x in by['RX'].values())
 return dict(status='ANALYZED',decision='PASS_PHYSICAL_MISMATCH_ONLY' if qualified else 'FAIL_FIXED_QUALITY_COUPLING_CLOSED',overall=total,strata=by,source_L_unique=6300,source_RX_holdout=None,Ag_efficacy_proved=False,As_accuracy_benefit_proved=False,joint_goal_complete=False,target_access=False,automatic_training=False)
