"""Source physics summaries;no model fitting or prediction."""
import math,statistics
from collections import Counter
from . import design as d

def group_summary(rows):
 n=len(rows);bias=[z['whole']['quality_snr_db']-z['native']['quality_snr_db'] for z in rows];bd=[abs(z['whole']['quality_snr_db']-z['interior']['quality_snr_db']) for z in rows];flips=[z['whole']['lock']!=z['native']['lock'] for z in rows];robust=sum(z['whole']['lock']==z['interior']['lock'] and z['whole']['lock']!=z['native']['lock'] for z in rows)
 return dict(count=n,median_bias_db=statistics.median(bias),median_abs_bias_db=statistics.median(map(abs,bias)),min_bias_db=min(bias),max_bias_db=max(bias),whole_lock_flips=sum(flips),whole_lock_flip_fraction=sum(flips)/n,interior_lock_flips=sum(z['interior']['lock']!=z['native']['lock'] for z in rows),robust_same_flip_count=robust,boundary_robust_flip_fraction=robust/sum(flips) if sum(flips) else 0.,median_whole_vs_interior_abs_db=statistics.median(bd),min_interior_count=min(z['interior_count'] for z in rows),mean_whole_total_MSE=statistics.mean(z['whole_total_MSE'] for z in rows),mean_interior_total_MSE=statistics.mean(z['interior_total_MSE'] for z in rows),max_fixed_receiver_MSE=max(z['fixed_receiver_MSE'] for z in rows),mean_abs_residual_CFO_change_hz=statistics.mean(abs(z['whole']['frequency_hz']-z['native']['frequency_hz']) for z in rows),whole_AGC_limited=sum(z['whole']['agc_limited'] for z in rows),native_AGC_limited=sum(z['native']['agc_limited'] for z in rows),white_to_whole_lock={a+'->'+b:sum(z['native']['lock']==a and z['whole']['lock']==b for z in rows) for a in ('locked','degraded','unlocked') for b in ('locked','degraded','unlocked')})
def criteria(groups,minimum):
 checks={'all_interior_min128':minimum>=128}
 for draw in range(2):
  for scene in d.PRIMARY:
   g=groups[str(draw)+'/'+scene];tag=str(draw)+'_'+scene
   checks.update({tag+'_magnitude':g['median_abs_bias_db']>=1.,tag+'_lock_flips':g['whole_lock_flip_fraction']>=.05,tag+'_boundary_median':g['median_whole_vs_interior_abs_db']<=.5,tag+'_robust_flips':g['boundary_robust_flip_fraction']>=.5})
 return checks
