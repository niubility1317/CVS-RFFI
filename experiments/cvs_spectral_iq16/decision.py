"""Fixed sufficient-statistic ratios and default-head preservation thresholds."""
import math
from . import design as d
def aggregate(batches):
 def ratio(num,den):
  a=sum(v[num] for v in batches);b=sum(v[den] for v in batches)
  if b<=1e-20:
   if a>1e-20:raise ValueError('Undefined nonzero fidelity error')
   return 0.
  return a/b
 return dict(count=sum(v['count'] for v in batches),finite_PSD_relative_RMSE=ratio('spectral_error_SSE','spectral_oracle_change_SSE')**.5,boundary_error_fraction=ratio('boundary_error_energy','oracle_IQ_change_energy'),interior_relative_MSE_max=max(v['interior_relative_MSE'] for v in batches),RMS_relative_error_max=max(v['RMS_relative_error_max'] for v in batches),restore_gain_dB_min=min(v['restore_gain_dB_min'] for v in batches),restore_gain_dB_max=max(v['restore_gain_dB_max'] for v in batches),finite_delta_mean=[sum(v['finite_delta_sum'][j] for v in batches)/sum(v['count'] for v in batches) for j in range(32)],oracle_delta_mean=[sum(v['oracle_delta_sum'][j] for v in batches)/sum(v['count'] for v in batches) for j in range(32)])
def qualify(stat,kernels,diagnostics,scores):
 if set(scores)!={v['view'] for v in d.views()} or set(kernels)!=set(scores)-{'identity'} or set(diagnostics)!=set(kernels):raise ValueError('All fixed source views/controls required')
 c=d.config()['criteria'];raw=scores['identity'];drops={};matches=[]
 for view in d.views()[1:]:
  name=view['view'];score=scores[name];drop=100*(raw['accuracy']-score['accuracy']);parts={name2:{k:100*(raw['by_'+name2][k]['accuracy']-v['accuracy']) for k,v in score['by_'+name2].items()} for name2 in ('RX','TX','day')};drops[name]=dict(accuracy_drop_pp=drop,strata_drop_pp=parts)
  if view['kind']=='structured':
   control=name.replace('structured','control',1);a=kernels[name]['perturbation_energy'];b=kernels[control]['perturbation_energy'];matches.append(abs(a-b)/max(a,b,1e-30))
 conditions=dict(stable_RX=stat['relative_MSE'] is not None and stat['relative_MSE']<c['crossfit_relative_MSE_lt'] and stat['better_folds']>=c['crossfit_better_folds_min'] and stat['prediction_SSE']<stat['control_SSE'],kernel_realization=all(v['kernel_relative_RMSE']<=c['kernel_relative_RMSE_max'] and v['grid_response_max_abs_dB']<=d.DB+1e-8 for v in kernels.values()),matched_control=max(matches)<=1e-8,finite_IQ_realization=all(v['count']==5040 and v['finite_PSD_relative_RMSE']<=c['finite_PSD_relative_RMSE_max'] and v['boundary_error_fraction']<=c['boundary_error_fraction_max'] and v['interior_relative_MSE_max']<=c['interior_relative_MSE_max'] and v['RMS_relative_error_max']<=c['packet_RMS_relative_error_max'] for v in diagnostics.values()),source_identity=all(v['accuracy_drop_pp']<=.5+1e-10 and all(x<=1.+1e-10 for group in v['strata_drop_pp'].values() for x in group.values()) for k,v in drops.items() if k.startswith('structured')))
 return dict(decision='PASS' if all(conditions.values()) else 'FAIL',conditions=conditions,identity_drops=drops,matched_control_max_relative_energy_error=max(matches),Ag_efficacy_proved=False,automatic_training=False,joint_goal_complete=False)
