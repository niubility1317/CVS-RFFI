"""Read only fixed source artifacts; target results cannot enter seed qualification."""
import json,math
from pathlib import Path
from . import design as d

def compute():
 p=d.BASE/'source_features.json'
 if not p.is_file():return dict(status='WAITING_FOR_FIXED_SOURCE_ARTIFACTS',run_id=d.RUN,target_results_read=False)
 f=d.read(p)
 if f['status']!='VERIFIED' or f['run_id']!=d.RUN or f['models']!=2 or f['missing'] or len(f['paired'])!=1 or f['head_fit'] or f['head_updates'] or f['query_read'] or f['U_IQ_read'] or f['U_truth_read']:raise ValueError('Actual source paired feature evidence incomplete')
 pair=f['paired'][0]
 if not pair['actual_initial_states_identical'] or not pair['source_features']['model_parameters_buffers_unchanged'] or pair['actual_final_parameter_difference']['feature_parameters']!=176065:raise ValueError('Actual feature boundary/source state differs')
 z=pair['source_features']['records'][0]
 if z['dimension']!='overall' or z['count']!=720 or not math.isfinite(z['joint160_L2_mean']):raise ValueError('Actual L720 source comparison differs')
 rows=[]
 for row in d.training_rows():
  c=d.config(row);root=Path(c['output_root']);done=d.read(root/'completion.json');frozen=d.read(root/'source_frozen.json');diag=d.read(root/'source_style_epoch_diagnostic.json')
  if done['config']!=c or done['status']!='SOURCE_TRAINED' or done['epoch']!=50 or done['logged_steps']!=11100 or done['optimizer_steps']!=11100 or done['target_access'] or done['target_evaluated'] or frozen['selection']!='final_only' or frozen['config']!=c:raise ValueError('Fixed final source budget/provenance differs')
  if [r['epoch'] for r in diag['records']]!=list(range(1,51)) or diag['query_read'] or diag['head_fit'] or diag['U_truth_read']:raise ValueError('Fixed complete style epoch evidence differs')
  enabled=c['identity_mixstyle']['enabled'];layers={}
  for n in d.STYLE_LAYERS:
   counts={k:sum(r['layers'][n][k] for r in diag['records']) for k in ('calls','packet_calls','eligible_packets','changed_packets')}
   if counts['calls']!=(11100 if enabled else 0) or counts['packet_calls']!=(1420800 if enabled else 0) or not 0<=counts['changed_packets']<=counts['eligible_packets']<=counts['packet_calls']:raise ValueError('Actual source-only hook counts differ')
   layers[n]=dict(**counts,changed_ratio=counts['changed_packets']/counts['packet_calls'] if counts['packet_calls'] else 0.)
  if any(r['teacher_style_calls'] or r['U_style_calls'] or r['eval_style_calls'] for r in diag['records']):raise ValueError('Unregistered role intervention')
  loss=done['final_training_L_CE']
  if not math.isfinite(loss):raise ValueError('Invalid source loss')
  rows.append(dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],final_L_CE=loss,layers=layers))
 primary=next(r for r in rows if r['arm']==d.PRIMARY);threshold=d.config(d.training_rows()[0])['source_falsification']
 checks=dict(final_L_CE=all(0<=r['final_L_CE']<=threshold['final_L_CE_max'] for r in rows),actual_layers=all(v['changed_ratio']>=threshold['minimum_layer_changed_ratio'] for v in primary['layers'].values()),actual_source_z=z['joint160_L2_mean']>=threshold['minimum_actual_source_z_L2'])
 return dict(status='PASS' if all(checks.values()) else 'FAIL',run_id=d.RUN,seed=d.SEEDS[0],checks=checks,rows=rows,joint160_source_pair=z,thresholds=threshold,target_results_read=False,target_IQ_read=False,target_truth_read=False,U_truth_read=False,head_fit=False,source_only=True,additional_seed_launch=False,decision_basis='Only the criteria frozen in the first-seed registration; no target selection, parameter tuning, stopping or rerun.')

def main():
 result=compute()
 if result['status']=='WAITING_FOR_FIXED_SOURCE_ARTIFACTS':print(json.dumps(result));return
 out=d.BASE/'source_qualification.json'
 if out.is_file():
  if d.read(out)!=result:raise ValueError('Existing immutable source qualification differs')
 else:d.write(out,result)
 print(json.dumps(result))
if __name__=='__main__':main()
