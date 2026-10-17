"""Complete fixed E30 source logs and scores; no raw IQ, truth or model access."""
from pathlib import Path
from collections import Counter
import json,math
from experiments.cvs_practical_power30.analyze_logs import source_logs,json_ready,read_json
from experiments.cvs_clock_factorial20.analyze_logs import compare_sequences,records

def collect(run_root,original_training_root,control_root):
 run_root=Path(run_root);original_training_root=Path(original_training_root);control_root=Path(control_root);rid='CAL_REFRESH-s2026092701';b=run_root/rid
 if (read_json(run_root/'completion.json') or {}).get('status')!='ARTIFACTS_COMPLETE':raise ValueError('Fixed seven-view prediction and score matrix must finish')
 for row in (b,control_root):
  if (read_json(row/'completion.json') or {}).get('status')!='TRAINED_DEFAULT_PREDICTED_SCORED':raise ValueError('Both default-head rows must complete')
 source=b/'source';old=original_training_root/'FULL_AS-s2026092701/source'
 candidate=source_logs(source,run_root/'logs'/(rid+'.stdout.log'),require_detail=True);nominal=source_logs(old,original_training_root/'logs/FULL_AS-s2026092701.stdout.log')
 sampling=compare_sequences(source/'training_sequence.jsonl.gz',old/'training_sequence.jsonl.gz')
 if not sampling['same_complete_sequence']:raise ValueError('Original full-source L/U sequence differs')
 step_state={};objective_max_error=0.;objective_records=0
 for r in records(source/'reference_steps.jsonl.gz',step_state):
  objective_records+=1
  if r['step']!=objective_records or r['cumulative_step']!=4440+objective_records or r['L_forwards']!=4 or r['As_packets']!=128 or not r['same_IQ_each_pair'] or r['ordinary_redraws']!=0 or not r['all_satellite_L_supervised'] or r['SupCon_weight']!=.1 or len(r['SupCon_details'])!=4:raise ValueError('Actual paired objective/cost rule differs')
  error=max(abs(r['total_L']-(.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'])),abs(r['common_RD']-(r['common_CE']+r['common_KL'])),abs(r['extra_RD']-(r['extra_CE']+r['extra_KL'])))
  objective_max_error=max(objective_max_error,error)
  if not math.isfinite(error) or error>1e-5:raise ValueError('Actual L objective normalization differs')
 if step_state['partial'] or objective_records!=2220:raise ValueError('Full objective logs required')
 counts=candidate['training_result'];channel=counts['channel']
 if (counts['actual_steps'],counts['cumulative_steps'],counts['L_forward_packets'],counts['As_training_observations'],channel['calibration_calls'],channel['packets'],channel['calibration_state_count'],channel['session_count'])!=(2220,6660,1136640,284160,284160,284160,150,15):raise ValueError('Actual complete budget/calibration coverage differs')
 if channel['physical_channel_prepasses']!=0 or channel['extra_physical_filters']!=0 or not channel['true_hardware_unchanged']:raise ValueError('Additional channel/filter/hardware change')
 expected={f'rx{rx}:day{day}:eq1' for rx in (1,3,4,6,8) for day in (1,2,3)}
 if set(channel['sessions'])!=expected or set(channel['calibrations'])!={f'e{e}/{s}' for e in range(21,31) for s in expected}:raise ValueError('Missing or extra true/epoch calibration state')
 error=[];changed=0
 for key,cal in channel['calibrations'].items():
  epoch,session=key.split('/',1);true=channel['sessions'][session]
  if any(cal[k]!=true[k] for k in ('session_id','amplitude_imbalance_db','quadrature_error_deg','oscillator_cfo_hz')):raise ValueError('True source injected receiver changed')
  if set(cal)!=set(true) or any(not math.isfinite(cal[k]) for k in cal if k!='session_id'):raise ValueError('Invalid calibration estimate')
  error.append(dict(epoch=int(epoch[1:]),session=session,amplitude_error_db=cal['estimated_amplitude_imbalance_db']-true['amplitude_imbalance_db'],phase_error_deg=cal['estimated_quadrature_error_deg']-true['quadrature_error_deg']))
  changed+=int(cal['estimated_amplitude_imbalance_db']!=true['estimated_amplitude_imbalance_db'] or cal['estimated_quadrature_error_deg']!=true['estimated_quadrature_error_deg'])
 same_keys=[k for k in candidate['native_args'] if k.startswith('aug_') or k in ('epochs','label_epochs','pseudo_epochs','seed','lr','weight_decay','optimizer','use_aug','use_ema_teacher','pseudo_domain_gate','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_strong_agreement','baseline_ckpt','freeze_backbone','a1_scratch_only','source_fold_held_rx','source_fold_training_receivers')]
 if any(candidate['native_args'][k]!=nominal['native_args'][k] for k in same_keys):raise ValueError('Matched actual ordinary/budget/U/parent args differ')
 for name,path,data in (('CAL_REFRESH',b,candidate),('NOMINAL_AS',control_root,nominal)):
  data['prediction']=read_json(path/'prediction/complete.json');data['score']=read_json(path/'scores/metrics.json');data['row_completion']=read_json(path/'completion.json')
 return dict(status='CLOSED_LOGS_COLLECTED',run_id=run_root.name,rows={'CAL_REFRESH':candidate,'NOMINAL_AS':nominal},sampling=sampling,actual_shared_args=len(same_keys),objective_records=objective_records,objective_max_error=objective_max_error,calibration_states=error,changed_estimates_vs_original=changed,result=read_json(run_root/'result.json'),completion=read_json(run_root/'completion.json'),IQ_reads=0,CP_loads=0,NN=0,new_updates=0,raw_truth_reads=0)

if __name__=='__main__':
 import sys
 missing=[];out=json_ready(collect(*sys.argv[1:]),missing=missing);out['nonfinite_optional_metadata_paths']=missing
 print(json.dumps(out,ensure_ascii=False,allow_nan=False))
