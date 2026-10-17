"""Read every step/epoch from the closed two-row matrix; zero IQ/CP/NN access."""
from pathlib import Path
import math
from .log_io import source_logs,json_ready,read_json,compare_sequences,records
MODES=('INDEPENDENT','ANTITHETIC')
def collect(run_root):
 root=Path(run_root)
 if read_json(root/'completion.json')['status']!='ARTIFACTS_COMPLETE':raise ValueError('Full fixed matrix must complete')
 rows={};all_records={};objective_max_error=0.;mask_totals={}
 for mode in MODES:
  rid=mode+'-s2026092701';p=root/rid;s=p/'source'
  row=source_logs(s,root/'logs'/(rid+'.stdout.log'),require_detail=True)
  row.update(prediction=read_json(p/'prediction/complete.json'),score=read_json(p/'scores/metrics.json'),row_completion=read_json(p/'completion.json'))
  if row['row_completion']['status']!='TRAINED_DEFAULT_PREDICTED_SCORED':raise ValueError('All declared views must finish')
  state={};rr=list(records(s/'reference_steps.jsonl.gz',state));ds={};dd=list(records(s/'dropout_steps.jsonl.gz',ds))
  if state['partial'] or ds['partial'] or len(rr)!=2220 or len(dd)!=2220:raise ValueError('Complete reference and dropout records required')
  for step,(r,mask) in enumerate(zip(rr,dd),1):
   if r['step']!=step or r['cumulative_step']!=4440+step or r['L_forwards']!=4 or r['As_packets']!=128 or not r['same_IQ_each_pair'] or r['ordinary_redraws'] or not r['all_satellite_L_supervised'] or r['SupCon_weight']!=.1 or len(r['SupCon_details'])!=4:raise ValueError('Original L supervision/normalization differs')
   error=max(abs(r['total_L']-(.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'])),abs(r['common_RD']-r['common_CE']-r['common_KL']),abs(r['extra_RD']-r['extra_CE']-r['extra_KL']))
   if not math.isfinite(error) or error>1e-5:raise ValueError('L loss scale differs')
   objective_max_error=max(error,objective_max_error)
   if {k:v for k,v in mask.items() if k!='cumulative_step'}!=r['dropout_coupling'] or mask!=dict(step=step,mode=mode,active_calls_per_view=13,random_tensors=52,random_elements=13565952,view_count=4,additional_NN_forwards=0,cumulative_step=4440+step):raise ValueError('Actual private mask cost/order differs')
  result=row['training_result'];ch=result['channel'];drop=result['dropout']
  if (result['actual_steps'],result['cumulative_steps'],result['L_forward_packets'],result['As_training_observations'],ch['packets'],ch['session_count'],drop['steps'],drop['random_tensors'],drop['random_elements'])!=(2220,6660,1136640,284160,284160,15,2220,52*2220,13565952*2220):raise ValueError('Actual budget/As/mask coverage differs')
  if ch['dropped'] or ch['confidence_filter'] or ch['ordinary_before_satellite'] or ch['formal_test_changed']:raise ValueError('Complete original As changed')
  rows[mode]=row;all_records[mode]=rr;mask_totals[mode]=drop
 a,b=(rows[m]['native_args'] for m in MODES);diff={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
 if diff!={'run_id','candidate_id','output_dir'} and diff!={'candidate_id','output_dir'}:raise ValueError('Actual native args differ:'+str(diff))
 paths=[root/(m+'-s2026092701')/'source' for m in MODES]
 sampling=compare_sequences(paths[0]/'training_sequence.jsonl.gz',paths[1]/'training_sequence.jsonl.gz')
 if not sampling['same_complete_sequence']:raise ValueError('Physical L/U sequence differs')
 if read_json(paths[0]/'source_contract.json')['role_ids']!=read_json(paths[1]/'source_contract.json')['role_ids']:raise ValueError('Physical source roles differ')
 if rows[MODES[0]]['training_result']['channel']!=rows[MODES[1]]['training_result']['channel']:raise ValueError('Same As prior/session/coverage required')
 for r,t in zip(all_records[MODES[0]],all_records[MODES[1]]):
  for k in ('ordinary_raw_MSE','second_raw_MSE','scene_counts','state_counts'):
   if r[k]!=t[k]:raise ValueError('Same generated ordinary/As input evidence differs:'+k)
 return dict(status='CLOSED_LOGS_COLLECTED',run_id=root.name,rows=rows,sampling=sampling,actual_native_arg_count=len(a),actual_native_arg_differences=sorted(diff),objective_records_each=2220,objective_max_error=objective_max_error,actual_mask_totals=mask_totals,result=read_json(root/'result.json'),completion=read_json(root/'completion.json'),IQ_reads=0,CP_loads=0,NN=0,new_updates=0,raw_truth_reads=0)
if __name__=='__main__':
 import sys,json
 missing=[];out=json_ready(collect(sys.argv[1]),missing=missing);out['nonfinite_optional_metadata_paths']=missing;print(json.dumps(out,ensure_ascii=False,allow_nan=False))
