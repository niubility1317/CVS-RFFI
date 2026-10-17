"""All persisted logs and fixed paired scores;no new IQ/checkpoint/rawtruth/model access."""
from pathlib import Path
import json,math

def json_ready(value,path='',missing=None):
 if missing is None:missing=[]
 if isinstance(value,float) and not math.isfinite(value):missing.append(path);return None
 if isinstance(value,dict):return {k:json_ready(v,path+'/'+str(k),missing) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [json_ready(v,path+'/'+str(i),missing) for i,v in enumerate(value)]
 return value
from experiments.cvs_clock_factorial20.analyze_logs import read_json,records,numeric_logs,sequence,compare_sequences,stdout_scan

def source_logs(source,stdout,expected=2220,require_detail=False):
 source=Path(source);state={};epochs=list(records(source/'epoch_metrics.jsonl',state));steps=numeric_logs(source/'step_metrics.jsonl.gz','metrics');seq=sequence(source/'training_sequence.jsonl.gz')
 if state['partial'] or [x['epoch'] for x in epochs]!=list(range(1,11)) or steps['records']!=expected or steps['partial'] or not steps['step_order']['consecutive'] or seq['partial'] or seq['unexpected_roles']:raise ValueError('Closed ten-epoch logs required')
 if seq['unique_physical']!={'L':6300,'U':56700} or seq['packets']!={'L':284160,'U':567000}:raise ValueError('Full original source coverage differs')
 for key,value in (('train/optimizer_step_applied',1),('train/pseudo_truth_available',0),('train/skipped_nonfinite_grad',0),('train/skipped_nonfinite_loss',0),('concat/L_forwards',4),('concat/As_packets',128),('concat/SupCon_weight',.1)):
  x=steps['totals'][key]
  if x['minimum']!=value or x['maximum']!=value:raise ValueError('Actual source update/U/view rule differs:'+key)
 detail={k:numeric_logs(source/(k+'.jsonl.gz')) for k in ('reference_steps','unit_steps')}
 if require_detail and any(x['records']!=expected or x['partial'] for x in detail.values()):raise ValueError('Missing candidate detailed source records')
 return dict(epoch_metrics=epochs,step_metrics=steps,training_sequence=seq,source_role_counts={k:len(v) for k,v in read_json(source/'source_contract.json')['role_ids'].items()},native_args=read_json(source/'actual_native_args.json'),initialization=read_json(source/'initialization.json'),EMA_initialization=read_json(source/'EMA_initialization.json'),optimizer_initialization=read_json(source/'optimizer_initialization.json'),source_completion=read_json(source/'completion.json'),training_result=read_json(source/'training_result.json'),progress=read_json(source/'reference_progress.json'),actual_channel=read_json(source/'actual_channel.json'),ordinary_actual=read_json(source/'ordinary_augmentation_actual.json'),native_terminal=read_json(source/'phase1_terminal_status.json'),stdout=stdout_scan(Path(stdout)),**detail)

def collect(run_root,original_training_root,control_root):
 run_root=Path(run_root);original_training_root=Path(original_training_root);control_root=Path(control_root);row='QUALITY_POWER-s2026092701';b=run_root/row
 if (read_json(run_root/'completion.json') or {}).get('status')!='ARTIFACTS_COMPLETE':raise ValueError('New prediction/score matrix must finish first')
 if (read_json(b/'completion.json') or {}).get('status')!='TRAINED_DEFAULT_PREDICTED_SCORED' or (read_json(control_root/'completion.json') or {}).get('status')!='TRAINED_DEFAULT_PREDICTED_SCORED':raise ValueError('Both fixed default-head rows required')
 src=b/'source';old=original_training_root/'FULL_AS-s2026092701'/'source'
 candidate=source_logs(src,run_root/'logs'/(row+'.stdout.log'),require_detail=True);nominal=source_logs(old,original_training_root/'logs/FULL_AS-s2026092701.stdout.log')
 sampling=compare_sequences(src/'training_sequence.jsonl.gz',old/'training_sequence.jsonl.gz')
 if not sampling['same_complete_sequence']:raise ValueError('Matched source physical sequence differs')
 counts=candidate['training_result'];channel=counts['channel']
 if (counts['actual_steps'],counts['cumulative_steps'],counts['L_forward_packets'],counts['As_training_observations'],channel['quality_prefilter_calls'],channel['packets'])!=(2220,6660,1136640,284160,284160,284160) or channel['physical_channel_prepasses']!=0:raise ValueError('Actual complete budget or single-pass quality cost differs')
 same_keys=[k for k in candidate['native_args'] if k.startswith('aug_') or k in ('epochs','label_epochs','pseudo_epochs','seed','lr','weight_decay','optimizer','use_aug','use_ema_teacher','pseudo_domain_gate','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_strong_agreement','baseline_ckpt','freeze_backbone','a1_scratch_only','source_fold_held_rx','source_fold_training_receivers')]
 if any(candidate['native_args'][k]!=nominal['native_args'][k] for k in same_keys):raise ValueError('Matched actual ordinary/budget/U/parent args differ')
 for name,path,data in (('QUALITY_POWER',b,candidate),('NOMINAL_AS',control_root,nominal)):
  data['prediction']=read_json(path/'prediction/complete.json');data['score']=read_json(path/'scores/metrics.json');data['row_completion']=read_json(path/'completion.json')
 return dict(status='CLOSED_LOGS_COLLECTED',run_id=run_root.name,rows={'QUALITY_POWER':candidate,'NOMINAL_AS':nominal},sampling=sampling,actual_shared_args=len(same_keys),result=read_json(run_root/'result.json'),completion=read_json(run_root/'completion.json'),IQ_reads=0,CP_loads=0,NN=0,new_updates=0,raw_truth_reads=0)

if __name__=='__main__':
 import sys
 missing=[];value=json_ready(collect(*sys.argv[1:]),missing=missing);value['nonfinite_optional_metadata_paths']=missing
 print(json.dumps(value,ensure_ascii=False,allow_nan=False))
