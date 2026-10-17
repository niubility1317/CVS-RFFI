"""Read complete fixed source logs and already-scored results;no IQ/CP/NN/truth."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
from experiments.cvs_practical_power30.analyze_logs import source_logs,json_ready,read_json
from experiments.cvs_clock_factorial20.analyze_logs import compare_sequences,records
PROFILE_SEED=20261017711

def shape_id(pid,epoch,step):
 import numpy as np
 namespace=f'FcrossRX/train/e{epoch}/step{step}/L_S'
 raw=json.dumps([PROFILE_SEED,namespace,'PDP_shape',tuple(map(int,pid))],ensure_ascii=False,separators=(',',':'))
 seed=int.from_bytes(hashlib.sha256(raw.encode('utf8')).digest()[:8],'little')
 return int(np.random.default_rng(seed).integers(0,2))

def profile_coverage(path,channel):
 counts=Counter();step_state={};steps=0;by_step=[];by_rx=Counter();unique=set()
 for r in records(Path(path),step_state):
  if r['role']!='L':continue
  steps+=1;step=4440+steps;epoch=20+(steps-1)//222+1;cc=Counter()
  if len(r['ids'])!=128:raise ValueError('Every actual L128 observation required')
  for pid in r['ids']:
   if len(pid)!=5 or pid[0]!=1 or pid[1] not in (1,3,4,6,8) or pid[2] not in (1,2,3):raise ValueError('Original full source physical IDs required')
   k=shape_id(pid,epoch,step);cc[k]+=1;counts[k]+=1;by_rx[(pid[1],pid[2],k)]+=1;unique.add(tuple(pid))
  by_step.append(dict(epoch=epoch,step=step,original=cc[0],equispaced=cc[1]))
 if step_state['partial'] or steps!=2220 or sum(counts.values())!=284160 or len(unique)!=6300 or channel['profile_steps']!=by_step or channel['profile_counts']!={str(k):v for k,v in counts.items()}:raise ValueError('Saved profile coverage differs from all original source observations')
 observed=Counter()
 for r in channel['profile_strata']:
  if r['scene'] not in ('practical_low_suburban','practical_mid_urban','practical_low_urban') or r['state'] not in ('LOS','PARTIAL','BLOCKED') or r['lock'] not in ('locked','degraded','unlocked'):raise ValueError('Unexpected profile stratum')
  observed[(r['RX'],r['day'],r['shape'])]+=r['count']
 if observed!=by_rx or any(by_rx[(rx,day,k)]<=0 for rx in (1,3,4,6,8) for day in (1,2,3) for k in (0,1)):raise ValueError('All original RX/day profile counts required')
 return dict(records=steps,As_packets=sum(counts.values()),unique_L_physical=len(unique),shape_counts=dict(counts),all_private_draws_exact=True,per_RX_day_shape=[dict(RX=k[0],day=k[1],shape=k[2],count=v) for k,v in sorted(by_rx.items())],raw_IQ_reads=0)

def collect(run_root,original_training_root,control_root):
 run_root=Path(run_root);original_training_root=Path(original_training_root);control_root=Path(control_root);rid='PDP_SHAPE-s2026092701';b=run_root/rid
 if (read_json(run_root/'completion.json') or {}).get('status')!='ARTIFACTS_COMPLETE':raise ValueError('All fixed sevenview predictions and scores must finish')
 for row in (b,control_root):
  if (read_json(row/'completion.json') or {}).get('status')!='TRAINED_DEFAULT_PREDICTED_SCORED':raise ValueError('Both fixed default-head rows required')
 source=b/'source';old=original_training_root/'FULL_AS-s2026092701/source'
 candidate=source_logs(source,run_root/'logs'/(rid+'.stdout.log'),require_detail=True);nominal=source_logs(old,original_training_root/'logs/FULL_AS-s2026092701.stdout.log')
 sampling=compare_sequences(source/'training_sequence.jsonl.gz',old/'training_sequence.jsonl.gz')
 if not sampling['same_complete_sequence']:raise ValueError('Matched original L/U sequence differs')
 state={};objective_max_error=0.;objective_records=0
 for r in records(source/'reference_steps.jsonl.gz',state):
  objective_records+=1
  if r['step']!=objective_records or r['cumulative_step']!=4440+objective_records or r['L_forwards']!=4 or r['As_packets']!=128 or not r['same_IQ_each_pair'] or r['ordinary_redraws']!=0 or not r['all_satellite_L_supervised'] or r['SupCon_weight']!=.1 or len(r['SupCon_details'])!=4:raise ValueError('Actual paired objective/cost rule differs')
  error=max(abs(r['total_L']-(.5*r['common_RD']+.5*r['extra_RD']+.1*r['SupCon'])),abs(r['common_RD']-r['common_CE']-r['common_KL']),abs(r['extra_RD']-r['extra_CE']-r['extra_KL']))
  objective_max_error=max(objective_max_error,error)
  if not math.isfinite(error) or error>1e-5:raise ValueError('Actual objective normalization differs')
 if state['partial'] or objective_records!=2220:raise ValueError('Full objective logs required')
 counts=candidate['training_result'];channel=counts['channel']
 if (counts['actual_steps'],counts['cumulative_steps'],counts['L_forward_packets'],counts['As_training_observations'],channel['profile_calls'],channel['packets'],channel['session_count'])!=(2220,6660,1136640,284160,284160,284160,15):raise ValueError('Actual complete budget/profile coverage differs')
 if channel['physical_channel_prepasses'] or channel['extra_physical_filters'] or channel['amplitude_renormalization'] or not channel['true_hardware_and_calibration_unchanged'] or not channel['exact_ensemble_RMS_preserved']:raise ValueError('Unexpected extra pass or normalization')
 expected={f'rx{rx}:day{day}:eq1' for rx in (1,3,4,6,8) for day in (1,2,3)}
 if set(channel['sessions'])!=expected:raise ValueError('Missing original physical RX sessions')
 import numpy as np
 for session,hw in channel['sessions'].items():
  raw=json.dumps([20261016423,'receiver',session],ensure_ascii=False,separators=(',',':'));seed=int.from_bytes(hashlib.sha256(raw.encode('utf8')).digest()[:8],'little');rng=np.random.default_rng(seed)
  amp=float(rng.uniform(-.2,.2));quad=float(rng.uniform(-1.,1.));cfo=float(rng.normal(0.,200.));expected_hw=dict(session_id=session,amplitude_imbalance_db=amp,quadrature_error_deg=quad,oscillator_cfo_hz=cfo,estimated_amplitude_imbalance_db=amp+float(rng.normal(0.,.05)),estimated_quadrature_error_deg=quad+float(rng.normal(0.,.2)))
  if hw!=expected_hw:raise ValueError('Original injected hardware or calibration changed')
 coverage=profile_coverage(source/'training_sequence.jsonl.gz',channel)
 same_keys=[k for k in candidate['native_args'] if k.startswith('aug_') or k in ('epochs','label_epochs','pseudo_epochs','seed','lr','weight_decay','optimizer','use_aug','use_ema_teacher','pseudo_domain_gate','pseudo_temporal_gate','pseudo_temporal_mode','pseudo_strong_agreement','baseline_ckpt','freeze_backbone','a1_scratch_only','source_fold_held_rx','source_fold_training_receivers')]
 if any(candidate['native_args'][k]!=nominal['native_args'][k] for k in same_keys):raise ValueError('Matched actual ordinary/budget/U/parent args differ')
 for name,path,data in (('PDP_SHAPE',b,candidate),('NOMINAL_AS',control_root,nominal)):
  data['prediction']=read_json(path/'prediction/complete.json');data['score']=read_json(path/'scores/metrics.json');data['row_completion']=read_json(path/'completion.json')
 return dict(status='CLOSED_LOGS_COLLECTED',run_id=run_root.name,rows={'PDP_SHAPE':candidate,'NOMINAL_AS':nominal},sampling=sampling,actual_shared_args=len(same_keys),objective_records=objective_records,objective_max_error=objective_max_error,profile_coverage=coverage,original_hardware_and_calibration_states_verified=15,result=read_json(run_root/'result.json'),completion=read_json(run_root/'completion.json'),IQ_reads=0,CP_loads=0,NN=0,new_updates=0,raw_truth_reads=0)

if __name__=='__main__':
 import sys
 missing=[];out=json_ready(collect(*sys.argv[1:]),missing=missing);out['nonfinite_optional_metadata_paths']=missing
 print(json.dumps(out,ensure_ascii=False,allow_nan=False))
