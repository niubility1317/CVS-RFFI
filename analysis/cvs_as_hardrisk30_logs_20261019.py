"""Read every native and risk source log;stdlib only,no model/IQ/query/truth."""
import gzip,json,math
from pathlib import Path
from collections import Counter,defaultdict
from itertools import zip_longest
ARMS=('H01','R01')
FIELDS=('reference_steps','shape_steps','unit_steps','hardrisk_steps','step_metrics')
def numeric(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def read(p):return json.loads(p.read_text()) if p.is_file() else {}
def records(p):
 if not p.is_file():return []
 opener=gzip.open if p.suffix=='.gz' else open
 with opener(p,'rt',encoding='utf-8-sig') as f:return [json.loads(line) for line in f if line.strip()]
def stat(values):
 vals=[float(v) for v in values if numeric(v)]
 return dict(count=len(vals),nonfinite_or_missing=len(values)-len(vals),mean=sum(vals)/len(vals) if vals else None,min=min(vals) if vals else None,max=max(vals) if vals else None,sum=sum(vals))
def summarize(base):
 base=Path(base);rows={};sequences={};As_sequences={}
 for arm in ARMS:
  source=base/(arm+'-s2026092701')/'source';data={k:records(source/(k+'.jsonl.gz')) for k in FIELDS};native=[v.get('metrics',{}) for v in data['step_metrics']];risk=data['hardrisk_steps'];seq=records(source/'training_sequence.jsonl.gz');sequences[arm]=seq;As_sequences[arm]=[v.get('sample_ids') for v in risk];roles={}
  for role in ('L','U'):
   rr=[v for v in seq if v.get('role')==role];roles[role]=dict(batches=len(rr),physical_packets=sum(len(v.get('ids',[])) for v in rr),batch_histogram=dict(Counter(str(len(v.get('ids',[]))) for v in rr)))
  checks=defaultdict(Counter)
  def check(k,v):checks[k]['match' if v else 'mismatch']+=1
  for i,r in enumerate(risk,1):
   check('ordered_step',r['step']==i);check('mode',r['mode']==('HARD' if arm=='H01' else 'SHUFFLE'));check('64_As_all_supervised',r['selected_packets']==r['As_packets']==64);check('weight_bounds',r['weight_min']>=.75-1e-6 and r['weight_max']<=1.25+1e-6);check('total_weight64',abs(r['weight_sum']-64)<1e-4);check('perTX_mass',all(abs(g['weight_sum']-g['count'])<1e-4 for g in r['class_groups']) and sum(g['count'] for g in r['class_groups'])==64);check('CEonly',abs(r['loss_delta']-.5*r['mixed_CE_delta'])<3e-6 and abs(r['mixed_RD_new']-r['mixed_RD_old']-r['mixed_CE_delta'])<3e-6);check('no_new_NN_VJP',r['extra_NN']==0 and r['extra_CE_operator_calls']==2 and r['extra_CE_logit_packets']==128);check('same_IQ',r['same_IQ_each_pair'] and r['ordinary_redraws']==0);check('rank_detached',r['ranking_gradient_stopped']);check('ordinary_KL_unchanged',not r['ordinary_CE_changed'] and not r['KL_changed'])
  for r in data['reference_steps']:check('sixL_ORD128',r['L_forwards']==6 and r['same_IQ_each_pair'] and r['ordinary_redraws']==0 and r['crossview_mode']=='ORD128' and r['SupCon_weight']==.1 and r['As_packets']==64 and r['all_satellite_L_supervised'])
  for i,r in enumerate(native,1):check('U_truth_hidden',r.get('train/pseudo_truth_available')==0);check('EMA_successful_counter',r.get('train/a1_ema_successful_updates')==i);check('finite_native_total',numeric(r.get('train/loss')));check('native_gradient_surgery_off',r.get('train/os_gradient_surgery_active')==0 and r.get('train/os_budget_controller_active')==0)
  epochs=[]
  for ep in range(10):
   chunk=native[ep*222:(ep+1)*222];rch=risk[ep*222:(ep+1)*222];epochs.append(dict(local_epoch=ep+1,cumulative_epoch=ep+21,steps=len(chunk),native={k:stat([r.get(k) for r in chunk]) for k in ('train/loss','train/loss_labeled','train/loss_unlabeled','train/loss_entropy','train/pseudo_selected','train/pseudo_total')},risk={k:stat([r.get(k) for r in rch]) for k in ('selected_CE_old','selected_CE_new','loss_delta','mixed_CE_delta','weight_min','weight_max','weight_sum')}))
  result=read(source/'result.json');stdout=base/'logs'/(arm+'-s2026092701.log');text=stdout.read_text(errors='replace') if stdout.is_file() else '';costkeys=('optimizer_steps','L_NN','L_As','U_steps','U_packets','U_student_NN','student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','wall_seconds','EMA_local_successful_updates','EMA_cumulative_updates')
  rows[arm]=dict(stream_records={k:len(v) for k,v in data.items()},epoch_records=len(records(source/'epoch_metrics.jsonl')),observational_checks={k:dict(v) for k,v in checks.items()},roles=roles,cost={k:result.get(k) for k in costkeys},risk_compute=result.get('within_TX_risk'),epochs=epochs,stdout=dict(lines=len(text.splitlines()),tracebacks=text.count('Traceback (most recent call last):'),OOM=text.count('CUDA out of memory')),all_streams_complete=all(len(v)==2220 for v in data.values()),sample_IDs_exported=False)
 same_seq=sequences['H01']==sequences['R01'];sameAs=As_sequences['H01']==As_sequences['R01'];complete=all(r['all_streams_complete'] and r['epoch_records']==10 for r in rows.values())
 return dict(status='SOURCE_LOGS_COMPLETE' if complete else 'SOURCE_LOGS_PARTIAL',rows=rows,source_physical_sequence_equal=same_seq,As_physical_sequence_equal=sameAs,model_seed_count=1,NN=0,CP=0,IQ=0,truth_reads=0,updates=0,observational_only=True)
