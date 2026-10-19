"""Complete E30 extra-CE log readout; stdlib only, zero model/data/truth access."""
import gzip,json,math
from pathlib import Path
from collections import Counter,defaultdict
ARMS=('N01','C01','S01')
FIELDS=('reference_steps','shape_steps','unit_steps','extraCE_steps','step_metrics')
def read(p):return json.loads(p.read_text()) if p.is_file() else {}
def records(p):
 if not p.is_file():return []
 with (gzip.open if p.suffix=='.gz' else open)(p,'rt',encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def stat(v):
 a=[float(x) for x in v if isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)]
 return dict(count=len(a),missing_or_nonfinite=len(v)-len(a),mean=sum(a)/len(a) if a else None,min=min(a) if a else None,max=max(a) if a else None,sum=sum(a))
def summarize(base):
 base=Path(base);rows={};seqs={}
 for arm in ARMS:
  p=base/(arm+'-s2026092701')/'source';data={k:records(p/(k+'.jsonl.gz')) for k in FIELDS};native=[r.get('metrics',{}) for r in data['step_metrics']];extra=data['extraCE_steps'];seq=records(p/'training_sequence.jsonl.gz');seqs[arm]=seq;check=defaultdict(Counter)
  def verify(k,v):check[k]['match' if v else 'mismatch']+=1
  for i,r in enumerate(extra,1):
   verify('ordered_step',r['step']==i and r['cumulative_step']==4440+i);verify('extra_view',r['mode']==('HQ' if arm=='S01' else 'RAW'));verify('all_labels_sameIQ_no_consistency',r['all_extra_labels_supervised'] and r['same_IQ_two_CE'] and r['no_cross_view_consistency']);verify('CEonly_formula',abs(r['CE']-.5*(r['CE0']+r['CE1']))<2e-6 and abs(r['loss_delta']-.25*r['CE'])<2e-6 and abs(r['total_L']-r['original_total_L']-r['loss_delta'])<2e-6);verify('NN_scale',r['NN']==128 and r['CE_operator_calls']==2 and r['total_L_nominal_scale']==1.5 and r['extra_KL']==r['extra_teacher_NN']==0);verify('extra_physics_count',r['physical_snapshots_extra']==(0 if arm=='N01' else 64));verify('HQ_routing',r['HQ_consumed']==(64 if arm=='S01' else 0) and r['HQ_shadow']==(64 if arm=='C01' else 0));verify('native_quality_dependency',r['quality_records']==(0 if arm=='N01' else 64) and (arm=='N01' or abs(r['quality_delta_db_sum']-384)<1e-6))
  for r,s,e in zip(data['reference_steps'],data['shape_steps'],extra):
   verify('original6L_retained',r['L_forwards']==8 and r['same_IQ_each_pair'] and r['ordinary_redraws']==0 and r['crossview_mode']=='ORD128' and r['SupCon_weight']==.1 and r['As_packets']==(0 if arm=='N01' else 64) and r['all_satellite_L_supervised']);expected=.5*r['common_RD']+.5*r['extra_RD']+.25*s['shape_RD']+.1*r['SupCon']+e['loss_delta'];verify('full_loss_formula',abs(expected-r['total_L'])<1e-5)
  for i,r in enumerate(native,1):verify('U_truth_hidden',r.get('train/pseudo_truth_available')==0);verify('EMA_successful_counter',r.get('train/a1_ema_successful_updates')==i);verify('finite_native_loss',isinstance(r.get('train/loss'),(int,float)) and math.isfinite(r['train/loss']));verify('gradient_surgery_off',r.get('train/os_gradient_surgery_active')==r.get('train/os_budget_controller_active')==0)
  result=read(p/'result.json');roles={}
  for role in ('L','U'):
   ss=[r for r in seq if r.get('role')==role];roles[role]=dict(batches=len(ss),packets=sum(len(r.get('ids',[])) for r in ss),batch_histogram=dict(Counter(str(len(r.get('ids',[]))) for r in ss)))
  if result:verify('actual_L_and_U_NN',result['L_NN']==1989120 and result['student_train_NN']==1989120+567000 and result['teacher_eval_NN']==567000 and result['optimizer_steps']==2220 and result['EMA_local_successful_updates']==2220)
  epochs=[]
  for ep in range(10):
   nn=native[ep*222:(ep+1)*222];ee=extra[ep*222:(ep+1)*222];epochs.append(dict(local_epoch=ep+1,cumulative_epoch=ep+21,steps=len(nn),native={k:stat([r.get(k) for r in nn]) for k in ('train/loss','train/loss_labeled','train/loss_unlabeled','train/loss_entropy','train/pseudo_selected','train/pseudo_total')},extraCE={k:stat([r.get(k) for r in ee]) for k in ('CE','loss_delta','physical_CPU_wall_seconds','quality_records','HQ_consumed','HQ_shadow')}))
  log=base/'logs'/(arm+'-s2026092701.log');txt=log.read_text(errors='replace') if log.is_file() else '';costkeys=('optimizer_steps','L_NN','L_As','U_steps','U_packets','student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN','wall_seconds','EMA_local_successful_updates','EMA_cumulative_updates')
  rows[arm]=dict(stream_records={k:len(v) for k,v in data.items()},epoch_records=len(records(p/'epoch_metrics.jsonl')),checks={k:dict(v) for k,v in check.items()},roles=roles,cost={k:result.get(k) for k in costkeys},extra_compute=result.get('extra_view'),epochs=epochs,stdout=dict(lines=len(txt.splitlines()),tracebacks=txt.count('Traceback (most recent call last):'),OOM=txt.count('CUDA out of memory')),all_streams_complete=all(len(v)==2220 for v in data.values()))
 complete=all(r['all_streams_complete'] and r['epoch_records']==10 for r in rows.values());same=all(seqs[a]==seqs['N01'] for a in ARMS)
 return dict(status='SOURCE_LOGS_COMPLETE' if complete else 'SOURCE_LOGS_PARTIAL',rows=rows,source_physical_sequences_equal=same,model_seed_count=1,NN=0,CP=0,IQ=0,truth_reads=0,updates=0,observational_only=True)
