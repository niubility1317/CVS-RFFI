"""Same-seed clean default-head results; preserve each arm and all negatives."""
from pathlib import Path
from statistics import mean,stdev
from . import design as d
from .provenance import source_proof as current_proof
from experiments.cvs_f_crossbatch50 import design as prior
from experiments.cvs_f_crossbatch50.provenance import source_proof as prior_proof
def stats(v):return dict(n=len(v),mean=mean(v) if v else None,sample_SD=stdev(v) if len(v)>1 else None)
def aggregate(base,reference_base):
 base=Path(base);reference_base=Path(reference_base);rows=[dict(row_id='MEM4-s2026092701',treatment='MEM4',model_seed=2026092701)]+d.read(base/'frozen_matrix.json')['rows'];records=[];paired=[];cost=[]
 if [r['model_seed'] for r in rows]!=list(d.ALL_SEEDS):raise ValueError('All three seeds required; no partial aggregation')
 for row in rows:
  path=(base.parent/d.REUSED_MEMORY_RUN if row['model_seed']==2026092701 else base)/row['row_id'];src=d.read(path/'source/local_source_complete.json');c=src['config'];(prior.validate if row['model_seed']==2026092701 else d.validate)(c);source_proof=prior_proof if row['model_seed']==2026092701 else current_proof;source_proof(c);score=d.read(path/'scores/metrics.json');pred=d.read(path/'prediction/complete.json');ref=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'scores/metrics.json');refpred=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'prediction/complete.json')
  for value in (score,ref):
   if value['status']!='MODEL_PREDICTED_SCORED' or value['views']!=['clean'] or not value['truth_last'] or not value['independent_CPU_score'] or value['count']!=168000:raise ValueError('Complete independent clean scores required')
  for value in (pred,refpred):
   if value['status']!='PREDICTIONS_COMPLETE' or value['views']!=['clean'] or value['truth_read'] or value['query_fit'] or value['head_fit_updates']:raise ValueError('Frozen default-head clean predictions required')
  if score['config']!=c:raise ValueError('Same-row config differs')
  cur=score['records']['clean'];old=ref['records']['clean'];records.append(dict(row_id=row['row_id'],treatment=row['treatment'],model_seed=row['model_seed'],current=cur,reused_BASE=old));per=[100*(x-y) for x,y in zip(cur['per_TX_recall'],old['per_TX_recall'])]
  paired.append(dict(treatment=row['treatment'],model_seed=row['model_seed'],accuracy_pp=100*(cur['accuracy']-old['accuracy']),macro_f1_pp=100*(cur['macro_f1']-old['macro_f1']),per_TX_pp=per,weak_TX_mean_pp=(per[1]+per[3])/2,other_TX_mean_pp=sum(per[i] for i in (0,2,4,5))/4,worst_RX_pp=100*(cur['worst_RX']-old['worst_RX']),worst_day_pp=100*(cur['worst_day']-old['worst_day'])))
  ch=d.read(path/'source/crossbatch_training_observation.json')
  cost.append(dict(row_id=row['row_id'],epochs=50,optimizer_steps=11100,source_seconds=src['elapsed_seconds'],prediction_seconds=pred['seconds'],peak_cuda_MiB=src['peak_cuda_MiB'],source_packet_reads=src['actual_source_packet_reads'],extra_L_feature_forwards=ch['extra_L_packets'],extra_model_forwards=ch['extra_model_forwards'],mean_historical_keys=ch['mean_historical_keys'],max_history_age=ch['max_history_age'],per_TX_keys=ch['per_TX_keys'],satellite_examples=0,head_fit_updates=0,BASE_training_reused=True))
 metrics=('accuracy_pp','macro_f1_pp','weak_TX_mean_pp','other_TX_mean_pp','worst_RX_pp','worst_day_pp')
 by_arm={a:dict(paired_summary={k:stats([r[k] for r in paired if r['treatment']==a]) for k in metrics},positive_seeds=dict(overall=sum(r['accuracy_pp']>0 for r in paired if r['treatment']==a),weak=sum(r['weak_TX_mean_pp']>0 for r in paired if r['treatment']==a))) for a in d.ARMS}
 absolute={k:stats([100*r['current'][k] for r in records]) for k in ('accuracy','macro_f1','worst_RX','worst_day')}
 per_TX=[dict(TX=name,absolute_recall_percent=stats([100*r['current']['per_TX_recall'][i] for r in records]),paired_pp=stats([r['per_TX_pp'][i] for r in paired]),positive_seeds=sum(r['per_TX_pp'][i]>0 for r in paired)) for i,name in enumerate(d.CLASSES)]
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,reference_run=reference_base.name,records=records,paired=paired,by_arm=by_arm,absolute=absolute,per_TX=per_TX,all_three_seeds_required=True,first_seed_reused=True,cost=cost,source_expansion=d.read(base/'source_expansion.json'),head_fit_updates=0,target_feedback=False,truth_last=True,limitation='Exposed exploratory Phase1 clean benchmark. User requested all3 after seed01 target exposure; old source-promotion failure preserved. No retuning/reselection/retraining01. Fixed remaining02/03 both scored regardless result. Defaulthead joint. Not independent blind confirmation, Phase2 or satellite evidence.')
 d.write(base/'result.json',result);return result
