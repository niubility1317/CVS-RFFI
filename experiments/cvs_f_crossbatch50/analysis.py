"""Same-seed clean default-head results; preserve each arm and all negatives."""
from pathlib import Path
from statistics import mean,stdev
from . import design as d
def stats(v):return dict(n=len(v),mean=mean(v) if v else None,sample_SD=stdev(v) if len(v)>1 else None)
def aggregate(base,reference_base):
 base=Path(base);reference_base=Path(reference_base);rows=d.read(base/'frozen_matrix.json')['rows'];records=[];paired=[];cost=[]
 for row in rows:
  path=base/row['row_id'];src=d.read(path/'source/local_source_complete.json');c=src['config'];d.validate(c);score=d.read(path/'scores/metrics.json');pred=d.read(path/'prediction/complete.json');ref=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'scores/metrics.json');refpred=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'prediction/complete.json')
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
 direct=[]
 for seed in d.SEEDS:
  arms={r['treatment']:r for r in paired if r['model_seed']==seed}
  if set(arms)==set(d.ARMS):direct.append(dict(model_seed=seed,**{k:arms['MEM4'][k]-arms['ANCHOR'][k] for k in metrics},per_TX_pp=[x-y for x,y in zip(arms['MEM4']['per_TX_pp'],arms['ANCHOR']['per_TX_pp'])]))
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,reference_run=reference_base.name,records=records,paired=paired,by_arm=by_arm,MEM4_minus_ANCHOR=direct,cost=cost,source_expansion=d.read(base/'source_expansion.json'),head_fit_updates=0,target_feedback=False,truth_last=True,limitation='Exposed exploratory Phase1 clean benchmark. Source promotion fixed before new candidate target. Default head joint throughout. No target tuning, independent blind-test, Phase2, or satellite claim.')
 d.write(base/'result.json',result);return result
