"""All fixed seeds, same-seed BASE and magnitude-control contrasts; no reranking."""
from pathlib import Path
from statistics import mean,stdev
from . import design as d
from .provenance import source_proof
def stats(v):return dict(n=len(v),mean=mean(v) if v else None,sample_SD=stdev(v) if len(v)>1 else None,positive_count=sum(x>0 for x in v))
def delta(cur,old):
 per=[100*(x-y) for x,y in zip(cur['per_TX_recall'],old['per_TX_recall'])]
 return dict(accuracy_pp=100*(cur['accuracy']-old['accuracy']),macro_f1_pp=100*(cur['macro_f1']-old['macro_f1']),per_TX_pp=per,weak_TX_mean_pp=(per[1]+per[3])/2,other_TX_mean_pp=sum(per[i] for i in (0,2,4,5))/4,worst_RX_pp=100*(cur['worst_RX']-old['worst_RX']),worst_day_pp=100*(cur['worst_day']-old['worst_day']))
def summarize_pairs(rows):
 keys=('accuracy_pp','macro_f1_pp','weak_TX_mean_pp','other_TX_mean_pp','worst_RX_pp','worst_day_pp')
 return dict(metrics={k:stats([r[k] for r in rows]) for k in keys},per_TX=[dict(TX=t,**stats([r['per_TX_pp'][i] for r in rows])) for i,t in enumerate(d.CLASSES)])
def verdict(main,direction):
 if [r['model_seed'] for r in main]!=list(d.SEEDS) or [r['model_seed'] for r in direction]!=list(d.SEEDS):raise ValueError('Complete fixed three-seed comparisons required')
 weak=all(mean(r['per_TX_pp'][i] for r in main)>0 and sum(r['per_TX_pp'][i]>0 for r in main)>=2 for i in (1,3));stable=all(r['weak_TX_mean_pp']>0 for r in main);health=mean(r['accuracy_pp'] for r in main)>=0 and mean(r['worst_RX_pp'] for r in main)>=-.5 and all(mean(r['per_TX_pp'][i] for r in main)>=-.25 for i in (0,2,4,5));beyond=mean(r['weak_TX_mean_pp'] for r in direction)>0 and sum(r['weak_TX_mean_pp']>0 for r in direction)>=2
 return dict(primary_STAT_GRAD_vs_BASE=dict(PASS=weak and stable and health,both_weak_means_positive_and_each2of3=weak,weakmean_positive3of3=stable,other_TX_overall_worstRX_health=health),direction_vs_NORM_ONLY=dict(PASS=beyond),all3_required=True,rule='Primary: each weakTX mean>0 and at least2/3positive; weakmean>0 all3; overallmean>=BASE; each otherTX mean>=BASE-.25pp; worstRXmean>=BASE-.5pp. Attribution separately: STAT_GRAD-NORM_ONLY weakmean>0 and >=2/3positive. Fixed before target; no target-driven retuning/ranking/retraining.')
def aggregate(base,reference_base):
 base=Path(base);reference_base=Path(reference_base);rows=d.read(base/'frozen_matrix.json')['rows']
 if rows!=d.rows():raise ValueError('All fixed six rows required')
 records=[];paired=[];cost=[]
 for row in rows:
  path=base/row['row_id'];src=d.read(path/'source/local_source_complete.json');c=src['config'];d.validate(c);source_proof(c);score=d.read(path/'scores/metrics.json');pred=d.read(path/'prediction/complete.json');ref=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'scores/metrics.json');refpred=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'prediction/complete.json')
  for value in (score,ref):
   if value['status']!='MODEL_PREDICTED_SCORED' or value['views']!=['clean'] or not value['truth_last'] or not value['independent_CPU_score'] or value['count']!=168000:raise ValueError('Complete independent clean scores required')
  for value in (pred,refpred):
   if value['status']!='PREDICTIONS_COMPLETE' or value['views']!=['clean'] or value['truth_read'] or value['query_fit'] or value['head_fit_updates']:raise ValueError('Frozen default-head clean predictions required')
  if score['config']!=c:raise ValueError('Same-row score config differs')
  cur=score['records']['clean'];old=ref['records']['clean'];records.append(dict(**row,current=cur,reused_BASE=old));paired.append(dict(**row,**delta(cur,old)));ch=d.read(path/'source/statistic_gradient_observation.json')
  cost.append(dict(row_id=row['row_id'],epochs=50,optimizer_steps=11100,source_seconds=src['elapsed_seconds'],prediction_seconds=pred['seconds'],peak_cuda_MiB=src['peak_cuda_MiB'],source_packet_reads=src['actual_source_packet_reads'],statistic_observation=ch,BASE_training_reused=True))
 by_arm={a:dict(paired_summary=summarize_pairs([r for r in paired if r['treatment']==a]),absolute={k:stats([100*r['current'][k] for r in records if r['treatment']==a]) for k in ('accuracy','macro_f1','worst_RX','worst_day')}) for a in d.ARMS}
 direction=[]
 for seed in d.SEEDS:
  cur=next(r['current'] for r in records if r['treatment']=='STAT_GRAD' and r['model_seed']==seed);old=next(r['current'] for r in records if r['treatment']=='NORM_ONLY' and r['model_seed']==seed);direction.append(dict(model_seed=seed,**delta(cur,old)))
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,reference_run=reference_base.name,records=records,paired=paired,by_arm=by_arm,direction_pairs=direction,direction_summary=summarize_pairs(direction),scientific_verdict=verdict([r for r in paired if r['treatment']=='STAT_GRAD'],direction),cost=cost,all_three_seeds_required=True,source_expansion=d.read(base/'source_expansion.json'),head_fit_updates=0,target_feedback=False,truth_last=True,limitation='Exposed exploratory Phase1 clean benchmark. Source errors chose weakTX1/3 before this fixed matrix. All6rows run regardless performance; matched local BASE3 reused. No independent blind-test/Phase2/satellite claim. Norm control at current descriptor gradient, not parameter gradient or cross-trajectory equality.')
 d.write(base/'result.json',result);return result
