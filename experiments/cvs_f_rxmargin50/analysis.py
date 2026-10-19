"""Reuse frozen matched BASE predictions; summarize actual fixed margin rows."""
from pathlib import Path
from statistics import mean,stdev
from . import design as d

def stats(v):return dict(n=len(v),mean=mean(v) if v else None,sample_SD=stdev(v) if len(v)>1 else None)
def aggregate(base,reference_base):
 base=Path(base);reference_base=Path(reference_base);rows=d.read(base/'frozen_matrix.json')['rows'];records=[];paired=[];cost=[]
 for row in rows:
  src=d.read(base/row['row_id']/'source/local_source_complete.json');c=src['config'];d.validate(c);score=d.read(base/row['row_id']/'scores/metrics.json');pred=d.read(base/row['row_id']/'prediction/complete.json');ref=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'scores/metrics.json');refpred=d.read(reference_base/('BASE-s'+str(row['model_seed']))/'prediction/complete.json')
  for v in (score,ref):
   if v['status']!='MODEL_PREDICTED_SCORED' or v['views']!=['clean'] or not v['truth_last'] or not v['independent_CPU_score'] or v['count']!=168000:raise ValueError('Complete independent clean scores required')
  for v in (pred,refpred):
   if v['status']!='PREDICTIONS_COMPLETE' or v['views']!=['clean'] or v['truth_read'] or v['query_fit'] or v['head_fit_updates']:raise ValueError('Frozen default-head clean predictions required')
  if score['config']!=c:raise ValueError('Same-row score config differs')
  cur=score['records']['clean'];old=ref['records']['clean'];records.append(dict(row_id=row['row_id'],model_seed=row['model_seed'],RX_MARGIN=cur,reused_BASE=old));per=[100*(x-y) for x,y in zip(cur['per_TX_recall'],old['per_TX_recall'])]
  paired.append(dict(model_seed=row['model_seed'],accuracy_pp=100*(cur['accuracy']-old['accuracy']),macro_f1_pp=100*(cur['macro_f1']-old['macro_f1']),per_TX_pp=per,weak_TX_mean_pp=(per[1]+per[3])/2,other_TX_mean_pp=sum(per[i] for i in (0,2,4,5))/4,worst_RX_pp=100*(cur['worst_RX']-old['worst_RX']),worst_day_pp=100*(cur['worst_day']-old['worst_day'])))
  cost.append(dict(row_id=row['row_id'],epochs=50,optimizer_steps=11100,source_seconds=src['elapsed_seconds'],prediction_seconds=pred['seconds'],peak_cuda_MiB=src['peak_cuda_MiB'],source_packet_reads=src['actual_source_packet_reads'],registered_U_exposure=963900,satellite_examples=0,head_fit_updates=0,BASE_training_reused=True))
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,reference_run=reference_base.name,records=records,paired=paired,paired_summary={k:stats([r[k] for r in paired]) for k in ('accuracy_pp','macro_f1_pp','weak_TX_mean_pp','other_TX_mean_pp','worst_RX_pp','worst_day_pp')},positive_seeds=dict(overall=sum(r['accuracy_pp']>0 for r in paired),weak=sum(r['weak_TX_mean_pp']>0 for r in paired)),cost=cost,source_expansion=d.read(base/'source_expansion.json'),head_fit_updates=0,target_feedback=False,truth_last=True,limitation='Exposed exploratory Phase1 benchmark. Candidate and source promotion fixed before new candidate target access. All negative results retained. No Phase2 or satellite deployment claim.')
 d.write(base/'result.json',result);return result
