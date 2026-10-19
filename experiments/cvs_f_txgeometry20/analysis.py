"""Complete registered same-row results, paired statistics and negative outcomes."""
from pathlib import Path
from statistics import mean,stdev
from . import design as d

def stats(v):return dict(n=len(v),mean=mean(v) if v else None,sample_SD=stdev(v) if len(v)>1 else None)
def aggregate(base):
 base=Path(base);matrix=d.read(base/'frozen_matrix.json')['rows'];records=[];cost=[]
 for row in matrix:
  c=d.config(row);s=d.read(base/row['row_id']/'scores/metrics.json');src=d.read(base/row['row_id']/'source/completion.json');prediction=d.read(base/row['row_id']/'prediction/complete.json')
  if s['status']!='MODEL_PREDICTED_SCORED' or s['config']!=c or not s['truth_last'] or s['views']!=list(d.VIEWS):raise ValueError('Incomplete frozen matrix results')
  records.append(dict(**row,metrics=s['records']));cost.append(dict(row_id=row['row_id'],epochs=20,optimizer_steps=src['optimizer_steps'],source_seconds=src['seconds'],prediction_seconds=prediction['seconds'],peak_cuda_MiB=src['peak_cuda_MiB'],L_exposure=20*222*128,U_exposure=src['U_exposure'],satellite_examples=src['satellite_examples'],satellite_applied_examples=src['satellite_applied_examples']))
 summary={};contrasts={}
 for arm in d.ARMS:
  rows=[r for r in records if r['arm']==arm];summary[arm]={}
  for view in d.VIEWS:
   rr=[r['metrics'][view] for r in rows];summary[arm][view]=dict(accuracy=stats([v['accuracy'] for v in rr]),macro_f1=stats([v['macro_f1'] for v in rr]),worst_RX=stats([v['worst_RX'] for v in rr]),worst_day=stats([v['worst_day'] for v in rr]),TX_recall=[stats([v['per_TX_recall'][k] for v in rr]) for k in range(6)])
 for arm in d.ARMS[1:]:
  contrasts[arm+'-BASE']={}
  for view in d.VIEWS:
   paired=[]
   for seed in d.SEEDS:
    a=next((r for r in records if r['model_seed']==seed and r['arm']==arm),None);b=next((r for r in records if r['model_seed']==seed and r['arm']=='BASE'),None)
    if a is None or b is None:continue
    aa,bb=a['metrics'][view],b['metrics'][view];paired.append(dict(model_seed=seed,accuracy_pp=100*(aa['accuracy']-bb['accuracy']),macro_f1_pp=100*(aa['macro_f1']-bb['macro_f1']),per_TX_recall_pp=[100*(x-y) for x,y in zip(aa['per_TX_recall'],bb['per_TX_recall'])],worst_RX_pp=100*(aa['worst_RX']-bb['worst_RX']),worst_day_pp=100*(aa['worst_day']-bb['worst_day'])))
   differences=[p['accuracy_pp'] for p in paired];contrasts[arm+'-BASE'][view]=dict(per_seed=paired,accuracy_difference_pp=stats(differences),positive_seeds=sum(x>0 for x in differences))
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,records=records,summary=summary,contrasts=contrasts,cost=cost,source_expansion=d.read(base/'source_expansion.json'),head_fit_updates=0,target_feedback=False,truth_last=True,limitation='Exposed ManySig Phase1 benchmark; local E20 supervised clean+sat recipe, not paired with historical50 oldSSL85.11%; one seed where source-falsified, do not report as complete three-seed estimate; no new blind or Phase2 claim')
 d.write(base/'result.json',result);return result
