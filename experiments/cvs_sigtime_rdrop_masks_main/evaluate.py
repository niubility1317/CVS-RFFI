import os
from pathlib import Path
from . import design as d
from .provenance import source_proof,own_model
from experiments.cvs_feature_sequence50 import predict as shared_predict
def predict(row):return shared_predict.predict(row,design=d,proof=source_proof,load=own_model)
def preflight(row):return shared_predict.preflight(row,design=d,proof=source_proof)
def score(row):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU scorer required')
 c,ids=preflight(row)
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
 out=d.BASE/row/'scores'
 if out.exists():raise FileExistsError('Preserve existing row scores')
 y,rx,days=truth_arrays(d.read(d.TRUTH),ids)
 with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as value:decisions=value['NATIVE_HEAD'].copy()
 records,day_records=metric_rows(c,decisions,ids,y,rx,days);records+=day_records
 if len(records)!=18:raise ValueError('Eighteen complete same-row clean strata required')
 for record in records:record['readout']='NATIVE_HEAD'
 out.mkdir();np.savez_compressed(out/'correctness.npz',ids=ids,NATIVE_HEAD=decisions==y)
 d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
 d.write(d.BASE/row/'completion.json',dict(status='MODEL_PREDICTED_SCORED',config=c,count=168000,metric_records=18,truth_last=True,new_training_steps=c['epochs']*222,head_fit_updates=0))
 print('SCORED '+row,flush=True)
def aggregate(records):
 from statistics import mean,stdev
 overall=[r for r in records if r['dimension']=='overall']
 if len(overall)!=3 or {r['model_seed'] for r in overall}!=set(d.SEEDS):raise ValueError('Complete three fixed model seeds required')
 details=[]
 for r in overall:
  group=[x for x in records if x['row_id']==r['row_id']]
  details.append(dict(model_seed=r['model_seed'],accuracy=r['accuracy'],macro_f1=r['macro_f1'],errors=r['query_count']-sum(r['confusion'][k][k] for k in range(6)),worst_RX_accuracy=min(x['accuracy'] for x in group if x['dimension']=='receiver'),worst_day_accuracy=min(x['accuracy'] for x in group if x['dimension']=='day')))
 return dict(seed_count=3,seeds=sorted(d.SEEDS),per_seed=details,**{k+'_mean':mean(r[k] for r in details) for k in ('accuracy','macro_f1','worst_RX_accuracy','worst_day_accuracy')},accuracy_sample_SD=stdev(r['accuracy'] for r in details))

def qualify(summaries,paired,interactions):
 a=summaries['E50_ANTI_RDROP'];p=paired['E50_ANTI_RDROP minus E50_INDEP_RDROP'];i=interactions['50']
 if a['seed_count']!=3 or p['seed_count']!=3 or i['seed_count']!=3:raise ValueError('Fixed complete three-seed main evidence required')
 return dict(qualified=a['accuracy_mean']>=.84 and p['mean_pp']>0 and p['positive_seeds']>=2 and i['mean_errors']<0,E50_clean_mean_at_least84=a['accuracy_mean']>=.84,E50_same_budget_RDrop_mean_gain_pp=p['mean_pp'],E50_positive_seeds=p['positive_seeds'],E50_mean_error_interaction=i['mean_errors'],target_feedback=False,goal_completed=False)

def finish():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only frozen aggregation required')
 import numpy as np
 from statistics import mean,stdev
 from .history import prove
 records=[];locations={};physical=[]
 for row in d.rows():
  rid=row['row_id'];preflight(rid);root=d.BASE/rid;done=d.read(root/'completion.json')
  if done['status']!='MODEL_PREDICTED_SCORED' or done['config']!=d.config(row):raise ValueError('All12 new rows must finish before aggregation')
  locations[(row['arm'],row['model_seed'])]=root
  group=d.read(root/'scores/metrics.json')['records']
  if len(group)!=18:raise ValueError('Complete same-row strata required')
  records.extend(dict(z,analysis_arm=row['arm'],physical_run_id=d.RUN,reused_artifact=False) for z in group);physical.append(dict(row_id=rid,physical_run_id=d.RUN,reused=False))
 for row in d.historical_rows():
  root=prove(row);arm=f'E{row["epochs"]}_INDEP_'+('RDROP' if row['mode']=='RDROP' else 'CE');locations[(arm,row['model_seed'])]=root
  scored=d.read(root/'scores/metrics.json');group=scored['records']
  if len(group)!=18 or not scored['truth_last']:raise ValueError('Historical18 truth-last strata required')
  records.extend(dict(z,analysis_arm=arm,physical_run_id=row['run_id'],reused_artifact=True) for z in group);physical.append(dict(row_id=row['row_id'],physical_run_id=row['run_id'],reused=True))
 modes=('INDEP_CE','ANTI_CE','INDEP_RDROP','ANTI_RDROP');arms=tuple(f'E{e}_{m}' for e in d.BUDGETS for m in modes)
 summaries={a:aggregate([z for z in records if z['analysis_arm']==a]) for a in arms};correctness={};reference=None
 for (arm,seed),root in locations.items():
  with np.load(root/'prediction/predictions.npz',allow_pickle=False) as pred,np.load(root/'scores/correctness.npz',allow_pickle=False) as scored:
   ids=pred['ids'];actual=scored['NATIVE_HEAD'].copy()
   if reference is None:reference=ids.copy()
   if not np.array_equal(reference,ids) or not np.array_equal(ids,scored['ids']) or actual.shape!=(168000,) or actual.dtype!=np.bool_:raise ValueError('Same frozen physical decisions required')
   overall=next(z for z in records if z['analysis_arm']==arm and z['model_seed']==seed and z['dimension']=='overall')
   if abs(float(actual.mean())-overall['accuracy'])>1e-12:raise ValueError('Scored accuracy differs from saved decisions')
   correctness[(arm,seed)]=actual
 comparisons=tuple((f'E{e}_{a}',f'E{e}_{b}') for e in d.BUDGETS for a,b in (('ANTI_RDROP','INDEP_RDROP'),('ANTI_CE','INDEP_CE'),('INDEP_RDROP','INDEP_CE'),('ANTI_RDROP','ANTI_CE'),('ANTI_RDROP','INDEP_CE')))+tuple((f'E50_{m}',f'E20_{m}') for m in modes)
 pairs=[];paired={}
 for after,before in comparisons:
  key=after+' minus '+before;values=[]
  for seed in d.SEEDS:
   a,b=correctness[(before,seed)],correctness[(after,seed)];repair=int((~a&b).sum());broken=int((a&~b).sum());delta=(repair-broken)/1680;values.append(delta)
   pairs.append(dict(comparison=key,model_seed=seed,corrected_errors=repair,introduced_errors=broken,accuracy_difference_pp=delta))
  paired[key]=dict(seed_count=3,mean_pp=mean(values),sample_SD_pp=stdev(values),positive_seeds=sum(v>0 for v in values))
 interactions={}
 for e in d.BUDGETS:
  per=[]
  for seed in d.SEEDS:
   err={m:168000-int(correctness[(f'E{e}_{m}',seed)].sum()) for m in modes};per.append(dict(model_seed=seed,errors=(err['ANTI_RDROP']-err['INDEP_RDROP'])-(err['ANTI_CE']-err['INDEP_CE'])))
  interactions[str(e)]=dict(seed_count=3,per_seed=per,mean_errors=mean(z['errors'] for z in per))
 if len(records)!=432 or len(locations)!=24:raise ValueError('Exact fixed24 comparison rows required')
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,summary=summaries,records=records,paired_errors=pairs,paired_summary=paired,error_interactions=interactions,qualification=qualify(summaries,paired,interactions),physical_rows=physical,planned_totals=d.totals(),new_rows=12,reused_rows=12,truth_last=True,head_fit_updates=0,target_feedback=False,goal_completed=False,limitation='Historical baseline artifacts reused without retraining; all12 fixed new models scratch. Exposed clean benchmark, no newTX/satellite/Phase2 or global novelty claim.')
 d.write(d.BASE/'result.json',result);d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',new_rows=12,reused_rows=12,metric_records=432,truth_last=True,head_fit_updates=0));print('PAIRED_MASKS_MAIN_MATRIX_COMPLETE',flush=True)
