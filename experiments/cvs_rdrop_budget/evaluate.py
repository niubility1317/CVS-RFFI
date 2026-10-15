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
def finish():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Final aggregation stays CPU only')
 import numpy as np
 from statistics import mean,stdev
 records=[];pairs=[]
 for row in d.rows():
  c=d.config(row);preflight(row['row_id']);done=d.read(d.BASE/row['row_id']/'completion.json')
  if done['status']!='MODEL_PREDICTED_SCORED' or done['new_training_steps']!=c['epochs']*222:raise ValueError('Complete registered budget row required')
  records+=d.read(d.BASE/row['row_id']/'scores/metrics.json')['records']
 comparisons=(('E20_RDROP','E20_TWO_CE'),('E50_RDROP','E50_TWO_CE'),('E50_TWO_CE','E20_TWO_CE'),('E50_RDROP','E20_RDROP'))
 for after,before in comparisons:
  for seed in d.SEEDS:
   with np.load(d.BASE/(before+'-s'+str(seed))/'scores/correctness.npz',allow_pickle=False) as a,np.load(d.BASE/(after+'-s'+str(seed))/'scores/correctness.npz',allow_pickle=False) as b:
    if not np.array_equal(a['ids'],b['ids']):raise ValueError('Paired budget/mode physical IDs differ')
    repair=int((~a['NATIVE_HEAD']&b['NATIVE_HEAD']).sum());broken=int((a['NATIVE_HEAD']&~b['NATIVE_HEAD']).sum())
    pairs.append(dict(comparison=after+' minus '+before,model_seed=seed,corrected_errors=repair,introduced_errors=broken,accuracy_difference_pp=(repair-broken)/1680))
 summaries={a:aggregate([r for r in records if r['arm']==a]) for a in d.ARMS};paired={}
 for after,before in comparisons:
  key=after+' minus '+before;values=[p['accuracy_difference_pp'] for p in pairs if p['comparison']==key]
  paired[key]=dict(seed_count=3,mean_pp=mean(values),sample_SD_pp=stdev(values),positive_seeds=sum(x>0 for x in values))
 if len(records)!=216:raise ValueError('Twelve rows and216 complete strata required')
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,summary=summaries,records=records,paired_errors=pairs,paired_summary=paired,totals=d.totals(),primary='E50_RDROP',
  E50_RDROP_goal84_check=summaries['E50_RDROP']['accuracy_mean']>=.84,E20_RDROP_goal84_check=summaries['E20_RDROP']['accuracy_mean']>=.84,
  truth_last=True,head_fit_updates=0,target_feedback=False,goal_completed=False,
  limitation='Exposed fixed ManySig clean benchmark; budget effect includes preregistered LR horizon and13+7 vs33+17 stages; no newTX/satellite/Phase2 or global novelty claim')
 d.write(d.BASE/'result.json',result);d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',rows=12,metric_records=216,truth_last=True,head_fit_updates=0));print('RDROP_BUDGET_MATRIX_COMPLETE',flush=True)
