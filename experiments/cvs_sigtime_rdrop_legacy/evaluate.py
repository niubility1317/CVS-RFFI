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
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU aggregation required')
 import numpy as np
 from statistics import mean,stdev
 state=d.read(d.BASE/'queue_state.json');skipped=state.get('source_skipped',[]);records=[];pairs=[];completed=[]
 skipped_ids={r['row_id'] for r in skipped}
 for row in d.rows():
  rid=row['row_id']
  if rid in skipped_ids:continue
  c=d.config(row);preflight(rid);done=d.read(d.BASE/rid/'completion.json')
  if done['status']!='MODEL_PREDICTED_SCORED' or done['new_training_steps']!=c['epochs']*222:raise ValueError('Complete original SIG_TIME default-head row required')
  records+=d.read(d.BASE/rid/'scores/metrics.json')['records'];completed.append(rid)
 comparisons=tuple((f'E{e}_{a}',f'E{e}_{b}') for e in d.BUDGETS for a,b in (('RDROP','ORIGINAL'),('RDROP','TWO_CE'),('TWO_CE','ORIGINAL')))+tuple((f'E50_{mode}',f'E20_{mode}') for mode in d.MODES)
 for after,before in comparisons:
  for seed in d.SEEDS:
   ra,rb=(f'{arm}-s{seed}' for arm in (after,before))
   if ra not in completed or rb not in completed:continue
   with np.load(d.BASE/rb/'scores/correctness.npz',allow_pickle=False) as a,np.load(d.BASE/ra/'scores/correctness.npz',allow_pickle=False) as b:
    if not np.array_equal(a['ids'],b['ids']):raise ValueError('Same physical test row required')
    repair=int((~a['NATIVE_HEAD']&b['NATIVE_HEAD']).sum());broken=int((a['NATIVE_HEAD']&~b['NATIVE_HEAD']).sum())
   pairs.append(dict(comparison=after+' minus '+before,model_seed=seed,corrected_errors=repair,introduced_errors=broken,accuracy_difference_pp=(repair-broken)/1680))
 summaries={};paired={}
 for arm in d.ARMS:
  group=[r for r in records if r['arm']==arm];overall=[r for r in group if r['dimension']=='overall']
  if len(overall)==3:summaries[arm]=aggregate(group)
  else:summaries[arm]=dict(seed_count=len(overall),complete_three_seeds=False,seeds=[r['model_seed'] for r in overall],per_seed=overall,accuracy_mean=mean(r['accuracy'] for r in overall) if overall else None,accuracy_sample_SD=None)
 for after,before in comparisons:
  key=after+' minus '+before;values=[p['accuracy_difference_pp'] for p in pairs if p['comparison']==key]
  paired[key]=dict(seed_count=len(values),mean_pp=mean(values) if values else None,sample_SD_pp=stdev(values) if len(values)>1 else None,positive_seeds=sum(v>0 for v in values))
 all_complete=len(completed)==18
 if len(records)!=18*len(completed) or set(completed)|skipped_ids!={r['row_id'] for r in d.rows()}:raise ValueError('Every registered row must complete or have explicit source failure skip')
 result=dict(status='ARTIFACTS_COMPLETE' if all_complete else 'SOURCE_FALSIFIED',run_id=d.RUN,summary=summaries,records=records,paired_errors=pairs,paired_summary=paired,planned_totals=d.totals(),actual_rows=len(completed),source_skipped=skipped,primary='E50_RDROP',
  E50_RDROP_goal84_check=summaries['E50_RDROP']['seed_count']==3 and summaries['E50_RDROP']['accuracy_mean']>=.84,E20_RDROP_goal84_check=summaries['E20_RDROP']['seed_count']==3 and summaries['E20_RDROP']['accuracy_mean']>=.84,
  original_SIGTIME_recipe=True,truth_last=True,head_fit_updates=0,target_feedback=False,goal_completed=False,
  limitation='Historical83.0393% is unpaired reference; new ORIGINAL is the causal same-recipe control. Original LS.01/normal augmentation/RAW_HZ retained. Exposed clean benchmark; no newTX/satellite/Phase2/global novelty claim')
 d.write(d.BASE/'result.json',result);d.write(d.BASE/'completion.json',dict(status=result['status'],registered_rows=18,rows=len(completed),metric_records=len(records),truth_last=True,head_fit_updates=0));print('ORIGINAL_SIGTIME_RDROP_MATRIX_COMPLETE',flush=True)
