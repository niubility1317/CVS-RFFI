"""Reuse completed default-head references; predict new checkpoints then score independently."""
import os,shutil
from pathlib import Path
from . import design as d
from .provenance import source_proof,own_model
from experiments.cvs_feature_sequence50 import predict as shared
COUNT=168000

def predict(rid):
 c=d.config(next(r for r in d.rows() if r['row_id']==rid))
 if not c['baseline_reuse']:return shared.predict(rid,design=d,proof=source_proof,load=own_model)
 import numpy as np
 from experiments.cvs_report_ssl.evaluate import physical_ids
 source=source_proof(c);old=source.parent/'prediction';out=d.BASE/rid/'prediction';done=d.read(old/'complete.json');actual=d.read(source/'completion.json')['config']
 if done['status']!='PREDICTIONS_COMPLETE' or done['config']!=actual or done['count']!=COUNT or done['readouts']!=['NATIVE_HEAD'] or done['views']!=['clean'] or done['truth_read'] or done['query_fit'] or done['head_fit_updates']:raise ValueError('Original complete immutable RDrop prediction required')
 ids=physical_ids()
 with np.load(old/'predictions.npz',allow_pickle=False) as p:
  if set(p.files)!={'ids','NATIVE_HEAD'} or not np.array_equal(p['ids'],ids) or p['NATIVE_HEAD'].shape!=(COUNT,) or p['NATIVE_HEAD'].dtype.kind not in 'iu' or p['NATIVE_HEAD'].min()<0 or p['NATIVE_HEAD'].max()>=6:raise ValueError('Original reference physical decisions differ')
 out.mkdir(parents=True,exist_ok=False);shutil.copy2(old/'predictions.npz',out/'predictions.npz')
 d.write(out/'provenance.json',dict(status='VERIFIED',config=c,checkpoint=str(source/'final_ssdg.pth'),baseline_native_reused=True,default_classifier_only=True,separate_head_training=False,backbone_frozen=True,truth_read=False,query_fit=False,historical_benchmark_exposed=True))
 d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,count=COUNT,views=['clean'],readouts=['NATIVE_HEAD'],truth_read=False,query_fit=False,head_fit_updates=0,reused_from=str(old),new_target_forwards=0))
 print('PREDICTIONS_REUSED '+rid,flush=True)

def preflight(rid):return shared.preflight(rid,design=d,proof=source_proof)

def score(rid):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU score required')
 import numpy as np
 c,ids=preflight(rid);out=d.BASE/rid/'scores'
 if out.exists():raise FileExistsError('Existing score preserved')
 if c['baseline_reuse']:
  source=source_proof(c);old=source.parent/'scores';v=d.read(old/'metrics.json')
  if not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=18:raise ValueError('Complete independently scored reference required')
  records=[dict(r,row_id=rid,arm=c['arm']) for r in v['records']]
  with np.load(old/'correctness.npz',allow_pickle=False) as correct:
   if not np.array_equal(correct['ids'],ids) or correct['NATIVE_HEAD'].shape!=(COUNT,) or correct['NATIVE_HEAD'].dtype.kind!='b':raise ValueError('Original scored correctness physical IDs differ')
  out.mkdir();shutil.copy2(old/'correctness.npz',out/'correctness.npz');reused=True
 else:
  from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
  y,rx,days=truth_arrays(d.read(d.TRUTH),ids)
  with np.load(d.BASE/rid/'prediction/predictions.npz',allow_pickle=False) as p:decisions=p['NATIVE_HEAD'].copy()
  records,day_records=metric_rows(c,decisions,ids,y,rx,days);records+=day_records
  if len(records)!=18:raise ValueError('Complete 18 physical same-row strata required')
  for r in records:r['readout']='NATIVE_HEAD'
  out.mkdir();np.savez_compressed(out/'correctness.npz',ids=ids,NATIVE_HEAD=decisions==y);reused=False
 d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED',reference_reused=reused))
 d.write(d.BASE/rid/'completion.json',dict(status='MODEL_PREDICTED_SCORED',config=c,count=COUNT,metric_records=18,truth_last=True,new_training_steps=0 if reused else 11100,head_fit_updates=0,reference_reused=reused));print('SCORED '+rid,flush=True)

def finish():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Independent CPU aggregation required')
 from .gate import decision_uploaded
 if not decision_uploaded(d.read(d.BASE/'source_qualification.json')):raise ValueError('Source-only decision must be uploaded before any new target summary')
 import numpy as np
 from statistics import mean,stdev
 state=d.read(d.BASE/'queue_state.json');skip=state['source_skipped'];omitted={r['row_id'] for r in skip};records=[];completed=[]
 for row in d.rows():
  rid=row['row_id']
  if rid in omitted:continue
  c,ids=preflight(rid);done=d.read(d.BASE/rid/'completion.json')
  if done['status']!='MODEL_PREDICTED_SCORED' or done['new_training_steps']!=(0 if c['baseline_reuse'] else 11100):raise ValueError('Complete fixed native row required')
  records+=d.read(d.BASE/rid/'scores/metrics.json')['records'];completed.append(rid)
 summary={}
 for arm in d.ARMS:
  allrows=[r for r in records if r['arm']==arm];details=[]
  for r in allrows:
   if r['dimension']!='overall':continue
   row=[v for v in allrows if v['row_id']==r['row_id']];details.append(dict(model_seed=r['model_seed'],accuracy=r['accuracy'],macro_f1=r['macro_f1'],worst_RX_accuracy=min(v['accuracy'] for v in row if v['dimension']=='receiver'),worst_day_accuracy=min(v['accuracy'] for v in row if v['dimension']=='day')))
  summary[arm]=dict(seed_count=len(details),per_seed=details,accuracy_mean=mean(v['accuracy'] for v in details) if details else None,accuracy_sample_SD=stdev(v['accuracy'] for v in details) if len(details)>1 else None,macro_f1_mean=mean(v['macro_f1'] for v in details) if details else None,worst_RX_mean=mean(v['worst_RX_accuracy'] for v in details) if details else None,worst_day_mean=mean(v['worst_day_accuracy'] for v in details) if details else None)
 comparisons={}
 def stats(values):return dict(seed_count=len(values),mean_pp=mean(values) if values else None,sample_SD_pp=stdev(values) if len(values)>1 else None,positive_seeds=sum(v>0 for v in values))
 for name,after,before in (('B-A','B','A'),('E-C','E','C'),('E-D','E','D'),('F-E','F','E'),('F-B','F','B')):
  per=[]
  for seed in d.SEEDS:
   ar,br=(f'{v}-s{seed}' for v in (after,before))
   if ar not in completed or br not in completed:continue
   with np.load(d.BASE/br/'scores/correctness.npz',allow_pickle=False) as a,np.load(d.BASE/ar/'scores/correctness.npz',allow_pickle=False) as b:
    if not np.array_equal(a['ids'],b['ids']):raise ValueError('Paired physical IDs differ')
    fixed=int((~a['NATIVE_HEAD']&b['NATIVE_HEAD']).sum());broken=int((a['NATIVE_HEAD']&~b['NATIVE_HEAD']).sum())
   per.append(dict(model_seed=seed,corrected_errors=fixed,introduced_errors=broken,accuracy_difference_pp=(fixed-broken)/1680))
  comparisons[name]=dict(per_seed=per,**stats([v['accuracy_difference_pp'] for v in per]))
 interactions=[]
 for seed in d.SEEDS:
  values={a:next((v['accuracy'] for v in summary[a]['per_seed'] if v['model_seed']==seed),None) for a in ('A','B','E','F')}
  if all(v is not None for v in values.values()):interactions.append(dict(model_seed=seed,interaction_pp=100*(values['F']-values['E']-values['B']+values['A'])))
 comparisons['F-E-B+A']=dict(per_seed=interactions,**stats([v['interaction_pp'] for v in interactions]))
 costs=[]
 for row in d.rows():
  if row['row_id'] not in completed:continue
  c=d.config(row);p=source_proof(c);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json')
  costs.append(dict(row_id=row['row_id'],arm=row['arm'],model_seed=row['model_seed'],reference_reused=c['baseline_reuse'],elapsed_source_seconds=done['elapsed_seconds'],total_parameters=resolved['total_parameters'],new_epochs=0 if c['baseline_reuse'] else 50,new_optimizer_steps=0 if c['baseline_reuse'] else 11100,historical_reference_epochs_separate=50 if c['baseline_reuse'] else 0,head_fit_updates=0))
 full=len(completed)==18;goal=full and summary['F']['accuracy_mean']>=.84
 result=dict(status='ARTIFACTS_COMPLETE' if full else 'SOURCE_FALSIFIED',run_id=d.RUN,rows=len(completed),new_models=sum(not d.config(r)['baseline_reuse'] and r['row_id'] in completed for r in d.rows()),metric_records=len(records),prediction_count=len(completed)*COUNT,summary=summary,records=records,comparisons=comparisons,costs=costs,source_qualification=d.read(d.BASE/'source_qualification.json'),source_skipped=skip,head_fit_updates=0,truth_last=True,target_feedback=False,goal84=goal,planned_totals=d.totals(),interpretation='B-A tests X3 on RDrop. E-C and E-D test unified vs separate policies; E-D also changes units, clipping metric and decay. F-E and F-B test combination increments. Positive F-E-B+A suggests complementary effects; three seeds do not establish statistical significance.',limitation='Fixed paired matrix with all negative results; exposed ManySig clean, not new blind/newTX/Phase2. No implied addition of historical gains.')
 d.write(d.BASE/'result.json',result);d.write(d.BASE/'completion.json',dict(status=result['status'],registered_rows=18,rows=len(completed),new_models=result['new_models'],prediction_count=result['prediction_count'],metric_records=len(records),truth_last=True,head_fit_updates=0));print('RDROP_AF_COMBINATION_COMPLETE',flush=True)
