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
 from experiments.cvs_rdrop_sinc_units.evaluate import preflight as original_preflight
 original_preflight(d.base_config(c['model_seed'])['row_id'])
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
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU aggregation required')
 from statistics import mean,stdev
 import numpy as np
 records=[];costs=[];summary={};comparisons={}
 for row in d.rows():
  rid=row['row_id'];c,ids=preflight(rid);done=d.read(d.BASE/rid/'completion.json')
  if done['status']!='MODEL_PREDICTED_SCORED' or done['new_training_steps']!=(0 if c['baseline_reuse'] else 11100):raise ValueError('Complete fixed target row required')
  records+=d.read(d.BASE/rid/'scores/metrics.json')['records'];source=source_proof(c);old=d.read(source/'completion.json');costs.append(dict(row_id=rid,arm=c['arm'],model_seed=c['model_seed'],source_seconds=old['elapsed_seconds'],new_epochs=0 if c['baseline_reuse'] else 50,new_steps=0 if c['baseline_reuse'] else 11100,historical_epochs_separate=50 if c['baseline_reuse'] else 0))
 def stats(v):return dict(mean=mean(v),sample_SD=stdev(v),n=len(v))
 for arm in d.ARMS:
  details=[]
  for seed in d.SEEDS:
   rr=[r for r in records if r['arm']==arm and r['model_seed']==seed];r=next(r for r in rr if r['dimension']=='overall');details.append(dict(model_seed=seed,accuracy=r['accuracy'],macro_f1=r['macro_f1'],worst_RX=min(r['accuracy'] for r in rr if r['dimension']=='receiver'),worst_day=min(r['accuracy'] for r in rr if r['dimension']=='day')))
  summary[arm]=dict(per_seed=details,**{k:stats([r[k] for r in details]) for k in ('accuracy','macro_f1','worst_RX','worst_day')})
 for after,before in (('ALL_TX','BASE'),('CROSS_RX','BASE'),('CROSS_RX','ALL_TX')):
  per=[]
  for seed in d.SEEDS:
   ar=after+'-s'+str(seed);br=before+'-s'+str(seed)
   with np.load(d.BASE/ar/'scores/correctness.npz',allow_pickle=False) as a,np.load(d.BASE/br/'scores/correctness.npz',allow_pickle=False) as b:
    if not np.array_equal(a['ids'],b['ids']):raise ValueError('Paired physical IDs differ')
    fixed=int((a['NATIVE_HEAD']&~b['NATIVE_HEAD']).sum());broken=int((~a['NATIVE_HEAD']&b['NATIVE_HEAD']).sum())
   x=next(v for v in summary[after]['per_seed'] if v['model_seed']==seed);y=next(v for v in summary[before]['per_seed'] if v['model_seed']==seed);per.append(dict(model_seed=seed,corrected_errors=fixed,introduced_errors=broken,accuracy_difference_pp=(fixed-broken)/1680.,macro_f1_difference_pp=100*(x['macro_f1']-y['macro_f1']),worst_RX_difference_pp=100*(x['worst_RX']-y['worst_RX']),worst_day_difference_pp=100*(x['worst_day']-y['worst_day'])))
  comparisons[after+'-'+before]=dict(per_seed=per,accuracy_difference_pp=stats([v['accuracy_difference_pp'] for v in per]),positive_seeds=sum(v['accuracy_difference_pp']>0 for v in per))
 result=dict(status='ARTIFACTS_COMPLETE',run_id=d.RUN,rows=9,new_models=6,reused_F_models=3,new_epochs=300,new_optimizer_steps=66600,clean_decisions=1512000,new_clean_decisions=1008000,metric_records=len(records),summary=summary,comparisons=comparisons,records=records,costs=costs,head_fit_updates=0,truth_last=True,target_feedback=False,all_three_seeds_preregistered_before_new_target_results=True,limitation='Exposed ManySig clean default-head target performance; no newTX, Phase2 or satellite deployment claim; retain all negative results')
 d.write(d.BASE/'result.json',result);d.write(d.BASE/'completion.json',dict(status='ARTIFACTS_COMPLETE',rows=9,new_models=6,new_epochs=300,new_optimizer_steps=66600,truth_last=True,head_fit_updates=0));print('F_TARGET_MATRIX_COMPLETE',flush=True)
