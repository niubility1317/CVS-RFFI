"""All original L6300 statistics only;no checkpoint,network,augmentation or target."""
import os,sys,time
from collections import Counter
import numpy as np
from . import design as d
from .core import observe,fit

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU source statistics only')
 import torch
 torch.set_num_threads(2);out=d.resolve(d.BASE);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  contract=d.read(d.resolve(d.SOURCE));prep=d.resolve(d.PREPARATION);ready=d.read(prep/'preparation_complete.json')
  if ready['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or ready['target_access'] or ready['identity_updates']!=0:raise ValueError('Existing legal source preparation differs')
  packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False)
  for actual in (packed['contract'],ready['source_contract']):
   if any(actual.get(k)!=v for k,v in contract.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source role handles differ')
  L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
  y,rx,day=(np.asarray(L[k].tolist(),dtype=np.int64) for k in ('y','rx','day'));ids=L['ids']
  if len(ids)!=6300 or len(set(ids))!=6300 or set(ids)!=set(contract['role_ids']['L_s']) or set(ids)&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('All and only original L6300 required')
  if {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':6300,'U_s':56700,'V':27000} or L['x'].shape!=(6300,2,256) or L['x'].dtype!=torch.float32 or Counter(zip(rx.tolist(),day.tolist(),y.tolist()))!=Counter({(r,z,t):70 for r in d.FIT for z in (1,2,3) for t in range(6)}):raise ValueError('Full original90 sourcecells required;no sourceRX holdout')
  d.write(out/'resolved_config.json',dict(config=d.config(),source_role_counts={k:len(v) for k,v in contract['role_ids'].items()},execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),cache_other_source_features_deserialized_unused=True,U_truth_used=False,V_used=False,target_access=False,CP=0,NN=0,updates=0))
  x=np.asarray(L['x'].tolist(),dtype=np.float64);profiles,coeff,power=observe(x);del x,L
  cells=np.empty((6,3,5,2));cell_records=[]
  for tx in range(6):
   for z in range(1,4):
    for ri,r in enumerate(d.FIT):
     selected=(y==tx)&(day==z)&(rx==r);v=coeff[selected];cells[tx,z-1,ri]=v.mean(0)
     cell_records.append(dict(TX=tx,day=z,RX=r,count=int(selected.sum()),mean_coeff=v.mean(0).tolist(),variance_coeff=v.var(0).tolist(),mean_log_profile=profiles[selected].mean(0).tolist()))
  np.savez_compressed(out/'source_observations.npz',ids=np.asarray(ids),labels=y,RX=rx,day=day,log_profiles=profiles,coefficients=coeff,packet_power=power)
  d.write(out/'progress.json',dict(observed=6300,source_cells=90,CP=0,NN=0,updates=0,target_access=False))
  result=fit(cells);result.update(status='ANALYZED_SOURCE_STATISTICS_ONLY',config=d.config(),cells=cell_records,power_min=float(power.min()),power_max=float(power.max()),source_cache_loads=1,source_IQ_unique=6300,source_cells=90,source_RX_holdout=None,source_cache_other_roles_features_deserialized_but_unused=True,U_truth_used=False,V_used=False,CP=0,NN=0,new_IQ_views=0,head_fit_updates=0,training_updates=0,teacher_forwards=0,target_access=False,AGC_causality_proved=False,Ag_efficacy_proved=False,As_efficacy_proved=False,joint_goal_complete=False,automatic_training=False,elapsed_seconds=time.perf_counter()-start,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip())
  d.write(out/'result.json',result);d.write(out/'completion.json',{k:result[k] for k in ('status','decision','source_IQ_unique','source_cells','CP','NN','training_updates','target_access','automatic_training')});print('COMPLETE '+result['decision'],flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False));raise
