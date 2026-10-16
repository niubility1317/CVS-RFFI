"""Read the existing source carrier;use only fitL for zero-model statistics."""
import os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import observe,analyze

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only source diagnostic required')
 import torch
 torch.set_num_threads(2);c=d.config();d.validate(c);runtime=d.read(d.ROOT/'runtime.json')
 if runtime['run_id']!=d.RUN:raise ValueError('Runtime belongs to another run')
 out=d.resolve(d.BASE,runtime);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  preparation=d.resolve(d.PREPARATION,runtime);contract=d.read(d.resolve(d.SOURCE,runtime));prepared=d.read(preparation/'preparation_complete.json')
  if prepared.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared.get('target_access') is not False or prepared.get('identity_updates')!=0:raise ValueError('Existing source preparation differs')
  packed=torch.load(preparation/'source_cache.pt',map_location='cpu',weights_only=False)
  for actual in (packed['contract'],prepared['source_contract']):
   if any(actual.get(k)!=v for k,v in contract.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source role handles differ')
  L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
  ids=set(L['ids'])
  if len(ids)!=6300 or len(L['ids'])!=6300 or ids!=set(contract['role_ids']['L_s']) or ids&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('Original authorized L6300 role required')
  mask=L['rx']!=1;x=np.asarray(L['x'][mask].tolist(),dtype=np.float64);tx,rx,day=(np.asarray(L[k][mask].tolist(),dtype=np.int64) for k in ('y','rx','day'));fit_ids=[i for i,keep in zip(L['ids'],mask.tolist()) if keep];del L
  if len(fit_ids)!=5040 or len(set(fit_ids))!=5040 or set(rx)!=set(d.FIT):raise ValueError('ExactlyfitL5040 fourRX required')
  print('INPUT PASS originalsource roles;fitL5040 only statistically used;no model/checkpoint/target',flush=True)
  d.write(out/'resolved_config.json',dict(config=c,executed_commit=(d.ROOT/'release_commit.txt').read_text().strip(),python=sys.executable,torch_version=torch.__version__,numpy_version=np.__version__,source_roles_exact_match=True,source_carrier_L_deserialized=6300,fit_L_unique=5040,source_cache_other_roles_and_features_deserialized_but_unused=True,U_truth_read=False,held_RX1_statistically_used=False,model_constructed=False,checkpoint_loads=0,target_constructed=False))
  saved=x.copy();observation=observe(x)
  if not np.array_equal(x,saved):raise ValueError('Source IQ modified during statistics')
  result=analyze(observation,tx,rx,day);d.write(out/'cycle_covariance.json',result);d.write(out/'fit_L_ids.json',fit_ids)
  d.write(out/'packet_observations.json',dict(shape=observation['shape'].tolist(),coherence=observation['coherence'].tolist(),residual_ratio=observation['residual_ratio'].tolist(),zero_residual=observation['zero_residual'].tolist(),TX=tx.tolist(),RX=rx.tolist(),day=day.tolist()))
  completion=dict(status='ANALYZED',scientific_decision=result['decision'],fit_L_packets=5040,source_cells=72,folds=18,network_forwards=0,optimizer_steps=0,checkpoint_loads=0,head_fit_updates=0,model_constructed=False,U_truth_read=False,U_statistically_used=False,V_statistically_used=False,held_RX1_statistically_used=False,target_access=False,IQ_modified=False,automatic_training=False,Ag_efficacy_proved=False,joint_goal_complete=False,seconds=time.perf_counter()-start)
  d.write(out/'completion.json',completion);print('COMPLETE '+str(completion),flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),no_automatic_retry=True));raise
if __name__=='__main__':run()
