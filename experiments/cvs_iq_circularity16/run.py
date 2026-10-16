"""CPU-only source association assay; no parameter/model/checkpoint construction."""
import os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import observe,crossfit

def run():
 if len(sys.argv)!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('No overrides;CPU only')
 c=d.config();d.validate(c);out=Path(d.BASE);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  from .checks import run as checks
  checked=checks();d.write(out/'synthetic_checks.json',checked);print('SMOKE PASS synthetic circularity/roles;0NN;no CP applies to G0',flush=True)
  import torch
  torch.set_num_threads(2)
  prepared=d.read(Path(d.PREPARATION)/'preparation_complete.json')
  if prepared.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared.get('target_access') is not False or prepared.get('identity_updates')!=0:raise ValueError('Existing validated source preparation differs')
  packed=torch.load(Path(d.PREPARATION)/'source_cache.pt',map_location='cpu',weights_only=False)
  contract=d.read(d.SOURCE)
  for actual in (packed['contract'],prepared['source_contract']):
      if any(actual.get(k)!=v for k,v in contract.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Existing source role contract differs')
  L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
  ids=set(L['ids'])
  if len(ids)!=6300 or len(L['ids'])!=6300 or ids!=set(contract['role_ids']['L_s']) or ids&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('Only legal original L6300 can supply fit-RX observations')
  mask=(L['rx']!=1)
  # Filter before IQ conversion/statistics; held RX1 is not used to infer a DC bank.
  x=np.asarray(L['x'][mask].tolist(),dtype=np.float64)
  tx,rx,day=(np.asarray(L[k][mask].tolist(),dtype=np.int64) for k in ('y','rx','day'))
  fit_ids=[identity for identity,keep in zip(L['ids'],mask.tolist()) if keep];del L
  if len(fit_ids)!=5040 or len(set(fit_ids))!=5040:raise ValueError('Exact four-RX legal L fit set required')
  print('INPUT PASS fit L5040;heldRX1/U/V unused;no checkpoint/model/target',flush=True)
  d.write(out/'resolved_config.json',dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),python=sys.executable,torch_version=torch.__version__,numpy_version=np.__version__,source_roles_exact_match=True,fit_L_unique=5040,fit_RX=sorted(set(rx.tolist())),held_RX1_used=False,old_cache_other_roles_and_features_deserialized_but_never_used=True,network_constructed=False,checkpoint_access=False,target_constructed=False,core_file=__import__('experiments.cvs_iq_circularity16.core',fromlist=['core']).__file__))
  observations=observe(x);result=crossfit(observations,tx,rx,day)
  result.update(IQ_packets_statistically_used=5040,held_RX1_used=False,network_forwards=0,optimizer_steps=0,source_only=True,target_access=False,automatic_training=False)
  d.write(out/'circularity_crossfit.json',result);d.write(out/'circularity_cells.json',result['cells']);d.write(out/'fit_L_ids.json',fit_ids)
  np.savez_compressed(out/'observations.npz',**observations,TX=tx,RX=rx,day=day)
  complete=dict(status='ANALYZED',scientific_decision=result['decision'],fit_L_packets=5040,source_cells=72,folds=18,wrongRX_controls=9,TX_half_partitions=10,network_forwards=0,checkpoint_loads=0,optimizer_steps=0,U_used=False,V_used=False,held_RX1_used=False,target_access=False,automatic_training=False,seconds=time.perf_counter()-start)
  d.write(out/'completion.json',complete);print('COMPLETE '+str(complete),flush=True)
 except Exception as e:
  d.write(out/'failure.json',dict(status='FAILED',error=repr(e),no_automatic_retry=True));raise
if __name__=='__main__':run()
