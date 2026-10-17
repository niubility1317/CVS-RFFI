"""Use only original full L IQ;zero model/checkpoint/query access."""
import gzip,json,os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import compare,summarize

def run():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Source physical diagnostic is CPU only')
 import torch
 from leo_practical.channel import Config,receiver_for_session,stable_seed
 import leo_practical.channel as ch
 import leo_practical.residual as residual
 torch.set_num_threads(2);c=d.config();d.validate(c);rt=d.read(d.ROOT/'runtime.json');out=d.resolve(d.BASE,rt);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  prep=d.resolve(d.PREPARATION,rt);contract=d.read(d.resolve(d.SOURCE,rt));prepared=d.read(prep/'preparation_complete.json')
  if prepared.get('status')!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or prepared.get('target_access') is not False or prepared.get('identity_updates')!=0:raise ValueError('Existing preparation not legal source data')
  packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False)
  for actual in (packed['contract'],prepared['source_contract']):
   if any(actual.get(k)!=v for k,v in contract.items()) or actual.get('native_role_comparison')!='EXACT_MATCH':raise ValueError('Source contract not exact')
  L={k:packed['labeled'][k] for k in ('x','y','rx','day','ids')};del packed
  if len(L['ids'])!=6300 or len(set(L['ids']))!=6300 or set(L['ids'])!=set(contract['role_ids']['L_s']) or set(L['ids'])&(set(contract['role_ids']['U_s'])|set(contract['role_ids']['V'])):raise ValueError('All and only original L6300 required')
  x=np.asarray(L['x'].tolist(),dtype=np.float64);tx,rx,day=(np.asarray(L[k].tolist(),dtype=np.int64) for k in ('y','rx','day'))
  if x.shape!=(6300,2,256) or not np.isfinite(x).all() or set(rx)!=set(d.FIT) or set(day)!={1,2,3} or set(tx)!=set(range(6)):raise ValueError('Full original source shape/roles differ')
  configs={s:Config(fs_hz=c['fs_hz'],fc_hz=c['fc_hz'],scenario=s,processing_route='residual',mode='post_sync',equalization_enabled=False,reference_snr_db=30.,boundary='reflect') for s in d.SCENES}
  d.write(out/'resolved_config.json',dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),channel_module=ch.__file__,residual_module=residual.__file__,source_L=6300,source_RX_holdout=None,source_cache_other_roles_features_deserialized_but_unused=True,checkpoint_loads=0,network_forwards=0,U_truth_read=False,target_access=False))
  observations=[];sessions={}
  with gzip.open(out/'physical_pairs.jsonl.gz','xt',encoding='utf8',compresslevel=3) as stream:
   for i,pid in enumerate(L['ids']):
    session=f'rx{int(rx[i])}:day{int(day[i])}:eq1'
    for scene,cfg in configs.items():
     seed=stable_seed(d.CHANNEL_SEED,'F/practical-quality/source-L',scene,pid);hw=receiver_for_session(cfg,d.RECEIVER_SEED,session)
     if session in sessions and sessions[session]!=hw:raise ValueError('Hardware session must not depend on TX')
     sessions[session]=hw;r=compare(x[i,0]+1j*x[i,1],cfg,seed,hw);r.update(physical_id=pid,RX=int(rx[i]),TX=int(tx[i]),day=int(day[i]),scene=scene);observations.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n')
    if (i+1)%100==0:
     stream.flush();d.write(out/'progress.json',dict(source_packets=i+1,physical_pairs=len(observations),seconds=time.perf_counter()-started));print('SOURCE_PROGRESS',i+1,len(observations),flush=True)
  result=summarize(observations);d.write(out/'result.json',result);d.write(out/'completion.json',dict(status='ANALYZED',decision=result['decision'],physical_pairs=len(observations),residual_channel_calls=2*len(observations),source_L_unique=6300,source_cache_loads=1,model_constructed=False,checkpoint_loads=0,NN=0,training_updates=0,teacher_forwards=0,head_fit_updates=0,U_truth_read=False,V_statistically_used=False,target_access=False,seconds=time.perf_counter()-started));print('SOURCE_PHYSICAL_ANALYSIS_COMPLETE',result['decision'],flush=True)
 except Exception as exc:d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False));raise
