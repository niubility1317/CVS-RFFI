"""Frozen, row-shared Phase1 weak-channel observations; no truth or fitting."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from experiments.cvs_state_test_now import design as d

WEAK=('leo_clear_weak','leo_low_elev_weak','leo_rain_weak')
ROOT=d.BASE/'weak_views'

def channel_batch(x,scenario,generator):
 from sat_channel import SatSimConfig,apply_sat_gnd_channel_batch
 from training_controls import sat_channel_config_for_scenario
 cfg=SatSimConfig(**dict(sat_channel_config_for_scenario(scenario),fs_hz=25000000.,fc_hz=2462000000.))
 return apply_sat_gnd_channel_batch(x,cfg,gen=generator,return_meta=False)[0],asdict(cfg)

def build():
 from experiments.cvs_state_test_now.evaluate import physical_ids
 # Called only after all source checkpoints are frozen. IDs never connect truth.
 if d.read(d.BASE/'source_matrix_frozen.json')['status']!='ALL_SOURCE_FROZEN':raise ValueError('Source not frozen')
 ids=physical_ids(d);raw=np.load(Path(d.CAPSULE)/'clean.npy',mmap_mode='r',allow_pickle=False)
 if raw.shape!=(len(ids),2,256):raise ValueError('Clean array shape differs')
 ROOT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);started=time.perf_counter();configs={}
 for j,scene in enumerate(WEAK):
  gen=torch.Generator().manual_seed(2026100912+j)
  arr=np.lib.format.open_memmap(ROOT/(scene+'.npy'),mode='w+',dtype=np.float32,shape=raw.shape)
  for i in range(0,len(ids),256):
   a=np.ascontiguousarray(raw[i:i+256],dtype=np.float32)
   x=torch.frombuffer(bytearray(a.tobytes()),dtype=torch.float32).reshape(a.shape)
   y,cfg=channel_batch(x,scene,gen)
   if not torch.isfinite(y).all():raise ValueError('Nonfinite weak observation')
   arr[i:i+len(x)]=np.asarray(y.tolist(),dtype=np.float32)
  arr.flush();configs[scene]=cfg;del arr
 np.savez(ROOT/'index.npz',ids=ids)
 d.write(ROOT/'manifest.json',dict(status='VALIDATED_ONCE',count=len(ids),scenes=list(WEAK),
  seed_by_scene={s:2026100912+j for j,s in enumerate(WEAK)},configs=configs,batch_size=256,
  physical_ids_ref=d.CAPSULE+'/index.npz',source_capsule=d.CAPSULE,
  rows_share_observations=True,truth_read=False,query_fit=False,
  scope='Phase1 independent view stress tests, not Phase2 support/query capsule',elapsed_seconds=time.perf_counter()-started))

def verified():
 from experiments.cvs_state_test_now.evaluate import physical_ids
 m=d.read(ROOT/'manifest.json');ids=physical_ids(d)
 with np.load(ROOT/'index.npz',allow_pickle=False) as ix:
  if not np.array_equal(ix['ids'],ids):raise ValueError('Weak IDs differ')
 if m['status']!='VALIDATED_ONCE' or m['count']!=len(ids) or m['scenes']!=list(WEAK) or m['truth_read']:
  raise ValueError('Weak view manifest differs')
 return m
