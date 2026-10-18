"""Recover only JSON aggregates from preserved source statistics;never reopens IQ."""
import json,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import evaluate

def run(base,commit):
 start=time.perf_counter();B=Path(base);out=B/'json_recovery_v1'
 failure=d.read(B/'failure.json')
 if failure.get('status')!='FAILED' or 'not JSON serializable' not in failure.get('error',''):raise ValueError('Only original documented JSON failure is recoverable here')
 resolved=d.read(B/'resolved_config.json');d.validate(resolved['config'])
 if resolved['target_access'] or resolved['checkpoint_access'] or resolved['network_constructed']:raise ValueError('Only original legal source-only saved statistics permitted')
 out.mkdir(exist_ok=False)
 from .checks import run as checks
 d.write(out/'checks.json',checks())
 meta=d.read(B/'source_ids.json');floor=0.;zeros=0
 with np.load(B/'observations.npz',allow_pickle=False) as obs:
  for role,n,percell in [('L_s',6300,70),('V',27000,300)]:
   q=obs[role+'_shape'];f=obs[role+'_amplitude_floor_fraction'];z=obs[role+'_zero_RMS'];m=meta[role]
   if q.shape!=(n,6) or f.shape!=(n,) or z.shape!=(n,) or not np.isfinite(q).all() or not np.isfinite(f).all():raise ValueError('Saved complete source statistics required')
   for t in range(6):
    for rx in d.RX:
     for day in (1,2,3):
      idx=[i for i in range(n) if (m['y'][i],m['rx'][i],m['day'][i])==(t,rx,day)]
      if len(idx)!=percell:raise ValueError('All original source cells required')
      floor=max(floor,float(f[idx].mean()))
   zeros+=int(z.sum())
 with np.load(B/'cell_quantiles.npz',allow_pickle=False) as cells:result,fits=evaluate(cells['L_s'],cells['V'],dict(max_floor_fraction_each_cell=floor,zero_RMS_packets=zeros))
 with np.load(B/'fitted_readout.npz',allow_pickle=False) as saved:
  if set(saved.files)!=set(fits) or any(np.max(np.abs(saved[k]-fits[k]))>1e-12 for k in fits):raise ValueError('Recovered fits differ from pre-failure saved actual fits')
 result.update(status='SOURCE_RADIAL_QUANTILE_READOUT_RECOVERED',run_id=d.RUN,max_floor_fraction_each_cell=floor,zero_RMS_packets=zeros,source_L_unique=6300,source_V_unique=27000,all_source_RX=list(d.RX),source_RX_holdout=None,L_V_physical_disjoint=True,U_used=False,U_truth_read=False,V_updates=0,head_fit_updates=0,CP=0,NN=0,updates=0,teacher_NN=0,target_access=False,original_failure_preserved=True,recovery_commit=commit)
 d.write(out/'result.json',result)
 completion=dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY_RECOVERED',decision=result['decision'],original_execution_commit=resolved['execution_commit'],recovery_commit=commit,original_failure_preserved=True,IQ_reads=0,CP=0,NN=0,updates=0,teacher_NN=0,target_access=False,U_used=False,original_source_IQ_observations=33300,source_cache_loads_original=1,source_cache_loads_recovery=0,packet_quantile_values=199800,cell_quantile_values=1080,scalar_fits_recomputed=144,additive_vectors_recomputed=72,original_fits_exactly_reproduced=True,training_IQ_views=0,automatic_training=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start)
 d.write(out/'completion.json',completion)
 return dict(result=result,completion=completion)
