"""One immutable CPU source-distribution readout,no model/IQ augmentation."""
import os,sys,time
from pathlib import Path
import numpy as np
from . import design as d
from .core import observe,cube,evaluate
from .permissions import select

def run():
 if len(sys.argv)!=1 or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fixed CPU source entry required')
 c=d.config();d.validate(c);runtime=d.read(d.ROOT/'runtime.json');project=runtime['project'];out=d.resolve(c['output_root'],project);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();d.write(out/'active.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,owner=d.OWNER))
 try:
  from .checks import run as checks
  d.write(out/'checks.json',checks());print('SMOKE PASS radial algebra/source roles;no model or checkpoint consumed',flush=True)
  import torch
  torch.set_num_threads(2);contract=d.read(d.resolve(d.SOURCE,project));prep=d.resolve(d.PREPARATION,project);ready=d.read(prep/'preparation_complete.json');packed=torch.load(prep/'source_cache.pt',map_location='cpu',weights_only=False);selected=select(packed,ready,contract);del packed
  arrays={};metadata={};means={};cells={};quality={};max_floor=0.;zeros=0
  for role,per_cell in [('L_s',70),('V',300)]:
   r=selected.pop(role);meta={k:r[k].tolist() for k in ('y','rx','day')};meta['ids']=list(r['ids']);metadata[role]=meta;chunks=[]
   for first in range(0,len(r['x']),256):chunks.append(observe(np.array(r['x'][first:first+256].tolist(),dtype=np.float64)))
   obs={k:np.concatenate([v[k] for v in chunks]) for k in chunks[0]};del chunks,r;arrays[role]=obs;means[role],cells[role]=cube(obs['shape'],meta,per_cell);cell_quality=[]
   for cell in cells[role]:
    mask=(np.array(meta['y'])==cell['TX'])&(np.array(meta['rx'])==cell['RX'])&(np.array(meta['day'])==cell['day']);frac=float(obs['amplitude_floor_fraction'][mask].mean());cell_quality.append(dict(**cell,amplitude_floor_fraction=frac));max_floor=max(max_floor,frac)
   nz=int(obs['zero_RMS'].sum());zeros+=nz;quality[role]=dict(zero_RMS_packets=nz,floor_fraction_quantiles=np.quantile(obs['amplitude_floor_fraction'],[0,.1,.5,.9,1]).tolist(),cells=cell_quality,packets=len(obs['shape']),all_packets_retained=True)
  result,fits=evaluate(means['L_s'],means['V'],dict(max_floor_fraction_each_cell=max_floor,zero_RMS_packets=zeros));result.update(status='SOURCE_RADIAL_QUANTILE_READOUT_COMPLETE',run_id=d.RUN,source_L_unique=6300,source_V_unique=27000,all_source_RX=list(d.RX),source_RX_holdout=None,cells=cells,quality=quality,max_floor_fraction_each_cell=max_floor,zero_RMS_packets=zeros,L_V_physical_disjoint=True,U_used=False,U_truth_read=False,V_updates=0,head_fit_updates=0,CP=0,NN=0,updates=0,teacher_NN=0)
  d.write(out/'resolved_config.json',dict(config=c,execution_commit=(d.ROOT/'release_commit.txt').read_text().strip(),source_roles_exact_match=True,checkpoint_access=False,network_constructed=False,target_access=False,unused_U_features_deserialized_not_consumed=True));d.write(out/'source_ids.json',metadata);np.savez_compressed(out/'observations.npz',**{role+'_'+k:v for role,obs in arrays.items() for k,v in obs.items()});np.savez_compressed(out/'cell_quantiles.npz',**means);np.savez_compressed(out/'fitted_readout.npz',**fits);d.write(out/'result.json',result)
  d.write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE_SOURCE_ONLY',run_id=d.RUN,decision=result['decision'],L=6300,V=27000,source_RX=list(d.RX),source_RX_holdout=None,source_cells_per_role=90,folds=18,unique_IQ_observations=33300,source_cache_loads=1,packet_quantile_values=199800,cell_quantile_values=1080,L_only_scalar_fits=144,L_only_additive_vectors=72,IQ_modified_for_training=False,IQ_views_generated=0,CP=0,NN=0,teacher_NN=0,updates=0,head_fit_updates=0,U_used=False,U_truth_read=False,V_updates=0,target_access=False,automatic_training=False,joint_goal_complete=False,wall_seconds=time.perf_counter()-start));print('SOURCE_RADIAL_COMPLETE '+result['decision'],flush=True)
 except Exception as exc:
  d.write(out/'failure.json',dict(status='FAILED',error=repr(exc),automatic_retry=False));raise
if __name__=='__main__':run()
