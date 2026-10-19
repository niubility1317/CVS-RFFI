"""Frozen local default-head predictions; this module never opens scoring truth."""
import os,time
from pathlib import Path
import numpy as np
import torch
from . import design as d,model as api
from experiments.cvs_equivariant_identity.precision import numerical_context,FULL_FP32_POLICY
from cvsrffi.eval import apply_sat_channel_for_scenario
from cvsrffi.tensors import make_torch_generator
from cvsrffi.practical_adapter import set_evaluation_context
COUNT=168000

def observation_ids(profile):
 p=Path(profile['dataset']['target_capsule']);meta=d.read(p/'manifest.json')
 if any(meta.get(k)!=v for k,v in dict(status='VALIDATED_ONCE',count=COUNT,classes=list(d.CLASSES),source_target_disjoint=True,unique_ids=True,channel='residual/post_sync/noeq').items()):raise ValueError('Reuse only matching validated Phase1 observations')
 with np.load(p/'index.npz',allow_pickle=False) as ix:ids=ix['ids'].copy()
 if ids.shape!=(COUNT,) or len(set(ids.tolist()))!=COUNT:raise ValueError('Target ID coverage differs')
 return ids

def source_proof(c,base):
 done=d.read(Path(base)/c['row_id']/'source/completion.json')
 expected=dict(status='SOURCE_FROZEN',config=c,epoch=20,optimizer_steps=d.UPDATES,selection='fixed_final20',target_access=False,truth_access=False,head_fit_updates=0,U_exposure=0)
 if any(done.get(k)!=v for k,v in expected.items()):raise ValueError('Source freeze/provenance invalid')
 return done

def predict(c,profile,base):
 d.validate(c);base=Path(base);source=base/c['row_id']/'source';done=source_proof(c,base)
 if not (base/'source_expansion.json').is_file():raise ValueError('Fix source-only expansion decision before target IQ')
 decision=d.read(base/'source_expansion.json')
 if decision.get('target_access') is not False or decision.get('status')!='SOURCE_EXPANSION_FIXED':raise ValueError('Source-only expansion decision missing')
 out=base/c['row_id']/'prediction';out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(c['model_seed'])
 with numerical_context(FULL_FP32_POLICY):
  cp=torch.load(source/'final.pt',weights_only=False,map_location='cpu')
  expected=dict(config=c,commit=done['commit'],epoch=20,optimizer_steps=d.UPDATES,checkpoint_selection='fixed_final20',source_roles='EXACT_MATCH',scratch_only=True,ancestors=[],target_access=False,default_head_joint_training=True,head_fit_updates=0)
  if any(cp.get(k)!=v for k,v in expected.items()):raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
  m=api.build(c,'cuda:0');m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()];buffers={n:b.clone() for n,b in m.named_buffers()}
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
  if z['tx_logits'].shape!=(2,6) or not torch.isfinite(z['tx_logits']).all():raise ValueError('Real final-checkpoint no-query smoke failed')
  ids=observation_ids(profile);array=np.load(Path(profile['dataset']['target_capsule'])/'clean.npy',mmap_mode='r',allow_pickle=False)
  if array.shape!=(COUNT,2,256) or array.dtype!=np.float32:raise ValueError('Validated clean shape differs')
  args=api.args(c);args.practical_buffer_input=True;args.practical_buffer_output=True;args.practical_execution_fast=False;args.practical_eval_cache_dir=str(base/'practical_view_cache');values={};seconds={};torch.cuda.reset_peak_memory_stats()
  for vi,view in enumerate(d.VIEWS):
   preds=np.empty(COUNT,dtype=np.int64);start=time.perf_counter();generator=make_torch_generator(torch.device('cuda:0'),c['evaluation_seed']+vi*100003)
   with torch.no_grad():
    for begin in range(0,COUNT,256):
     x=torch.from_numpy(np.array(array[begin:begin+256],copy=True)).to('cuda:0')
     if view!='clean':
      set_evaluation_context(ids[begin:begin+len(x)].tolist());x,_=apply_sat_channel_for_scenario(x,view,args,gen=generator)
     if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite registered test observation')
     outp=m.forward_identity_only(x);logits=outp['tx_logits']
     if logits.shape!=(len(x),6) or not torch.isfinite(logits).all() or not torch.isfinite(outp['id_feat_joint']).all():raise FloatingPointError('Invalid frozen prediction')
     preds[begin:begin+len(x)]=logits.argmax(1).cpu().numpy()
   values[view]=preds;torch.cuda.synchronize();seconds[view]=time.perf_counter()-start
   d.write(out/(view+'-complete.json'),dict(status='VIEW_PREDICTED',count=COUNT,view=view,truth_read=False,seconds=seconds[view]))
   print('VIEW_PREDICTED '+c['row_id']+' '+view,flush=True)
  if versions!=[p._version for p in m.parameters()] or any(not torch.equal(v,dict(m.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Frozen prediction updated model state')
  np.savez_compressed(out/'predictions.npz',ids=ids,**values)
  d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,commit=done['commit'],count=COUNT,views=list(d.VIEWS),truth_read=False,query_fit=False,head_fit_updates=0,checkpoint=str(source/'final.pt'),per_view_seconds=seconds,seconds=time.perf_counter()-started,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2,practical_observations='Native practical v3 all6 views from existing cleanIQ; per-physical opaque ID namespace target_fixed_v3; channel seed392005/receiver2027; shared reference implementation cache, buffered IO; legacy3scene satellite.npy unused'))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--profile',required=True);p.add_argument('--base',required=True);a=p.parse_args();c=d.config(next(r for r in d.rows() if r['row_id']==a.row));predict(c,d.read(a.profile),a.base)
