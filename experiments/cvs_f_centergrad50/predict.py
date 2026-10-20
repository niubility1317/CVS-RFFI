"""Frozen default-head clean prediction only; never opens truth."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from . import design as d,model as api
from .provenance import source_proof,validate_checkpoint
from experiments.cvs_f_txgeometry20.predict import observation_ids,COUNT
from experiments.cvs_equivariant_identity.precision import numerical_context

def predict(c,profile,base):
 d.validate(c);done=source_proof(c);base=Path(base);decision=d.read(base/'source_expansion.json')
 if decision!=d.fixed_authorization() or d.read(base/'frozen_matrix.json')!=dict(rows=d.rows(),user_fixed_extension=decision,target_feedback=False):raise ValueError('Fixed three-seed matrix required before target')
 out=base/c['row_id']/'prediction';out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  checkpoint=torch.load(Path(c['output_root'])/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(checkpoint,c);model=api.build(c,'cuda:0');model.load_state_dict(checkpoint['model'],strict=True);model.eval().requires_grad_(False);versions=[p._version for p in model.parameters()];buffers={n:v.clone() for n,v in model.named_buffers()}
  with torch.no_grad():smoke=model.forward_identity_only(torch.zeros(2,2,256,device='cuda:0'))
  if smoke['tx_logits'].shape!=(2,6) or not torch.isfinite(smoke['tx_logits']).all():raise ValueError('Actual final50 no-query checkpoint smoke failed')
  ids=observation_ids(profile);array=np.load(Path(profile['dataset']['target_capsule'])/'clean.npy',mmap_mode='r',allow_pickle=False)
  if array.shape!=(COUNT,2,256) or array.dtype!=np.float32:raise ValueError('Existing clean capsule differs')
  pred=np.empty(COUNT,dtype=np.int64);torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():
   for begin in range(0,COUNT,256):
    x=torch.from_numpy(np.array(array[begin:begin+256],copy=True)).to('cuda:0');outp=model.forward_identity_only(x);logits=outp['tx_logits']
    if logits.shape!=(len(x),6) or not torch.isfinite(logits).all() or not torch.isfinite(outp['id_feat_joint']).all():raise FloatingPointError('Invalid default-head clean output')
    pred[begin:begin+len(x)]=logits.argmax(1).cpu().numpy()
  if versions!=[p._version for p in model.parameters()] or any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Frozen clean prediction updated model state')
  np.savez_compressed(out/'predictions.npz',ids=ids,clean=pred);torch.cuda.synchronize();d.write(out/'complete.json',dict(status='PREDICTIONS_COMPLETE',config=c,commit=done['source_commit'],views=['clean'],count=COUNT,truth_read=False,query_fit=False,head_fit_updates=0,seconds=time.perf_counter()-start,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2,checkpoint='same-row fixed_final50 student, default joint head',satellite_test=False))
 print('CLEAN_PREDICTED '+c['row_id'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);p.add_argument('--profile',required=True);p.add_argument('--metadata',required=True);p.add_argument('--base',required=True);a=p.parse_args();predict(d.config(next(r for r in d.rows() if r['row_id']==a.row),a.base,a.profile,a.metadata),d.read(a.profile),a.base)
