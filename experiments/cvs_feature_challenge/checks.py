import argparse,json
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from . import design as d
from .objective import mask_for,objective,layer_info
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_stage2_decouple.source import seed_all

def rejected(fn):
 try:fn()
 except (ValueError,RuntimeError):return
 raise AssertionError('Illegal input was accepted')
def run(device='cpu'):
 torch.set_num_threads(2)
 for row in d.rows():
  c=d.config(row);d.validate(c);bad=deepcopy(c);bad['source_U']=True;rejected(lambda:d.validate(bad));bad=deepcopy(c);bad['test_views']=['clean','practical_high'];rejected(lambda:d.validate(bad))
 f=torch.tensor([[[1.,1.],[2.,2.],[3.,3.],[4.,4.]]],device=device,requires_grad=True);g=torch.ones_like(f)
 mask,score,k=mask_for(f,g,'GRAD_MASK',1);assert k==1 and mask.flatten().tolist()==[1.,1.,1.,0.] and not mask.requires_grad
 r,_,_=mask_for(f,g,'RANDOM_MASK',77);assert torch.equal(r,mask_for(f,g,'RANDOM_MASK',77)[0]) and int((r==0).sum())==1
 rejected(lambda:mask_for(f,g[:,:,:1],'GRAD_MASK',1));rejected(lambda:mask_for(f,g,'CE30',1))
 outputs=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==d.SEEDS[0]));seed_all(c['model_seed']);m=build(c['source_loader_config'],device);pg,_=groups(m,c['source_loader_config']);params=[p for group in pg for p in group['params']];x=torch.randn(8,2,256,device=device);y=torch.arange(8,device=device,dtype=torch.long)%6
  if any(isinstance(v,nn.modules.batchnorm._BatchNorm) for v in m.modules()):raise ValueError('Double-forward challenge requires explicit BN policy')
  info=layer_info(m,x[:4]);m.train();before=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state(x.device).clone() if x.is_cuda else None;loss,parts,detail=objective(m,x,y,None,None,None,None,c,1111);after=torch.get_rng_state().clone();after_cuda=torch.cuda.get_rng_state(x.device).clone() if x.is_cuda else None;loss.backward()
  assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in params) and all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
  assert detail['challenge_forwards']==int(arm!='CE30') and detail['rank_grad_calls']==int(arm!='CE30')
  hooks=sum(len(mod._forward_hooks) for mod in m.modules());assert hooks==0
  pg[0]['params'][0].grad=None
  # The extra masked forward must leave global RNG at the clean-forward state.
  seed_all(c['model_seed']);torch.set_rng_state(before)
  if x.is_cuda:torch.cuda.set_rng_state(before_cuda,x.device)
  # CPU RNG check is exact for all registered dropout layers.
  m(x);assert torch.equal(torch.get_rng_state(),after)
  if x.is_cuda:assert torch.equal(torch.cuda.get_rng_state(x.device),after_cuda)
  if arm!='CE30':
   middle,_,middle_detail=objective(m,x,y,None,None,None,None,c,1112);middle.backward();assert middle_detail['layer_index']==1 and torch.isfinite(middle)
  assert sum(len(mod._forward_hooks) for mod in m.modules())==0
  rejected(lambda:objective(m,x,y,x,None,None,None,c,1111))
  m.eval();state={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m(x);singles=torch.cat([m(x[i:i+1]) for i in range(len(x))])
  assert torch.allclose(z,singles,atol=1e-4,rtol=1e-4) and all(torch.equal(v,m.state_dict()[k]) for k,v in state.items())
  if arm!='CE30':rejected(lambda:objective(m,x,y,None,None,None,None,c,1111))
  assert sum(len(mod._forward_hooks) for mod in m.modules())==0
  outputs.append(dict(arm=arm,loss=float(loss.detach()),parts=parts,detail=detail,actual_layers=info,parameters=sum(p.numel() for p in params),single_packet_inference=True,query_IQ_or_truth_read=False))
 return dict(status='PASS',rows=9,protocol_negatives=True,mask_exact=True,native_gradients=True,inference_mask_absent=True,results=outputs)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
