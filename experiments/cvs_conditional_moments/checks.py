import argparse,json
from copy import deepcopy
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .alignment import capture,penalty

def run(device='cpu'):
 torch.set_num_threads(2);records=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));model=build(c,None,device);pg,_=groups(model,c);params=[p for g in pg for p in g['params']];torch.manual_seed(57)
  x=torch.randn(144,2,256,device=device);y=torch.arange(144,device=device)%6;rx=torch.tensor([3,4,6,8],device=device).repeat_interleave(36)
  with capture(model) as maps:z=model(x)
  align,detail=penalty(maps,y,rx,c);grads=torch.autograd.grad(align,params,allow_unused=True,retain_graph=True)
  if not torch.isfinite(align) or not align>0 or not any(g is not None and g.abs().sum()>0 for g in grads):raise AssertionError('Actual shallow/middle alignment gradient missing')
  weight_index=next(i for i,p in enumerate(params) if p is model.head.weight)
  if grads[weight_index] is not None:raise AssertionError('Moment penalty must not train classifier weights')
  perm=torch.randperm(len(y),device=device);pa,_=penalty({k:h[perm] for k,h in maps.items()},y[perm],rx[perm],c)
  lp,_=penalty(maps,(y+1)%6,rx,c)
  if not torch.allclose(align,pa,atol=1e-6) or not torch.allclose(align,lp,atol=1e-6):raise AssertionError('Packet permutation/allclass symmetry differs')
  loss=F.cross_entropy(z,y)+.1*align;loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params) or not model.head.weight.grad.abs().sum()>0:raise AssertionError('Cos CE or source gradient differs')
  if sum(p.numel() for p in params)!=177025:raise AssertionError('Training mechanism changed model/head capacity')
  model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
  with torch.no_grad():
   out=model.forward_identity_only(x[:8]);joint=out['id_feat_joint'];expected=30*F.linear(F.normalize(joint,dim=1,eps=1e-4),F.normalize(model.head.weight,dim=1,eps=1e-4));single=torch.cat([model(t[None]) for t in x[:8]])
   refit=30*F.linear(F.normalize(joint,dim=1,eps=1e-4),F.normalize(torch.randn(9,160,device=device),dim=1,eps=1e-4))
  if not torch.equal(expected,out['tx_logits']) or not torch.allclose(single,out['tx_logits'],rtol=1e-4,atol=1e-4) or refit.shape!=(8,9) or any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise AssertionError('Ordinarycos/refit interface/perpacket inference differs')
  if any(m._forward_hooks for m in model.modules()):raise AssertionError('Training feature hooks persisted into inference')
  bad=deepcopy(c);bad['classifier']['routing']=True
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Forbidden classifier routing accepted')
  for field,value in [('epochs',100),('source_U',True),('held_rx',0),('training_target_access',True)]:
   bad=deepcopy(c);bad[field]=value
   try:d.validate(bad)
   except ValueError:pass
   else:raise AssertionError('Illegal source config accepted')
  try:penalty(maps,y,torch.full_like(rx,c['held_rx']),c)
  except ValueError:pass
  else:raise AssertionError('Held RX fitted')
  try:
   with capture(model):raise RuntimeError('check exception cleanup')
  except RuntimeError:pass
  if any(m._forward_hooks for m in model.modules()):raise AssertionError('Exception leaked hooks')
  records.append(dict(arm=arm,parameters=177025,alignment=float(align),layers=detail,ordinarycos=True,refit9class=True,head_penalty_gradient=False))
 return dict(status='PASS',records=records,negative_cases=12,eval_packet_independence=True,no_persistent_alignment_state=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
