import json
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from experiments.cvs_e20_cross_style.style import _paths

def run(device='cpu'):
 torch.set_num_threads(2);records=[];generator=torch.Generator().manual_seed(3026);x=torch.randn(4,2,256,generator=generator).to(device);y=torch.tensor([0,1,2,3],device=device);reference=None
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));model=build(c,device);pg,names=groups(model,c);model.eval()
  modules,paths=_paths(model);identity=modules[paths['time_down'].rsplit('.',1)[0]];outputs=[]
  handle=identity.t3.register_forward_hook(lambda m,i,o:outputs.append(tuple(o.shape)))
  z=model.forward_identity_only(x);handle.remove()
  if z['id_feat_joint'].shape!=(4,160) or z['tx_logits'].shape!=(4,6) or model.head.weight.shape!=(6,160) or model._identity_input_policy['explicit_CFO_identity_input']:raise ValueError('Sampling identity160/ordinarycos interface')
  expected_length=64 if arm=='CARRY_POINT_GN' else 32
  if outputs!=[(4,96,expected_length)]:raise ValueError('Actual sampling time layout differs')
  F.cross_entropy(z['tx_logits'],y).backward()
  if identity.t3.dw.weight.grad is None or not torch.isfinite(identity.t3.dw.weight.grad).all() or identity.t3.dw.weight.grad.norm()==0:raise ValueError('Replacement identity DW not reachable byCE')
  trainable={id(p) for p in model.parameters() if p.requires_grad};grouped=[p for group in pg for p in group['params']]
  if trainable!={id(p) for p in grouped} or len(grouped)!=len({id(p) for p in grouped}):raise ValueError('Sampling optimizer parameter mismatch')
  model.zero_grad(set_to_none=True);saved={k:v.clone() for k,v in model.state_dict().items()};fresh=build(c,device);fresh.load_state_dict(saved,strict=True);fresh.eval()
  with torch.no_grad():restored=fresh(x)
  if not torch.equal(z['tx_logits'].detach(),restored):raise ValueError('Strict sampling checkpoint roundtrip differs')
  records.append(dict(arm=arm,t3_shape=list(outputs[0]),parameters=sum(p.numel() for p in grouped),identity_CE_reaches_t3=True,ordinarycos160=True,source_U=False,V_forward=False))
 return dict(status='PASS',records=records,target_access=False,checkpoint_read=False,strict_state_roundtrip=True,optimizer_parameters_exact=True)

if __name__=='__main__':print(json.dumps(run()))
