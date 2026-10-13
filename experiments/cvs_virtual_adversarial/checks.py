import argparse,json
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from . import design as d
from .objective import objective,consistency,layer_info,packet_norm,unit,scaled,divergence
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_stage2_decouple.source import seed_all

def rejected(fn):
 try:fn()
 except (ValueError,RuntimeError):return
 raise AssertionError('Illegal input accepted')
def run(device='cpu'):
 torch.set_num_threads(2)
 for row in d.rows():
  c=d.config(row);d.validate(c);bad=deepcopy(c);bad['vat']['epsilon_relative']=.5;rejected(lambda:d.validate(bad));bad=deepcopy(c);bad['semi']['confidence_gate']=True;rejected(lambda:d.validate(bad))
 t=torch.randn(8,2,256,device=device);direction=unit(t);delta=scaled(direction,t,.02);assert torch.allclose(packet_norm(delta)/packet_norm(t),torch.full((8,),.02,device=device,dtype=torch.float64),rtol=1e-5,atol=1e-6)
 z=torch.randn(8,6,device=device);q=z.double().softmax(1).detach();assert abs(float(divergence(z,q)))<1e-12
 outputs=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==d.SEEDS[0]));seed_all(c['model_seed']);model=build(c['source_loader_config'],device);pg,_=groups(model,c['source_loader_config']);params=[p for group in pg for p in group['params']]
  assert not any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules())
  x=torch.randn(8,2,256,device=device);y=torch.arange(8,device=device,dtype=torch.long)%6;ux=torch.randn_like(x) if c['source_U'] else None;ids=[f'U-fixture-{i}' for i in range(8)] if ux is not None else None;info=layer_info(model,x[:4]);model.train()
  loss,parts,detail=objective(model,x,y,ux,ids,None,None,c,1111);assert torch.isfinite(loss);loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params);assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
  assert detail['U_packets']==(8 if c['source_U'] else 0) and detail['search_grad_calls']==int(c['source_U'])
  if c['source_U']:
   for p in params:p.grad=None
   modes=[m.training for m in model.modules()];cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(device).clone() if x.is_cuda else None
   first,dd=consistency(model,ux,c,1112);assert all(p.grad is None for p in params);assert [m.training for m in model.modules()]==modes and torch.equal(cpu,torch.get_rng_state())
   if x.is_cuda:assert torch.equal(cuda,torch.cuda.get_rng_state(device))
   again,dd2=consistency(model,ux,c,1112);assert torch.allclose(first,again,rtol=1e-5,atol=1e-8);first.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in params)

   assert abs(dd['perturbation_ratio_mean']-.02)<1e-6 and dd['perturbation_ratio_max']<=.020001
   if arm=='VAT_FEATURE':assert detail['layer_index']==0 and dd['layer_index']==1
   rejected(lambda:consistency(model,ux,c,1110));rejected(lambda:objective(model,x,y,ux,ids,None,object(),c,1111))
  else:rejected(lambda:objective(model,x,y,x,[str(i) for i in range(8)],None,None,c,1111))
  assert sum(len(m._forward_hooks) for m in model.modules())==0
  model.eval();state={k:v.clone() for k,v in model.state_dict().items()}
  with torch.no_grad():logits=model(x);individual=torch.cat([model(x[i:i+1]) for i in range(len(x))])
  assert torch.allclose(logits,individual,atol=1e-4,rtol=1e-4) and all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
  rejected(lambda:objective(model,x,y,ux,ids,None,None,c,1111))
  outputs.append(dict(arm=arm,loss=float(loss.detach()),parts=parts,detail=detail,actual_layers=info,parameters=sum(p.numel() for p in params)))
 return dict(status='PASS',rows=12,legal_U_interface=True,detached_search=True,teacher_no_gradient=True,native_identity_gradient=True,bounded_per_packet_radius=True,inference_training_hooks_absent=True,query_IQ_or_truth_read=False,results=outputs)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 from experiments.cvs_equivariant_identity.precision import numerical_context
 with numerical_context(d.FULL_FP32_POLICY):v=run(a.device)
 d.write(a.output,v);print(json.dumps(v))
