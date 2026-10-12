import argparse,json
from pathlib import Path
import torch
from . import design as d
from .model import build,groups,assignments
def run(device='cpu'):
 torch.set_num_threads(2);records=[];fit=dict(ids=[str(i) for i in range(5040)],rx=torch.arange(4).repeat_interleave(1260))
 for field,value in (('source_U',True),('training_target_access',True),('updates',6661),('held_rx',9)):
  bad=dict(d.config(d.rows()[0]),**{field:value})
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Changed protocol/budget accepted '+field)
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm));a=assignments(c,fit);m=build(c,None,device);pg,names=groups(m,c);params=[p for g in pg for p in g['params']];x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;assigned=torch.arange(128,device=device)%4
  logits,heads=m.training_logits(x);aux=heads[torch.arange(128,device=device),assigned];loss=torch.nn.functional.cross_entropy(logits,y)+.25*torch.nn.functional.cross_entropy(aux,y);loss.backward()
  if not torch.isfinite(loss) or not torch.isfinite(m.experts.grad).all() or any(not torch.isfinite(p.grad).all() for p in params if p.grad is not None):raise ValueError('Actualnative+expert gradient differs')
  m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m(x[:8]);single=torch.cat([m(t[None]) for t in x[:8]]);g,h=m.training_logits(x[:8]);expected=.5*g.softmax(1)+.5*h.softmax(-1).mean(1)
  if not torch.allclose(z.exp(),expected,atol=1e-6) or not torch.allclose(z,single,atol=1e-4,rtol=1e-4) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Fixed queryindependent formula differs')
  records.append(dict(arm=arm,trainable_parameters=sum(p.numel() for p in params),head_partition_counts=a.bincount(minlength=4).tolist(),headloss_gradient_nonzero=int((m.experts.grad.abs().sum((1,2))>0).sum())))
 if any(r['trainable_parameters']!=180865 for r in records):raise ValueError('Matchedheadparametercount differs')
 return dict(status='PASS',records=records,query_RX_needed=False,source_only=True,expertgradient=True,formula=True,protocol_negative_cases=4)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
