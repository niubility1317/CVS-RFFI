import argparse,json
from copy import deepcopy
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .average import MeanParameters
from experiments.cvs_nocfo_foundation.model import build,groups
def run(device='cpu'):
 torch.set_num_threads(2);c=d.config(d.rows()[0]);m=build(c['source_loader_config'],device);pg,_=groups(m,c['source_loader_config']);avg=MeanParameters(m,pg);params=dict(m.named_parameters());snapshots=[]
 if any('BatchNorm' in type(v).__name__ for v in m.modules()):raise ValueError('NoBN SWA30 protocol changed')
 for k in range(10):
  with torch.no_grad():
   for name in avg.names:params[name].add_(.00001)
  snapshots.append(params[avg.names[0]].detach().clone());avg.update(m)
 state=avg.state(m)
 if len(avg.names)!=67 or not torch.allclose(state[avg.names[0]],torch.stack(snapshots).mean(0),atol=1e-7):raise AssertionError('Same trajectory arithmetic mean')
 m.load_state_dict(state,strict=True);x=torch.randn(128,2,256,device=device);y=torch.arange(128,device=device)%6;loss=F.cross_entropy(m(x),y);loss.backward()
 if not torch.isfinite(loss):raise ValueError('Actual SWA model gradient')
 m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():
  out=m.forward_identity_only(x[:8]);expected=30*F.linear(F.normalize(out['id_feat_joint'],dim=1,eps=1e-4),F.normalize(m.head.weight,dim=1,eps=1e-4));single=torch.cat([m(t[None]) for t in x[:8]]);refit=30*F.linear(F.normalize(out['id_feat_joint'],dim=1,eps=1e-4),F.normalize(torch.randn(9,160,device=device),dim=1,eps=1e-4))
 if not torch.equal(expected,out['tx_logits']) or not torch.allclose(single,out['tx_logits'],atol=1e-4,rtol=1e-4) or refit.shape!=(8,9) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise AssertionError('Cos/refit/no query state')
 for k,v in [('epochs',100),('source_U',True),('training_target_access',True),('held_rx',0)]:
  bad=deepcopy(c);bad[k]=v
  try:d.validate(bad)
  except ValueError:pass
  else:raise AssertionError('Invalid SWA source config')
 bad=MeanParameters(m,pg)
 try:bad.state(m)
 except ValueError:pass
 else:raise AssertionError('Missing snapshots accepted')
 return dict(status='PASS',parameters=sum(p.numel() for g in pg for p in g['params']),snapshots=10,average_tensors=len(avg.names),BN=False,ordinarycos=True,refit9class=True,negative_cases=5,learning_rates=[d.learning_rate(e) for e in range(1,31)])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run(a.device);d.write(a.output,v);print(json.dumps(v))
