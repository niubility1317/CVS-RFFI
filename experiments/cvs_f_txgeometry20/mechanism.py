"""Same-IQ RDrop plus source-only metric learning on actual deployed features."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from experiments.cvs_f_txrisk.feature import penalty,loss
from experiments.cvs_f_supcon50.mechanism import LEVELS,PREFIX,schedule
from experiments.cvs_sigtime_rdrop_legacy.model import terms

@contextmanager
def capture_maps(m,n):
 values={};handles=[]
 try:
  for level in LEVELS:
   def capture(module,args,out,level=level):
    if level in values:raise ValueError('Feature level repeated inside one actual forward')
    values[level]=out[:n]
   handles.append(m.get_submodule(PREFIX+level).register_forward_hook(capture))
  yield values
  if tuple(values)!=LEVELS:raise ValueError('Actual shallow and middle F maps missing')
 finally:
  for h in handles:h.remove()

def objective(m,x,y,rx,epoch,c):
 """Two identical post-augmentation concat inputs; sat CE only, clean metric/RDrop."""
 n=len(y)
 if x.shape!=(2*n,2,256) or n!=128 or not m.training or not torch.is_grad_enabled():raise ValueError('Real L128 plus sat128 student-only objective required')
 weight=schedule(epoch);geo=bool(c['geometry']);parts=[];outputs=[];joints=[];details=[]
 for phase in range(2):
  with capture_maps(m,n) as maps:out=m.forward_identity_only(x)
  if out['id_feat_joint'].shape!=(2*n,160) or out['tx_logits'].shape!=(2*n,6):raise ValueError('Default F joint160/head graph differs')
  outputs.append(out['tx_logits'])
  if weight:
   v,info=penalty(maps,y,rx,'CROSS_RX',c['weights']);parts.append(v);details.extend(info)
   if geo:
    j,info=loss(out['id_feat_joint'][:n],y,rx,'CROSS_RX',c['weights']);joints.append(j)
 total,ce,kl,_,_=terms(outputs[0][:n],outputs[1][:n],y,1.)
 sat=.5*(F.cross_entropy(outputs[0][n:],y,label_smoothing=.01)+F.cross_entropy(outputs[1][n:],y,label_smoothing=.01))
 aux=torch.stack(parts).mean() if parts else total*0
 joint=torch.stack(joints).mean() if joints else total*0
 total=total+.68*sat+weight*aux+.5*weight*joint
 return total,dict(clean_CE=float(ce.detach()),RDrop_KL=float(kl.detach()),satellite_CE=float(sat.detach()),shallow_middle_loss=float(aux.detach()),joint160_loss=float(joint.detach()),shallow_middle_weight=weight,joint160_weight=.5*weight if geo else 0.,valid_anchors=[i['valid_anchors'] for i in details],paired_forwards=2,extra_forwards_for_metric=0)
