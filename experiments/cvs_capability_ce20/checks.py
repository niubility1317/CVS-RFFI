import copy
from collections import Counter
import torch
from . import design as d
from .loss import objective,allocate
def run():
 torch.manual_seed(2061);s=torch.randn(128,6,requires_grad=True);t=torch.randn(128,6,requires_grad=True);y=torch.arange(128)%6;rx=torch.tensor([d.FIT[(i//6)%5] for i in range(128)])
 rng=torch.get_rng_state().clone();loss,a=objective(s,t,y,rx,'CAPABILITY',1);other,b=objective(s,t,y,rx,'SHUFFLED',1)
 if not torch.equal(rng,torch.get_rng_state()):raise AssertionError('Control consumed global training RNG')
 g=torch.autograd.grad(loss,(s,t),allow_unused=True);w=torch.tensor(a['packet_weights']);q=torch.nn.functional.one_hot(y,6)*.99+.01/6
 if g[1] is not None or not torch.allclose(g[0],w[:,None]*(s.detach().softmax(1)-q)/128,atol=1e-7) or not bool((w>=.5).all()) or abs(float(w.mean())-1)>1e-6:raise AssertionError('Detached normalized all-label CE gradient differs')
 ca=Counter((int(y[i]),int(rx[i])) for i in range(128) if a['applied_mask'][i]);cb=Counter((int(y[i]),int(rx[i])) for i in range(128) if b['applied_mask'][i])
 if ca!=cb or a['teacher_correct_count']!=b['teacher_correct_count'] or b['mask_changed_count']==0:raise AssertionError('Within TX/RX same-count control differs')
 for correct in (True,False):
  logits=torch.full((128,6),-20.);logits.scatter_(1,(y if correct else (y+1)%6)[:,None],20.)
  for mode in ('CAPABILITY','SHUFFLED'):
   weights,_=allocate(logits,y,rx,mode,1)
   if not torch.equal(weights,torch.ones(128)):raise AssertionError('All/none-correct fallback differs')
 perm=torch.tensor([4,1,5,0,2,3]);inv=perm.argsort();_,p=objective(s[:,perm],t[:,perm],inv[y],rx,'SHUFFLED',1)
 if p['applied_mask']!=b['applied_mask']:raise AssertionError('True class relabeling changes within-group random control')
 negatives=0
 for change in ({'epochs':50},{'checkpoint_sources':['old.pth']},{'model_seed':2},{'training_mechanism':{}},{'data_contract':{}}):
  c=copy.deepcopy(d.config(d.rows()[0]));c.update(change)
  try:d.validate(c)
  except ValueError:negatives+=1
  else:raise AssertionError('Unauthorized recipe accepted')
 return dict(status='PASS',all_As_positive_CE=True,As_mean_weight_one=True,teacher_detached=True,formula_gradient=True,TX_RX_counts_preserved=True,global_RNG_unchanged=True,class_relabeling=True,all_none_fallback=True,protocol_negatives=negatives)
if __name__=='__main__':print(run())
