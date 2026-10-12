from contextlib import contextmanager
import torch
from torch.nn import functional as F
from experiments.cvs_finite_action.operators import apply
from . import design as d

@contextmanager
def deterministic(model):
 modes=[(m,m.training) for m in model.modules()]
 try:
  model.eval();yield
 finally:
  for m,mode in modes:m.training=mode

def robust_loss(logits,y):
 if logits.ndim!=3 or logits.shape[0]!=5 or logits.shape[1]!=len(y) or logits.shape[2]!=6:raise ValueError('Five-view six-class actual margin required')
 true=logits.gather(2,y[None,:,None].expand(5,-1,1))
 margins=true-logits;worst=margins.min(0).values
 # Correct-class margin is exactly zero. Other classes use the same rule.
 return torch.logsumexp(-worst,dim=1).mean(),worst

def objective(model,x,y,c):
 main=model(x);ce=F.cross_entropy(main,y);small=x[:32];sy=y[:32]
 views=[small]+[apply(small,kind,a,c['model_seed']) for kind,a in c['actions']]
 packed=torch.cat([small]*5 if c['arm']=='CLEAN_REPLAY' else views)
 with deterministic(model):extra=model(packed).reshape(5,len(small),6)
 if c['arm']=='FINITE_MARGIN':aux,margin=robust_loss(extra,sy)
 else:
  aux=F.cross_entropy(extra.flatten(0,1),sy.repeat(5));_,margin=robust_loss(extra,sy)
 loss=ce+c['loss_weight']*aux
 wrong=margin.masked_fill(F.one_hot(sy,6).bool(),torch.inf).min(1).values
 return loss,dict(CE=float(ce.detach()),extra_loss=float(aux.detach()),total_loss=float(loss.detach()),L_accuracy=float((main.argmax(1)==y).float().mean()),finite_min_margin=float(wrong.detach().mean()))
