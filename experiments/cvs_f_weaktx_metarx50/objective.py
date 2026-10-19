"""Feature-only MLDG adaptation with current visible L and a fixed existing W."""
from collections.abc import Mapping
import torch
from torch.nn import functional as F
from torch.func import functional_call
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
W='encoder.core.id_backbone.cls_head.weight'
COORDINATES=('encoder.core.id_backbone.sinc.low_coordinate','encoder.core.id_backbone.sinc.band_coordinate')
RXS=(1,3,4,6,8)
WEAK=(1,3)
def active(step):return step>1110 and step%16==0
def anchor_for(step):
 if not active(step):raise ValueError('Outside fixed meta calendar')
 return RXS[((step//16)-70)%5]
def balanced_ce(logits,labels):
 if logits.shape!=(len(labels),6) or labels.dtype!=torch.long or not len(labels) or not bool(((labels>=0)&(labels<6)).all()) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite current labeled six-TX source logits required')
 each=F.cross_entropy(logits,labels,label_smoothing=.01,reduction='none')
 return torch.stack([each[labels==k].mean() for k in labels.unique()]).mean()
def masks(labels,rx,anchor):
 if labels.dtype!=torch.long or rx.dtype!=torch.long or labels.shape!=(128,) or rx.shape!=labels.shape or not bool(((labels>=0)&(labels<6)).all()) or not all(int(r) in RXS for r in rx.detach().cpu().tolist()) or anchor not in RXS:raise ValueError('Only actual current L128 TX/source RX permitted')
 inner=rx==anchor;held=(rx!=anchor)&((labels==1)|(labels==3))
 return inner,held

def objective(model,x,y,rx,anchor,mode):
 if mode not in ('EXTRA_RX_CE','META_RX') or x.shape!=(128,2,256) or not model.training or not torch.is_grad_enabled():raise ValueError('Registered ordered source-L objective only')
 inner,held=masks(y,rx,anchor)
 if not bool(inner.any()) or not all(bool(((y==c)&held).any()) for c in WEAK):return x.sum()*0.,dict(eligible=False,anchor_RX=anchor,extra_L_packets=0,inner_gradient_tensors=0)
 backbone=model.id_backbone;parameters=dict(backbone.named_parameters());buffers={n:v.detach().clone() for n,v in backbone.named_buffers()};modes=[(m,m.training) for m in backbone.modules()]
 if W not in parameters or any(n not in parameters for n in COORDINATES):raise ValueError('Existing default W/unified Sinc leaves missing')
 parameters[W]=parameters[W].detach()
 names=[n for n,p in parameters.items() if n!=W and n not in COORDINATES and p.requires_grad];leaves=[parameters[n] for n in names]
 try:
  backbone.eval()
  with fresh_parameter_caches(backbone):
   first=functional_call(backbone,(parameters,buffers),(x,),dict(return_aux=True),strict=True)
   if not isinstance(first,Mapping):raise ValueError('Actual identity backbone output required')
   inner_loss=balanced_ce(first['logits'][inner],y[inner]);current_held=balanced_ce(first['logits'][held],y[held])
   gradients=torch.autograd.grad(inner_loss,leaves,create_graph=mode=='META_RX',retain_graph=True,allow_unused=True)
   norm=torch.stack([g.detach().square().sum() for g in gradients if g is not None]).sum().sqrt()
   if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('Missing/nonfinite actual inner feature gradient')
   alpha=.01 if mode=='META_RX' else 0.;virtual=dict(parameters)
   for n,p,g in zip(names,leaves,gradients):
    if g is not None:virtual[n]=p-alpha*g if alpha else p
   # The two physical q leaves remain at the exact current values in the
   # virtual SGD step. Real main/outer gradients still reach both leaves and
   # the unchanged actual physical-unit AdamW applies their real update.
   second=functional_call(backbone,(virtual,{n:v.clone() for n,v in buffers.items()}),(x[held],),dict(return_aux=True),strict=True)
   held_loss=balanced_ce(second['logits'],y[held]);loss=.5*(inner_loss+held_loss)
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite meta objective')
  return loss,dict(eligible=True,anchor_RX=anchor,inner_L_packets=int(inner.sum()),held_weak_L_packets=int(held.sum()),extra_L_packets=128+int(held.sum()),inner_classes=int(y[inner].unique().numel()),held_classes=2,inner_gradient_tensors=sum(g is not None for g in gradients),inner_gradient_norm=float(norm),inner_alpha=alpha,inner_loss=float(inner_loss.detach()),held_current_loss=float(current_held.detach()),held_virtual_loss=float(held_loss.detach()),virtual_held_gain=float((current_held-held_loss).detach()),second_order=mode=='META_RX',virtual_Sinc_update=False,default_W_aux_detached=True,extra_optimizer_steps=0)
 finally:
  for m,flag in modes:m.training=flag
