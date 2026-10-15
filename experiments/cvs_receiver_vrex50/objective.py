import torch
from torch.nn import functional as F
RX=(1,3,4,6,8)
def risks(logits,y,rx,*,role='L_s'):
 if role!='L_s' or logits.ndim!=2 or logits.shape!=(len(y),6) or not len(y) or not logits.is_floating_point():raise ValueError('Labelledsource default logits required')
 if y.ndim!=1 or rx.shape!=y.shape or y.dtype!=torch.long or rx.dtype!=torch.long or not torch.isfinite(logits).all() or (y<0).any() or (y>=6).any() or set(rx.tolist())-set(RX):raise ValueError('InvalidsourceTX/RX')
 ce=F.cross_entropy(logits,y,label_smoothing=.01,reduction='none');counts=torch.stack([(rx==r).sum() for r in RX]);means=torch.stack([ce[rx==r].sum()/counts[k].clamp_min(1) for k,r in enumerate(RX)])
 present=counts>0;balanced=means[present].mean();variance=(means[present]-balanced).square().mean()
 return balanced,variance,counts,means
def objective(logits,y,rx,coefficient,*,role='L_s'):
 if coefficient not in (0.,1.):raise ValueError('Only fixedzero/one coefficients registered')
 balanced,variance,counts,means=risks(logits,y,rx,role=role);selected=balanced if coefficient==0 else balanced+variance
 return selected,balanced,variance,counts,means
