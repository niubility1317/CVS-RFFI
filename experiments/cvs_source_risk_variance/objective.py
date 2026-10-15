import torch
from torch.nn import functional as F
RX=(1,3,4,6,8)
def risk_variance(logits,y,rx,*,role='L_s'):
 if role!='L_s' or logits.ndim!=2 or logits.shape!=(len(y),6) or len(y)==0:raise ValueError('Actual labelled-source default logits required')
 if y.shape!=rx.shape or y.ndim!=1 or y.dtype!=torch.long or rx.dtype!=torch.long:raise ValueError('Aligned source labels and receiver IDs required')
 if not torch.isfinite(logits).all() or (y<0).any() or (y>=6).any() or set(rx.tolist())!=set(RX):raise ValueError('All five legal source receivers required')
 per=F.cross_entropy(logits,y,label_smoothing=.01,reduction='none')
 risks=torch.stack([per[rx==r].mean() for r in RX]);mean=risks.mean();var=(risks-mean).square().mean()
 if not torch.isfinite(var):raise FloatingPointError('Nonfinite risk variance')
 return mean,var,risks
