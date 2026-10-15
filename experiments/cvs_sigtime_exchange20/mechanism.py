import torch
from torch.nn import functional as F
from experiments.cvs_dropout_transfer20.mechanism import donors,virtual_features,logits

def loss(z1,z2,W,y,donor,classify=None):
 if W.shape!=(6,160) or y.shape!=(len(z1),) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()):raise ValueError('Original default head and visible L labels required')
 if classify is None:classify=lambda z:logits(z,W)
 plus,minus,center,delta=virtual_features(z1,z2,donor)
 ce=.5*(F.cross_entropy(classify(plus),y,label_smoothing=.01)+F.cross_entropy(classify(minus),y,label_smoothing=.01))
 before=z1.detach().double().square().sum()+z2.detach().double().square().sum()
 after=plus.detach().double().square().sum()+minus.detach().double().square().sum()
 error=float((after-before).abs()/before.clamp_min(1e-12))
 if error>2e-6:raise ValueError('Pooled two-view energy changed')
 return ce,dict(energy_relative_error=error,mean_center_L2=center.detach().norm(dim=1).mean(),mean_residual_L2=delta.detach().norm(dim=1).mean())
