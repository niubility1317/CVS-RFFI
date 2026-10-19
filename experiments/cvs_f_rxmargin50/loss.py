"""Current-L feature angles, no fitted prototypes/head or persistent state."""
import torch
from torch.nn import functional as F
from experiments.cvs_f_supcon50.mechanism import masks
MARGIN=.05
TEMPERATURE=.05

def angular_margin(z,y,rx):
 if z.ndim!=2 or z.shape!=(len(y),160) or not torch.isfinite(z).all():raise ValueError('Finite deployed joint160 required')
 positive,_,_=masks(y,rx,'CROSS_RX');negative=(y[:,None]!=y[None,:])&(rx[:,None]==rx[None,:]);valid=positive.any(1)&negative.any(1)
 if not bool(valid.any()):return z.sum()*0.,dict(valid_anchors=0,total_anchors=len(y),positive_min=0,positive_max=0,active_fraction=0.,classes=0)
 u=F.normalize(z,dim=1,eps=1e-4);cosine=(u@u.T)[valid];pos=positive[valid];neg=negative[valid];labels=y[valid];count=pos.sum(1)
 # Count-normalized soft minimum emphasizes cross-RX difficult positives.
 # Hard negatives share the anchor RX, preventing RX difference alone from
 # serving as the negative identity cue. Satisfied margins produce zero loss.
 positive_similarity=-TEMPERATURE*(torch.logsumexp((-cosine/TEMPERATURE).masked_fill(~pos,-torch.inf),1)-count.to(z.dtype).log())
 negative_similarity=cosine.masked_fill(~neg,-torch.inf).max(1).values
 violation=F.relu(MARGIN+negative_similarity-positive_similarity)
 classes=labels.unique();value=torch.stack([violation[labels==c].mean() for c in classes]).mean()
 if not torch.isfinite(value):raise FloatingPointError('Nonfinite RX-matched angular loss')
 per_class=[dict(class_id=int(c),valid_anchors=int((labels==c).sum()),active_fraction=float((violation[labels==c]>0).float().mean()),loss=float(violation[labels==c].mean().detach())) for c in classes]
 return value,dict(valid_anchors=int(valid.sum()),total_anchors=len(y),positive_min=int(count.min()),positive_max=int(count.max()),active_fraction=float((violation>0).float().mean()),classes=len(classes),positive_similarity=float(positive_similarity.mean().detach()),negative_similarity=float(negative_similarity.mean().detach()),per_class=per_class)
