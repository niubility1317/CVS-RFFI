import torch
from torch.nn import functional as F

def objective(clean,second,teacher,labels):
 if clean.shape!=second.shape or clean.shape!=teacher.shape or clean.ndim!=2 or clean.shape[1]!=6 or labels.shape!=(len(clean),) or labels.dtype!=torch.long or not len(clean) or bool(((labels<0)|(labels>=6)).any()) or not all(bool(torch.isfinite(z).all()) for z in (clean,second,teacher)):raise ValueError('Finite paired default-head logits and legal L labels required')
 target=teacher.detach();mask=(target.argmax(1)==labels).to(clean.dtype)
 kl=(target.softmax(1)*(target.log_softmax(1)-clean.log_softmax(1))).sum(1)
 ce=F.cross_entropy(clean,labels);asce=F.cross_entropy(second,labels);keep=(mask*kl).mean();total=ce+asce+keep
 return total,dict(clean_CE=float(ce.detach()),second_CE=float(asce.detach()),keep_KL=float(keep.detach()),parent_correct=int(mask.sum()),denominator=len(labels),total_loss=float(total.detach()))
