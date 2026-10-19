"""Same-IQ class transfer; all selected As retain their original supervised CE."""
import torch
from torch.nn import functional as F

def correction(z0,z1,y,teacher_logits,mode):
 if mode not in ('CE','CSIL') or z0.shape!=z1.shape or z0.shape!=teacher_logits.shape or z0.shape!=(len(y),6) or len(y)!=64 or teacher_logits.requires_grad:raise ValueError('Only same-IQ64 frozen-class-teacher correction')
 if any(not torch.isfinite(v).all() for v in (z0,z1,teacher_logits)):raise ValueError('Finite logits required')
 p,q=z0.log_softmax(1),z1.log_softmax(1);oldkl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
 target=teacher_logits.detach().softmax(1);accept=(teacher_logits.argmax(1)==y).detach()
 per=.5*sum(F.kl_div(F.log_softmax(z/6,dim=1),target,reduction='none').sum(1) for z in (z0,z1))
 kd=(per*accept).mean();loss=-.25*oldkl+(.25*6*kd if mode=='CSIL' else kd*0)
 return loss,dict(As_KL_removed=float(oldkl.detach()),KD=float(kd.detach()),teacher_correct=int(accept.sum()),teacher_packets=64,loss_delta=float(loss.detach()),KD_active=mode=='CSIL',new_student_NN=0)
