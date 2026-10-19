import torch
from . import design as d
def allocate(teacher,y,rx,mode,step):
 if teacher.shape!=(len(y),6) or y.dtype!=torch.long or rx.shape!=y.shape or rx.dtype!=torch.long or mode not in ('SHUFFLED','CAPABILITY') or step not in range(1,4441) or not bool(((y>=0)&(y<6)).all()) or set(rx.tolist())-set(d.FIT) or not torch.isfinite(teacher).all():raise ValueError('Finite default six-class teacher and legal labeled source metadata required')
 pred=teacher.detach().argmax(1);mask=pred==y;applied=mask.detach().cpu().clone();groups={}
 for i,key in enumerate(zip(y.detach().cpu().tolist(),rx.detach().cpu().tolist())):groups.setdefault(key,[]).append(i)
 if mode=='SHUFFLED':
  generator=torch.Generator(device='cpu').manual_seed(d.MASK_SEED+step)
  original=applied.clone()
  for indices in groups.values():
   idx=torch.tensor(indices,dtype=torch.long);order=torch.randperm(len(indices),generator=generator);applied[idx]=original[idx[order]]
 applied=applied.to(teacher.device);count=int(applied.sum());w=torch.ones(len(y),device=teacher.device,dtype=teacher.dtype) if count==0 else .5+.5*applied.to(teacher.dtype)*(len(y)/count)
 return w.detach(),dict(teacher_prediction=pred.tolist(),teacher_correct=mask.tolist(),applied_mask=applied.tolist(),teacher_correct_count=int(mask.sum()),mask_changed_count=int((applied!=mask).sum()),packet_weights=w.tolist(),weight_sum=float(w.sum()),private_seed=d.MASK_SEED+step)
def objective(student,teacher,y,rx,mode,step):
 if student.shape!=(len(y),6) or not torch.isfinite(student).all():raise ValueError('Finite real six-class student required')
 w,info=allocate(teacher,y,rx,mode,step);ce=torch.nn.functional.cross_entropy(student,y,reduction='none',label_smoothing=.01)
 loss=(w*ce).mean();info.update(packet_CE=ce.detach().tolist(),extra_CE=float(ce.detach().mean()),weighted_CE=float(loss.detach()))
 return loss,info
