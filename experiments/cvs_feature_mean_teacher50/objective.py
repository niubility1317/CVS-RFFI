import torch
from torch.nn import functional as F
def consistency(student,teacher):
 if student.ndim!=2 or student.shape!=teacher.shape or student.shape[1]!=160 or not len(student):raise ValueError('Actual same-packet joint160 features required')
 if teacher.requires_grad or not student.requires_grad or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():raise ValueError('Detached finite ownEMA teacher and differentiable student required')
 a=F.normalize(student.float(),dim=1,eps=1e-8);b=F.normalize(teacher.detach().float(),dim=1,eps=1e-8)
 return .5*(a-b).square().sum(1).mean()
def coefficient(epoch,mode):
 if mode not in ('ZERO','FEATURE') or not 1<=epoch<=50:raise ValueError('Unregistered feature consistency schedule')
 return .1*min(1.,max(0.,(epoch-33)/5)) if mode=='FEATURE' else 0.
