import torch
from torch.nn import functional as F
MODES=('ALL_GRAPH','XR_GRAPH')
def relation(student,teacher_prob,rx,mode,*,role='L_s'):
 if role not in ('L_s','U_s') or mode not in MODES:raise ValueError('Illegal graph role or mode')
 n=len(student)
 if student.ndim!=2 or student.shape!=(n,160) or n<2 or teacher_prob.shape!=(n,6) or rx.shape!=(n,):raise ValueError('Actual joint160/probability/sourceRX shapes required')
 if not student.requires_grad or teacher_prob.requires_grad or rx.dtype!=torch.long:raise ValueError('Differentiable deployed features/detached teacher probabilities/integerRX required')
 if not torch.isfinite(student).all() or not torch.isfinite(teacher_prob).all() or (teacher_prob<0).any() or not torch.allclose(teacher_prob.sum(1),torch.ones(n,device=student.device),atol=1e-6):raise ValueError('Finite normalized teacher probability required')
 if not torch.isin(rx,torch.tensor([1,3,4,6,8],device=rx.device)).all():raise ValueError('Non-source receiver')
 if (student.detach().norm(dim=1)<=1e-8).any():raise ValueError('Undefined feature direction')
 mask=~torch.eye(n,dtype=torch.bool,device=student.device)
 if mode=='XR_GRAPH':mask=mask&(rx[:,None]!=rx[None,:])
 target=(teacher_prob.detach().float()@teacher_prob.detach().float().T)*mask
 degree=target.sum(1);valid=degree>1e-12
 target=target/degree.clamp_min(1e-12)[:,None]
 z=F.normalize(student.float(),dim=1,eps=1e-8)
 log_s=(z@z.T/.2).masked_fill(~mask,-1e9).log_softmax(1)
 rows=(target*(target.clamp_min(1e-30).log()-log_s)).sum(1)
 loss=rows[valid].mean() if valid.any() else student.sum()*0
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite graph KL')
 return loss,dict(target=target.detach(),allowed=mask.detach(),valid=valid.detach())
