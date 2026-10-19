"""Transfer competitors only; the true-class student logit is excluded exactly."""
import torch
def nontrue(student,teacher,y):
 if student.shape!=teacher.shape or student.shape!=(len(y),6) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not torch.isfinite(student).all() or not torch.isfinite(teacher).all():raise ValueError('Finite six-class legal labeled tensors required')
 keep=torch.arange(6,device=y.device)[None,:]!=y[:,None]
 s=(student/6.)[keep].reshape(-1,5).log_softmax(1)
 t=teacher.detach()[keep].reshape(-1,5).log_softmax(1)
 accepted=(teacher.detach().argmax(1)==y).to(s.dtype)
 value=6.*(accepted*(t.exp()*(t-s)).sum(1)).mean()
 return value,dict(nontrue_KL=float(value.detach()),teacher_correct=int(accepted.sum()),denominator=len(y),student_temperature=6.,true_class_in_auxiliary=False)
