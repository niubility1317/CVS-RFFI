"""One detached teacher target; no urban student qualification or fitted head."""
import torch
from torch.nn import functional as F
EPS=1e-4


def labeled_reliability(teacher_logits,labels):
    if teacher_logits.requires_grad or labels.ndim!=1 or teacher_logits.shape!=(len(labels),6) or bool((labels<0).any()):
        raise ValueError('L_s requires detached default teacher logits and legal source labels')
    return (teacher_logits.argmax(1)==labels).to(teacher_logits.dtype)


def validate(student,target,weight,width):
    if student.ndim!=2 or student.shape[1]!=160 or target.shape!=(len(student),width) or target.requires_grad:
        raise ValueError('Actual joint160 and stop-gradient teacher required')
    if weight.shape!=(len(student),) or weight.requires_grad or not bool(torch.isfinite(weight).all()) or bool(((weight<0)|(weight>1)).any()):
        raise ValueError('Detached source reliability weight in [0,1] required')
    if not bool(torch.isfinite(student).all() and torch.isfinite(target).all()):raise ValueError('Nonfinite identity transfer')


def direction(student,target,weight):
    validate(student,target,weight,160)
    values=1-(F.normalize(student,dim=1,eps=EPS)*F.normalize(target,dim=1,eps=EPS)).sum(1)
    return (weight*values).mean()


def output_kl(student,teacher_logits,weight,head):
    validate(student,teacher_logits,weight,6)
    if head.weight.shape!=(6,160) or head.scale!=30.:raise ValueError('Default cosine head scale changed')
    logits=head.scale*F.linear(F.normalize(student,dim=1,eps=EPS),F.normalize(head.weight.detach(),dim=1,eps=EPS))
    values=F.kl_div(F.log_softmax(logits,dim=1),F.softmax(teacher_logits,dim=1),reduction='none').sum(1)
    return (weight*values).mean()
