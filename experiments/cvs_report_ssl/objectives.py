"""Separate detached identity reliability and packet-local physical quality."""
import torch
from torch.nn import functional as F


def identity_weights(teacher_probabilities,threshold=.95):
    if not 0<threshold<1:raise ValueError('Invalid identity confidence threshold')
    return (teacher_probabilities.detach().amax(1)>=threshold).to(teacher_probabilities.dtype)


def soft_identity_kl(student_logits,teacher_probabilities,weights):
    if student_logits.shape!=teacher_probabilities.shape or student_logits.ndim!=2 or student_logits.shape[1]!=6 or weights.shape!=(len(student_logits),):
        raise ValueError('Six-class soft identity target shape differs')
    q=teacher_probabilities.detach();w=weights.detach()
    if not torch.isfinite(q).all() or bool((q<0).any()) or not torch.allclose(q.sum(1),torch.ones_like(w),atol=1e-5):raise ValueError('Invalid teacher distribution')
    if not torch.isfinite(w).all() or bool(((w<0)|(w>1)).any()):raise ValueError('Invalid identity reliability')
    per_sample=F.kl_div(F.log_softmax(student_logits,1),q,reduction='none').sum(1)
    return (per_sample*w).sum()/len(student_logits)


def physical_weights(quality,task):
    if quality.ndim!=2 or quality.shape[1]!=3 or task not in ('linear','repetition'):
        raise ValueError('Expected independent repetition-quality task weights')
    q=quality.detach()
    if task=='linear':return (.25+.75*q[:,0]).clamp(0,1)
    return (q[:,0]/(1+q[:,1])).clamp(0,1)
