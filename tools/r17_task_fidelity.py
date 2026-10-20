"""Prospective source-only R17 task distillation; no file access or head fitting.

This primitive is not enabled in R03. A future runner must declare frozen F50
knowledge use and verify its source provenance before supplying the reader.
"""
import torch
from torch.nn import functional as F


def task_terms(student_features,student_logits,teacher_features,teacher_logits):
    if (student_features.ndim!=2 or student_features.shape[1]!=160 or len(student_features)==0
            or student_logits.shape!=(len(student_features),6)
            or teacher_features.shape!=student_features.shape or teacher_logits.shape!=student_logits.shape
            or any(not torch.isfinite(v).all() for v in (student_features,student_logits,teacher_features,teacher_logits))):
        raise ValueError('Finite paired original F160 features and six-class logits required')
    reference_features=teacher_features.detach();reference_logits=teacher_logits.detach()
    reference_logp=F.log_softmax(reference_logits,dim=1)
    kl=(reference_logp.exp()*(reference_logp-F.log_softmax(student_logits,dim=1))).sum(1).clamp_min(0)
    # Half the squared unit-feature distance equals cosine distance and is
    # exactly zero for identical nonzero feature tensors.
    feature=.5*(F.normalize(student_features,dim=1,eps=1e-8)-F.normalize(reference_features,dim=1,eps=1e-8)).square().sum(1)
    return kl+feature,dict(task_KL=kl,task_feature_distance=feature)


def frozen_task_fidelity(student_iq,teacher_iq,reader):
    if (student_iq.ndim!=3 or student_iq.shape[1:]!=(2,256) or teacher_iq.shape!=student_iq.shape
            or not torch.isfinite(student_iq).all() or not torch.isfinite(teacher_iq).all()):
        raise ValueError('Finite same-input paired R17/student IQ2x256 required')
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Existing default-head reader must stay frozen in eval mode')
    with torch.no_grad():
        reference_features=reader.features(teacher_iq.detach())
        reference_logits=reader.classify_features(reference_features)
    student_features=reader.features(student_iq)
    student_logits=reader.classify_features(student_features)
    return task_terms(student_features,student_logits,reference_features,reference_logits)
