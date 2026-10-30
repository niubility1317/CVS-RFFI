"""Continuous original-head posterior consistency for authorized source F guidance."""
import torch
from torch.nn import functional as F


def frozen_posterior_loss(reader,restored,clean_features):
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Fixed eval F with all parameters frozen required')
    if restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean_features.shape!=(len(restored),160):
        raise ValueError('Matched IQ2x256 and original160D clean features required')
    with torch.no_grad():
        target=F.log_softmax(reader.classify_features(clean_features.detach()),dim=1)
    actual_features=reader.features(restored)
    if actual_features.shape!=clean_features.shape:raise ValueError('Original F feature shape differs')
    logits=reader.classify_features(actual_features)
    if logits.shape!=target.shape or logits.shape!=(len(restored),6):raise ValueError('Same original six-class default head required')
    return F.kl_div(F.log_softmax(logits,dim=1),target,log_target=True,reduction='none').sum(1)
