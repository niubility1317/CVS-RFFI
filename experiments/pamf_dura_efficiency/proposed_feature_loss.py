"""Unlaunched F-guidance prototype; real F training use needs user authorization."""
import torch
from torch.nn import functional as F


def frozen_feature_loss(reader,restored,clean_features):
    """Differentiate through received reconstruction, never through F or target."""
    if reader.training or any(parameter.requires_grad for parameter in reader.parameters()):
        raise ValueError('Fixed eval F with all parameters frozen required')
    if restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean_features.shape!=(len(restored),160):
        raise ValueError('Matched IQ2x256 and original160D clean reference features required')
    reference=clean_features.detach()
    actual=reader.features(restored)
    if actual.shape!=reference.shape:raise ValueError('Original F feature shape differs')
    return 1-F.cosine_similarity(actual,reference,dim=1,eps=1e-8)
