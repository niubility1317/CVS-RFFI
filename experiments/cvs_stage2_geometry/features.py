"""Exact final normalized joint160 used by the unchanged cosine classifier."""
import torch
from torch.nn import functional as F


def identity(model, x, verify=False):
    out = model.forward_identity_only(x)
    joint = out['id_feat_joint']
    if joint.shape != (len(x), 160):
        raise ValueError('Expected final joint160, including reference response')
    z = F.normalize(joint, dim=1, eps=1e-4)
    logits = out['tx_logits']
    if verify:
        head = model.id_backbone.encoder.core.id_backbone.cls_head
        expected = 30. * F.linear(z, F.normalize(head.weight, dim=1, eps=1e-4))
        if head.scale != 30. or not torch.allclose(logits, expected, atol=2e-5, rtol=2e-6):
            raise ValueError('Captured joint160 does not reproduce actual classifier')
        if (joint.norm(dim=1) <= 1e-4).any():
            raise ValueError('Joint below normalization floor; unit-sphere contract invalid')
        if not torch.isfinite(z).all():
            raise FloatingPointError('Nonfinite joint160')
    return logits, z
