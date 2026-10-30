"""Fixed original-head visible feature directions; no fitted projection."""
import torch
from torch.nn import functional as F


def head_feature_basis(reader):
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Fixed eval F with all parameters frozen required')
    weight=reader.core.id_backbone.cls_head.weight.detach().to(device='cpu',dtype=torch.float64)
    if weight.shape!=(6,160) or not torch.isfinite(weight).all():
        raise ValueError('Original finite six by160 head required')
    weight=F.normalize(weight,dim=1,eps=1e-4)
    centered=weight-weight.mean(0,keepdim=True)
    _,singular,vh=torch.linalg.svd(centered,full_matrices=False)
    rank=int((singular>singular.max()*1e-7).sum())
    if rank<1 or rank>5:raise ValueError('Nondegenerate original head contrast directions required')
    return vh[:rank].T.contiguous().detach()


def projected_feature_error(actual,reference,basis):
    if actual.ndim!=2 or actual.shape[1]!=160 or reference.shape!=actual.shape:
        raise ValueError('Paired original160D features required')
    if basis.ndim!=2 or basis.shape[0]!=160 or not 1<=basis.shape[1]<=5:
        raise ValueError('Fixed original head contrast basis required')
    difference=F.normalize(actual,dim=1,eps=1e-4)-F.normalize(reference.detach(),dim=1,eps=1e-4)
    visible=difference@basis.detach().to(device=actual.device,dtype=actual.dtype)
    # Equal expected scale to1-cos under an isotropic unit-feature residual.
    return (160/(2*basis.shape[1]))*visible.square().sum(1)


def frozen_head_feature_loss(reader,restored,reference,basis):
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Fixed eval F with all parameters frozen required')
    if restored.ndim!=3 or restored.shape[1:]!=(2,256):raise ValueError('Matched IQ2x256 required')
    return projected_feature_error(reader.features(restored),reference,basis)
