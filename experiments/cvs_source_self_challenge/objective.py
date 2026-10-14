import torch
from torch.nn import functional as F


def feature_mask(gradient, *, role='L_s'):
    if role != 'L_s':
        raise ValueError('Source true-label diagnostic accepts only L_s')
    if gradient.ndim != 3 or gradient.shape[1:] != (96,32) or not len(gradient) or not torch.isfinite(gradient).all():
        raise ValueError('Finite actual t2 derivative N96x32 required')
    score = gradient.detach().mean(1)
    cutoff = torch.quantile(score, 2/3, dim=1, keepdim=True, interpolation='linear')
    mask = score < cutoff
    if not ((mask.sum(1)>0)&(mask.sum(1)<32)).all():
        raise ValueError('Top-gradient temporal mask is not identifiable')
    return mask


def selected_masks(original, spatial, y, mask, seed, *, role='L_s'):
    if role != 'L_s':
        raise ValueError('Only legal source identity labels may choose diagnostic packets')
    n=len(y)
    if original.shape!=(n,6) or spatial.shape!=(n,6) or y.dtype!=torch.long or mask.shape!=(n,32) or mask.dtype!=torch.bool:
        raise ValueError('Aligned actual source logits, labels and temporal mask required')
    if n<3 or not torch.isfinite(original).all() or not torch.isfinite(spatial).all() or (y<0).any() or (y>=6).any():
        raise ValueError('Invalid source identity observations')
    drop=(original.detach().softmax(1)-spatial.detach().softmax(1)).gather(1,y[:,None]).squeeze(1)
    cutoff=torch.quantile(drop,2/3,interpolation='linear')
    selected=drop>=cutoff
    if not selected.any() or selected.all():
        raise ValueError('Batch self-challenge is not identifiable')
    final=mask.clone();final[~selected]=True
    random=torch.ones_like(final);generator=torch.Generator(device='cpu').manual_seed(seed)
    for k in range(n):
        random[k]=final[k,torch.randperm(32,generator=generator).to(final.device)]
    if not torch.equal(final.sum(1),random.sum(1)):
        raise ValueError('Matched random control changed masked exposure')
    return final,random,selected,drop


def ce(logits,y):
    return F.cross_entropy(logits,y,label_smoothing=.01)


def gradient_stats(reference, other):
    a,b=reference.detach().double().flatten(),other.detach().double().flatten()
    aa,bb=float(a.norm()),float(b.norm())
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite source parameter derivative')
    return dict(reference_norm=aa,other_norm=bb,ratio=bb/aa if aa else None,
                cosine=float(a@b)/(aa*bb) if aa*bb else None)
