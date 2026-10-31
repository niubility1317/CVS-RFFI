"""Fixed source-clean local metric in the original frozen head-visible feature space."""
import math
import torch
from torch.nn import functional as F


def visible_features(reader,iq,basis):
    return F.normalize(reader.features(iq),dim=1,eps=1e-4)@basis.detach().to(iq)*math.sqrt(160/10)


def clean_jacobian(reader,clean,basis):
    if reader.training or any(p.requires_grad for p in reader.parameters()):
        raise ValueError('Frozen eval original F required')
    if clean.ndim!=3 or clean.shape[1:]!=(2,256) or basis.shape!=(160,5):
        raise ValueError('Source IQ2x256 and original rank5 head basis required')
    with torch.enable_grad():
        x=clean.detach().clone().requires_grad_(True);value=visible_features(reader,x,basis)
        if value.shape!=(len(x),5):raise ValueError('Original normalized160D features required')
        jac=torch.stack([torch.autograd.grad(value[:,k].sum(),x,retain_graph=k<4)[0] for k in range(5)],dim=1)
    if not torch.isfinite(value).all() or not torch.isfinite(jac).all():raise FloatingPointError('Finite source F Jacobian required')
    return value.detach(),jac.detach()


def tangent_loss(restored,clean,jacobian):
    if restored.ndim!=3 or restored.shape[1:]!=(2,256) or clean.shape!=restored.shape or jacobian.shape!=(len(restored),5,2,256):
        raise ValueError('Matched source IQ and fixed rank5 clean Jacobian required')
    delta=restored-clean.detach()
    projected=(jacobian.detach()*delta[:,None]).sum((2,3))
    return projected.square().sum(1)


def eligibility(rows):
    if set(rows)!={'1','20'}:raise ValueError('Both fixed source cache views required')
    checks=[];ratios=[]
    for view,row in rows.items():
        rx=row['rx_mean_tangent']
        if row['count']!=512 or set(rx)!=set(map(str,[1,3,4,6,8])):raise ValueError('Full512 and all sourceRX required')
        ratio=row['median_tangent_to_IQ_norm'];ratios.append(ratio)
        passed=(row['defined_tangent']==512 and row['defined_IQ']==512
            and row['mean_tangent_cosine']>=.5 and row['mean_tangent_cosine']>=row['mean_IQ_cosine']+.1
            and row['nonnegative_tangent']>=384 and all(v is not None and v>=0 for v in rx.values())
            and ratio is not None and math.isfinite(ratio) and ratio>0)
        checks.append(dict(view=view,passed=bool(passed)))
    qualified=all(v['passed'] for v in checks)
    return dict(eligible=qualified,views=checks,fixed_source_weight=.25/max(ratios) if qualified else None,automatic_training=False)
