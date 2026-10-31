"""Local physical/SIG parameter derivatives, distinct from IQ gradient caps."""
import numpy as np
import torch


def split_parameters(physical,signature,parameters):
    if physical.ndim!=1 or signature.shape!=physical.shape:
        raise ValueError('Matched per-record physical and SIG surrogates required')
    sg=torch.autograd.grad(signature.mean(),parameters,retain_graph=True,allow_unused=True)
    pg=torch.autograd.grad(physical.mean(),parameters,allow_unused=True)
    def flatten(values):
        return torch.cat([(torch.zeros_like(p) if v is None else v.detach()).flatten() for p,v in zip(parameters,values)])
    p,s=flatten(pg),flatten(sg)
    if not torch.isfinite(p).all() or not torch.isfinite(s).all():raise ValueError('Finite local parameter derivatives required')
    return p,s,sum(v is None for v in pg),sum(v is None for v in sg)


def statistics(physical,signature):
    if (physical.ndim!=2 or physical.shape!=signature.shape or physical.shape[1]<1
        or not np.isfinite(physical).all() or not np.isfinite(signature).all()):
        raise ValueError('Matched finite physical/SIG gradient matrices required')
    p=physical.astype(np.float64);s=signature.astype(np.float64);t=p+s
    pn=np.linalg.norm(p,axis=1);sn=np.linalg.norm(s,axis=1);tn=np.linalg.norm(t,axis=1)
    ratio_defined=pn>1e-12;cos_defined=pn*sn>1e-24;total_defined=pn*tn>1e-24
    def values(a,defined):return [float(v) if d else None for v,d in zip(a,defined)]
    return dict(physical_norm=pn.tolist(),SIG_norm=sn.tolist(),total_norm=tn.tolist(),ratio_defined=ratio_defined.tolist(),
        SIG_to_physical_ratio=values(sn/np.maximum(pn,1e-12),ratio_defined),
        physical_SIG_cosine=values((p*s).sum(1)/np.maximum(pn*sn,1e-24),cos_defined),physical_SIG_cosine_defined=cos_defined.tolist(),
        total_physical_cosine=values((t*p).sum(1)/np.maximum(pn*tn,1e-24),total_defined),total_physical_cosine_defined=total_defined.tolist())


def component_summary(physical,signature):
    per_RX=statistics(physical,signature)
    pooled=statistics(physical.astype(np.float64).mean(0,keepdims=True),signature.astype(np.float64).mean(0,keepdims=True))
    def cancellation(key,pooled_key):
        denominator=float(np.mean(per_RX[key]));return float(pooled[pooled_key][0]/denominator) if denominator>1e-12 else None
    return dict(per_RX=per_RX,equal_RX_pool=pooled,
                physical_cross_RX_norm_retention=cancellation('physical_norm','physical_norm'),SIG_cross_RX_norm_retention=cancellation('SIG_norm','SIG_norm'))


def decision(views):
    if set(views)!=set(('1','20')):raise ValueError('Both fixed source views required')
    checks={}
    for view,full in views.items():
        r=full['per_RX'];p=full['equal_RX_pool']
        checks[view]=dict(physical_directions_defined=all(r['ratio_defined']) and p['ratio_defined'][0],
            at_least_four_RX_below_one_tenth=sum(v is not None and v<.1 for v in r['SIG_to_physical_ratio'])>=4,
            pooled_below_one_tenth=p['SIG_to_physical_ratio'][0] is not None and p['SIG_to_physical_ratio'][0]<.1)
    return dict(supports_small_SIG_parameter_contribution=all(all(v.values()) for v in checks.values()),checks=checks,automatic_training=False)
