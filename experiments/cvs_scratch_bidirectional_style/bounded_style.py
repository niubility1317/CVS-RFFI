"""L-only bounded source-statistic directions; detached moments, live features."""
import math
import numpy as np
import torch

TRANSFORM='bounded_logstats_v1'
LOG_BOUND=math.log(2.)

def transform(value,labels,receivers,days,ids,seed,p=1.,alpha=.1,eta=1.,direction=1):
    if p!=1. or alpha!=.1 or not 0<=eta<=1 or direction not in (-1,1):
        raise ValueError('Unregistered bounded-style settings')
    if (not torch.is_tensor(value) or value.ndim!=3 or value.is_complex()
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('Finite real B,C,T feature maps required')
    y,rx,day=[v.detach().cpu().tolist() for v in (labels,receivers,days)]
    if any(len(v)!=len(value) for v in (ids,y,rx,day)) or any(v not in range(6) for v in y):
        raise ValueError('Known L identity and matched occurrence metadata required')
    rng=np.random.default_rng(int(seed));rng.random()  # Preserve original private draw order.
    eligible=[[j for j in range(len(ids)) if y[j]==y[i] and rx[j]!=rx[i]
               and day[j]==day[i] and ids[j]!=ids[i]] for i in range(len(ids))]
    donor=[int(rng.choice(v)) if v else i for i,v in enumerate(eligible)]
    movement=eta*(1-rng.beta(alpha,alpha,len(ids)))
    movement=np.where(np.asarray([bool(v) for v in eligible]),movement,0.)
    a=torch.tensor(movement.tolist(),dtype=value.dtype,device=value.device)[:,None,None]
    mu=value.mean(-1,keepdim=True).detach()
    sigma=(value.var(-1,unbiased=False,keepdim=True)+1e-6).sqrt().detach()
    delta_mean=(mu[donor]-mu)/sigma
    delta_log=sigma[donor].log()-sigma.log()
    mean_step=direction*a*delta_mean.clamp(-1.,1.)
    log_step=direction*a*delta_log.clamp(-LOG_BOUND,LOG_BOUND)
    scale=log_step.exp()
    output=(value-mu)*scale+mu+sigma*mean_step
    output=torch.where(a!=0,output,value)
    if not torch.isfinite(output).all():raise FloatingPointError('Nonfinite bounded-style output')
    ratio=(output.detach()-value.detach()).flatten(1).norm(dim=1)/value.detach().flatten(1).norm(dim=1).clamp_min(1e-6)
    details=dict(enabled=True,eligible_packets=sum(bool(v) for v in eligible),
        changed_packets=int((ratio>0).sum()),changed_ratio=float((ratio>0).float().mean()),
        movement_mean=float(a.mean()),movement_max=float(a.max()),delta_ratio=float(ratio.mean()),donors=donor,
        transform=TRANSFORM,direction=int(direction),mean_shift_in_sigma_max=float(mean_step.abs().max()),
        scale_ratio_min=float(scale.min()),scale_ratio_max=float(scale.max()),
        mean_clipped_fraction=float((delta_mean.abs()>1.).float().mean()),
        log_scale_clipped_fraction=float((delta_log.abs()>LOG_BOUND).float().mean()),
        moments_detached=not mu.requires_grad and not sigma.requires_grad)
    return output,details
