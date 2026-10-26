"""Cross-receiver identity relations within or across two actual IQ views."""
import torch
from torch.nn import functional as F


def relation_loss(first,second,target,accepted,receivers,mode):
    n=len(first)
    if (mode not in ('separate','bridge') or first.shape!=(n,160) or second.shape!=first.shape
            or target.shape!=(n,6) or accepted.shape!=(n,) or accepted.dtype!=torch.bool
            or receivers.shape!=(n,) or receivers.dtype!=torch.long or n==0
            or not torch.isfinite(first).all() or not torch.isfinite(second).all()
            or not torch.isfinite(target).all() or bool((target<0).any())
            or not torch.allclose(target.sum(1),torch.ones(n,device=target.device),atol=1e-5,rtol=1e-5)):
        raise ValueError('Two finite aligned joint160 views and hidden-U teacher probabilities required')
    # The held RX and rejected records cannot affect feature normalization or pair membership.
    mask=accepted.detach();labels=target.detach().argmax(1)[mask];rx=receivers.detach()[mask]
    h=torch.cat((first[mask],second[mask]));labels=labels.repeat(2);rx=rx.repeat(2)
    m=int(mask.sum());views=torch.arange(2,device=first.device).repeat_interleave(m)
    same=labels[:,None]==labels[None,:]
    allowed=(views[:,None]==views[None,:]) if mode=='separate' else (views[:,None]!=views[None,:])
    positive=allowed&same&(rx[:,None]!=rx[None,:]);negative=allowed&~same
    valid=positive.any(1)&negative.any(1)
    if bool(valid.any()):
        z=(h-h.detach().mean(0))/h.detach().std(0,unbiased=False).clamp_min(1e-4)
        z=F.normalize(z,dim=1,eps=1e-4);scores=(z@z.T/.2)[valid]
        pos=positive[valid];count=pos.sum(1)
        per_anchor=(torch.logsumexp(scores.masked_fill(~(positive|negative)[valid],-torch.inf),1)
                    -(scores*pos).sum(1)/count-count.to(z.dtype).log())
        # Full original U-batch normalization preserves actual missing/invalid exposure.
        value=per_anchor.sum()/(2*n)
    else:
        value=(first.sum()+second.sum())*0.
    detail=dict(accepted_packets=m,valid_anchors=int(valid.sum()),positive_pairs=int(positive.sum()),
                negative_pairs=int(negative.sum()),original_U_packets=n,mode=mode)
    return value,detail


def extra(student,views,domain,target,accepted,receivers,mode):
    if len(views)!=2:raise ValueError('Exactly two extra student IQ forwards required')
    features=[student.forward_identity_only(v,domain_labels=domain)['z_id'] for v in views]
    value,detail=relation_loss(*features,target,accepted,receivers,mode)
    return value,detail
