"""One As64 objective: cross-RX positives and within-environment negatives."""
import torch
from torch.nn import functional as F

LEVELS=('time_down','t2.pool')
RX=(1,3,4,6,8)
TEMPERATURE=.2


def masks(y,rx,day,eq):
    if (not all(torch.is_tensor(v) and v.dtype==torch.long and v.ndim==1 for v in (y,rx,day,eq))
            or len(y)==0 or not all(v.shape==y.shape and v.device==y.device for v in (rx,day,eq))
            or not bool(((y>=0)&(y<6)).all()) or not all(int(v) in RX for v in rx.detach().cpu().tolist())
            or not bool(((day>=1)&(day<=3)).all()) or not bool((eq==1).all())):
        raise ValueError('Visible L TX and legal original RX/day/eq vectors required')
    same=y[:,None]==y[None,:]
    group=(rx[:,None]==rx[None,:])&(day[:,None]==day[None,:])&(eq[:,None]==eq[None,:])
    positive=same&(rx[:,None]!=rx[None,:])
    negative=(~same)&group
    valid=positive.any(1)&negative.any(1)
    return positive,negative,positive|negative,valid


def penalty(maps,y,rx,day,eq):
    if tuple(maps)!=LEVELS:
        raise ValueError('Both actual As shallow/middle maps required')
    positive,negative,denominator,valid=masks(y,rx,day,eq)
    count=positive.sum(1);negative_count=negative.sum(1);values=[];details=[]
    for level in LEVELS:
        h=maps[level]
        if (not torch.is_tensor(h) or h.ndim!=3 or len(h)!=len(y) or h.device!=y.device
                or not h.is_floating_point() or not bool(torch.isfinite(h).all())):
            raise ValueError('Complete finite live As maps and matching metadata required')
        descriptor=torch.cat((h.mean(-1),h.std(-1,unbiased=False)),1)
        z=(descriptor-descriptor.detach().mean(0))/descriptor.detach().std(0,unbiased=False).clamp_min(1e-4)
        z=F.normalize(z,dim=1,eps=1e-4)
        if bool(valid.any()):
            logits=(z@z.T/TEMPERATURE)[valid]
            number=count[valid]
            value=(torch.logsumexp(logits.masked_fill(~denominator[valid],-torch.inf),1)
                -(logits*positive[valid]).sum(1)/number-number.to(z.dtype).log()).mean()
        else:
            value=h.sum()*0.
        if not bool(torch.isfinite(value)):
            raise FloatingPointError('Nonfinite environment-conditioned As contrast')
        values.append(value)
        details.append(dict(level=level,loss=float(value.detach()),total_anchors=len(y),
            valid_anchors=int(valid.sum()),zero_anchors=int((~valid).sum()),
            positive_pairs=int(positive.sum()),negative_pairs=int(negative.sum()),
            valid_positive_pairs=int(positive[valid].sum()),valid_negative_pairs=int(negative[valid].sum()),
            positive_min=int(count[valid].min()) if bool(valid.any()) else 0,
            positive_max=int(count[valid].max()) if bool(valid.any()) else 0,
            negative_min=int(negative_count[valid].min()) if bool(valid.any()) else 0,
            negative_max=int(negative_count[valid].max()) if bool(valid.any()) else 0,
            temperature=TEMPERATURE,descriptor_shape=list(descriptor.shape),
            positive_rule='sameTX differentRX anyday',negative_rule='differentTX sameRX/day/eq',
            same_RX_positive_pairs=0,cross_group_negative_pairs=0,all_labeled_CE_retained=True))
    return torch.stack(values).mean(),details
