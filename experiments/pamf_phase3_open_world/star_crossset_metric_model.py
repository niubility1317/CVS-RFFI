"""L-anchor metric KL against L/OE candidates; no hard OE-OE relationships."""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F
from .star_metric_residual_model import (
    MetricResidual,oe_structure,SEED,PARAMETERS,TEMPERATURE,_metadata,_unit,_average,
)


def _set_prior(anchor_rx,anchor_day,rx,day,dtype):
    _,cells=torch.unique(torch.stack((rx,day),1),dim=0,return_inverse=True)
    onehot=F.one_hot(cells,num_classes=int(cells.max())+1).to(dtype)
    valid=(anchor_rx[:,None]!=rx[None])&(anchor_day[:,None]!=day[None])
    counts=valid.to(dtype)@onehot;domains=(counts>0).sum(1)
    if bool(domains.eq(0).any()):raise ValueError('Every L anchor needs candidates from both L and OE sets across RX AND day')
    logweight=math.log(.5)-domains[:,None].to(dtype).log()-counts[:,cells].clamp_min(1).log()
    return valid,onehot,cells,logweight,domains


def crossset_metric(z_L,z_OE,y_L,rx_L,day_L,rx_OE,day_OE):
    """Two half-mass candidate priors; positives are only legal same-class L."""
    _metadata(z_L,rx_L,day_L,y_L);_metadata(z_OE,rx_OE,day_OE)
    if z_L.device!=z_OE.device:raise ValueError('Both student embedding sets must share a device')
    with torch.autocast(device_type=z_L.device.type,enabled=False):
        lvalid,onehot,cells,lweight,ldomains=_set_prior(rx_L,day_L,rx_L,day_L,z_L.dtype)
        ovalid,_,_,oweight,odomains=_set_prior(rx_L,day_L,rx_OE,day_OE,z_L.dtype)
        known=_unit(z_L);outside=_unit(z_OE)
        logits=torch.cat((known@known.T/TEMPERATURE+lweight,known@outside.T/TEMPERATURE+oweight),1)
        valid=torch.cat((lvalid,ovalid),1);logits=logits.masked_fill(~valid,-torch.inf)
        positive=lvalid&(y_L[:,None]==y_L[None]);counts=positive.to(z_L.dtype)@onehot;domains=(counts>0).sum(1)
        if bool(domains.eq(0).any()):raise ValueError('Every L anchor needs a same-old-class cross-RX/day L positive')
        target_L=positive.to(z_L.dtype)/counts[:,cells].clamp_min(1)/domains[:,None]
        target=torch.cat((target_L,torch.zeros_like(ovalid,dtype=z_L.dtype)),1)
        logtarget=target.clamp_min(torch.finfo(z_L.dtype).tiny).log()
        logstudent=logits.log_softmax(1).masked_fill(~valid,0.)
        loss=_average((target*(logtarget-logstudent)).sum(1),torch.stack((y_L,rx_L,day_L),1))
    return dict(loss=loss,L_anchors=len(z_L),OE_candidates=len(z_OE),positive_pairs=int(positive.sum()),
        minimum_L_candidate_domains=int(ldomains.min()),minimum_OE_candidate_domains=int(odomains.min()),
        candidate_set_priors=[.5,.5],OE_OE_hard_negatives=False)
