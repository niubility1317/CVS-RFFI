"""Source-native SupCon with cross-receiver/day positives and optional OE negatives."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from . import ground_centered_core as centered
from . import logit_energy_core as native

ARMS=('KCON','OECON')
TEMPERATURE=.1
CONTRAST_WEIGHT=.1


def _features(value,n=None,device=None):
    if (not torch.is_tensor(value) or value.dtype!=torch.float32 or value.ndim!=2 or not len(value)
            or value.shape[1]!=160 or (n is not None and len(value)!=n)
            or (device is not None and value.device!=device) or not bool(torch.isfinite(value).all())):
        raise ValueError('Aligned finite FP32 native raw160 source features required')
    return value


def contrastive(rawL,y,domainL,rawOE,arm):
    """Mean positive log probability; OE is only outside-six membership."""
    if arm not in ARMS:raise ValueError('Only matched KCON/OECON source arms are allowed')
    L=_features(rawL);OE=_features(rawOE,device=L.device)
    y=native._codes(y,len(L),L.device,6,'known L labels')
    d=native._codes(domainL,len(L),L.device,15,'public L RX/day domains')
    # These are exactly the original cosine head's coordinates, including eps.
    u=F.normalize(L,dim=1,eps=1e-4);v=F.normalize(OE,dim=1,eps=1e-4)
    same=y[:,None]==y[None]
    diagonal=torch.eye(len(L),dtype=torch.bool,device=L.device)
    positive=same&(d[:,None]//3!=d[None]//3)&(d[:,None]%3!=d[None]%3)
    negative=~same
    allowed=positive|negative
    counts=positive.sum(1)
    if bool((counts==0).any()):raise ValueError('Every known anchor needs a same-TX cross-RX and cross-day positive')
    similarities=(u@u.T)/TEMPERATURE;oe_similarities=(u@v.T)/TEMPERATURE
    numerator=similarities.masked_fill(~positive,0.).sum(1)/counts
    known_denominator=torch.logsumexp(similarities.masked_fill(~allowed,-torch.inf),dim=1)
    with_OE_denominator=torch.logaddexp(known_denominator,torch.logsumexp(oe_similarities,dim=1))
    known=(known_denominator-numerator).mean();outside=(with_OE_denominator-numerator).mean()
    selected=known if arm=='KCON' else outside
    with torch.no_grad():
        off=~torch.eye(len(OE),dtype=torch.bool,device=OE.device)
        mean_OE_cos=(v@v.T)[off].mean() if len(OE)>1 else v.new_zeros(())
        diagnostics=dict(positive_cosine_mean=(similarities* TEMPERATURE)[positive].mean(),
            known_negative_cosine_mean=(similarities*TEMPERATURE)[negative].mean() if bool(negative.any()) else L.new_zeros(()),
            OE_negative_cosine_mean=(oe_similarities*TEMPERATURE).mean(),OE_offdiagonal_cosine_mean=mean_OE_cos)
    return dict(contrast=selected,known_only_contrast=known,with_OE_contrast=outside,
        with_OE_minus_known=outside-known,contrast_weight=CONTRAST_WEIGHT,contrast_temperature=TEMPERATURE,
        anchor_count=len(L),positive_pair_count=int(positive.sum()),known_negative_pair_count=int(negative.sum()),
        same_class_excluded_pair_count=int((same&~positive&~diagonal).sum()),
        selected_OE_negative_pair_count=len(L)*len(OE) if arm=='OECON' else 0,
        available_OE_negative_pair_count=len(L)*len(OE),OE_positive_pair_count=0,OE_true_labels_used=False,
        selected_contrast='known_only' if arm=='KCON' else 'known_and_OE',**diagnostics)


def compute(rawL,logitsL,y,teacher,domainL,rawOE,logitsOE,domainOE,arm):
    L=native._logits(logitsL);OE=native._logits(logitsOE)
    _features(rawL,len(L),L.device);_features(rawOE,len(OE),L.device)
    base=centered.compute(L,y,teacher,domainL,OE,domainOE)
    contrast=contrastive(rawL,y,domainL,rawOE,arm)
    return dict(base,**contrast,total=base['total']+CONTRAST_WEIGHT*contrast['contrast'])
