"""Reuse nonphysical U logits for label-free RDrop on the rejected subset."""
import torch


RECIPE=dict(ordinary_pair_weight=.5,mixed_nonphysical_pair_weight=.25,raw_pair_weight=.25,
    CE_normalizer=1.25,hard_label_gate=.95,physical_As_KL_weight=0.,
    hard_CE_or_contrast_on_rejected=False,extra_model_forwards=0,new_head=False)


def rejected_pair_kl(a,b,accepted):
    if (a.ndim!=2 or a.shape!=b.shape or accepted.shape!=a.shape[:1]
            or accepted.dtype!=torch.bool or a.device!=b.device or accepted.device!=a.device
            or not a.is_floating_point() or a.dtype!=b.dtype or not bool(torch.isfinite(a).all())
            or not bool(torch.isfinite(b).all())):
        raise ValueError('Finite paired logits and the unchanged boolean raw acceptance mask required')
    if not len(a):return (a.sum()+b.sum())*0.
    p,q=a.log_softmax(1),b.log_softmax(1)
    kl=.5*((p.exp()-q.exp())*(p-q)).sum(1)
    return (kl*(~accepted)).mean()


def rejected_rdrop(logits,accepted,physical):
    """Return the extra base loss before the existing external U weight of six."""
    if (len(logits)!=6 or accepted.ndim!=1 or not len(accepted) or len(accepted)%2
            or accepted.dtype!=torch.bool or physical.dtype!=torch.bool or physical.shape!=accepted.shape
            or physical.device!=accepted.device or int(physical.sum())*2!=len(accepted)
            or any(v.ndim!=2 or v.shape!=logits[0].shape or len(v)!=len(accepted) for v in logits)):
        raise ValueError('Six current U logits and the original exact mixed-half partition required')
    ordinary=rejected_pair_kl(logits[0],logits[1],accepted)
    mixed=rejected_pair_kl(logits[2][~physical],logits[3][~physical],accepted[~physical])
    raw=rejected_pair_kl(logits[4],logits[5],accepted)
    loss=(.5*ordinary+.25*mixed+.25*raw)/1.25
    rejected=int((~accepted).sum());nonphysical=int((~accepted&~physical).sum())
    return loss,dict(rejected_U=rejected,rejected_nonphysical_half=nonphysical,
        rejected_RDrop_pairs=2*rejected+nonphysical,reused_prediction_views=2*(2*rejected+nonphysical),
        ordinary_KL=float(ordinary.detach()),mixed_nonphysical_KL=float(mixed.detach()),raw_KL=float(raw.detach()),
        base_loss=float(loss.detach()),physical_As_KL_weight=0.,hard_label_or_CE_on_rejected=False,
        extra_model_forwards=0,new_head=False)
