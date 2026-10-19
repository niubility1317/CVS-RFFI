"""Train-only pseudo identity and leave-one-receiver-out feature targets."""
import torch
import torch.nn.functional as F
from . import design as d


def rx_indices(rx):
    if rx.ndim!=1 or rx.dtype!=torch.long:raise ValueError('Integer source RX vector required')
    values=torch.tensor(d.RX,device=rx.device)
    matches=rx[:,None]==values[None,:]
    if not matches.any(1).all():raise ValueError('Only registered source RX allowed')
    return matches.long().argmax(1)


class BankBuilder:
    def __init__(self):
        self.sums=torch.zeros(5,6,3,160,dtype=torch.float64)
        self.counts=torch.zeros(5,6,dtype=torch.long)
        self.seen=0;self.accepted=0

    @torch.no_grad()
    def update(self,features,rx,logits):
        n=len(features)
        if (n==0 or features.shape!=(n,3,160) or logits.shape!=(n,3,6) or len(rx)!=n or
            not torch.isfinite(features).all() or not torch.isfinite(logits).all()):
            raise ValueError('Finite teacher-specific source features/logits required')
        features=features.detach().cpu();logits=logits.detach().cpu();rx=rx.detach().cpu()
        ri=rx_indices(rx);prob=logits.softmax(-1);confidence,classes=prob.max(-1)
        accepted=(classes==classes[:,:1]).all(1)&(confidence>=d.CONFIDENCE).all(1)
        pseudo=classes[:,0];unit=F.normalize(features.double(),dim=-1)
        for group in range(5):
            for label in range(6):
                mask=accepted&(ri==group)&(pseudo==label)
                self.sums[group,label]+=unit[mask].sum(0);self.counts[group,label]+=mask.sum()
        self.seen+=n;self.accepted+=int(accepted.sum())
        return pseudo,accepted

    @torch.no_grad()
    def finalize(self):
        group_means=self.sums/self.counts.clamp_min(1)[:,:,None,None]
        prototypes=torch.zeros_like(group_means);valid=torch.zeros(5,6,dtype=torch.bool)
        for own in range(5):
            for label in range(6):
                others=self.counts[:,label]>0;others[own]=False
                if others.any():
                    value=group_means[others,label].mean(0)
                    if (value.norm(dim=-1)>1e-8).all():
                        prototypes[own,label]=F.normalize(value,dim=-1);valid[own,label]=True
        return IdentityBank(prototypes.float(),valid,self.counts.clone(),self.seen,self.accepted)


class IdentityBank:
    def __init__(self,prototypes,valid,counts,seen,accepted):
        self.prototypes=prototypes.detach();self.valid=valid.detach();self.counts=counts.detach()
        self.seen=seen;self.accepted=accepted

    @torch.no_grad()
    def lookup(self,rx,pseudo,accepted,device):
        ri=rx_indices(rx.cpu());pseudo=pseudo.detach().cpu();accepted=accepted.detach().cpu()
        if (pseudo.shape!=rx.shape or accepted.shape!=rx.shape or pseudo.dtype!=torch.long or
            accepted.dtype!=torch.bool or ((pseudo<0)|(pseudo>5)).any()):
            raise ValueError('Detached source pseudo identity/mask required')
        return self.prototypes[ri,pseudo].to(device), (accepted&self.valid[ri,pseudo]).to(device)

    def state(self):
        return dict(source_records=self.seen,consensus_accepted=self.accepted,
            accepted_fraction=self.accepted/max(self.seen,1),pseudo_RX_class_counts=self.counts.tolist(),
            otherRX_prototype_available=self.valid.tolist(),native_confidence=d.CONFIDENCE,
            class_source='unanimous original native heads only,not TX truth',TX_labels_read=0,target_reads=0,head_fit_updates=0,
            inference_uses_bank=False)


def identity_terms(embedding,logits,pseudo,accepted,prototype,prototype_mask,arm):
    if arm not in d.ARMS:raise ValueError('Registered identity arm required')
    n=len(pseudo)
    if (embedding.shape!=(n,160) or logits.shape!=(n,6) or prototype.shape!=embedding.shape or
        accepted.shape!=(n,) or prototype_mask.shape!=(n,)):
        raise ValueError('Observed-half identity targets required')
    native=F.cross_entropy(logits,pseudo.detach(),reduction='none')*accepted.detach()
    cosine=1-F.cosine_similarity(embedding,prototype.detach(),dim=-1)
    alignment=cosine*prototype_mask.detach() if arm==d.ARMS[1] else torch.zeros_like(native)
    return native,alignment
