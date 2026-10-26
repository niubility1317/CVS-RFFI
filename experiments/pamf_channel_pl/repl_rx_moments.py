"""Detached L-only receiver moments and class-mean-protected U extrapolation."""
import torch
from torch.nn import functional as F


class LabeledReceiverMoments:
    def __init__(self,means,counts,receivers=(1,3,4,6,8),decay=.9):
        self.receivers=tuple(receivers);self.decay=decay
        if (means.shape!=(6,len(self.receivers),160) or counts.shape!=means.shape[:2] or counts.dtype!=torch.long
                or len(set(self.receivers))!=len(self.receivers) or decay!=.9
                or not bool(torch.isfinite(means).all()) or bool((counts<=0).any())
                or bool((means.norm(dim=-1)>1.0001).any())):
            raise ValueError('Complete finite normalized-L class/RX moments and fixed decay .9 required')
        self.means=means.detach().float().clone();self.counts=counts.detach().long().clone()
        self.updates=0;self.exposures=0;self.metrics={}

    def _rx(self,rx):
        allowed=torch.tensor(self.receivers,device=rx.device,dtype=torch.long)
        match=rx[:,None]==allowed[None]
        if rx.ndim!=1 or rx.dtype!=torch.long or not bool(match.any(1).all()):raise ValueError('Registered source RX required')
        return match.long().argmax(1)

    @torch.no_grad()
    def observe(self,features,labels,rx,visible,role='L_s'):
        n=len(features)
        if (role!='L_s' or features.shape!=(n,160) or labels.shape!=(n,) or labels.dtype!=torch.long
                or visible.shape!=(n,) or visible.dtype!=torch.bool or not bool(visible.all())
                or bool(((labels<0)|(labels>=6)).any()) or not bool(torch.isfinite(features).all())
                or bool((features.norm(dim=1)<=1e-8).any())):
            raise ValueError('Only current visible true L features may update RX moments')
        index=self._rx(rx);z=F.normalize(features.detach(),dim=1)
        self.means=self.means.to(z.device);self.counts=self.counts.to(z.device)
        for cls in range(6):
            for r in range(len(self.receivers)):
                mask=(labels==cls)&(index==r);count=int(mask.sum())
                if count:
                    self.means[cls,r].mul_(self.decay).add_(z[mask].mean(0),alpha=1-self.decay)
                    self.counts[cls,r]+=count
        self.updates+=1;self.exposures+=n

    @torch.no_grad()
    def protected_offsets(self,device,dtype):
        self.means=self.means.to(device);self.counts=self.counts.to(device)
        means=self.means.double();centers=means.mean(1);bias=means.mean(0)-means.mean((0,1))
        _,s,vh=torch.linalg.svd(centers-centers.mean(0),full_matrices=False)
        rank=int((s>max(float(s[0])*1e-6,1e-12)).sum());basis=vh[:rank]
        protected=bias-(bias@basis.T)@basis
        total=(means-centers[:,None]).square().sum(-1).mean();shared=bias.square().sum(-1).mean();energy=protected.square().sum(-1).mean()
        self.metrics=dict(class_rank=rank,protected_RX_fraction=float(energy/total.clamp_min(1e-20)),
            shared_RX_fraction=float(shared/total.clamp_min(1e-20)),moment_updates=self.updates,L_stat_exposures=self.exposures)
        return protected.to(dtype)

    def extrapolate(self,h,pseudo,accepted,rx,partner,valid):
        n=len(h)
        if (h.shape!=(n,160) or not bool(torch.isfinite(h).all()) or bool((h.norm(dim=1)<=1e-8).any())
                or any(v.shape!=(n,) for v in (pseudo,accepted,rx,partner,valid)) or pseudo.dtype!=torch.long
                or partner.dtype!=torch.long or accepted.dtype!=torch.bool or valid.dtype!=torch.bool
                or bool(((pseudo<0)|(pseudo>=6)).any()) or (n and (int(partner.min())<0 or int(partner.max())>=n))):
            raise ValueError('Aligned actual full features and accepted same-pseudo source pairs required')
        index=self._rx(rx)
        if bool((valid&(~accepted|~accepted[partner]|(pseudo!=pseudo[partner])|(rx==rx[partner]))).any()):
            raise ValueError('Invalid identity-protected receiver pair')
        offset=self.protected_offsets(h.device,h.dtype)
        direction=(offset[index]-offset[index[partner]]).detach()
        value=F.normalize(h,dim=1)+.25*direction*valid[:,None]
        self.metrics['eligible_pairs']=int(valid.sum());self.metrics['alpha25_mean_displacement']=float((.25*direction[valid]).norm(dim=1).mean()) if bool(valid.any()) else 0.
        return value

    def state_dict(self):
        return dict(receivers=self.receivers,decay=self.decay,means=self.means.detach().cpu().clone(),counts=self.counts.detach().cpu().clone(),
            updates=self.updates,exposures=self.exposures)

    def load_state_dict(self,state):
        if tuple(state['receivers'])!=self.receivers or state['decay']!=self.decay or state['updates']<0 or state['exposures']<0:
            raise ValueError('Own L moment state and fixed receiver coordinates required')
        restored=type(self)(state['means'],state['counts'],self.receivers,self.decay)
        self.means,self.counts=restored.means,restored.counts;self.updates=state['updates'];self.exposures=state['exposures']
