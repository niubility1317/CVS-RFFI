"""No TX truth or fitted parameters; this module is never needed for query state."""
from dataclasses import dataclass
import numpy as np
import torch

ARMS=('samerx_style','crossrx_style')
SOURCE_RX=(1,3,4,6,8)
P=.5;ALPHA=.1;EPS=1e-6;LAYERS=(0,2)


@dataclass(frozen=True)
class StylePlan:
    donor:torch.Tensor
    coefficient:torch.Tensor
    active:torch.Tensor
    common_eligible:np.ndarray
    attempted:np.ndarray
    physical_active:np.ndarray

    def counts(self):
        return dict(physical_records=len(self.physical_active),eligible=int(self.common_eligible.sum()),
                    attempted=int(self.attempted.sum()),active=int(self.physical_active.sum()),
                    paired_IQ_applications_per_layer=2*int(self.physical_active.sum()))


def make_plan(rx,pseudo,accepted,rng,arm,device='cpu',p=P,alpha=ALPHA):
    rx=np.asarray(rx);pseudo=np.asarray(pseudo);accepted=np.asarray(accepted);n=len(rx)
    if (arm not in ARMS or n==0 or rx.shape!=(n,) or pseudo.shape!=(n,) or accepted.shape!=(n,) or
        rx.dtype.kind not in 'iu' or pseudo.dtype.kind not in 'iu' or accepted.dtype!=np.bool_ or
        not set(rx.tolist()).issubset(SOURCE_RX) or (pseudo<0).any() or (pseudo>5).any() or
        not isinstance(rng,np.random.Generator) or not np.isfinite(p) or not 0<=p<=1 or not np.isfinite(alpha) or alpha<=0):
        raise ValueError('Registered source RX/pseudo identity/mask and fixed mixing inputs required')
    same=[];cross=[];index=np.arange(n)
    for i in range(n):
        identity=accepted[i]&accepted&(pseudo==pseudo[i])&(index!=i)
        same.append(np.flatnonzero(identity&(rx==rx[i])))
        cross.append(np.flatnonzero(identity&(rx!=rx[i])))
    common=np.asarray([bool(len(a) and len(b)) for a,b in zip(same,cross)],dtype=np.bool_)
    # Fixed draw counts keep physical selection and coefficients identical across arms.
    partner_draw=rng.random(n);active_draw=rng.random(n)
    coefficient=rng.beta(alpha,alpha,n).astype(np.float32)
    attempted=common&(active_draw<p)
    active=attempted&(coefficient<1.)  # Float32 lambda=1 is an exact no-op,logged separately.
    donor=index.copy();candidates=same if arm==ARMS[0] else cross
    for i in np.flatnonzero(active):
        choices=candidates[i];donor[i]=choices[min(int(partner_draw[i]*len(choices)),len(choices)-1)]
    coefficient=np.where(active,coefficient,1.).astype(np.float32)
    return StylePlan(torch.tensor(np.concatenate((donor,donor+n)).tolist(),dtype=torch.long,device=device),
        torch.tensor(np.tile(coefficient,2).tolist(),dtype=torch.float32,device=device)[:,None,None],
        torch.tensor(np.tile(active,2).tolist(),dtype=torch.bool,device=device),common,attempted,active)


def mix_statistics(features,plan):
    if (features.ndim!=3 or features.dtype!=torch.float32 or len(features)!=len(plan.active) or
        plan.donor.shape!=(len(features),) or plan.donor.dtype!=torch.long or
        plan.coefficient.shape!=(len(features),1,1) or plan.active.dtype!=torch.bool or
        plan.donor.device!=features.device or plan.coefficient.device!=features.device or plan.active.device!=features.device):
        raise ValueError('Current source observed+clean half-batch and matching plan required')
    mean=features.mean(-1,keepdim=True).detach()
    std=(features.var(-1,keepdim=True,unbiased=False)+EPS).sqrt().detach()
    coefficient=plan.coefficient.detach()
    mixed_mean=coefficient*mean+(1-coefficient)*mean[plan.donor]
    mixed_std=coefficient*std+(1-coefficient)*std[plan.donor]
    mixed=(features-mean)/std*mixed_std+mixed_mean
    return torch.where(plan.active[:,None,None],mixed,features)
