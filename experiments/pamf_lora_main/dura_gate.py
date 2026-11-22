"""Source-trained bypass classifier for LoRa DURA, independent of identity."""
import torch
from torch import nn
from torch.nn import functional as F

from .dura_repair import PacketRestorer, no_tone
from .refine_g import ActiveDura


class ImpairmentGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.tracker=PacketRestorer('consensus')
        self.net=nn.Sequential(nn.Linear(11,24),nn.SiLU(),nn.Linear(24,1))
        self.register_buffer('feature_mean',torch.zeros(11))
        self.register_buffer('feature_std',torch.ones(11))
        self.register_buffer('clean_boundary',torch.tensor(0.))

    @torch.no_grad()
    def features(self,x):
        w,_,q,median,_=self.tracker.track(x)
        r=w.abs().square().mean(-1).clamp_min(1e-8).sqrt()
        residual=no_tone(w)/r[...,None]
        variation=(residual-residual.mean(1,keepdim=True)).abs().square().mean((1,2))
        envelope=w.abs()/r[...,None]
        jump=(residual[...,1:]-residual[...,:-1]).abs().square().mean((1,2))
        env_var=(envelope-envelope.mean(1,keepdim=True)).square().mean((1,2))
        sd=envelope.std(-1)
        coarse=self.tracker.reference.coherence(x.flatten(0,1)).reshape(len(x),6).median(1).values
        return torch.stack((median,q.min(1).values,q.max(1).values,q.std(1),r.std(1)/r.mean(1),
                  variation,jump,env_var,sd.mean(1),sd.std(1),coarse),dim=1)

    def logits(self,x):
        return self.net((x-self.feature_mean)/self.feature_std).flatten()

    @torch.no_grad()
    def impairment(self,x):
        a=self.features(x)
        margin=self.logits(a)-self.clean_boundary
        return torch.where(a[:,0]>.20,margin,torch.zeros_like(margin))


def fit_gate(gate,clean,practical,epochs,seed,calibration):
    """Inputs are feature vectors from source IQ only; no TX or F arguments."""
    # Calibration accepts additional source U clean observations. Those are
    # used solely to prevent receiver fingerprints from triggering the gate.
    x=torch.cat((clean,practical))
    y=torch.cat((x.new_zeros(len(clean)),x.new_ones(len(practical))))
    with torch.no_grad():
        gate.feature_mean.copy_(x.mean(0))
        gate.feature_std.copy_(x.std(0).clamp_min(1e-5))
    rng=torch.Generator(device=x.device).manual_seed(seed)
    opt=torch.optim.AdamW(gate.net.parameters(),lr=.002,weight_decay=.01)
    history=[]
    for _ in range(epochs):
        total=x.new_zeros(())
        for ids in torch.randperm(len(x),device=x.device,generator=rng).split(512):
            ys=y[ids]
            z=gate.logits(x[ids])
            loss=F.binary_cross_entropy_with_logits(z,ys,weight=torch.where(ys==0,5.,1.))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total+=loss.detach()*len(ids)
        history.append(float(total/len(x)))
    with torch.no_grad():
        edge=max(gate.logits(clean).max(),gate.logits(calibration).max())
        gate.clean_boundary.copy_(edge+1e-4)
    return history


class RiskDura(ActiveDura):
    """Only change the bypass decision; reuse the frozen chirp DURA proposal."""
    def __init__(self,gate,extend):
        super().__init__()
        self.gate=gate
        self.extend=extend

    @torch.no_grad()
    def strength(self,x):
        old=super().strength(x)
        margin=self.gate.impairment(x)
        if self.extend:
            extra=.30*margin.clamp(0.,1.)
            old=torch.where(old>0,old,extra)
        return torch.where(margin>0,old,torch.zeros_like(old))
