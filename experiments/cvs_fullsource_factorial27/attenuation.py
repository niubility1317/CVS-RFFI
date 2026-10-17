"""A bounded IQ observation perturbation, not a fitted RX hardware model."""
from collections import Counter
import numpy as np
import torch

KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
WIDTH=16
MAX_DEPTH=.5
RELATIVE_LIMIT=.1

@torch.no_grad()
def attenuate(x,starts):
    if x.ndim!=3 or x.shape[1:]!=(2,256) or x.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(x).all()):
        raise ValueError('Finite float32/64 IQ256 required')
    if starts.shape!=(len(x),) or starts.dtype!=torch.long or starts.device!=x.device or bool(((starts<0)|(starts>256-WIDTH)).any()):
        raise ValueError('One complete nonwrapping local window per packet required')
    k=torch.arange(256,device=x.device)[None,:]-starts[:,None]
    inside=(k>=0)&(k<WIDTH)
    w=torch.sin(torch.pi*(k.double()+.5)/WIDTH).square()*inside
    w=w.to(x.dtype)
    energy=x.double().square().sum((1,2))
    weighted=(x.double()*w[:,None,:].double()).square().sum((1,2))
    # Float32 margin prevents rounded output from exceeding the fixed 10% bound.
    depth=torch.minimum(torch.full_like(energy,MAX_DEPTH),
        RELATIVE_LIMIT*(1-2e-6)*torch.sqrt(energy/weighted.clamp_min(torch.finfo(torch.float64).tiny)))
    depth=torch.where(weighted>0,depth,torch.full_like(depth,MAX_DEPTH)).to(x.dtype)
    out=x-x*(depth[:,None,None]*w[:,None,:])
    delta=(out.double()-x.double()).square().sum((1,2))
    denom=energy.clamp_min(torch.finfo(torch.float64).tiny)
    relative=torch.where(energy>0,torch.sqrt(delta/denom),torch.zeros_like(energy))
    removed=torch.where(energy>0,1-out.double().square().sum((1,2))/denom,torch.zeros_like(energy))
    outside=(out-x).masked_select((~inside)[:,None,:].expand_as(x))
    outside_error=float(outside.abs().max()) if outside.numel() else 0.
    if outside_error!=0. or bool((relative>RELATIVE_LIMIT+1e-7).any()) or bool((removed<-1e-12).any()) or not bool(torch.isfinite(out).all()):
        raise ValueError('Local observation attenuation violated its fixed finite bound')
    return out,dict(depth=depth.tolist(),relative_L2=relative.tolist(),removed_power_fraction=removed.tolist(),
        zero_energy=(energy==0).tolist(),outside_max_error=outside_error)

class LocalAttenuation:
    def __init__(self,seed,fit=(1,3,4,6,8)):
        self.seed=int(seed);self.fit=tuple(fit);self.calls=0;self.packets=0
        self.ids=set();self.starts=Counter();self.max_relative=0.;self.max_depth=0.;self.max_removed=0.
        self.sum_relative=0.;self.sum_removed=0.;self.zero_energy=0
    def apply(self,x,meta,step):
        if x.shape!=(128,2,256) or type(step) is not int or step!=self.calls+1 or step not in range(1,6661):
            raise ValueError('One ordered original L128 observation slot required')
        vals=[meta[k].tolist() for k in KEYS]
        if any(len(v)!=128 for v in vals):raise ValueError('Complete original physical metadata required')
        ids=list(zip(*vals))
        if any(p[0]!=1 or p[1] not in self.fit or p[2] not in (1,2,3) or any(int(v)!=v or v<0 for v in p) for p in ids):
            raise ValueError('Only current allowed fitL physical IDs')
        # Position never depends on TX, RX, loss, confidence or waveform quality.
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]))
        starts=rng.integers(0,257-WIDTH,size=128).tolist()
        out,stats=attenuate(x,torch.tensor(starts,device=x.device,dtype=torch.long))
        self.calls=step;self.packets+=128;self.ids.update(ids);self.starts.update(starts)
        self.max_relative=max(self.max_relative,max(stats['relative_L2']));self.max_depth=max(self.max_depth,max(stats['depth']))
        self.max_removed=max(self.max_removed,max(stats['removed_power_fraction']))
        self.sum_relative+=sum(stats['relative_L2']);self.sum_removed+=sum(stats['removed_power_fraction']);self.zero_energy+=sum(stats['zero_energy'])
        return out,[list(p) for p in ids],dict(starts=starts,**stats)
    def summary(self):
        return dict(calls=self.calls,packets=self.packets,physical_IDs=len(self.ids),width=WIDTH,start_positions=len(self.starts),
            start_counts={str(k):v for k,v in sorted(self.starts.items())},max_relative_L2=self.max_relative,max_depth=self.max_depth,
            max_removed_power_fraction=self.max_removed,mean_relative_L2=self.sum_relative/max(1,self.packets),
            mean_removed_power_fraction=self.sum_removed/max(1,self.packets),zero_energy_packets_kept=self.zero_energy,
            outside_max_error=0.,TX_used_for_selection=False,RX_used_for_selection=False,global_RNG_advance=False,
            new_normalization=False,time_indices_changed=False,boundary_extension=False,added_noise=False,synthetic_RX_hardware=False)
