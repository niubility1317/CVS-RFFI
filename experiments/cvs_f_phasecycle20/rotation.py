from collections import Counter
import numpy as np
import torch

KEYS=('eq_i','rx_i','day_i','sig_i','base_index')
def rotate(x,k):
    if x.ndim!=3 or x.shape[1:]!=(2,256) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):raise ValueError('Finite original IQ[batch,2,256] required')
    if k.shape!=(len(x),) or k.dtype!=torch.long or k.device!=x.device or bool(((k<0)|(k>3)).any()):raise ValueError('One C4 element per physical packet required')
    i,q=x[:,0],x[:,1]
    values=torch.stack((x,torch.stack((-q,i),1),-x,torch.stack((q,-i),1)),1)
    return values[torch.arange(len(x),device=x.device),k]

class Cycle:
    def __init__(self,seed,fit=(3,4,6,8)):
        self.seed=int(seed);self.fit=tuple(fit);self.rng=np.random.default_rng(self.seed);self.offsets={};self.visits=Counter();self.coverage=Counter();self.calls=0;self.packets=0
        self.max_power_error=0.;self.max_inverse_error=0.
    def apply(self,x,meta):
        if x.shape!=(128,2,256):raise ValueError('Only original L128 enters C4')
        vals=[meta[k].tolist() for k in KEYS]
        if any(len(v)!=len(x) for v in vals):raise ValueError('One physical metadata row per IQ required')
        ids=list(zip(*vals))
        if any(p[0]!=1 or p[1] not in self.fit or p[2] not in (1,2,3) or any(int(v)!=v or v<0 for v in p) for p in ids):raise ValueError('Original allowed source L metadata required')
        rotations=[]
        for p in ids:
            # No true label, waveform statistic or loss enters phase selection.
            if p not in self.offsets:self.offsets[p]=int(self.rng.integers(3))
            offset=self.offsets[p]
            k=1+(offset+self.visits[p])%3
            self.visits[p]+=1;self.coverage[(p,k)]+=1;rotations.append(k)
        k=torch.tensor(rotations,device=x.device,dtype=torch.long);out=rotate(x,k)
        power=float((x.square().sum(1)-out.square().sum(1)).abs().max())
        inverse=float((x-rotate(out,(4-k)%4)).abs().max())
        if power!=0. or inverse!=0.:raise ValueError('C4 changed power or original complex information')
        self.calls+=1;self.packets+=len(x);self.max_power_error=max(self.max_power_error,power);self.max_inverse_error=max(self.max_inverse_error,inverse)
        return out,ids,rotations
    def summary(self):
        imbalance=max((max(self.coverage[(p,k)] for k in (1,2,3))-min(self.coverage[(p,k)] for k in (1,2,3)) for p in self.visits),default=0)
        return dict(calls=self.calls,packets=self.packets,physical_IDs=len(self.visits),physical_ID_rotations=len(self.coverage),
            max_per_ID_rotation_count_imbalance=imbalance,max_pointwise_power_error=self.max_power_error,max_exact_inverse_error=self.max_inverse_error,
            TX_used_for_selection=False,global_RNG_advance=False,new_normalization=False,boundary_change=False,added_noise=False)
