"""Bounded label-independent linear channel diversity on labeled source IQ."""
import math
import torch
import torch.nn.functional as F

def bounded_echo(x,*,seed,epoch,step,probability=.5,maximum=.15,ramp_epochs=40,delays=(1,2,4)):
    if x.ndim!=3 or x.shape[1]!=2:raise ValueError('Expected Bx2xT real IQ')
    if not 0<=probability<=1 or not 0<=maximum<1 or ramp_epochs<1:raise ValueError('Invalid channel bound')
    if any(d<1 or d>=x.shape[-1] for d in delays):raise ValueError('Invalid delay')
    g=torch.Generator().manual_seed(int(seed)+int(epoch)*100003+int(step)*101)
    n=len(x);active=torch.rand(n,generator=g)<probability
    strength=maximum*min(1.,max(0.,(epoch-1)/ramp_epochs))
    radius=torch.rand(n,len(delays),generator=g)
    radius=radius/radius.sum(1,keepdim=True).clamp_min(1e-12)
    radius*=strength*torch.rand(n,1,generator=g)*active[:,None]
    theta=2*math.pi*torch.rand(n,len(delays),generator=g)
    re=(radius*theta.cos()).to(device=x.device,dtype=x.dtype)
    im=(radius*theta.sin()).to(device=x.device,dtype=x.dtype)
    value=x.clone()
    for j,delay in enumerate(delays):
        shifted=F.pad(x[...,:-delay],(delay,0))
        rotated=torch.stack((-shifted[:,1],shifted[:,0]),dim=1)
        value=value+re[:,j,None,None]*shifted+im[:,j,None,None]*rotated
    # Preserve packet energy; do not inject PA/IQ-imbalance or transmitter noise.
    scale=(x.square().mean((1,2),keepdim=True)/value.square().mean((1,2),keepdim=True).clamp_min(1e-12)).sqrt()
    value=value*scale
    mask=(active & (strength>0)).to(x.device)
    value=torch.where(mask[:,None,None],value,x)
    return value,dict(dg_selected=int((active & (strength>0)).sum()),dg_l1_bound=strength,
                      dg_actual_max=float(radius.sum(1).max()),dg_input_samples=n)

def satellite_keep(seed,epoch,step,probability):
    if not 0<probability<=1:raise ValueError('Invalid probability')
    g=torch.Generator().manual_seed(int(seed)+77003+int(epoch)*200003+int(step)*211)
    return bool(torch.rand((),generator=g)<probability)
