"""Source-only local complex projection teacher for physical fusion."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def local_projection_target(anchor, physical, clean, window=17):
    if anchor.ndim != 3 or anchor.shape[1:] != (2, 256) or physical.shape != anchor.shape or clean.shape != anchor.shape:
        raise ValueError('Matching source IQ windows required')
    if window < 1 or window % 2 != 1 or window > 255:
        raise ValueError('Odd local window required')
    if not all(torch.isfinite(v).all() for v in (anchor, physical, clean)):
        raise FloatingPointError('Finite source IQ required')
    a=torch.complex(anchor[:,0],anchor[:,1])
    p=torch.complex(physical[:,0],physical[:,1])
    c=torch.complex(clean[:,0],clean[:,1])
    delta=p-a; residual=c-a
    product=delta.conj()*residual
    def mean(x):
        return F.avg_pool1d(x[:,None],window,stride=1,padding=window//2)[:,0]
    energy=mean(delta.abs().square())
    floor=a.abs().square().mean(1,keepdim=True).clamp_min(1e-12)*1e-8
    real=(mean(product.real)/energy.clamp_min(floor)).clamp(-1,1)
    imag=(mean(product.imag)/energy.clamp_min(floor)).clamp(-1,1)
    estimate=a+torch.complex(real,imag)*delta
    return torch.stack((estimate.real,estimate.imag),1)


def local_teacher_loss(output, anchor, physical, clean, groups):
    if output.shape != anchor.shape or groups.shape != (len(output),):
        raise ValueError('One output and receiver group per source record required')
    target=local_projection_target(anchor,physical,clean)
    energy=anchor.detach().square().mean((1,2)).clamp_min(1e-8)
    per_record=(output-target).square().mean((1,2))/energy
    return torch.stack([per_record[groups==g].mean() for g in groups.unique()]).mean()
