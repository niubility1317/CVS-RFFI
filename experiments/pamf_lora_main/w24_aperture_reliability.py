"""Receive-only local noise cues from the part of native IQ invisible to W24."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InvisibleIQ(nn.Module):
    """Remove the row-space projection of the exact linear W24 aperture."""

    def __init__(self, indices, taps, edges, *, width):
        super().__init__()
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import splu

        indices, taps, edges = (np.asarray(a.detach().cpu().tolist()) for a in (indices,taps,edges))
        chirps, center, _ = indices.shape
        row = np.arange(chirps*(center+edges.shape[-1])).reshape(chirps,-1)
        columns = np.concatenate((indices.ravel(),edges.ravel()))
        rows = np.concatenate((np.broadcast_to(row[:,:center,None],indices.shape).ravel(),row[:,center:].ravel()))
        coefficients = np.concatenate((np.broadcast_to(taps,indices.shape).ravel(),np.ones(edges.size)))
        aperture = csr_matrix((coefficients,(rows,columns)),shape=(row.size,width),dtype=np.float64)
        aperture.eliminate_zeros()
        gram = splu((aperture@aperture.T).tocsc())
        dual = np.empty((row.size,width),dtype=np.float64)
        # Solve blocks of right-hand sides, keeping the temporary native matrix small.
        for offset in range(0,row.size,128):
            count = min(128,row.size-offset)
            basis = np.zeros((row.size,count))
            axis = np.arange(count)
            basis[offset+axis,axis] = 1.
            dual[offset:offset+count] = (aperture.T @ gram.solve(basis)).T
        observe = np.ascontiguousarray(aperture.toarray().T,dtype=np.float64)
        # The archived CUDA runtime predates the NumPy 2 tensor bridge.
        self.register_buffer('observe',torch.frombuffer(memoryview(observe),dtype=torch.float64).view(observe.shape),
                             persistent=False)
        self.register_buffer('lift',torch.frombuffer(memoryview(dual),dtype=torch.float64).view(dual.shape),
                             persistent=False)

    @torch.no_grad()
    def forward(self, raw):
        # Small singular values near native edges make single-precision lifting inaccurate.
        iq = torch.view_as_real(raw).transpose(1,2).flatten(0,1).double()
        hidden = (iq - (iq @ self.observe) @ self.lift).unflatten(0,(len(raw),2)).transpose(1,2)
        return torch.view_as_complex(hidden.to(raw.real.dtype).contiguous())


class VisibleIQ(torch.autograd.Function):
    """Orthogonal projection with the same adjoint in backward."""

    @staticmethod
    def forward(ctx, raw, projector):
        ctx.projector = projector
        return raw - projector(raw)

    @staticmethod
    def backward(ctx, gradient):
        return gradient - ctx.projector(gradient.contiguous()), None


def reliability(invisible, rms, *, localized):
    residual = invisible.unflatten(-1,(8,1024)) / rms
    bucket = residual.abs().square().unflatten(-1,(16,64)).mean(-1)
    if localized:
        power = F.interpolate(bucket,size=1024,mode='linear',align_corners=False)
    else:
        power = bucket.mean(-1,keepdim=True).expand(-1,-1,1024)
    return (power.clamp(1e-7,4.).log()/5.).unsqueeze(-2)


class LocalizedWhite:
    """Matched source-only nonstationarity with fixed per-packet noise RMS."""

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, noise):
        raw = noise.flatten(1)
        opts = dict(device=raw.device,dtype=raw.real.dtype,generator=self.rng)
        active = torch.rand((len(raw),1),**opts) < .5
        knots = torch.randn((len(raw),1,32),**opts).clamp(-2.,2.)
        gain = F.interpolate(knots,size=raw.shape[-1],mode='linear',align_corners=False)[:,0].exp()
        altered = raw*gain
        weight = (raw.abs().square().sum(-1,keepdim=True) /
                  altered.abs().square().sum(-1,keepdim=True).clamp_min(1e-12)).sqrt()
        return torch.where(active,altered*weight,raw).reshape_as(noise)
