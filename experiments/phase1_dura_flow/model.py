"""DURA-anchored conditional IQ flow with source-only velocity matching."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase1_dura_blendgate.model import align_to_anchor


class DuraFlow(nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.entry = nn.Conv1d(9, width, 7, padding=3)
        self.blocks = nn.ModuleList(nn.Conv1d(width, width, 5, dilation=d, padding=2*d) for d in (1, 2, 4, 8))
        self.filter_real = nn.Conv1d(1, 2, 7, padding=3, bias=False)
        self.filter_imag = nn.Conv1d(1, 2, 7, padding=3, bias=False)
        self.gate_output = nn.Conv1d(width, 18, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def velocity(self, received, anchor, state, log_variance, time):
        n = len(received)
        if (received.ndim != 3 or received.shape[1:] != (2, 256)
                or anchor.shape != received.shape or state.shape != received.shape
                or log_variance.shape != (n,) or time.shape != (n,)):
            raise ValueError('Matching IQ windows, blind noise and per-record time required')
        if not all(torch.isfinite(v).all() for v in (received, anchor, state, log_variance, time)):
            raise FloatingPointError('Finite per-record evidence required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        x = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        a = torch.complex(anchor[:, 0], anchor[:, 1]) / scale[:, 0]
        z = torch.complex(state[:, 0], state[:, 1]) / scale[:, 0]
        cx = (z * x.conj())
        ca = (z * a.conj())
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8) / 8)[:, None].expand(-1, 256)
        features = torch.stack((x.abs().clamp_max(16).log1p(), a.abs().clamp_max(16).log1p(),
                                z.abs().clamp_max(16).log1p(), cx.real.clamp(-16,16), cx.imag.clamp(-16,16),
                                ca.real.clamp(-16,16), ca.imag.clamp(-16,16), noise,
                                time[:, None].expand(-1, 256)), dim=1)
        h = F.silu(self.entry(features))
        for layer in self.blocks:
            h = h + .2 * F.silu(layer(h))
        def filtered(q):
            real = self.filter_real(q.real[:, None]) - self.filter_imag(q.imag[:, None])
            imag = self.filter_imag(q.real[:, None]) + self.filter_real(q.imag[:, None])
            return torch.complex(real, imag)
        basis = torch.cat((x[:, None], a[:, None], z[:, None], filtered(x), filtered(a), filtered(z)), dim=1)
        weight = self.gate_output(h).tanh().reshape(n, 2, 9, 256)
        velocity = (torch.complex(weight[:,0], weight[:,1]) * basis).sum(1) / 9
        return torch.stack((velocity.real, velocity.imag), dim=1) * scale

    def forward(self, received, initial, log_variance):
        state = initial
        for i in range(4):
            time = received.new_full((len(received),), i / 4)
            state = state + self.velocity(received, initial, state, log_variance, time) / 4
        return state


def flow_matching_loss(model, received, anchor, clean, log_variance, groups, *, time=None):
    n = len(received)
    if clean.shape != received.shape or anchor.shape != received.shape or groups.shape != (n,):
        raise ValueError('Paired IQ and one receiver group per record required')
    if time is None:
        time = torch.rand(n, device=received.device, dtype=received.dtype)
    if time.shape != (n,) or not torch.isfinite(time).all() or not ((time >= 0) & (time <= 1)).all():
        raise ValueError('One finite source interpolation time in [0,1] required')
    anchor = anchor.detach()
    with torch.no_grad():
        endpoint = align_to_anchor(clean.detach(), anchor, preserve_edges=True)
        target = endpoint - anchor
        state = anchor + time[:, None, None] * target
        energy = endpoint.square().mean((1,2)).clamp_min(1e-8)
    prediction = model.velocity(received.detach(), anchor, state, log_variance.detach(), time.detach())
    per = (prediction - target).square().mean((1,2)) / energy
    loss = torch.stack([per[groups == g].mean() for g in groups.unique()]).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Finite source flow-matching loss required')
    return loss
