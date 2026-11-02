"""Match a restored signal to a separately simulated source view."""
import torch
from torch.nn import functional as F

CYCLE_COUNTS = [0, 0.0]


def next_view(received):
    if received.ndim < 1 or len(received) == 0 or len(received) % 3:
        raise ValueError('Three equally sized source view blocks are required')
    n = len(received) // 3
    return torch.cat((received[n:2*n], received[2*n:], received[:n]), dim=0)


def cycle_residual(output, target):
    if output.ndim != 3 or output.shape[1:] != (2, 256) or target.shape != output.shape:
        raise ValueError('Matching 256-point IQ windows are required')
    if not torch.isfinite(output).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Finite source views are required')
    target_complex = torch.complex(target[:, 0, 64:-64], target[:, 1, 64:-64]).detach()
    with torch.no_grad():
        pad = F.pad(output.detach(), (64, 64), mode='reflect')
        windows = torch.complex(pad[:, 0], pad[:, 1]).unfold(-1, 256, 1)[:, :, 64:-64]
        gain = (windows.conj() * target_complex[:, None]).sum(-1) / windows.abs().square().sum(-1).clamp_min(1e-20)
        error = (gain[:, :, None] * windows - target_complex[:, None]).abs().square().mean(-1)
        choice = error.argmin(1)
    pad = F.pad(output, (64, 64), mode='reflect')
    windows = torch.complex(pad[:, 0], pad[:, 1]).unfold(-1, 256, 1)
    selected = windows[torch.arange(len(output), device=output.device), choice, 64:-64].reshape(len(output), 4, 32)
    wanted = target_complex.reshape(len(output), 4, 32)
    with torch.no_grad():
        gain = (selected.detach().conj() * wanted).sum(-1) / selected.detach().abs().square().sum(-1).clamp_min(1e-20)
    residual = (gain[:, :, None] * selected - wanted).abs().square().mean((1, 2))
    return residual / wanted.abs().square().mean((1, 2)).clamp_min(1e-8)


def crossview_objective(received, output, groups):
    if groups.shape != (len(received),):
        raise ValueError('One receiver group per source view is required')
    per = cycle_residual(output, next_view(received))
    CYCLE_COUNTS[0] += len(per)
    CYCLE_COUNTS[1] += float(per.detach().sum())
    return 0.5 * torch.stack([per[groups == group].mean() for group in groups.unique()]).mean()
