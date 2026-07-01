import math
from typing import Optional

import torch
import torch.nn.functional as F


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-7) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(x, dim=dim, eps=float(eps))


def angular_distance_deg(z: torch.Tensor, mu: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    z_n = safe_normalize(z, dim=-1, eps=eps)
    mu_n = safe_normalize(mu.to(device=z_n.device, dtype=z_n.dtype), dim=-1, eps=eps)
    cos = (z_n * mu_n).sum(dim=-1).clamp(-1.0 + float(eps), 1.0 - float(eps))
    return torch.rad2deg(torch.acos(cos))


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    a_n = safe_normalize(a, dim=-1, eps=eps)
    b_n = safe_normalize(b.to(device=a_n.device, dtype=a_n.dtype), dim=-1, eps=eps)
    t = t.to(device=a_n.device, dtype=a_n.dtype)
    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0 + float(eps), 1.0 - float(eps))
    omega = torch.acos(dot)
    so = torch.sin(omega).clamp_min(float(eps))
    out = torch.sin((1.0 - t) * omega) / so * a_n + torch.sin(t * omega) / so * b_n
    return safe_normalize(out, dim=-1, eps=eps)


def random_tangent_directions(
    mu: torch.Tensor,
    n: int,
    *,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    mu_n = safe_normalize(mu.view(1, -1), dim=1, eps=eps).squeeze(0)
    raw = torch.randn((int(n), mu_n.numel()), generator=generator, device=mu_n.device, dtype=mu_n.dtype)
    proj = raw - (raw @ mu_n).view(-1, 1) * mu_n.view(1, -1)
    return safe_normalize(proj, dim=1, eps=eps)


def rotate_on_sphere(mu: torch.Tensor, tangent_dir: torch.Tensor, angle_deg: torch.Tensor | float) -> torch.Tensor:
    mu_n = safe_normalize(mu.view(1, -1), dim=1).squeeze(0)
    u = safe_normalize(tangent_dir, dim=-1)
    if not torch.is_tensor(angle_deg):
        angle = torch.full((u.shape[0], 1), math.radians(float(angle_deg)), device=u.device, dtype=u.dtype)
    else:
        angle = torch.deg2rad(angle_deg.to(device=u.device, dtype=u.dtype)).view(-1, 1)
    out = torch.cos(angle) * mu_n.view(1, -1) + torch.sin(angle) * u
    return safe_normalize(out, dim=1)

