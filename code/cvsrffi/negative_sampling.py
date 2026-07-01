from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from cvsrffi.component_geometry import angular_distance_deg, random_tangent_directions, rotate_on_sphere, safe_normalize, slerp


@dataclass
class NegativeBatch:
    z: torch.Tensor
    kind: list[str]
    target: torch.Tensor
    source_class: list[int | None]
    source_component: list[int | None]
    debug: dict[str, Any]


def _generator(seed: Optional[int], device: torch.device) -> torch.Generator:
    gen = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    return gen


def sample_shell_negatives(bank, n_per_component: int = 8, gamma_deg: float = 1.0, seed: Optional[int] = None) -> NegativeBatch:
    comps = list(bank.iter_components())
    if not comps:
        return NegativeBatch(torch.empty(0, 0), [], torch.empty(0, dtype=torch.long), [], [], {"empty": True})
    device = comps[0].mu.device
    gen = _generator(seed, device)
    rows = []
    kinds = []
    src_cls = []
    src_comp = []
    for comp in comps:
        dirs = random_tangent_directions(comp.mu, int(n_per_component), generator=gen)
        lo = float(comp.r_accept_deg) + float(gamma_deg)
        hi = max(lo, float(comp.r_vac_deg))
        radii = torch.empty((int(n_per_component),), device=device).uniform_(lo, hi, generator=gen)
        z = rotate_on_sphere(comp.mu, dirs, radii)
        rows.append(z)
        kinds.extend(["shell"] * int(n_per_component))
        src_cls.extend([int(comp.class_id)] * int(n_per_component))
        src_comp.extend([int(comp.component_id)] * int(n_per_component))
    out = torch.cat(rows, dim=0)
    return NegativeBatch(out, kinds, torch.full((out.size(0),), -1, dtype=torch.long), src_cls, src_comp, {})


def sample_tail_outward_negatives(
    z_tail: torch.Tensor,
    comp_mu: torch.Tensor,
    alpha_range: tuple[float, float] = (1.1, 1.5),
    seed: Optional[int] = None,
) -> NegativeBatch:
    z = safe_normalize(z_tail.float(), dim=1)
    mu = safe_normalize(comp_mu.to(device=z.device, dtype=z.dtype).view(1, -1), dim=1)
    gen = _generator(seed, z.device)
    lo, hi = float(alpha_range[0]), float(alpha_range[1])
    alpha = torch.empty((z.size(0), 1), device=z.device, dtype=z.dtype).uniform_(lo, hi, generator=gen)
    out = safe_normalize(mu + alpha * (z - mu), dim=1)
    return NegativeBatch(out, ["tail_outward"] * out.size(0), torch.full((out.size(0),), -1, dtype=torch.long), [None] * out.size(0), [None] * out.size(0), {})


def _cross_class_pairs(bank):
    comps = list(bank.iter_components())
    pairs = []
    for i, a in enumerate(comps):
        for b in comps[i + 1 :]:
            if int(a.class_id) == int(b.class_id):
                continue
            d = float(angular_distance_deg(a.mu, b.mu).item())
            pairs.append((d, a, b))
    return [p[1:] for p in sorted(pairs, key=lambda x: x[0])]


def sample_interclass_slerp_negatives(
    bank,
    n_per_pair: int = 16,
    lambda_range: tuple[float, float] = (0.35, 0.65),
    max_pairs: Optional[int] = None,
    seed: Optional[int] = None,
) -> NegativeBatch:
    pairs = _cross_class_pairs(bank)
    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]
    if not pairs:
        return NegativeBatch(torch.empty(0, 0), [], torch.empty(0, dtype=torch.long), [], [], {"empty": True})
    device = pairs[0][0].mu.device
    gen = _generator(seed, device)
    rows = []
    kinds = []
    src_cls = []
    src_comp = []
    for a, b in pairs:
        t = torch.empty((int(n_per_pair), 1), device=device).uniform_(
            float(lambda_range[0]),
            float(lambda_range[1]),
            generator=gen,
        )
        z = slerp(a.mu.view(1, -1).expand(int(n_per_pair), -1), b.mu.view(1, -1).expand(int(n_per_pair), -1), t)
        rows.append(z)
        kinds.extend(["inter_class"] * int(n_per_pair))
        src_cls.extend([int(a.class_id)] * int(n_per_pair))
        src_comp.extend([int(a.component_id)] * int(n_per_pair))
    out = torch.cat(rows, dim=0)
    return NegativeBatch(out, kinds, torch.full((out.size(0),), -1, dtype=torch.long), src_cls, src_comp, {})


def sample_same_class_bridge_negatives(
    bank,
    density_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor | float]] = None,
    n_per_pair: int = 8,
    lambda_range: tuple[float, float] = (0.35, 0.65),
    density_threshold: Optional[float] = None,
    seed: Optional[int] = None,
) -> NegativeBatch:
    if density_fn is None or density_threshold is None:
        return NegativeBatch(torch.empty(0, 0), [], torch.empty(0, dtype=torch.long), [], [], {"skipped": "density_unavailable"})
    rows = []
    kinds = []
    src_cls = []
    src_comp = []
    gen = None
    for class_id, comps in bank.classes.items():
        for i, a in enumerate(comps):
            for b in comps[i + 1 :]:
                if gen is None:
                    gen = _generator(seed, a.mu.device)
                t = torch.empty((int(n_per_pair), 1), device=a.mu.device).uniform_(
                    float(lambda_range[0]),
                    float(lambda_range[1]),
                    generator=gen,
                )
                z = slerp(a.mu.view(1, -1).expand(int(n_per_pair), -1), b.mu.view(1, -1).expand(int(n_per_pair), -1), t)
                density = density_fn(z, int(class_id))
                density_t = density if torch.is_tensor(density) else torch.full((z.size(0),), float(density), device=z.device)
                keep = density_t.view(-1) < float(density_threshold)
                if bool(keep.any()):
                    rows.append(z[keep])
                    k = int(keep.sum().item())
                    kinds.extend(["same_class_low_density_bridge"] * k)
                    src_cls.extend([int(class_id)] * k)
                    src_comp.extend([int(a.component_id)] * k)
    if not rows:
        return NegativeBatch(torch.empty(0, 0), [], torch.empty(0, dtype=torch.long), [], [], {"empty": True})
    out = torch.cat(rows, dim=0)
    return NegativeBatch(out, kinds, torch.full((out.size(0),), -1, dtype=torch.long), src_cls, src_comp, {})
