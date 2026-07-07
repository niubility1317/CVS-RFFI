import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from cvsrffi.component_geometry import angular_distance_deg, safe_normalize


def _as_tensor(v: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(v):
        return v.detach().clone().to(dtype=dtype)
    return torch.as_tensor(v, dtype=dtype)


def _radius_to_deg(v: Any, default: float) -> float:
    try:
        if torch.is_tensor(v):
            value = float(v.detach().flatten()[0].item())
        else:
            value = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return math.degrees(value) if abs(value) <= math.pi else value


def _optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        value = float(v)
    except Exception:
        return None
    return value if math.isfinite(value) else None


@dataclass
class ComponentStats:
    class_id: int
    component_id: int
    source_domains: list[int]
    n_samples: int
    mu: torch.Tensor
    r_core_deg: float
    r_accept_deg: float
    r_tail_deg: float
    r_vac_deg: float
    density_core_min: Optional[float] = None
    density_tail_min: Optional[float] = None
    nll_core_max: Optional[float] = None
    nll_tail_max: Optional[float] = None
    nearest_other_angle_deg: Optional[float] = None
    enabled: bool = True


class VacuumGaussianPrototypeBank:
    def __init__(self, classes: Mapping[int, Iterable[ComponentStats]], *, feature_key: str = "z_id"):
        self.feature_key = str(feature_key or "z_id")
        self.classes: Dict[int, list[ComponentStats]] = {
            int(k): [c for c in comps if bool(c.enabled)] for k, comps in classes.items()
        }

    @classmethod
    def from_phase2_package(cls, path_or_dict: str | Path | Mapping[str, Any], feature_key: str = "z_id"):
        if isinstance(path_or_dict, (str, Path)):
            path = Path(path_or_dict)
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as f:
                    package = json.load(f)
            else:
                package = torch.load(path, map_location="cpu")
        else:
            package = dict(path_or_dict)
        feature = str(package.get("feature_key", feature_key) or feature_key)
        if "fusion_components" in package:
            return cls(_components_from_fusion_package(package), feature_key=feature)
        return cls(_components_from_legacy_package(package), feature_key=feature)

    def iter_components(self) -> Iterable[ComponentStats]:
        for comps in self.classes.values():
            for comp in comps:
                yield comp

    def get_component(self, class_id: int, component_id: int) -> ComponentStats:
        for comp in self.classes.get(int(class_id), []):
            if int(comp.component_id) == int(component_id):
                return comp
        raise KeyError(f"component not found: class={class_id} component={component_id}")

    def angular_distance_deg(self, z: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        return angular_distance_deg(z, mu)

    def nearest_own_component(self, z: torch.Tensor, class_id: int) -> ComponentStats:
        comps = self.classes.get(int(class_id), [])
        if not comps:
            raise KeyError(f"class has no enabled components: {class_id}")
        return min(comps, key=lambda comp: float(angular_distance_deg(z, comp.mu).item()))

    def nearest_other_component(self, z: torch.Tensor, exclude_class_id: int) -> ComponentStats:
        comps = [c for c in self.iter_components() if int(c.class_id) != int(exclude_class_id)]
        if not comps:
            raise KeyError("no other-class components are available")
        return min(comps, key=lambda comp: float(angular_distance_deg(z, comp.mu).item()))

    def tangent_mahalanobis_nll(self, z: torch.Tensor, component: ComponentStats) -> Optional[float]:
        if component.nll_core_max is None and component.nll_tail_max is None:
            return None
        d = float(angular_distance_deg(z, component.mu).item())
        scale = max(1e-6, float(component.r_accept_deg))
        return d / scale

    def knn_density(self, z: torch.Tensor, component: ComponentStats) -> Optional[float]:
        if component.density_core_min is None and component.density_tail_min is None:
            return None
        d = float(angular_distance_deg(z, component.mu).item())
        scale = max(1e-6, float(component.r_accept_deg))
        return math.exp(-(d / scale))

    def to_json_dict(self) -> Dict[str, Any]:
        classes = {}
        for class_id, comps in self.classes.items():
            classes[str(class_id)] = {"components": [_component_to_json(c) for c in comps]}
        return {"feature_key": self.feature_key, "classes": classes}


def _component_to_json(c: ComponentStats) -> Dict[str, Any]:
    return {
        "class_id": int(c.class_id),
        "component_id": int(c.component_id),
        "source_domains": list(c.source_domains),
        "n_samples": int(c.n_samples),
        "mu": c.mu.detach().cpu().tolist(),
        "r_core_deg": float(c.r_core_deg),
        "r_accept_deg": float(c.r_accept_deg),
        "r_tail_deg": float(c.r_tail_deg),
        "r_vac_deg": float(c.r_vac_deg),
        "density_p05": c.density_core_min,
        "density_p10": c.density_tail_min,
        "nll_p95": c.nll_core_max,
        "nll_tail_p95": c.nll_tail_max,
        "nearest_other_deg": c.nearest_other_angle_deg,
        "accept_enabled": bool(c.enabled),
    }


def _components_from_fusion_package(package: Mapping[str, Any]) -> Dict[int, list[ComponentStats]]:
    fused = package.get("fused_tx_prototypes")
    fused_t = _as_tensor(fused) if fused is not None else None
    out: Dict[int, list[ComponentStats]] = {}
    raw_components = package.get("fusion_components", [])
    if isinstance(raw_components, Mapping):
        iterable = sorted((int(k), v) for k, v in raw_components.items())
    else:
        iterable = list(enumerate(raw_components))
    for class_id, rows in iterable:
        comps = []
        for idx, row in enumerate(rows or []):
            comp_id = int(row.get("component_id", idx))
            mu_obj = row.get("mu", None)
            if mu_obj is None and fused_t is not None:
                mu_obj = fused_t[int(class_id), comp_id]
            if mu_obj is None:
                continue
            mu = safe_normalize(_as_tensor(mu_obj).view(1, -1), dim=1).squeeze(0)
            r_accept = _optional_float(row.get("r_accept_deg", row.get("accept_radius_deg")))
            if r_accept is None:
                r_accept = 10.0
            r_core = _optional_float(row.get("r_core_deg"))
            r_tail = _optional_float(row.get("r_tail_deg", row.get("radius_deg")))
            r_vac = _optional_float(row.get("r_vac_deg"))
            comps.append(
                ComponentStats(
                    class_id=int(class_id),
                    component_id=comp_id,
                    source_domains=[int(x) for x in row.get("source_domains", row.get("domains", []))],
                    n_samples=int(row.get("n_samples", row.get("count", 0)) or 0),
                    mu=mu,
                    r_core_deg=float(r_core if r_core is not None else min(r_accept, r_accept * 0.8)),
                    r_accept_deg=float(r_accept),
                    r_tail_deg=float(r_tail if r_tail is not None else max(r_accept, r_accept * 1.25)),
                    r_vac_deg=float(r_vac if r_vac is not None else max(r_accept, r_accept * 1.5)),
                    density_core_min=_optional_float(row.get("density_p05", row.get("density_core_min"))),
                    density_tail_min=_optional_float(row.get("density_p10", row.get("density_tail_min"))),
                    nll_core_max=_optional_float(row.get("nll_p95", row.get("nll_core_max"))),
                    nll_tail_max=_optional_float(row.get("nll_tail_p95", row.get("nll_tail_max"))),
                    nearest_other_angle_deg=_optional_float(row.get("nearest_other_deg")),
                    enabled=bool(row.get("accept_enabled", row.get("enabled", True))),
                )
            )
        out[int(class_id)] = comps
    return out


def _components_from_legacy_package(package: Mapping[str, Any]) -> Dict[int, list[ComponentStats]]:
    proto = _as_tensor(package.get("prototypes"))
    counts = package.get("prototype_counts", torch.ones(proto.size(0), dtype=torch.long))
    counts_t = _as_tensor(counts, dtype=torch.long).view(-1)
    radii = package.get("radii", {}) if isinstance(package.get("radii", {}), Mapping) else {}
    p95 = radii.get("p95", None)
    out: Dict[int, list[ComponentStats]] = {}
    for class_id in range(proto.size(0)):
        r_accept = _radius_to_deg(p95[class_id] if torch.is_tensor(p95) and p95.numel() > class_id else None, 10.0)
        out[class_id] = [
            ComponentStats(
                class_id=class_id,
                component_id=0,
                source_domains=[],
                n_samples=int(counts_t[class_id].item()) if counts_t.numel() > class_id else 0,
                mu=safe_normalize(proto[class_id].view(1, -1), dim=1).squeeze(0),
                r_core_deg=min(r_accept, r_accept * 0.8),
                r_accept_deg=r_accept,
                r_tail_deg=max(r_accept, r_accept * 1.25),
                r_vac_deg=max(r_accept, r_accept * 1.5),
                enabled=True,
            )
        ]
    return out
