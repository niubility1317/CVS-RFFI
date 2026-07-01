from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from cvsrffi.component_geometry import angular_distance_deg


@dataclass
class GateThresholds:
    energy_max_by_class: Optional[dict[int, float]] = None
    logit_margin_core_min: float = 0.0
    logit_margin_tail_min: float = 0.0
    geo_margin_core_min_deg: float = 2.0
    geo_margin_tail_min_deg: float = 4.0
    allow_tail_auto_accept: bool = False
    use_density_gate: bool = True
    use_nll_gate: bool = True
    use_energy_gate: bool = True
    use_geo_margin_gate: bool = True
    reject_nan: bool = True


def energy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return -float(temperature) * torch.logsumexp(logits.float() / float(temperature), dim=-1)


class LocalComponentHardGate:
    def __init__(self, bank, thresholds: GateThresholds):
        self.bank = bank
        self.th = thresholds

    def decide(self, z: torch.Tensor, logits: Optional[torch.Tensor] = None, energy: Optional[float] = None) -> Dict[str, Any]:
        z = z.detach().float().view(-1)
        if self.th.reject_nan and (not torch.isfinite(z).all()):
            return {"decision": "REJECT_NAN", "debug": {"gates": {}, "radius_region": "nan"}}
        if logits is None:
            raise ValueError("LocalComponentHardGate.decide requires logits for class selection")
        logits = logits.detach().float().view(-1)
        if self.th.reject_nan and (not torch.isfinite(logits).all()):
            return {"decision": "REJECT_NAN", "debug": {"gates": {}, "radius_region": "nan"}}

        top = torch.topk(logits, k=min(2, logits.numel()))
        pred = int(top.indices[0].item())
        second = int(top.indices[1].item()) if top.indices.numel() > 1 else None
        margin = float((top.values[0] - top.values[1]).item()) if top.values.numel() > 1 else float("inf")
        energy_val = float(energy) if energy is not None else float(energy_from_logits(logits.view(1, -1)).item())
        debug: Dict[str, Any] = {
            "pred_class": pred,
            "second_class": second,
            "component_id": None,
            "d_own_deg": None,
            "d_other_deg": None,
            "geo_margin_deg": None,
            "energy": energy_val,
            "logit_margin": margin,
            "density": None,
            "nll": None,
            "radius_region": "unknown",
            "gates": {},
        }

        if margin < float(self.th.logit_margin_core_min):
            debug["gates"]["logit_margin"] = False
            return {"decision": "REJECT_LOW_LOGIT_MARGIN", "class_id": pred, "debug": debug}
        debug["gates"]["logit_margin"] = True

        if self.th.use_energy_gate and self.th.energy_max_by_class is not None:
            max_energy = self.th.energy_max_by_class.get(pred)
            if max_energy is not None and energy_val > float(max_energy):
                debug["gates"]["energy"] = False
                return {"decision": "REJECT_HIGH_ENERGY", "class_id": pred, "debug": debug}
            debug["gates"]["energy"] = True if max_energy is not None else "skipped"
        else:
            debug["gates"]["energy"] = "skipped"

        own = self.bank.nearest_own_component(z, pred)
        debug["component_id"] = int(own.component_id)
        d_own = float(angular_distance_deg(z, own.mu).item())
        debug["d_own_deg"] = d_own
        try:
            other = self.bank.nearest_other_component(z, pred)
            d_other = float(angular_distance_deg(z, other.mu).item())
            geo_margin = d_other - d_own
            debug["d_other_deg"] = d_other
            debug["geo_margin_deg"] = geo_margin
        except KeyError:
            geo_margin = None
            debug["gates"]["geo_margin"] = "skipped"

        if d_own <= float(own.r_core_deg):
            region = "core"
        elif d_own <= float(own.r_accept_deg):
            region = "tail"
        else:
            region = "outside"
        debug["radius_region"] = region
        debug["gates"]["radius"] = region in ("core", "tail")
        if region == "outside":
            return {"decision": "REJECT_OUTSIDE_RADIUS", "class_id": pred, "debug": debug}

        if self.th.use_geo_margin_gate and geo_margin is not None:
            req = self.th.geo_margin_core_min_deg if region == "core" else self.th.geo_margin_tail_min_deg
            if geo_margin < float(req):
                debug["gates"]["geo_margin"] = False
                return {"decision": "REJECT_LOW_GEO_MARGIN", "class_id": pred, "debug": debug}
            debug["gates"]["geo_margin"] = True

        density = self.bank.knn_density(z, own)
        debug["density"] = density
        if self.th.use_density_gate and density is not None:
            min_density = own.density_core_min if region == "core" else own.density_tail_min
            if min_density is not None and density < min_density:
                debug["gates"]["density"] = False
                return {"decision": "REJECT_LOW_DENSITY", "class_id": pred, "debug": debug}
            debug["gates"]["density"] = True if min_density is not None else "skipped"
        else:
            debug["gates"]["density"] = "skipped"

        nll = self.bank.tangent_mahalanobis_nll(z, own)
        debug["nll"] = nll
        if self.th.use_nll_gate and nll is not None:
            max_nll = own.nll_core_max if region == "core" else own.nll_tail_max
            if max_nll is not None and nll > max_nll:
                debug["gates"]["nll"] = False
                return {"decision": "REJECT_HIGH_NLL", "class_id": pred, "debug": debug}
            debug["gates"]["nll"] = True if max_nll is not None else "skipped"
        else:
            debug["gates"]["nll"] = "skipped"

        if region == "core":
            return {"decision": "ACCEPT_KNOWN_CORE", "class_id": pred, "component_id": int(own.component_id), "debug": debug}
        if margin >= float(self.th.logit_margin_tail_min) and bool(self.th.allow_tail_auto_accept):
            return {"decision": "ACCEPT_KNOWN_TAIL_STRICT", "class_id": pred, "component_id": int(own.component_id), "debug": debug}
        return {"decision": "REVIEW_KNOWN_TAIL", "class_id": pred, "component_id": int(own.component_id), "debug": debug}

    def batch_decide(self, z_batch: torch.Tensor, logits_batch: Optional[torch.Tensor] = None, energy_batch: Optional[torch.Tensor] = None):
        rows = []
        for i in range(z_batch.size(0)):
            logits = logits_batch[i] if logits_batch is not None else None
            energy = float(energy_batch[i].item()) if energy_batch is not None else None
            rows.append(self.decide(z_batch[i], logits=logits, energy=energy))
        return rows

