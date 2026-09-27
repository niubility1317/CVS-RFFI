import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from cvsrffi.component_geometry import angular_distance_deg


@dataclass
class GateThresholds:
    energy_max_by_class: Optional[dict[int, float]] = None
    energy_temperature: float = 1.0
    energy_formula_id: str = "negative_logsumexp_temperature_v1"
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
    reject_zero_direction: bool = True
    max_radius_to_inter_ratio: float = 0.50

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]):
        energy_obj = values.get("energy_max_by_class")
        energy = None
        if isinstance(energy_obj, Mapping):
            energy = {int(key): float(value) for key, value in energy_obj.items()}
        return cls(
            energy_max_by_class=energy,
            energy_temperature=float(values.get("energy_temperature", 1.0)),
            energy_formula_id=str(values.get("energy_formula_id", "")),
            logit_margin_core_min=float(values.get("logit_margin_core_min", 0.0)),
            logit_margin_tail_min=float(values.get("logit_margin_tail_min", 0.0)),
            geo_margin_core_min_deg=float(values.get("geo_margin_core_min_deg", 2.0)),
            geo_margin_tail_min_deg=float(values.get("geo_margin_tail_min_deg", 4.0)),
            allow_tail_auto_accept=bool(values.get("allow_tail_auto_accept", False)),
            use_density_gate=bool(values.get("use_density_gate", True)),
            use_nll_gate=bool(values.get("use_nll_gate", True)),
            use_energy_gate=bool(values.get("use_energy_gate", True)),
            use_geo_margin_gate=bool(values.get("use_geo_margin_gate", True)),
            reject_nan=bool(values.get("reject_nan", True)),
            reject_zero_direction=bool(values.get("reject_zero_direction", False)),
            max_radius_to_inter_ratio=float(values.get("max_radius_to_inter_ratio", float("nan"))),
        )


def energy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return -float(temperature) * torch.logsumexp(logits.float() / float(temperature), dim=-1)


class LocalComponentHardGate:
    policy_id = "endpoint_accept_v1"

    def __init__(self, bank, thresholds: GateThresholds):
        manifest = getattr(bank, "endpoint_manifest", None)
        if not isinstance(manifest, Mapping):
            raise ValueError("LocalComponentHardGate requires a verified endpoint_accept_v1 artifact")
        artifact_thresholds = GateThresholds.from_mapping(manifest.get("gate_thresholds", {}))
        if thresholds != artifact_thresholds:
            raise ValueError("LocalComponentHardGate thresholds differ from the verified artifact")
        self.bank = bank
        self.th = thresholds
        self.boundary_version = str(manifest.get("boundary_version", ""))
        self.boundary_hash = str(manifest.get("boundary_hash", ""))

    @classmethod
    def from_phase1_package(
        cls,
        path_or_dict,
        thresholds: Optional[GateThresholds] = None,
        *,
        entry_point: str = "runtime_inference",
        runtime_identity: Optional[Mapping[str, Any]] = None,
    ):
        from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank
        from cvsrffi.phase2_prototypes import verify_endpoint_accept_v1_manifest

        if isinstance(path_or_dict, (str, Path)):
            path = Path(path_or_dict)
            if path.suffix.lower() == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    package = json.load(handle)
            else:
                package = torch.load(path, map_location="cpu")
        else:
            package = dict(path_or_dict)
        manifest = verify_endpoint_accept_v1_manifest(package)
        if entry_point not in {"train_export", "offline_eval", "runtime_inference"}:
            raise ValueError(f"unsupported endpoint_accept_v1 entry point: {entry_point}")
        entry = manifest.get("entry_points", {}).get(entry_point, {})
        if str(entry.get("boundary_hash", "")) != str(manifest.get("boundary_hash", "")):
            raise ValueError(f"endpoint_accept_v1 entry-point hash mismatch: {entry_point}")
        if entry_point in {"offline_eval", "runtime_inference"}:
            if not isinstance(runtime_identity, Mapping):
                raise ValueError(f"endpoint_accept_v1 {entry_point} requires actual runtime identity")
            artifact_identity = manifest.get("inference_identity", {})
            required_identity_keys = (
                "feature_key",
                "feature_dim",
                "source_checkpoint_sha256",
                "class_id_to_tx",
                "logit_class_order",
                "classification_head_contract",
                "checkpoint_load_strict",
            )
            mismatched = [
                key for key in required_identity_keys
                if runtime_identity.get(key) != artifact_identity.get(key)
            ]
            if mismatched:
                raise ValueError(
                    f"endpoint_accept_v1 {entry_point} runtime identity mismatch: {','.join(mismatched)}"
                )
        artifact_thresholds = GateThresholds.from_mapping(manifest.get("gate_thresholds", {}))
        if thresholds is not None and thresholds != artifact_thresholds:
            raise ValueError("endpoint_accept_v1 runtime thresholds differ from the versioned artifact")

        bank = VacuumGaussianPrototypeBank.from_phase2_package(
            package,
            require_endpoint_manifest=True,
        )
        return cls(bank, artifact_thresholds)

    @classmethod
    def from_train_export(cls, path_or_dict):
        return cls.from_phase1_package(path_or_dict, entry_point="train_export")

    @classmethod
    def from_offline_eval(cls, path_or_dict, *, runtime_identity: Mapping[str, Any]):
        return cls.from_phase1_package(
            path_or_dict, entry_point="offline_eval", runtime_identity=runtime_identity
        )

    @classmethod
    def from_runtime_inference(cls, path_or_dict, *, runtime_identity: Mapping[str, Any]):
        return cls.from_phase1_package(
            path_or_dict, entry_point="runtime_inference", runtime_identity=runtime_identity
        )

    @classmethod
    def for_diagnostic_unverified(cls, bank, thresholds: GateThresholds):
        """Build a non-exportable Phase1 proxy gate for legacy diagnostic scripts only."""

        gate = cls.__new__(cls)
        gate.bank = bank
        gate.th = thresholds
        gate.policy_id = "diagnostic_dynamic_gate_v0"
        gate.boundary_version = "unversioned_non_exportable"
        gate.boundary_hash = ""
        return gate

    def decide(self, z: torch.Tensor, logits: Optional[torch.Tensor] = None, energy: Optional[float] = None) -> Dict[str, Any]:
        identity_debug = {
            "endpoint_policy_id": self.policy_id,
            "endpoint_boundary_version": self.boundary_version,
            "endpoint_boundary_hash": self.boundary_hash,
            "gates": {},
            "radius_region": "nan",
        }
        if not torch.is_tensor(z) or z.dim() != 1 or int(z.numel()) != int(self.bank.feature_dim):
            identity_debug["feature_shape"] = tuple(z.shape) if torch.is_tensor(z) else None
            identity_debug["expected_feature_dim"] = int(self.bank.feature_dim)
            return {"decision": "REJECT_INVALID_FEATURE", "debug": identity_debug}
        z = z.detach().float()
        if self.th.reject_nan and (not torch.isfinite(z).all()):
            return {"decision": "REJECT_NAN", "debug": identity_debug}
        feature_norm = float(torch.linalg.vector_norm(z).item())
        identity_debug["feature_norm"] = feature_norm
        if self.th.reject_nan and not math.isfinite(feature_norm):
            return {"decision": "REJECT_NAN", "debug": identity_debug}
        if self.th.reject_zero_direction and feature_norm <= 1e-8:
            return {"decision": "REJECT_INVALID_FEATURE", "debug": identity_debug}
        if logits is None:
            return {"decision": "REJECT_INVALID_LOGITS", "debug": identity_debug}
        if not torch.is_tensor(logits) or logits.dim() != 1:
            identity_debug["logit_shape"] = tuple(logits.shape) if torch.is_tensor(logits) else None
            return {"decision": "REJECT_INVALID_LOGITS", "debug": identity_debug}
        logits = logits.detach().float()
        if self.th.reject_nan and (not torch.isfinite(logits).all()):
            return {"decision": "REJECT_NAN", "debug": identity_debug}
        expected_classes = len(getattr(self.bank, "classes", {}))
        if logits.numel() < 2 or logits.numel() != expected_classes:
            identity_debug["logit_count"] = int(logits.numel())
            identity_debug["expected_class_count"] = int(expected_classes)
            return {"decision": "REJECT_INVALID_LOGITS", "debug": identity_debug}

        top = torch.topk(logits, k=min(2, logits.numel()))
        pred = int(top.indices[0].item())
        second = int(top.indices[1].item()) if top.indices.numel() > 1 else None
        margin = float((top.values[0] - top.values[1]).item()) if top.values.numel() > 1 else float("inf")
        computed_energy = float(
            energy_from_logits(logits.view(1, -1), temperature=float(self.th.energy_temperature)).item()
        )
        if not math.isfinite(computed_energy):
            return {"decision": "REJECT_NAN", "debug": identity_debug}
        if energy is not None:
            supplied_energy = float(energy)
            if not math.isfinite(supplied_energy):
                return {"decision": "REJECT_NAN", "debug": identity_debug}
            if abs(supplied_energy - computed_energy) > 1e-5 * max(1.0, abs(computed_energy)):
                identity_debug["computed_energy"] = computed_energy
                identity_debug["supplied_energy"] = supplied_energy
                return {"decision": "REJECT_ENERGY_MISMATCH", "debug": identity_debug}
        energy_val = computed_energy
        debug: Dict[str, Any] = {
            "endpoint_policy_id": self.policy_id,
            "endpoint_boundary_version": self.boundary_version,
            "endpoint_boundary_hash": self.boundary_hash,
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
            debug["gates"]["geo_margin"] = False
            return {"decision": "REJECT_LOW_GEO_MARGIN", "class_id": pred, "debug": debug}

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
