"""Read-only NTRS diagnostics for independent checkpoint evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


_DISTRIBUTION_FIELDS = (
    "gate",
    "safe_gate",
    "alpha",
    "correction_energy",
    "physical_correction_energy",
    "support_distance",
    "correctability",
    "uncertainty",
    "subspace_residual",
    "class_attraction_cosine",
)


def _finite_flatten(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        return torch.empty(0, dtype=torch.float32)
    flat = value.detach().float().reshape(-1).cpu()
    return flat[torch.isfinite(flat)]


def _distribution_summary(values: list[torch.Tensor]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": float("nan"), "p95": float("nan")}
    stacked = torch.cat(values, dim=0)
    if int(stacked.numel()) == 0:
        return {"count": 0, "mean": float("nan"), "p95": float("nan")}
    return {
        "count": int(stacked.numel()),
        "mean": float(stacked.mean().item()),
        "p95": float(torch.quantile(stacked, 0.95).item()),
    }


def restore_ntrs_eval_epoch(model: Any, checkpoint: Mapping[str, Any]) -> int | None:
    """Restore the final training epoch so a rebuilt NTRS gate is not at S1."""

    try:
        epoch = int(checkpoint.get("epoch", 0))
    except (TypeError, ValueError):
        epoch = 0
    if epoch <= 0:
        saved_args = checkpoint.get("args", {})
        if isinstance(saved_args, Mapping):
            try:
                epoch = int(saved_args.get("epochs", 0))
            except (TypeError, ValueError):
                epoch = 0
    if epoch <= 0:
        return None
    for candidate in (model, getattr(model, "_orig_mod", None), getattr(model, "module", None)):
        setter = getattr(candidate, "set_ntrs_epoch", None)
        if callable(setter):
            setter(epoch)
            return epoch
    return None


def ntrs_prototypes_from_model(model: Any) -> Optional[torch.Tensor]:
    """Read the raw CosFace prototype matrix without changing model state."""

    for candidate in (model, getattr(model, "_orig_mod", None), getattr(model, "module", None)):
        if candidate is None:
            continue
        backbone = getattr(candidate, "id_backbone", None)
        cls_head = getattr(backbone, "cls_head", None)
        head = getattr(cls_head, "head", None)
        weight = getattr(head, "weight", None)
        if torch.is_tensor(weight):
            return weight.detach()
    return None


def ntrs_unknown_rescue_from_model(model: Any) -> bool:
    for candidate in (model, getattr(model, "_orig_mod", None), getattr(model, "module", None)):
        if candidate is not None and hasattr(candidate, "ntrs_unknown_rescue"):
            return bool(getattr(candidate, "ntrs_unknown_rescue"))
    return False


class NTRSTelemetryAccumulator:
    """Accumulate NTRS outputs and known-class transitions without mutation."""

    def __init__(self, *, prototypes: Optional[torch.Tensor], unknown_rescue: bool):
        self.prototypes = prototypes.detach().float().cpu() if torch.is_tensor(prototypes) else None
        self.unknown_rescue = bool(unknown_rescue)
        self._values = {"clean": defaultdict(list), "satellite": defaultdict(list)}
        self._correct = {
            role: {"raw": 0, "robust": 0, "fused": 0, "total": 0}
            for role in ("clean", "satellite")
        }
        self._transitions = {
            role: {"both_correct": 0, "rescued_correct": 0, "harmed_correct": 0, "both_wrong": 0}
            for role in ("clean", "satellite")
        }
        self._disagreement = {"clean": 0, "satellite": 0}
        self._path_correct = defaultdict(int)
        self._path_total = defaultdict(int)

    def _record_paths(self, output: Mapping[str, Any]) -> None:
        aux_id = output.get("aux_id") if isinstance(output.get("aux_id"), Mapping) else {}
        aux_phys = output.get("aux_phys") if isinstance(output.get("aux_phys"), Mapping) else {}
        aux_dom = output.get("aux_dom") if isinstance(output.get("aux_dom"), Mapping) else {}
        checks = {
            "physical_view_rate": bool(aux_phys.get("ntrs_physical_view", False)),
            "frequency_dual_view_rate": bool(aux_phys.get("ntrs_frequency_dual_view", False)),
            "pa_original_iq_rate": bool(aux_phys.get("ntrs_pa_uses_original_iq", False)),
            "identity_anchor_raw_iq_rate": not bool(aux_id.get("ntrs_physical_view", False)),
            "domain_raw_iq_rate": not bool(aux_dom.get("ntrs_physical_view", False)),
        }
        for name, passed in checks.items():
            self._path_total[name] += 1
            self._path_correct[name] += int(passed)

    def _record_class_attraction(self, role: str, output: Mapping[str, Any]) -> None:
        if self.prototypes is None:
            return
        anchor = output.get("ntrs_z_anchor")
        correction = output.get("ntrs_correction")
        raw_logits = output.get("ntrs_raw_logits")
        if not all(torch.is_tensor(value) for value in (anchor, correction, raw_logits)):
            return
        count = min(int(anchor.size(0)), int(correction.size(0)), int(raw_logits.size(0)))
        if count <= 0:
            return
        prototypes = self.prototypes.to(device=anchor.device, dtype=anchor.dtype)
        predicted = raw_logits[:count].detach().argmax(dim=1)
        if bool((predicted >= int(prototypes.size(0))).any()):
            return
        target = prototypes[predicted] - anchor[:count].detach()
        applied = -correction[:count].detach()
        valid = (target.norm(dim=1) > 1e-8) & (applied.norm(dim=1) > 1e-8)
        if bool(valid.any()):
            cosine = F.cosine_similarity(applied[valid].float(), target[valid].float(), dim=1)
            values = _finite_flatten(cosine)
            if int(values.numel()) > 0:
                self._values[role]["class_attraction_cosine"].append(values)

    def _record_output(
        self,
        role: str,
        output: Mapping[str, Any],
        labels: Optional[torch.Tensor],
    ) -> None:
        key_map = {
            "gate": "ntrs_gate",
            "safe_gate": "ntrs_safe_gate",
            "alpha": "ntrs_alpha",
            "correction_energy": "ntrs_correction_energy",
            "physical_correction_energy": "ntrs_physical_correction_energy",
            "support_distance": "ntrs_support_distance",
            "correctability": "ntrs_correctability",
            "uncertainty": "ntrs_uncertainty",
            "subspace_residual": "ntrs_subspace_residual",
        }
        for name, key in key_map.items():
            values = _finite_flatten(output.get(key))
            if int(values.numel()) > 0:
                self._values[role][name].append(values)
        self._record_class_attraction(role, output)
        self._record_paths(output)

        if not torch.is_tensor(labels):
            return
        raw = output.get("ntrs_raw_logits")
        robust = output.get("ntrs_robust_logits")
        fused = output.get("tx_logits")
        if not all(torch.is_tensor(value) and value.dim() == 2 for value in (raw, robust, fused)):
            return
        label = labels.detach().to(device=raw.device).view(-1).long()
        if any(int(value.size(0)) != int(label.numel()) for value in (raw, robust, fused)):
            return
        raw_prediction = raw.detach().argmax(dim=1)
        robust_prediction = robust.detach().argmax(dim=1)
        raw_correct = raw_prediction == label
        robust_correct = robust_prediction == label
        fused_correct = fused.detach().argmax(dim=1) == label
        self._correct[role]["raw"] += int(raw_correct.sum().item())
        self._correct[role]["robust"] += int(robust_correct.sum().item())
        self._correct[role]["fused"] += int(fused_correct.sum().item())
        self._correct[role]["total"] += int(label.numel())
        self._disagreement[role] += int((raw_prediction != robust_prediction).sum().item())
        self._transitions[role]["both_correct"] += int((raw_correct & robust_correct).sum().item())
        self._transitions[role]["rescued_correct"] += int(((~raw_correct) & robust_correct).sum().item())
        self._transitions[role]["harmed_correct"] += int((raw_correct & (~robust_correct)).sum().item())
        self._transitions[role]["both_wrong"] += int(((~raw_correct) & (~robust_correct)).sum().item())

    def update(
        self,
        clean_output: Mapping[str, Any],
        satellite_output: Mapping[str, Any],
        labels: Optional[torch.Tensor],
    ) -> None:
        self._record_output("clean", clean_output, labels)
        self._record_output("satellite", satellite_output, labels)

    def merge(self, other: "NTRSTelemetryAccumulator") -> None:
        if not isinstance(other, NTRSTelemetryAccumulator):
            raise TypeError(f"other must be NTRSTelemetryAccumulator, got {type(other)!r}")
        for role in self._values:
            for name in set(self._values[role]).union(other._values[role]):
                self._values[role][name].extend(other._values[role][name])
            for name in self._correct[role]:
                self._correct[role][name] += int(other._correct[role][name])
            for name in self._transitions[role]:
                self._transitions[role][name] += int(other._transitions[role][name])
            self._disagreement[role] += int(other._disagreement[role])
        for name in set(self._path_total).union(other._path_total):
            self._path_total[name] += int(other._path_total[name])
            self._path_correct[name] += int(other._path_correct[name])

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"clean": {}, "satellite": {}}
        for role in ("clean", "satellite"):
            for name in _DISTRIBUTION_FIELDS:
                result[role][name] = _distribution_summary(self._values[role][name])
            total = int(self._correct[role]["total"])
            for name in ("raw", "robust", "fused"):
                result[role][f"{name}_accuracy"] = (
                    float(self._correct[role][name] / total) if total > 0 else float("nan")
                )
            result[role]["count"] = total
            result[role]["transitions"] = dict(self._transitions[role])
            result[role]["raw_robust_disagreement_rate"] = (
                float(self._disagreement[role] / total)
                if total > 0
                else float("nan")
            )
        result["safety"] = {
            "unknown_rescue_enabled": bool(self.unknown_rescue),
            "unknown_transition_status": "N/A_NO_FROZEN_REJECTION_THRESHOLD",
        }
        result["paths"] = {
            name: float(self._path_correct[name] / self._path_total[name])
            if self._path_total[name] > 0
            else float("nan")
            for name in sorted(self._path_total)
        }
        return result


__all__ = [
    "NTRSTelemetryAccumulator",
    "ntrs_prototypes_from_model",
    "ntrs_unknown_rescue_from_model",
    "restore_ntrs_eval_epoch",
]
