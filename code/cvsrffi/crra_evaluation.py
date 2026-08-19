"""Read-only CRRA diagnostics for independent checkpoint evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


_SUMMARY_FIELDS = (
    "correction_energy",
    "alpha",
    "gate",
    "support_distance",
    "reliability_time",
    "reliability_freq",
    "reliability_pa",
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


def restore_crra_eval_epoch(model: Any, checkpoint: Mapping[str, Any]) -> int | None:
    """Restore the saved training epoch for a rebuilt model before CRRA evaluation.

    ``crra_epoch`` is runtime state rather than a tensor in ``state_dict``. A
    newly constructed model otherwise starts at epoch one and turns the
    scheduled CRRA gate off during independent final-checkpoint evaluation.
    """

    epoch_value = checkpoint.get("epoch", None)
    try:
        epoch = int(epoch_value)
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
        setter = getattr(candidate, "set_crra_epoch", None)
        if callable(setter):
            setter(epoch)
            return int(epoch)
    return None


class CRRATelemetryAccumulator:
    """Accumulate model-output telemetry without retaining or updating a model."""

    def __init__(self):
        self._values = {
            "clean": defaultdict(list),
            "satellite": defaultdict(list),
            "paired": defaultdict(list),
        }
        self._q_correct = {"clean": 0, "satellite": 0}
        self._q_total = {"clean": 0, "satellite": 0}

    def _record_output(self, role: str, output: Mapping[str, Any], labels: Optional[torch.Tensor]) -> None:
        mappings = {
            "correction_energy": "crra_correction_energy",
            "alpha": "crra_alpha",
            "gate": "crra_gate",
            "support_distance": "crra_support_distance",
        }
        for name, key in mappings.items():
            values = _finite_flatten(output.get(key))
            if int(values.numel()) > 0:
                self._values[role][name].append(values)

        reliability = output.get("crra_branch_reliability")
        if torch.is_tensor(reliability) and reliability.dim() == 2 and int(reliability.size(1)) >= 3:
            for index, name in enumerate(("reliability_time", "reliability_freq", "reliability_pa")):
                values = _finite_flatten(reliability[:, index])
                if int(values.numel()) > 0:
                    self._values[role][name].append(values)
        else:
            for name, key in (
                ("reliability_time", "crra_reliability_time"),
                ("reliability_freq", "crra_reliability_freq"),
                ("reliability_pa", "crra_reliability_pa"),
            ):
                values = _finite_flatten(output.get(key))
                if int(values.numel()) > 0:
                    self._values[role][name].append(values)

        logits = output.get("crra_condition_tx_adv_logits")
        if torch.is_tensor(logits) and torch.is_tensor(labels) and logits.dim() == 2:
            label = labels.detach().to(device=logits.device).view(-1).long()
            if int(label.numel()) == int(logits.size(0)):
                self._q_correct[role] += int((logits.detach().argmax(dim=1) == label).sum().item())
                self._q_total[role] += int(label.numel())

    def _record_paired_geometry(
        self,
        clean_output: Mapping[str, Any],
        satellite_output: Mapping[str, Any],
        labels: Optional[torch.Tensor],
    ) -> None:
        clean_z = clean_output.get("z_id")
        satellite_z = satellite_output.get("z_id")
        if not torch.is_tensor(clean_z) or not torch.is_tensor(satellite_z):
            return
        count = min(int(clean_z.size(0)), int(satellite_z.size(0)))
        if count <= 0:
            return
        clean_z = F.normalize(clean_z[:count].detach().float(), dim=1)
        satellite_z = F.normalize(satellite_z[:count].detach().float(), dim=1)
        cosine_distance = 1.0 - (clean_z * satellite_z).sum(dim=1)
        values = _finite_flatten(cosine_distance)
        if int(values.numel()) > 0:
            self._values["paired"]["view_cosine_distance"].append(values)
        if not torch.is_tensor(labels):
            return
        label = labels.detach().to(device=clean_z.device).view(-1)[:count]
        if int(label.numel()) != count:
            return
        combined = torch.cat([clean_z, satellite_z], dim=0)
        combined_labels = torch.cat([label, label], dim=0)
        radii = []
        for class_id in torch.unique(combined_labels).tolist():
            class_z = combined[combined_labels == int(class_id)]
            if int(class_z.size(0)) < 2:
                continue
            centre = F.normalize(class_z.mean(dim=0, keepdim=True), dim=1)
            radii.append((class_z - centre).norm(dim=1))
        if radii:
            values = _finite_flatten(torch.cat(radii, dim=0))
            if int(values.numel()) > 0:
                self._values["paired"]["cross_domain_class_radius"].append(values)

    def update(
        self,
        clean_output: Mapping[str, Any],
        satellite_output: Mapping[str, Any],
        labels: Optional[torch.Tensor],
    ) -> None:
        """Read detached outputs and append scalar diagnostics only."""

        self._record_output("clean", clean_output, labels)
        self._record_output("satellite", satellite_output, labels)
        self._record_paired_geometry(clean_output, satellite_output, labels)

    def merge(self, other: "CRRATelemetryAccumulator") -> None:
        """Append detached diagnostic observations from another accumulator."""

        if not isinstance(other, CRRATelemetryAccumulator):
            raise TypeError(f"other must be CRRATelemetryAccumulator, got {type(other)!r}")
        for role, fields in self._values.items():
            for name in set(fields).union(other._values[role]):
                fields[name].extend(other._values[role][name])
        for role in self._q_correct:
            self._q_correct[role] += int(other._q_correct[role])
            self._q_total[role] += int(other._q_total[role])

    def summary(self) -> dict[str, dict[str, dict[str, float | int]]]:
        result: dict[str, dict[str, dict[str, float | int]]] = {
            "clean": {},
            "satellite": {},
            "paired": {},
        }
        for role in ("clean", "satellite"):
            for name in _SUMMARY_FIELDS:
                result[role][name] = _distribution_summary(self._values[role][name])
            total = int(self._q_total[role])
            result[role]["q_tx_leakage_accuracy"] = {
                "count": total,
                "accuracy": float(self._q_correct[role] / total) if total > 0 else float("nan"),
            }
        for name in ("view_cosine_distance", "cross_domain_class_radius"):
            result["paired"][name] = _distribution_summary(self._values["paired"][name])
        return result


__all__ = ["CRRATelemetryAccumulator", "restore_crra_eval_epoch"]
