from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch


@dataclass
class ProtoEvidence:
    class_id: int
    prototype: torch.Tensor
    count: int
    margin: float
    entropy: float
    intra_var: float
    client_drift: float
    clean_sat_kl: float
    client_id: str
    style_id: Optional[int] = None
    mode: str = "clean"
    age: int = 0
    reliability: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.class_id = int(self.class_id)
        self.prototype = _normalize(self.prototype.detach().float().flatten())
        self.count = int(self.count)
        self.margin = float(self.margin)
        self.entropy = float(self.entropy)
        self.intra_var = float(self.intra_var)
        self.client_drift = float(self.client_drift)
        self.clean_sat_kl = float(self.clean_sat_kl)
        self.client_id = str(self.client_id)
        self.mode = str(self.mode)
        if self.reliability is None:
            self.reliability = compute_reliability(
                count=self.count,
                margin=self.margin,
                entropy=self.entropy,
                intra_var=self.intra_var,
                client_drift=self.client_drift,
                clean_sat_kl=self.clean_sat_kl,
            )
        else:
            self.reliability = max(0.0, min(1.0, float(self.reliability)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "prototype": [float(v) for v in self.prototype.cpu().tolist()],
            "count": self.count,
            "margin": self.margin,
            "entropy": self.entropy,
            "intra_var": self.intra_var,
            "client_drift": self.client_drift,
            "clean_sat_kl": self.clean_sat_kl,
            "client_id": self.client_id,
            "style_id": self.style_id,
            "mode": self.mode,
            "age": self.age,
            "reliability": float(self.reliability),
            "metadata": dict(self.metadata),
        }


class ProtoEvidenceBank:
    """Reliability-ranked multi-prototype evidence bank."""

    def __init__(self, *, max_per_class: int = 8):
        self.max_per_class = max(1, int(max_per_class))
        self._by_class: dict[int, list[ProtoEvidence]] = {}

    def add(self, evidence: ProtoEvidence) -> None:
        entries = self._by_class.setdefault(int(evidence.class_id), [])
        entries.append(evidence)
        entries.sort(key=lambda e: (float(e.reliability), int(e.count), -int(e.age)), reverse=True)
        del entries[self.max_per_class :]

    def update(self, items) -> None:
        for item in items:
            self.add(item)

    def get_class(self, class_id: int) -> tuple[ProtoEvidence, ...]:
        return tuple(self._by_class.get(int(class_id), ()))

    def age_one_round(self) -> None:
        for entries in self._by_class.values():
            for item in entries:
                item.age += 1

    def summary(self) -> dict[str, Any]:
        num_proto = sum(len(v) for v in self._by_class.values())
        reliabilities = [float(item.reliability) for entries in self._by_class.values() for item in entries]
        return {
            "num_classes": len(self._by_class),
            "num_prototypes": num_proto,
            "max_per_class": self.max_per_class,
            "mean_reliability": float(sum(reliabilities) / max(1, len(reliabilities))) if reliabilities else 0.0,
        }


def compute_reliability(
    *,
    count: int,
    margin: float,
    entropy: float,
    intra_var: float,
    client_drift: float,
    clean_sat_kl: float,
) -> float:
    count_score = min(1.0, max(0.0, float(count) / 10.0))
    margin_score = min(1.0, max(0.0, float(margin)))
    entropy_penalty = min(1.0, max(0.0, float(entropy)))
    var_penalty = min(1.0, max(0.0, float(intra_var)))
    drift_penalty = min(1.0, max(0.0, float(client_drift)))
    sat_penalty = min(1.0, max(0.0, float(clean_sat_kl)))
    score = 0.35 * count_score + 0.35 * margin_score + 0.30 * (1.0 - (entropy_penalty + var_penalty + drift_penalty + sat_penalty) / 4.0)
    return max(0.0, min(1.0, float(score)))


def _normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return x / torch.linalg.vector_norm(x, ord=2).clamp_min(float(eps))
