from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherDiscriminabilityUncertaintyGate(nn.Module):
    """Evidence-first sample gate with an explicit null route.

    Learned corrections are bounded, consume detached context and cannot create
    evidence when the physical quality term is absent.
    """

    def __init__(
        self,
        branch_count: int,
        correction_dim: int,
        delta_max: float = 0.15,
        use_learned_correction: bool = True,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.branch_count = int(branch_count)
        self.correction_dim = int(correction_dim)
        self.delta_max = float(delta_max)
        self.use_learned_correction = bool(use_learned_correction)
        self.eps = float(eps)
        self.log_coefficients = nn.Parameter(torch.zeros(self.branch_count, 4))
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(max(float(temperature), self.eps)))
        )
        if self.use_learned_correction:
            hidden = max(8, self.correction_dim)
            self.correction_net = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(self.correction_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(self.branch_count)
            )
        else:
            self.correction_net = None

    @staticmethod
    def _evidence_tensor(evidence: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in evidence:
            raise KeyError(f"missing gate evidence {key}")
        default = 1.0 if key == "U" else 0.0
        return torch.nan_to_num(
            evidence[key].float(), nan=default, posinf=default, neginf=default
        ).clamp(0.0, 1.0)

    def forward(
        self,
        evidence: Mapping[str, torch.Tensor],
        *,
        correction_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        i = self._evidence_tensor(evidence, "I")
        d = self._evidence_tensor(evidence, "D")
        s = self._evidence_tensor(evidence, "S")
        u = self._evidence_tensor(evidence, "U")
        if not (i.shape == d.shape == s.shape == u.shape):
            raise ValueError("I, D, S and U must share shape [B,branch_count]")
        if i.dim() != 2 or i.size(-1) != self.branch_count:
            raise ValueError("gate evidence must have shape [B,branch_count]")

        coefficients = F.softplus(self.log_coefficients) + self.eps
        physical_logits = (
            coefficients[:, 0].unsqueeze(0) * torch.log(i + self.eps)
            + coefficients[:, 1].unsqueeze(0) * torch.log(d + self.eps)
            + coefficients[:, 2].unsqueeze(0) * torch.log(s + self.eps)
            - coefficients[:, 3].unsqueeze(0) * u
        )
        quality = (i * d * s * (1.0 - u)).clamp(0.0, 1.0)

        if self.use_learned_correction:
            if correction_context is None:
                raise ValueError("correction_context is required when correction is enabled")
            if tuple(correction_context.shape[:2]) != tuple(i.shape):
                raise ValueError("correction_context must start with [B,branch_count]")
            if correction_context.size(-1) != self.correction_dim:
                raise ValueError("correction_context final dimension is incorrect")
            detached_context = correction_context.detach()
            raw_correction = torch.cat(
                [
                    network(detached_context[:, branch_index, :])
                    for branch_index, network in enumerate(self.correction_net)
                ],
                dim=-1,
            )
            correction = self.delta_max * torch.tanh(raw_correction) * quality
        else:
            correction = torch.zeros_like(physical_logits)

        branch_logits = physical_logits + correction
        maximum_quality = quality.max(dim=-1).values
        null_logit = 2.0 - 6.0 * maximum_quality
        temperature = self.log_temperature.exp().clamp(0.25, 4.0)
        probabilities = torch.softmax(
            torch.cat([null_logit.unsqueeze(-1), branch_logits], dim=-1) / temperature,
            dim=-1,
        )
        null_weight = probabilities[:, 0]
        weights = probabilities[:, 1:]
        entropy = -(
            probabilities * probabilities.clamp_min(self.eps).log()
        ).sum(dim=-1)
        return {
            "weights": weights,
            "null_weight": null_weight,
            "q_sample": 1.0 - null_weight,
            "physical_logits": physical_logits,
            "correction": correction,
            "entropy": entropy,
            "branch_usage": weights.mean(dim=0),
            "temperature": temperature,
        }


class NormalizedFiveBranchFusion(nn.Module):
    """Project each branch to a common unit sphere before convex fusion."""

    def __init__(
        self,
        branch_names: Sequence[str],
        input_dim: int,
        output_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.branch_names = tuple(branch_names)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.eps = float(eps)
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(self.input_dim, self.output_dim, bias=False)
                for name in self.branch_names
            }
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self, branches: Mapping[str, torch.Tensor], weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if weights.dim() != 2 or weights.size(-1) != len(self.branch_names):
            raise ValueError("weights must have shape [B,branch_count]")
        projected = []
        for name in self.branch_names:
            value = branches[name]
            if value.dim() != 2 or value.size(-1) != self.input_dim:
                raise ValueError(f"branch {name} has the wrong embedding shape")
            projected.append(F.normalize(self.projections[name](value), dim=-1, eps=self.eps))
        stacked = torch.stack(projected, dim=1)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        fused = F.normalize(self.output_norm(fused), dim=-1, eps=self.eps)
        diagnostics = {
            "projected_norms": stacked.norm(dim=-1),
            "weighted_branch_norms": stacked.norm(dim=-1) * weights,
        }
        return fused, diagnostics
