from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BranchInvariancePolicy:
    budgets: Mapping[tuple[str, str], float]
    default_budget: float = 0.10

    @classmethod
    def default(cls) -> "BranchInvariancePolicy":
        return cls(
            {
                ("time", "sto"): 0.01,
                ("time", "doppler"): 0.01,
                ("time", "rx_delay"): 0.01,
                ("time", "clock_skew"): 0.30,
                ("frequency", "rx_filter"): 0.01,
                ("frequency", "multipath"): 0.02,
                ("frequency", "tx_spectral_asymmetry"): 0.30,
                ("dac", "rx_iq_residual"): 0.02,
                ("dac", "tx_iq"): 0.30,
                ("pa", "agc"): 0.01,
                ("pa", "path_loss"): 0.01,
                ("pa", "pa_memory"): 0.30,
                ("joint", "common_channel"): 0.03,
            }
        )

    def budget(self, branch: str, direction: str) -> float:
        return float(self.budgets.get((str(branch), str(direction)), self.default_budget))


def branch_invariance_loss(
    sensitivities: Mapping[tuple[str, str], torch.Tensor],
    *,
    policy: BranchInvariancePolicy,
) -> dict[str, object]:
    violations: dict[tuple[str, str], torch.Tensor] = {}
    losses = []
    for key, sensitivity in sensitivities.items():
        excess = (sensitivity.float() - policy.budget(*key)).clamp_min(0.0)
        violations[key] = excess
        losses.append(F.smooth_l1_loss(excess, torch.zeros_like(excess)))
    loss = torch.stack(losses).mean() if losses else torch.tensor(0.0)
    return {"loss": loss, "violations": violations}
