from __future__ import annotations

from typing import Optional

import torch


def build_reliable_mask(
    prob: torch.Tensor,
    proto_margin: Optional[torch.Tensor] = None,
    *,
    tau_conf: float = 0.85,
    tau_margin: float = 0.05,
    prepost_kl: Optional[torch.Tensor] = None,
    prepost_kl_max: float = 1.0,
    use_prepost_agreement_filter: bool = True,
) -> torch.Tensor:
    conf = prob.max(dim=-1).values
    reliable = conf > float(tau_conf)
    if proto_margin is not None:
        reliable = reliable & (proto_margin > float(tau_margin))
    if use_prepost_agreement_filter and prepost_kl is not None:
        reliable = reliable & (prepost_kl < float(prepost_kl_max))
    return reliable
