from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch

from .style_packet import StyleDomainBatch


@dataclass(frozen=True)
class VirtualStyleView:
    x: torch.Tensor
    source: str
    style_id: int
    d_raw: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None


class VirtualDomainSampler:
    """Builds explicit constructed style-domain batches for FL-DG losses."""

    def __init__(self, *, clean_style_id: int = 0):
        self.clean_style_id = int(clean_style_id)

    def build_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        d_raw: Optional[torch.Tensor],
        views: Iterable[VirtualStyleView],
    ) -> StyleDomainBatch:
        xs = [x]
        ys = [y]
        raw_domains = [d_raw] if d_raw is not None else None
        style_domains = [torch.zeros((int(y.numel()),), dtype=torch.long, device=y.device)]
        sources = ["clean"]
        raw_style_ids = [self.clean_style_id]

        for view_idx, view in enumerate(views, start=1):
            vx = view.x.to(device=x.device, dtype=x.dtype)
            vy = view.y.to(device=y.device).long() if view.y is not None else y
            if int(vx.size(0)) != int(vy.numel()):
                raise ValueError("Virtual style view x/y batch sizes must match.")
            xs.append(vx)
            ys.append(vy)
            style_domains.append(torch.full((int(vy.numel()),), int(view_idx), dtype=torch.long, device=y.device))
            sources.append(str(view.source))
            raw_style_ids.append(int(view.style_id))
            if raw_domains is not None:
                raw_domains.append((view.d_raw.to(device=y.device).long() if view.d_raw is not None else d_raw))

        d_raw_out = torch.cat(raw_domains, dim=0) if raw_domains is not None else None
        return StyleDomainBatch(
            x=torch.cat(xs, dim=0),
            y=torch.cat(ys, dim=0).long(),
            d_raw=d_raw_out,
            d_style=torch.cat(style_domains, dim=0).long(),
            sources=tuple(sources),
            metadata={"num_views": len(sources), "raw_style_ids": tuple(raw_style_ids)},
        )
