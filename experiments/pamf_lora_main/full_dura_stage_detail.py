"""Supervise DURA's non-carrier detail from paired source IQ at all unfolded stages."""
import torch
from torch import nn

from .dura_repair import PacketRestorer, complex_iq, no_tone


DETAIL_WEIGHT=.10


class StageDetailLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tracker=PacketRestorer('consensus').eval()

    def forward(self, received, clean, stages):
        with torch.no_grad():
            _,rot,*_=self.tracker.track(received[:,None])
            rotation=rot[:,0].conj()
            target=complex_iq(clean[:,None])[:,0]*rotation
            target_detail=no_tone(target)
            scale=target_detail.abs().square().mean(-1)+.002*target.abs().square().mean(-1)
            scale=scale.clamp_min(1e-6)
        loss=[]
        for output in stages:
            recovered=complex_iq(output[:,None])[:,0]*rotation
            error=no_tone(recovered)-target_detail
            loss.append(error.abs().square().mean(-1)/scale)
        return DETAIL_WEIGHT*torch.stack(loss).mean(0)
