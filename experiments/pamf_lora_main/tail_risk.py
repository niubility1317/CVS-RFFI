"""Tail waveform loss within strata defined by received disturbance."""

import torch


def stratified_upper_quarter(risks, disturbance):
    """Average the worst quartile separately in the two receive-noise strata."""
    ordered = disturbance.flatten().argsort(stable=True)
    per_stage = risks.flatten(1)[:, ordered]
    return torch.stack([stratum.topk((stratum.shape[1]+3)//4, dim=1).values.mean(dim=1)
                        for stratum in per_stage.chunk(2, dim=1)]).mean(dim=0)
