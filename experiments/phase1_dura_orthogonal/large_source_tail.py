"""Fixed source-only hard-example IQ risk for robust restoration training."""
import torch

FRACTION = .25
WEIGHT = .5


def tail_iq_penalty(output, clean):
    """Weighted mean normalized IQ error over the worst quarter of a batch."""
    if output.shape != clean.shape or output.ndim != 3 or output.shape[1:] != (2, 256) or len(output) == 0:
        raise ValueError('Nonempty matching source IQ windows required')
    energy=clean.square().mean((1,2)).clamp_min(1e-8)
    per=(output-clean).square().mean((1,2))/energy
    count=max(1,(len(per)+3)//4)
    return WEIGHT*torch.topk(per,count).values.mean()
