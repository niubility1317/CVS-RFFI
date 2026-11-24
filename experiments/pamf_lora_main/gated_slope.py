"""Use the existing LoRa bypass to apply packet-shared slope refinement selectively."""

import torch

from .channel_blind import corruption_score
from .dura_repair import complex_iq
from .frame_leakage import slope_refinements
from .joint_unroll_source import coordinates


# Same source-U-only calibration as the frozen shared-covariance parent.
THRESHOLD = 0.045941177755594254


def selected_pooled_frame(tracker, iq, tracked):
    """Change the frame with an inverse mapping, retaining the observed RMS."""
    y, rotation, _, rms = tracked
    _, refined = slope_refinements(y,rotation,tracker.reference.chirp)
    observed = complex_iq(iq)*refined.conj()/rms
    dc_power = observed.mean(-1,keepdim=True).abs().square()
    quality = (dc_power / observed.abs().square().mean(-1,keepdim=True).clamp_min(1e-8)).clamp(0.,1.)
    return observed,refined,quality,rms


@torch.no_grad()
def guarded_coordinates(tracker, iq):
    tracked = coordinates(tracker,iq)
    active = corruption_score(tracked[0]) > THRESHOLD
    if not active.any():
        return tracked
    changed = selected_pooled_frame(tracker,iq[active],tuple(x[active] for x in tracked))
    return tuple(original.index_copy(0,active.nonzero().flatten(),refined)
                 for original,refined in zip(tracked,changed,strict=True))
