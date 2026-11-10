"""Conservative direction correction for a discrete reference-delay change."""
import torch


def stable_reference_directions(directions, raw_delay, perturbed_delay):
    """Retain every sample, but suppress its direction after a route jump."""
    if (directions.ndim != 2 or raw_delay.shape != directions.shape[:1]
            or perturbed_delay.shape != raw_delay.shape
            or raw_delay.device != directions.device or perturbed_delay.device != directions.device
            or not bool(torch.isfinite(directions).all())):
        raise ValueError('Finite directions and aligned delay selections required')
    stable = raw_delay == perturbed_delay
    return directions.detach() * stable[:, None].to(directions.dtype)
