"""Receive-only W24 readout shared by training and frozen prediction."""


def readout_gain(variance, rms):
    power = rms.square()
    noise = (variance * power).sum(dim=-2, keepdim=True) / power.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    return .5 + noise / (noise + .275**2)


def adaptive_final(observed, final, variance, rms):
    """Return the endpoint whose half edit uses the receive-estimated gain."""
    return observed + 2 * readout_gain(variance, rms) * (final - observed)
