"""Use held-out complex prediction dispersion to control LoRa recovery precision."""

from .lapped_posterior import LappedPosterior, analysis, other_mean


def held_eigenvalues(value, reference, spread, observation_noise):
    """Return twice the directional held-out variances in the complex scale."""
    supports = value.shape[1] - 1
    # For an audited chirp, subtract the mean built from the other supports.
    # This gives supports/(supports-1) times its deviation from the support mean.
    inflation = (supports / (supports - 1)) ** 2
    pseudo = other_mean(value.square()) - reference.square()
    axis = pseudo / pseudo.abs().clamp_min(1e-10)
    # Damp the sample pseudo-covariance from the small support set.
    separation = (supports - 1) / (supports + 1) * inflation * pseudo.abs()
    center = inflation * spread - observation_noise
    large = (center + separation).clamp_min(0.)
    small = (center - separation).clamp_min(0.)
    return large, small, axis


class HeldChirpPrecision(LappedPosterior):
    def __init__(self, directional):
        super().__init__(local=True)
        self.directional = directional

    def transformed(self, observed):
        # Keep the whole carrier outside the taper and the learned prior.
        return analysis(observed - observed.mean(-1, keepdim=True), self.window)

    def precision(self, value, reference, spread, noise, support_noise, variance, *, scalar_readout):
        large, small, axis = held_eigenvalues(value, reference, spread, support_noise)
        base = .02 + noise + variance
        scalar = .85 * noise / (base + .5 * (large + small))
        if not self.directional or scalar_readout:
            return scalar, None
        contrast = (small - large) / (2 * base + large + small)
        # The eigen-gain mean stays equal to the scalar control at every bin.
        return scalar, scalar * contrast * axis
