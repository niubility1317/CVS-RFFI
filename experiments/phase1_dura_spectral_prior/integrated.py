"""Replace each DURA prior correction with a frequency-coupled correction."""

from torch import nn

from experiments.phase1_dura_posterior.integrated import PosteriorInDura
from experiments.phase1_dura_spectral_prior.spectral_prior import SpectralPosteriorCorrection


class SpectralPosteriorInDura(PosteriorInDura):
    def __init__(self, dura):
        super().__init__(dura)
        self.corrections = nn.ModuleList([SpectralPosteriorCorrection() for _ in range(2)])
