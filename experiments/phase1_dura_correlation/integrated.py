"""Insert global temporal correlation corrections into both DURA stages."""
from torch import nn
from experiments.phase1_dura_posterior.integrated import PosteriorInDura
from experiments.phase1_dura_correlation.correlation_prior import CorrelationPriorCorrection


class CorrelationPosteriorInDura(PosteriorInDura):
    def __init__(self, dura):
        super().__init__(dura)
        self.corrections = nn.ModuleList([CorrelationPriorCorrection() for _ in range(2)])
