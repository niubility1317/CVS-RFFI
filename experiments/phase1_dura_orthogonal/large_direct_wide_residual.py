"""Direct time-frequency recovery with a wider per-sample residual bound."""
from experiments.phase1_dura_orthogonal.large_direct_timefreq_stable import StableLargeDirectTimeFreq

RESIDUAL_SCALE = 8.0


class WideResidualDirectTimeFreq(StableLargeDirectTimeFreq):
    def __init__(self, width=384, layers=8):
        super().__init__(width=width, layers=layers)
        self.residual_scale = RESIDUAL_SCALE
