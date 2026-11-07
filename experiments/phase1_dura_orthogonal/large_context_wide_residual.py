"""Source-only DURA refinement with a wider residual bound."""
from experiments.phase1_dura_orthogonal.large_context_stable import StableLargeContextDura

RESIDUAL_SCALE = 1.0


class WideResidualContextDura(StableLargeContextDura):
    def __init__(self, width=384, layers=8):
        super().__init__(width=width, layers=layers)
        self.residual_scale = RESIDUAL_SCALE
