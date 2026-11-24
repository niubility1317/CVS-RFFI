"""DURA two-stage capacity control on the common LoRa packet frame."""
from torch import nn

from experiments.pamf_r17_fidelity.models import share_long_weights
from experiments.pamf_r65_four_route.stage_polyphase import STAGE_TARGETS, StageDistilled
from experiments.pamf_reception_restore.models import Core
from .dura_repair import complex_iq, real_iq
from .relative_carrier import match_packet_power


WIDE_PRIOR_WIDTHS = (24, 40, 64, 96)


class PacketDura(nn.Module):
    """Run native DURA twice per chirp with the same packet scale and phase frame."""

    def __init__(self, *, wide=False):
        super().__init__()
        self.restorer = StageDistilled()
        if wide:
            parent_priors = nn.ModuleList([
                Core(incoming=9, widths=WIDE_PRIOR_WIDTHS)
                for _ in range(max(STAGE_TARGETS) + 1)
            ])
            share_long_weights(parent_priors)
            self.restorer.prior = nn.ModuleList([parent_priors[i] for i in STAGE_TARGETS])

    def forward(self, observed, _quality, rotation):
        native = real_iq(observed * rotation).flatten(0, 1)
        stages, _, _, coarse = self.solve(native, observed, rotation)

        def packet_frame(wave):
            phased = complex_iq(wave.unflatten(0, observed.shape[:2])) * rotation.conj()
            return match_packet_power(phased, observed)

        return packet_frame(coarse), packet_frame(stages[-1])

    def solve(self, native, _observed, _rotation=None):
        return self.restorer.stage_outputs(native)
