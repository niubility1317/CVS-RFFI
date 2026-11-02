"""Insert source-trained posterior corrections inside the two DURA stages."""

import torch
from torch import nn

from .prior_correction import PosteriorPriorCorrection


class PosteriorInDura(nn.Module):
    def __init__(self, dura):
        super().__init__()
        if getattr(dura, 'arm', None) != 'stage2' or len(dura.prior) != 2:
            raise ValueError('The two-stage DURA model is required')
        self.dura = dura.eval().requires_grad_(False)
        self.corrections = nn.ModuleList([PosteriorPriorCorrection() for _ in range(2)])

    def train(self, mode=True):
        super().train(mode)
        self.dura.eval()
        return self

    def forward(self, received, posterior, posterior_log_variance):
        from experiments.pamf_action_restore.models import SCALE
        from experiments.pamf_restoration.models import complex_iq, pair
        from experiments.pamf_uncertainty_restore.models import weighted_map

        with torch.no_grad():
            base, action, native = self.dura.estimate_context(received)
        variance = native[:, None].expand(-1, 256)
        observed, current = complex_iq(received), base
        logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256)/8
        for core, correction in zip(self.dura.prior, self.corrections):
            innovation = pair(action.adjoint(observed-action(complex_iq(current))))
            innovation = innovation/(innovation.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
            inputs = torch.cat((received, base, current, innovation, logvar), 1)
            prior = base + SCALE*core(inputs)
            prior = correction(received, current, prior, posterior, posterior_log_variance)
            current = pair(weighted_map(action, observed, complex_iq(prior), variance,
                                        steps=2, initial=complex_iq(current)))
        return current
