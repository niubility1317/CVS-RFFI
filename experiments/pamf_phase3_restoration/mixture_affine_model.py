"""Source-mixture restoration with saturation-resistant affine fusion."""
import torch
from .mixture_gate_model import MixtureGateRestorer
from .mixture_gate_posterior import mixture_components, combine_components


def affine_blend(base, components, adjustment):
    """Add a centered, unrestricted four-hypothesis IQ correction to base."""
    if (components.ndim != 3 or base.shape != components.shape[:2]
            or adjustment.shape != (components.shape[0], components.shape[2])):
        raise ValueError('Matched base, component IQ and adjustments required')
    centered = adjustment.to(components) - adjustment.to(components).mean(1, keepdim=True)
    return base + (components @ centered[:, :, None]).squeeze(-1)


class AffineMixtureRestorer(MixtureGateRestorer):
    def __init__(self, seed):
        super().__init__(seed)
        self.initialization_report.update(mode='scratch_single_stage_centered_affine_mixture',
            gate_semantics='unbounded centered affine coefficients, not probabilities',
            gate_initial_adjustment=0.)

    def forward(self, received, *, return_aux=False):
        aux = self.physical.estimate(received)
        with torch.no_grad():
            components, evidence, residual = mixture_components(
                aux['action_matrix'].detach(), received.detach().flatten(1).double(),
                self.physical.prior_mean.double(), self.physical.mixture_within.double(),
                self.physical.mixture_offsets.double(), self.physical.mixture_weights.double(),
                aux['variance'].detach().double())
            base, base_probability = combine_components(components, evidence, torch.zeros_like(evidence))
        adjustment, context = self.detail(received.detach(), components, base_probability, aux['variance'].detach())
        combined = affine_blend(base, components, adjustment).float().reshape_as(received)
        restored = self.detail.refiner(combined, context)
        info = dict(**aux, base_iq=base.float().reshape_as(received), relative_solve_residual=residual,
                    mixture_probability=base_probability.float(), mixture_adjustment=adjustment,
                    mixture_affine_iq=combined)
        return (restored, info) if return_aux else restored


def scratch_affine_mixture(seed):
    return AffineMixtureRestorer(seed)
