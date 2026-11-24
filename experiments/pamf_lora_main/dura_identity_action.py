"""Keep the original DURA priors and variance while using identity data fidelity."""

from experiments.pamf_action_restore.models import SCALE
from .dura_prior_width import PacketDura


class IdentityAction:
    @staticmethod
    def __call__(value):
        return value

    adjoint = __call__


def solve_identity_map(_action, observed, prior, variance, *, steps, initial):
    """Solve the same MAP objective exactly when the action is I."""
    prior_variance = SCALE**2
    return (prior_variance * observed + variance * prior) / (prior_variance + variance)


class IdentityActionDura(PacketDura):
    def __init__(self):
        super().__init__(wide=True)

    def solve(self, native, _observed, _rotation=None):
        coarse = self.restorer.frozen.coarse(native)
        variance = self.restorer.frozen.estimator.variance_only(native, coarse)
        context = (coarse, IdentityAction(), variance)
        return self.restorer.stage_outputs(native, context=context, solve_map=solve_identity_map)
