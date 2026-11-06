"""C4-equivariant physical estimation and detail recovery with one posterior solve."""
import torch

from .covariance_detail import CovarianceDetail
from .covariance_inverse import gaussian_posterior


def rotate_iq(iq, quarter):
    if quarter == 0:
        return iq
    if quarter == 2:
        return -iq
    i, q = iq.unbind(dim=1)
    return torch.stack((-q, i) if quarter == 1 else (q, -i), dim=1)


def operator_in_original_frame(operator, quarter):
    """Conjugate a real [I, Q] matrix by R^-1 A R using signed blocks."""
    if quarter % 2 == 0:
        return operator
    upper, lower = operator.chunk(2, dim=-2)
    ii, iq = upper.chunk(2, dim=-1)
    qi, qq = lower.chunk(2, dim=-1)
    return torch.cat((torch.cat((qq, -qi), dim=-1),
                      torch.cat((-iq, ii), dim=-1)), dim=-2)


def coordinate_prior(mean, covariance):
    """Moment-match the four equally weighted rotations of the source Gaussian."""
    mean = mean.double()
    moment = covariance.double() + torch.outer(mean, mean)
    return torch.zeros_like(mean), (moment + operator_in_original_frame(moment, 1)) * .5


def quarter_mean(values):
    a, b, c, d = values
    return ((a + b) + (c + d)) * .25


class CoordinateMAPDetail(CovarianceDetail):
    def __init__(self, seed):
        super().__init__(seed)
        self.initialization_report = dict(mode='scratch_C4_operator_and_detail_single_stage',
            model_seed=seed, detail_seed=seed+1, ancestor_checkpoints=[], training_stages=1,
            physical_estimation_TX_labels=False, continuous_F0_guidance_detail_only=True,
            source_prior_statistics_required=True)

    def forward(self, received, *, return_aux=False):
        orbit = [rotate_iq(received, quarter) for quarter in range(4)]
        estimates = [self.physical.estimate(observed) for observed in orbit]
        action = quarter_mean([operator_in_original_frame(estimate['action_matrix'], quarter)
                               for quarter, estimate in enumerate(estimates)])
        variance = quarter_mean([estimate['variance'] for estimate in estimates])
        # One solve on the averaged action. Paired IQ and F0 losses only update detail.
        with torch.no_grad():
            base, residual = gaussian_posterior(action.detach(), received.flatten(1).double(),
                self.physical.prior_mean.double(), self.physical.prior_covariance.double(), variance.detach().double())
            base = base.float().reshape_as(received)
            log_noise = variance.detach().log()[:, None, None].expand(-1, 1, received.shape[-1])
        details = []
        for quarter, observed in enumerate(orbit):
            restored = self.detail(rotate_iq(base, quarter), torch.cat((observed.detach(), log_noise), dim=1))
            details.append(rotate_iq(restored, (-quarter) % 4))
        restored = quarter_mean(details)
        if return_aux:
            return restored, dict(action_matrix=action, variance=variance, base_iq=base, relative_solve_residual=residual)
        return restored
