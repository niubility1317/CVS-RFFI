"""Reallocate the accepted U KL weight using source observations only."""
from functools import partial

import torch

from experiments.pamf_repl_anchor_kl import objective as anchor


@torch.no_grad()
def sample_preference(ordinary, raw, teacher, accepted, rx, signal):
    if signal == 'rx_balance':
        _, inverse, counts = torch.unique(rx[accepted], return_inverse=True, return_counts=True)
        preference = torch.zeros_like(accepted, dtype=torch.float32)
        preference[accepted] = counts[inverse].float().reciprocal()
        return preference

    views = torch.stack((*ordinary, *raw)).float()
    if signal == 'vote':
        return views.argmax(-1).eq(teacher.argmax(-1)).float().mean(0)
    if signal == 'view_gap':
        # Total variation from the evaluation distribution, averaged over four forwards.
        return .5 * (views.softmax(-1) - teacher.float().softmax(-1)).abs().sum(-1).mean(0)
    raise ValueError(f'Unknown KL weighting signal {signal!r}')


@torch.no_grad()
def sample_weights(preference, accepted, strength):
    count = accepted.sum().clamp_min(1)
    center = preference.masked_fill(~accepted, 0).sum() / count
    deviations = (preference - center).masked_fill(~accepted, 0)
    span = deviations.abs().max().clamp_min(1e-6)
    return (1 + (.75 * strength) * (deviations / span)) * accepted


def symmetric_kl(pair, weights):
    a, b = pair
    p, q = a.log_softmax(-1), b.log_softmax(-1)
    per_sample = .5 * ((p.exp() - q.exp()) * (p - q)).sum(-1)
    return (per_sample * weights).mean()


class AdaptiveKLREPL(anchor.AnchorKLREPL):
    def loss(self, student, device):
        self.adaptive_rx = None
        try:
            return super().loss(student, device)
        finally:
            self.adaptive_rx = None

    def unlabeled_student_forward(self, student, x, domain, pseudo, accepted, metadata):
        if self.adaptive_rx is None:
            self.adaptive_rx = metadata['rx_i'].to(device=x.device, dtype=torch.long)
        return super().unlabeled_student_forward(student, x, domain, pseudo, accepted, metadata)

    def compute_kl(self, ordinary, raw, teacher, accepted, recipe):
        rx = self.adaptive_rx
        if rx is None or rx.shape != accepted.shape:
            raise ValueError('Adaptive KL requires one source receiver per aligned U record')
        preference = sample_preference(ordinary, raw, teacher, accepted, rx, recipe['adaptive_signal'])
        weights = sample_weights(preference, accepted, recipe['adaptive_strength'])
        count = accepted.sum().clamp_min(1)
        self.metrics.update(
            HPL_adaptive_weight_sum=weights.sum(),
            HPL_adaptive_weight_mean=weights.sum() / count,
            HPL_adaptive_weight_max=weights.max(),
            HPL_adaptive_weight_min=weights.masked_fill(~accepted, 1).min() * accepted.any(),
            HPL_adaptive_weight_abs_deviation=((weights - 1).abs() * accepted).sum() / count,
            HPL_adaptive_preference_mean=(preference * accepted).sum() / count,
        )
        selected_pair = {'ordinary': ordinary, 'raw': raw}[recipe['adaptive_kl_view']]
        return recipe['weight'] * recipe['kl_factor'] * (.8 * symmetric_kl(selected_pair, weights))


installed = partial(anchor.installed, objective_class=AdaptiveKLREPL)
