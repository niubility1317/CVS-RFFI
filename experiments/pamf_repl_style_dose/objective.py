"""Prefer source-date style donors within the original cross-receiver pairs."""
from functools import partial

import torch

from experiments.pamf_repl_adaptive_kl.objective import AdaptiveKLREPL
from experiments.pamf_repl_anchor_kl import objective as anchor
from experiments.pamf_ssl_research.u_mixstyle_objective import partner_eligibility


@torch.no_grad()
def redirect_donors(original, preferred, ranks):
    # A uniform original draw conditioned on already satisfying the preference
    # is uniform within it. Only anchors needing a redirect use the new ranks.
    supported = preferred.any(1)
    original_preferred = preferred.gather(1, original[:, None]).squeeze(1)
    replacement = ranks.masked_fill(~preferred, -1.).argmax(1)
    return torch.where(supported & ~original_preferred, replacement, original)


class DateStyleREPL(AdaptiveKLREPL):
    donor_inputs = None
    donor_metrics = None

    def loss(self, student, device):
        try:
            value = super().loss(student, device)
            if self.donor_metrics is None or self.donor_inputs is not None:
                raise ValueError('Expected one shared style donor plan per original U batch')
            self.metrics.update(self.donor_metrics)
            return value
        finally:
            self.donor_inputs = None
            self.donor_metrics = None

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        mixed = super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)
        receivers = metadata['rx_i'].to(device=x.device, dtype=torch.long)
        days = metadata['day_i'].to(device=x.device, dtype=torch.long)
        if days.shape != pseudo.shape or bool((~torch.isin(days, days.new_tensor(self.c['source_days']))).any()):
            raise ValueError('Expected only aligned, registered source dates in U')
        self.donor_inputs = pseudo, accepted, receivers, days, step
        return mixed

    @torch.no_grad()
    def select_style_partners(self, original, valid):
        if self.donor_inputs is None or self.donor_metrics is not None:
            raise ValueError('Expected the current source metadata and a single donor selection')
        pseudo, accepted, receivers, days, step = self.donor_inputs
        self.donor_inputs = None
        eligible = partner_eligibility(pseudo, accepted, receivers)
        cross_day = days[:, None] != days[None, :]
        both, same = eligible & cross_day, eligible & ~cross_day
        rule = self.c['tuning_recipe']['date_donor']
        applied = original
        if rule != 'original':
            preferred = {'cross_day': both, 'same_day': same}[rule]
            generator = torch.Generator(device=original.device)
            generator.manual_seed(self.c['model_seed'] * 100_000 + step)
            ranks = torch.rand(preferred.shape, device=original.device, generator=generator)
            applied = redirect_donors(original, preferred, ranks)
        self.donor_metrics = dict(
            HPL_donor_eligible=valid.sum(),
            HPL_donor_same_supported=same.any(1).sum(),
            HPL_donor_cross_supported=both.any(1).sum(),
            HPL_donor_original_cross=(valid & (days != days[original])).sum(),
            HPL_donor_applied_cross=(valid & (days != days[applied])).sum(),
            HPL_donor_changed=(valid & (original != applied)).sum(),
        )
        return applied


installed = partial(anchor.installed, objective_class=DateStyleREPL)
