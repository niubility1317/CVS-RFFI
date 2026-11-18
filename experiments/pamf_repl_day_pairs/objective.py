"""Reweight existing source U positives without adding views or labels."""
from contextlib import contextmanager

import torch

from experiments.pamf_repl_view_dose import objective as view
from experiments.pamf_ssl_research import hard_pl_joint_objective as joint, hard_pl_objective as hard


@torch.no_grad()
def pair_distributions(labels, receivers, days, rule, strength):
    same_class = labels[:, None] == labels[None, :]
    positives = same_class & (receivers[:, None] != receivers[None, :])
    same_day = days[:, None] == days[None, :]
    both = positives & ~same_day
    base = positives.float() / positives.sum(1, keepdim=True).clamp_min(1)
    if rule == 'date_balance':
        counts = positives.float() @ same_day.float()
        preferred = positives / counts.clamp_min(1.)
    else:
        preferred = (positives & same_day if rule == 'same_date' else both).float()
    support = preferred.sum(1, keepdim=True)
    mix = strength * (support > 0)
    if rule == 'coverage':
        mix = strength * support / (support + 2.)
    preferred = preferred / support.clamp_min(1e-12)
    target = base + mix * (preferred - base)
    valid = positives.any(1) & (~same_class).any(1)
    return target, base, both, valid, mix.squeeze(1)


class DayPairsREPL(view.ViewDoseREPL):
    def loss(self, student, device):
        self.day_metadata = None
        self.day_stats = None
        try:
            value = super().loss(student, device)
            if self.day_metadata is None:
                raise ValueError('The original U batch must supply date metadata')
            if self.day_stats is None:
                if len(self.day_metadata):
                    raise ValueError('The configured U contrast route was not exercised')
                self.day_stats = dict.fromkeys(('valid', 'joint_available', 'uniform_joint_mass',
                                                'applied_joint_mass', 'blend'), 0.)
            self.metrics.update({f'HPL_pairs_{name}': metric for name, metric in self.day_stats.items()})
            return value
        finally:
            self.day_metadata = None
            self.day_stats = None

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        self.day_metadata = metadata['day_i'][accepted]
        return super().mixed_view(student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted)

    def positive_distribution(self, labels, receivers):
        days = self.day_metadata
        if days is None or days.shape != labels.shape:
            raise ValueError('Accepted pseudo labels, receivers and dates must align')
        recipe = self.c['tuning_recipe']
        target, base, both, valid, mix = pair_distributions(labels, receivers, days, recipe['day_rule'], recipe['day_strength'])
        if self.day_stats is None:
            count = valid.sum().clamp_min(1)
            self.day_stats = dict(
                valid=valid.sum(), joint_available=(valid & both.any(1)).sum(),
                uniform_joint_mass=((base * both).sum(1) * valid).sum() / count,
                applied_joint_mass=((target * both).sum(1) * valid).sum() / count,
                blend=(mix * valid).sum() / count,
            )
        return target


@contextmanager
def installed(config, holder):
    module = hard if config['tuning_recipe']['setting'] == 'ten_class' else joint
    original = module.contrast

    def day_contrast(maps, labels, receivers):
        target = holder['objective'].positive_distribution(labels, receivers)
        return original(maps, labels, receivers, positive_distribution=target)

    module.contrast = day_contrast
    try:
        with view.installed(config, holder, objective_class=DayPairsREPL):
            yield
    finally:
        module.contrast = original
