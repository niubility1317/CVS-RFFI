from contextlib import contextmanager

from experiments.pamf_repl_adaptive_kl.objective import AdaptiveKLREPL, anchor
from experiments.pamf_repl_day_pairs.objective import installed as day_scope
from experiments.pamf_repl_source_adapt.receiver_balance import positive_target
from experiments.pamf_repl_view_dose.objective import installed as view_scope
from experiments.pamf_ssl_research import hard_pl_objective as hard


class TenClassREPL(AdaptiveKLREPL):
    def compute_ce(self, raw_ce, recipe):
        return recipe['weight'] * recipe['raw_ce_dose'] * raw_ce

    def loss(self, student, device):
        self.balance_stats = None
        loss = super().loss(student, device)
        stats = self.balance_stats or dict(affected=0., tv=0.)
        self.metrics.update({f'HPL_balance_{key}': value for key, value in stats.items()})
        self.balance_stats = None
        return loss


@contextmanager
def installed(config, holder):
    recipe = config['tuning_recipe']
    if recipe['setting'] == 'source_rx_6':
        scope = day_scope if 'day_rule' in recipe else view_scope
        with scope(config, holder):
            yield
        return

    original = hard.contrast

    def balance(maps, labels, receivers):
        distribution, stats = positive_target(labels, receivers)
        holder['objective'].balance_stats = stats
        return original(maps, labels, receivers, positive_distribution=distribution)

    if recipe['rx_balance']:
        hard.contrast = balance
    try:
        with anchor.installed(config, holder, objective_class=TenClassREPL):
            yield
    finally:
        hard.contrast = original
