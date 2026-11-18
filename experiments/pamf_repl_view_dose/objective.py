"""Reuse the actual view loss graphs and selectively reweight CE or KL."""
from contextlib import contextmanager

from experiments.pamf_repl_uaux_soft import objective as soft
from experiments.pamf_ssl_research import hard_pl_objective as hard
from .plan import ORIGINAL_KL


VIEW_NAMES = ('ordinary', 'mixed', 'as_slot', 'raw')


def component(views, weights):
    return sum(value * weight for value, weight in zip(views, weights, strict=True))


def routed_components(views, parts, recipe):
    ce_views, kl_views = zip(*views, strict=True)
    ce = recipe['weight'] * component(ce_views, recipe['ce_views'])
    # Preserve the exact inherited KL graph on all CE-routing rows.
    if tuple(recipe['kl_views']) == ORIGINAL_KL:
        kl = parts['pair_kl']
    else:
        kl = recipe['weight'] * recipe['kl_factor'] * component(kl_views, recipe['kl_views'])
    return ce, kl


class ViewDoseREPL(soft.SoftREPL):
    def __init__(self, config):
        super().__init__(config)
        self.view_graphs = []

    def loss(self, student, device):
        if self.view_graphs:
            raise ValueError('Unconsumed view graphs from previous U batch')
        super().loss(student, device)
        if len(self.view_graphs) != len(VIEW_NAMES):
            raise ValueError('Expected the original four masked-pair calls')
        recipe = self.c['tuning_recipe']
        views, populations, accepted_rates = zip(*self.view_graphs, strict=True)
        base_ce, pair_kl = routed_components(views, self.parts, recipe)
        self.parts['pair_kl'] = pair_kl
        auxiliary = sum(self.parts.values())
        effective = base_ce + auxiliary
        self.metrics.update(HPL_pre_route_U_loss=self.pending_gradient_loss.detach(),
                            HPL_effective_CE_loss=base_ce.detach(),
                            HPL_effective_KL_loss=pair_kl.detach(),
                            HPL_effective_aux_loss=auxiliary.detach(),
                            HPL_loss=effective.detach(),
                            HPL_base_loss=(effective / self.settings['weight']).detach())
        for name, (ce, kl), n, accepted_rate in zip(VIEW_NAMES, views, populations, accepted_rates, strict=True):
            self.metrics[f'HPL_view_{name}_CE'] = ce.detach()
            self.metrics[f'HPL_view_{name}_KL'] = kl.detach()
            self.metrics[f'HPL_view_{name}_denominator'] = n
            self.metrics[f'HPL_view_{name}_acceptance'] = accepted_rate
        self.pending_gradient_loss = effective
        self.view_graphs.clear()
        return effective.detach() * 0.


@contextmanager
def installed(config, holder):
    original = soft.SoftREPL
    soft.SoftREPL = ViewDoseREPL
    try:
        with soft.installed(config, holder):
            priority_pair = hard.masked_pair

            def capture_pair(a, b, targets, accepted, kl_weight):
                output = priority_pair(a, b, targets, accepted, kl_weight)
                _, ce, kl = output
                holder['objective'].view_graphs.append(
                    ((ce, kl), accepted.numel(), accepted.float().mean().detach()))
                return output

            hard.masked_pair = capture_pair
            try:
                yield
            finally:
                hard.masked_pair = priority_pair
    finally:
        soft.SoftREPL = original
