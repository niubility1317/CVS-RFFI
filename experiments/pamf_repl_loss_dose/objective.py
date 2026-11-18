"""Independently scale the exact base CE and the four weighted auxiliary terms."""
from contextlib import contextmanager

from experiments.pamf_repl_uaux_soft import objective as soft


def apply_doses(full_u, parts, ce_dose, auxiliary_dose):
    auxiliary = parts['shallow'] + parts['pair_kl'] + parts['joint'] + parts['extrap_delta']
    base_ce = full_u - auxiliary
    effective = full_u if ce_dose == auxiliary_dose == 1. else (
        ce_dose * base_ce + auxiliary_dose * auxiliary)
    return effective, base_ce, auxiliary


class LossDoseREPL(soft.SoftREPL):
    def loss(self, student, device):
        super().loss(student, device)
        recipe = self.c['tuning_recipe']
        ce_dose, auxiliary_dose = recipe['ce_dose'], recipe['auxiliary_dose']
        original = self.pending_gradient_loss
        effective, base_ce, auxiliary = apply_doses(original, self.parts, ce_dose, auxiliary_dose)
        self.parts = {name: value * auxiliary_dose for name, value in self.parts.items()}
        self.pending_gradient_loss = effective
        self.metrics.update(HPL_pre_dose_U_loss=original.detach(),
                            HPL_effective_CE_loss=(ce_dose * base_ce).detach(),
                            HPL_effective_aux_loss=(auxiliary_dose * auxiliary).detach(),
                            HPL_loss=effective.detach(),
                            HPL_base_loss=(effective / self.settings['weight']).detach())
        return effective.detach() * 0.


@contextmanager
def installed(config, holder):
    cls = soft.SoftREPL
    soft.SoftREPL = LossDoseREPL
    try:
        with soft.installed(config, holder):
            yield
    finally:
        soft.SoftREPL = cls
