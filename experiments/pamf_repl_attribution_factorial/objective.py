"""Observe blind U decisions while retaining original component diagnostics."""
from contextlib import contextmanager
from experiments.pamf_repl_weight.objective import ObservedREPL
from experiments.pamf_repl_weight.pseudo_observation import PseudoObservationMixin


class AttributionREPL(PseudoObservationMixin, ObservedREPL):
    pass


@contextmanager
def installed(config, holder):
    from experiments.pamf_repl_component_guard import plan
    from experiments.pamf_repl_weight import objective
    old_objective = objective.ObservedREPL
    # objective.installed imports validate locally, from the weight plan module.
    from experiments.pamf_repl_weight import plan as weight_plan
    old_validate = weight_plan.validate
    objective.ObservedREPL = AttributionREPL
    weight_plan.validate = plan.validate
    try:
        with objective.installed(config, holder):
            try:
                yield
            finally:
                value = holder.get('objective')
                if value is not None:
                    obs = value.close_pseudo_observation()
                    if not config.get('hard_pl_smoke') and (obs['calls'] != value.calls
                            or obs['seen'] != value.seen or obs['accepted'] != value.accepted):
                        raise ValueError('Pseudo observations must match actual U execution')
    finally:
        objective.ObservedREPL = old_objective
        weight_plan.validate = old_validate
