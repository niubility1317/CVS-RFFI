"""Install per-component projection while preserving REPL diagnostics."""
from contextlib import contextmanager
from .projection import protected_components


@contextmanager
def installed(c, holder):
    from experiments.pamf_repl_weight_more import objective as more
    from experiments.pamf_repl_weight_more import plan as more_plan
    from experiments.pamf_ssl_research import hard_pl_guard_objective as guard
    from .plan import validate
    old_validate, old_protected = more_plan.validate, guard.protected_loss
    more_plan.validate = validate
    def guarded(labeled, unlabeled, named, *, max_u_ratio):
        objective = holder['objective']
        names = c['tuning_recipe']['project_components']
        parts = {name: objective.parts[name] for name in names}
        value, details, gradients, stats = protected_components(
            labeled, unlabeled, named, parts, max_u_ratio=max_u_ratio)
        objective.metrics.update({'HPL_component_guard_'+key: value for key,value in stats.items()})
        return value, details, gradients
    guard.protected_loss = guarded
    try:
        with more.installed(c, holder):
            yield
    finally:
        guard.protected_loss = old_protected
        more_plan.validate = old_validate
