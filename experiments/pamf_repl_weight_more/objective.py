"""Reuse existing REPL diagnostics; optionally alter paired KL weight."""
from contextlib import contextmanager


@contextmanager
def installed(c,holder):
    from experiments.pamf_repl_weight import objective, plan as original
    from experiments.pamf_ssl_research import hard_pl_objective
    from .plan import validate
    old_validate,old_pair=original.validate,hard_pl_objective.masked_pair
    factor=c['tuning_recipe'].get('kl_factor',1.)
    def masked_pair(a,b,y,accepted,kl_weight):
        return old_pair(a,b,y,accepted,kl_weight*factor)
    original.validate=validate;hard_pl_objective.masked_pair=masked_pair
    try:
        with objective.installed(c,holder):yield
    finally:
        original.validate=old_validate;hard_pl_objective.masked_pair=old_pair
