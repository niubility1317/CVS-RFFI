"""Vary only the U paired KL while preserving all other native components."""
from contextlib import contextmanager
from experiments.pamf_repl_attribution_factorial import objective as original


def factor_masked_pair(native, factor):
    def masked_pair(a,b,y,accepted,kl_weight):
        return native(a,b,y,accepted,kl_weight*factor)
    return masked_pair


@contextmanager
def installed(config, holder):
    from experiments.pamf_ssl_research import hard_pl_objective
    previous = hard_pl_objective.masked_pair
    hard_pl_objective.masked_pair = factor_masked_pair(previous, config['tuning_recipe']['kl_factor'])
    try:
        with original.installed(config,holder):
            yield
    finally:
        hard_pl_objective.masked_pair = previous
