"""Reuse full telemetry while adding an independently controlled CE guard."""
from contextlib import contextmanager
from experiments.pamf_repl_priority import objective as priority
from experiments.pamf_repl_uaux_soft.objective import SoftREPL
from .projection import guarded_loss

@contextmanager
def installed(c, holder):
    old_cls, old_reference = priority.PriorityREPL, priority.reference_loss
    recipe = c['tuning_recipe']
    def reference(labeled,unlabeled,ref,named,*,cap):
        if cap is None:
            return old_reference(labeled,unlabeled,ref,named,cap=None)
        parts = holder['objective'].parts
        other = sum(parts[k] for k in ('shallow','pair_kl','joint','extrap_delta') if k != recipe['reference'])
        return guarded_loss(labeled,unlabeled,ref,other,named,
                            alpha=recipe['alpha'],beta=recipe['beta'],gamma=recipe['gamma'])
    priority.PriorityREPL, priority.reference_loss = SoftREPL, reference
    try:
        with priority.installed(c,holder):
            yield
    finally:
        priority.PriorityREPL, priority.reference_loss = old_cls, old_reference
