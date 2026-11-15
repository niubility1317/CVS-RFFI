"""Use the same REPL objective and telemetry as the interrupted matrix."""
from contextlib import contextmanager


@contextmanager
def installed(c, holder):
    from experiments.pamf_repl_weight_more import objective, plan as original
    from .plan import validate
    old = original.validate
    original.validate = validate
    try:
        with objective.installed(c, holder):
            yield
    finally:
        original.validate = old
