"""Install the paper's matching split graph and frozen taper U mechanism."""
from contextlib import contextmanager

from experiments.pamf_paper_completion.model import installed as paper_scope
from .practical_gate import installed as practical_scope
from .validation_plan import validate


@contextmanager
def base_scope(c, holder, enabled, objective_factory):
    with paper_scope(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    with practical_scope(c, holder, base_scope=base_scope):
        yield
