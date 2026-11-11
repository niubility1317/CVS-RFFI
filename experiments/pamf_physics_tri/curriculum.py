"""Install the fixed Tri SSL after a natural-order source-only warm-up."""
from contextlib import contextmanager

from .certified_day import CertifiedDay
from .single_rx import SingleRxDay
from .practical_gate import installed as practical_scope


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .curriculum_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    factory = CertifiedDay if c['relation_certificate'] == 'branch' else SingleRxDay
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=factory):
        yield
