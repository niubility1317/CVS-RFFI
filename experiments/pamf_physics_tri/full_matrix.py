"""Fixed physical hybrid and branch prototype on remaining data settings."""
from contextlib import contextmanager

from .hybrid_contrast import HybridContrast
from .practical_gate import installed as practical_scope
from .source_prototype import SourcePrototype


@contextmanager
def paper_scope(c, holder, enabled, objective_factory):
    from experiments.pamf_paper_completion.model import installed
    from .full_matrix_plan import validate
    with installed(c, holder, ssl_enabled=enabled, objective_factory=objective_factory, validate=validate):
        yield


@contextmanager
def installed(c, holder):
    from .full_matrix_plan import validate
    row = validate(c)
    factory = SourcePrototype if row['arm'] == 'branch_prototype' else HybridContrast
    with practical_scope(c, holder, base_scope=paper_scope, objective_factory=factory):
        yield
