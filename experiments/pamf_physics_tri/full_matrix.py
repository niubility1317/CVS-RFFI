"""Fixed physical hybrid and branch prototype on remaining data settings."""
from contextlib import contextmanager

from .hybrid_contrast import HybridContrast
from .practical_gate import installed as practical_scope
from .source_prototype import SourcePrototype


class EmptyCertifiedBatch(ValueError):
    """A legal source batch contained no accepted certified physical packet."""


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


def smoke(rt):
    from pathlib import Path
    from experiments.cric_dura_comparison.io import write
    from . import full_matrix_plan as plan
    from .pair_runner import check_full_matrix
    from .runner import smoke as smoke_rows
    checks = []
    for row in plan.candidate_rows():
        for offset in range(10):
            try:
                item = smoke_rows(rt, rows=lambda: [row], configure=plan.configuration, check_parent=plan.parent,
                                  scope=installed, check_u=check_full_matrix, graph_check=plan.check_graph,
                                  segment_output=True, labeled_batch_offset=offset, write_result=False)
            except EmptyCertifiedBatch:
                continue
            item[0]['source_labeled_batch_offset'] = offset
            checks.extend(item)
            break
        else:
            raise ValueError(f'No certified physical U packet in ten source batches: {row["row_id"]}')
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          optimizer_updates=0, query_reads=0, U_truth_reads=0))
