"""Fixed ten-class factorial coverage comparison, with existing baseline reused."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import full_matrix_plan as original

RUN = '20261112-phase1-pamf-coverage-tenclass-m9-r01'
SEEDS = (2026092701, 2026092702, 2026092703)


def candidate_rows():
    return [dict(row_id=f'ten_class-coverage-{name}-s{seed}', setting='ten_class', protocol_id='ManyRx',
                 model_seed=seed, reference_memory=memory, tiered=tiered, views=list(VIEWS))
            for name, memory, tiered in (('reference', True, False), ('tiered', False, True),
                                         ('reference-tiered', True, True)) for seed in SEEDS]


def baseline_row(row):
    return next(r for r in original.candidate_rows()
                if r['setting'] == 'ten_class' and r['model_seed'] == row['model_seed']
                and r['arm'] == 'branch_prototype')


def proxy(c, row):
    old = baseline_row(row)
    root = Path(c['output_root'])
    segment = root.name if root.name in ('segment30', 'segment50') else None
    source = root.parent if segment else root
    old_root = source.parents[3] / original.RUN / 'F' / old['row_id'] / 'source'
    if segment:
        old_root /= segment
    return dict(c, run_id=original.RUN, row_id=old['row_id'], algorithm_id='pamf_tri_two_method_matrix_v1',
                output_root=str(old_root))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c.get('coverage_row') != row or c.get('run_id') != RUN or c.get('source_development'):
        raise ValueError('Only registered full-source coverage rows are allowed')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Independent output root required')
    recipe = dict(reference_memory=row['reference_memory'], tiered=row['tiered'], weak_weight=.25,
                  memory_per_class_day=8, memory_refresh_steps=20, labeled_diagnostic_interval=50)
    if c.get('coverage') != recipe:
        raise ValueError('Actual coverage recipe differs from fixed row')
    original.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if run_id != RUN or row not in candidate_rows():
        raise ValueError('Unregistered coverage row')
    c = original.configuration(context, baseline_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], coverage_row=dict(row),
             algorithm_id='pamf_tri_coverage_factorial_v1',
             coverage=dict(reference_memory=row['reference_memory'], tiered=row['tiered'], weak_weight=.25,
                           memory_per_class_day=8, memory_refresh_steps=20, labeled_diagnostic_interval=50),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    return original.parent(proxy(c, validate(c)))


check_graph = original.check_graph
