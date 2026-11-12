"""Balanced and source-physics-driven branch attenuation, three seeds each."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import certified_day_plan as controls
from . import single_rx_plan as base
from .plan import SEEDS

RUN = '20261112-phase1-pamf-tri-physical-branch-dropout-m9-r01'
MODES = ('balanced', 'physical', 'complementary')


def candidate_rows():
    return [dict(row_id=f'source_rx_1-tri-branchdrop-{mode}-s{seed}', setting=base.SETTING,
                 protocol_id=base.PROTOCOL, model_seed=seed, arm=mode, views=list(VIEWS))
            for mode in MODES for seed in SEEDS]


def old_row(row):
    return next(r for r in base.candidate_rows() if r['arm'] == 'day_physics' and r['model_seed'] == row['model_seed'])


def source_root(c):
    root = Path(c['output_root'])
    return root.parent if root.name in ('segment30', 'segment50') else root


def proxy(c, row):
    original = old_row(row)
    root = source_root(c)
    return dict(c, run_id=base.RUN, row_id=original['row_id'], algorithm_id='pamf_tri_single_rx_day_v1',
                output_root=str(root.parents[3] / base.RUN / 'F' / original['row_id'] / 'source'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != base.PROTOCOL or c['algorithm_id'] != 'pamf_tri_physical_branch_dropout_v1'
            or c['physical_branch_dropout'] != dict(mode=row['arm'], factor=.5)):
        raise ValueError('Unregistered source physical branch dropout')
    if source_root(c).parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Dedicated physical branch dropout output root required')
    base.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only nine registered branch routing rows allowed')
    c = base.configuration(context, old_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_physical_branch_dropout_v1',
             physical_branch_dropout=dict(mode=row['arm'], factor=.5),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    return base.parent(proxy(c, row))


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    equivalent = next(r for r in controls.candidate_rows()
                      if r['arm'] == 'branch' and r['model_seed'] == row['model_seed'])
    return controls.extra_controls(rt, equivalent)
