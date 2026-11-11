"""Fixed source-day relation certificates in the single-RX setting."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import single_rx_plan as base
from .plan import SEEDS

RUN = '20261111-phase1-pamf-tri-certified-day-m6-r01'
ARMS = ('branch', 'echo')


def candidate_rows():
    return [dict(row_id=f'{base.SETTING}-tri-certified-{arm}-s{seed}', setting=base.SETTING,
                 protocol_id=base.PROTOCOL, model_seed=seed, arm=arm, views=list(VIEWS))
            for arm in ARMS for seed in SEEDS]


def original_row(row):
    return next(r for r in base.candidate_rows() if r['arm'] == 'day_physics' and r['model_seed'] == row['model_seed'])


def proxy(c, row):
    old_row = original_row(row)
    root = Path(c['output_root'])
    return dict(c, run_id=base.RUN, row_id=old_row['row_id'], algorithm_id='pamf_tri_single_rx_day_v1',
                output_root=str(root.parents[3] / base.RUN / 'F' / old_row['row_id'] / 'source'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != base.PROTOCOL or c['relation_certificate'] != row['arm']
            or c['algorithm_id'] != 'pamf_tri_certified_day_v1'):
        raise ValueError('Unregistered cross-day relation certificate')
    base.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only six registered source-day certificate rows allowed')
    c = base.configuration(context, original_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_certified_day_v1',
             relation_certificate=row['arm'], output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Dedicated candidate output root required')
    return base.parent(proxy(c, row))


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    root = Path(rt['project']) / 'runs' / base.RUN / 'readouts' / original_row(row)['row_id']
    return [(f'day-physics-{base.PROTOCOL}-s{row["model_seed"]}', root / 'predictions', root / 'scores')]
