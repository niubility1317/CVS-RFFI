"""Source-channel same-packet identity in the deployed fusion space."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import single_rx_plan as base
from .plan import SEEDS

RUN = '20261111-phase1-pamf-tri-fused-physics-m3-r01'


def candidate_rows():
    return [dict(row_id=f'{base.SETTING}-tri-fused-physics-s{seed}', setting=base.SETTING,
                 protocol_id=base.PROTOCOL, model_seed=seed, views=list(VIEWS)) for seed in SEEDS]


def old_row(row):
    return next(r for r in base.candidate_rows() if r['arm'] == 'day_physics' and r['model_seed'] == row['model_seed'])


def proxy(c, row):
    original = old_row(row)
    root = Path(c['output_root'])
    return dict(c, run_id=base.RUN, row_id=original['row_id'], algorithm_id='pamf_tri_single_rx_day_v1',
                output_root=str(root.parents[3] / base.RUN / 'F' / original['row_id'] / 'source'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != base.PROTOCOL or c['algorithm_id'] != 'pamf_tri_fused_physics_v1'
            or c['fused_physics'] != dict(weight='original_joint', raw_detached=True)):
        raise ValueError('Unregistered same-packet fusion objective')
    base.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only the three registered fused-physics rows are allowed')
    c = base.configuration(context, old_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_fused_physics_v1',
             fused_physics=dict(weight='original_joint', raw_detached=True),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Unique fusion candidate output root required')
    return base.parent(proxy(c, row))


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    from .certified_day_plan import RUN as CERT_RUN
    runs = Path(rt['project']) / 'runs'
    seed = row['model_seed']
    day = runs / base.RUN / 'readouts' / old_row(row)['row_id']
    cert = runs / CERT_RUN / 'readouts' / f'{base.SETTING}-tri-certified-branch-s{seed}'
    return [(f'day-physics-{base.PROTOCOL}-s{seed}', day / 'predictions', day / 'scores'),
            (f'branch-certified-{base.PROTOCOL}-s{seed}', cert / 'predictions', cert / 'scores')]
