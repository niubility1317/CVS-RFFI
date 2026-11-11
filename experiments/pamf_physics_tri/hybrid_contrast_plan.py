"""Three fixed same-packet contrast structures with matched source seeds."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import fused_physics_plan as base
from .plan import SEEDS

RUN = '20261112-phase1-pamf-tri-hybrid-contrast-m9-r01'
MODES = ('symmetric', 'branch', 'fusion_branch')


def candidate_rows():
    return [dict(row_id=f'source_rx_1-tri-hybrid-{arm}-s{seed}', setting='source_rx_1',
                 protocol_id='m1-natural', model_seed=seed, arm=arm, views=list(VIEWS))
            for arm in MODES for seed in SEEDS]


def old_row(row):
    return next(r for r in base.candidate_rows() if r['model_seed'] == row['model_seed'])


def proxy(c, row):
    original = old_row(row)
    root = Path(c['output_root'])
    return dict(c, run_id=base.RUN, row_id=original['row_id'], algorithm_id='pamf_tri_fused_physics_v1',
                output_root=str(root.parents[3] / base.RUN / 'F' / original['row_id'] / 'source'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != 'm1-natural' or c['algorithm_id'] != 'pamf_tri_hybrid_contrast_v1'
            or c['hybrid_contrast'] != dict(mode=row['arm'], temperature=.2)):
        raise ValueError('Unregistered source physical contrast structure')
    base.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only nine registered physical contrast rows allowed')
    c = base.configuration(context, old_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_hybrid_contrast_v1',
             hybrid_contrast=dict(mode=row['arm'], temperature=.2),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Dedicated physical contrast output root required')
    return base.parent(proxy(c, row))


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    from .packet_contrast_plan import RUN as PACKET_RUN
    runs = Path(rt['project']) / 'runs'
    seed = row['model_seed']
    controls = list(base.extra_controls(rt, old_row(row)))
    fusion = runs / base.RUN / 'readouts' / old_row(row)['row_id']
    controls.append((f'fused-physics-m1-natural-s{seed}', fusion / 'predictions', fusion / 'scores'))
    for arm in ('contrast', 'joint'):
        packet = runs / PACKET_RUN / 'readouts' / f'source_rx_1-tri-packet-{arm}-s{seed}'
        controls.append((f'packet-{arm}-m1-natural-s{seed}', packet / 'predictions', packet / 'scores'))
    return controls
