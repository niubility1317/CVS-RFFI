"""Same-packet physical contrast, with and without cosine consistency."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import fused_physics_plan as base
from .plan import SEEDS

RUN = '20261111-phase1-pamf-tri-packet-contrast-m6-r01'
RECIPES = {'contrast': dict(cosine=False, temperature=.2), 'joint': dict(cosine=True, temperature=.2)}


def candidate_rows():
    return [dict(row_id=f'source_rx_1-tri-packet-{arm}-s{seed}', setting='source_rx_1',
                 protocol_id='m1-natural', model_seed=seed, arm=arm, views=list(VIEWS))
            for arm in RECIPES for seed in SEEDS]


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
            or c['protocol_id'] != 'm1-natural' or c['algorithm_id'] != 'pamf_tri_packet_contrast_v1'
            or c['packet_contrast'] != RECIPES[row['arm']]):
        raise ValueError('Unregistered same-packet physical contrast')
    base.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only six registered packet-contrast rows allowed')
    c = base.configuration(context, old_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_packet_contrast_v1',
             packet_contrast=RECIPES[row['arm']],
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Dedicated packet-contrast output root required')
    return base.parent(proxy(c, row))


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    runs = Path(rt['project']) / 'runs'
    entries = list(base.extra_controls(rt, old_row(row)))
    fusion = runs / base.RUN / 'readouts' / old_row(row)['row_id']
    entries.append((f'fused-physics-m1-natural-s{row["model_seed"]}', fusion / 'predictions', fusion / 'scores'))
    return entries
