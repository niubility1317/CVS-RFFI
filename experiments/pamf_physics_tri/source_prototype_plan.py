"""Fixed source-day prototype structures, matched over three seeds."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import hybrid_contrast_plan as base
from .plan import SEEDS

RUN = '20261112-phase1-pamf-tri-source-prototype-m9-r01'
MODES = ('fusion', 'branch', 'fusion_branch')


def candidate_rows():
    return [dict(row_id=f'source_rx_1-tri-prototype-{mode}-s{seed}', setting='source_rx_1',
                 protocol_id='m1-natural', model_seed=seed, arm=mode, views=list(VIEWS))
            for mode in MODES for seed in SEEDS]


def old_row(row):
    return next(r for r in base.candidate_rows() if r['arm'] == 'fusion_branch' and r['model_seed'] == row['model_seed'])


def source_root(c):
    root = Path(c['output_root'])
    return root.parent if root.name in ('segment30', 'segment50') else root


def proxy(c, row):
    original = old_row(row)
    root = source_root(c)
    return dict(c, run_id=base.RUN, row_id=original['row_id'], algorithm_id='pamf_tri_hybrid_contrast_v1',
                output_root=str(root.parents[3] / base.RUN / 'F' / original['row_id'] / 'source'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != 'm1-natural' or c['algorithm_id'] != 'pamf_tri_source_prototype_v1'
            or c['source_prototype'] != dict(mode=row['arm'], temperature=.2, coefficient='original_joint')):
        raise ValueError('Unregistered source prototype method')
    if source_root(c).parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Dedicated source-prototype root required')
    base.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only nine registered source prototype rows allowed')
    c = base.configuration(context, old_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_source_prototype_v1',
             source_prototype=dict(mode=row['arm'], temperature=.2, coefficient='original_joint'),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    return base.parent(proxy(c, row))


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    seed = row['model_seed']
    controls = list(base.extra_controls(rt, old_row(row)))
    parent = Path(rt['project']) / 'runs' / base.RUN / 'readouts' / old_row(row)['row_id']
    controls.append((f'hybrid-fusion-branch-m1-natural-s{seed}', parent / 'predictions', parent / 'scores'))
    return controls
