"""Fixed same-seed no-branch-vote comparison across three source settings."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import full_matrix_plan as original

RUN = '20261113-phase1-pamf-confidence-only-m9-r01'
SEEDS = (2026092701, 2026092702, 2026092703)


def candidate_rows():
    return [dict(row_id=f'{setting}-confidence-only-s{seed}', setting=setting, protocol_id=pid,
                 model_seed=seed, views=list(VIEWS))
            for setting, pid in (('ten_class', 'ManyRx'), ('labeled_1pct', 'label01'),
                                 ('source_rx_6', 'm6-natural')) for seed in SEEDS]


def baseline_row(row):
    return next(r for r in original.candidate_rows() if r['setting'] == row['setting']
                and r['model_seed'] == row['model_seed'] and r['arm'] == 'branch_prototype')


def proxy(c, row):
    old = baseline_row(row)
    root = Path(c['output_root'])
    segment = root.name if root.name in ('segment30', 'segment50') else None
    source = root.parent if segment else root
    old_root = source.parents[3] / original.RUN / 'F' / old['row_id'] / 'source'
    if segment:
        old_root /= segment
    return dict(c, run_id=original.RUN, row_id=old['row_id'], algorithm_id='pamf_tri_two_method_matrix_v1',
                output_root=str(old_root), practical_gate=dict(mode='branch', strength=.25, schedule='taper'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c.get('confidence_row') != row or c.get('run_id') != RUN or c.get('source_development'):
        raise ValueError('Only registered full-source confidence-only rows are allowed')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Independent output root required')
    if c.get('practical_gate') != dict(mode='all', strength=.25, schedule='taper', all_confident=True):
        raise ValueError('All confidence-accepted U samples must be routed without branch votes')
    original.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if run_id != RUN or row not in candidate_rows():
        raise ValueError('Unregistered confidence-only row')
    c = original.configuration(context, baseline_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], confidence_row=dict(row),
             algorithm_id='pamf_tri_confidence_only_v1',
             practical_gate=dict(mode='all', strength=.25, schedule='taper', all_confident=True),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    return original.parent(proxy(c, validate(c)))


check_graph = original.check_graph
