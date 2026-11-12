"""Fixed reduced Hybrid auxiliary weights with all confident U retained."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import full_matrix_plan as original

RUN = '20261113-phase1-pamf-hybrid-confidence-weight-m24-r01'
SEEDS = (2026092701, 2026092702, 2026092703)


def candidate_rows():
    return [dict(row_id=f'{setting}-hybrid-confidence-weight{tag}-s{seed}', setting=setting, protocol_id=pid,
                 model_seed=seed, auxiliary_weight=weight, views=list(VIEWS))
            for setting, pid in (('receiver_split_2', 'F2_rx_repartition_r01'), ('ten_class', 'ManyRx'),
                                 ('labeled_1pct', 'label01'), ('source_rx_6', 'm6-natural'))
            for tag, weight in (('025', .25), ('050', .5)) for seed in SEEDS]


def baseline_row(row):
    return next(r for r in original.candidate_rows() if r['setting'] == row['setting']
                and r['model_seed'] == row['model_seed'] and r['arm'] == 'hybrid')


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
    if row is None or c.get('hybrid_weight_row') != row or c.get('run_id') != RUN or c.get('source_development'):
        raise ValueError('Only fixed complete-source Hybrid weight rows are allowed')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Independent output root required')
    if (c.get('practical_gate') != dict(mode='all', strength=.25, schedule='taper', all_confident=True)
            or c.get('hybrid_auxiliary_weight') != row['auxiliary_weight'] or c.get('source_prototype') is not None):
        raise ValueError('All accepted U and the declared Hybrid auxiliary weight are required')
    original.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if run_id != RUN or row not in candidate_rows():
        raise ValueError('Unregistered Hybrid weight row')
    c = original.configuration(context, baseline_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], hybrid_weight_row=dict(row),
             algorithm_id='pamf_tri_hybrid_confidence_weight_v1', hybrid_auxiliary_weight=row['auxiliary_weight'],
             practical_gate=dict(mode='all', strength=.25, schedule='taper', all_confident=True),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    return original.parent(proxy(c, validate(c)))


check_graph = original.check_graph
