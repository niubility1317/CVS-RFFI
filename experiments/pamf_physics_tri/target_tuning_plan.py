"""Fixed three-seed full-source comparison of new prototype parameters."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import full_matrix_plan as original

RUN = '20261112-phase1-pamf-prototype-fullsource-tuning-m33-r01'
SEEDS = (2026092701, 2026092702, 2026092703)


def candidate_rows():
    variations = (
        ('ten_class', 'ManyRx', 'temp010', .1, None, None),
        ('ten_class', 'ManyRx', 'temp040', .4, None, None),
        ('ten_class', 'ManyRx', 'gate3', .2, 3, None),
        ('ten_class', 'ManyRx', 'weight050', .2, None, .5),
        ('ten_class', 'ManyRx', 'weight200', .2, None, 2.),
        ('labeled_1pct', 'label01', 'temp010', .1, None, None),
        ('labeled_1pct', 'label01', 'temp040', .4, None, None),
        ('source_rx_6', 'm6-natural', 'temp005', .05, None, None),
        ('source_rx_6', 'm6-natural', 'temp010', .1, None, None),
        ('source_rx_6', 'm6-natural', 'temp040', .4, None, None),
        ('source_rx_6', 'm6-natural', 'temp080', .8, None, None),
    )
    return [dict(row_id=f'{setting}-prototype-{name}-s{seed}', setting=setting, protocol_id=pid,
                 model_seed=seed, temperature=temp, required_votes=votes, prototype_weight=weight,
                 views=list(VIEWS))
            for setting, pid, name, temp, votes, weight in variations for seed in SEEDS]


def baseline_row(row):
    return next(r for r in original.candidate_rows()
                if r['setting'] == row['setting'] and r['model_seed'] == row['model_seed']
                and r['arm'] == 'branch_prototype')


def proxy(c, row):
    old = baseline_row(row)
    root = Path(c['output_root'])
    segment = root.name if root.name in ('segment30', 'segment50') else None
    source = root.parent if segment else root
    old_root = source.parents[3] / original.RUN / 'F' / old['row_id'] / 'source'
    if segment:
        old_root = old_root / segment
    value = dict(c, run_id=original.RUN, row_id=old['row_id'], algorithm_id='pamf_tri_two_method_matrix_v1',
                 output_root=str(old_root), practical_gate=dict(mode='branch', strength=.25, schedule='taper'),
                 source_prototype=dict(mode='branch', temperature=.2, coefficient='original_joint'))
    return value


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c.get('target_tuning_row') != row or c.get('run_id') != RUN:
        raise ValueError('Only the fixed full-source parameter matrix is allowed')
    if c.get('source_development'):
        raise ValueError('Full original source data required')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Independent output root required')
    gate = dict(mode='branch', strength=.25, schedule='taper')
    if row['required_votes'] is not None:
        gate['required_votes'] = row['required_votes']
    prototype = dict(mode='branch', temperature=row['temperature'], coefficient='original_joint')
    if row['prototype_weight'] is not None:
        prototype['weight'] = row['prototype_weight']
    if c.get('practical_gate') != gate or c.get('source_prototype') != prototype:
        raise ValueError('Actual candidate parameters differ from the fixed row')
    original.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if run_id != RUN or row not in candidate_rows():
        raise ValueError('Unregistered full-source parameter row')
    c = original.configuration(context, baseline_row(row), project, commit)
    gate = dict(mode='branch', strength=.25, schedule='taper')
    if row['required_votes'] is not None:
        gate['required_votes'] = row['required_votes']
    prototype = dict(mode='branch', temperature=row['temperature'], coefficient='original_joint')
    if row['prototype_weight'] is not None:
        prototype['weight'] = row['prototype_weight']
    c.update(run_id=RUN, row_id=row['row_id'], target_tuning_row=dict(row),
             algorithm_id='pamf_tri_fullsource_tuning_v1', practical_gate=gate, source_prototype=prototype,
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    return original.parent(proxy(c, row))


check_graph = original.check_graph
