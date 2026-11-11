"""Prespecified one-source-RX three-seed cross-day candidate and physical ablation."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .plan import SEEDS

RUN = '20261111-phase1-pamf-tri-single-rx-day-m6-r01'
SETTING, PROTOCOL = 'source_rx_1', 'm1-natural'
RECIPES = {'day': dict(mode='branch', strength=0., schedule='taper'),
           'day_physics': dict(mode='branch', strength=.25, schedule='taper')}


def candidate_rows():
    return [dict(row_id=f'{SETTING}-tri-{arm}-s{seed}', setting=SETTING, protocol_id=PROTOCOL,
                 model_seed=seed, arm=arm, views=list(VIEWS)) for arm in RECIPES for seed in SEEDS]


def original_row(row):
    return f'{SETTING}-factorial-ssl-full-permuted-s{row["model_seed"]}'


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != PROTOCOL or c['source_rxs'] != [1] or c['source_days'] != [1, 2, 3]
            or c['practical_gate'] != RECIPES[row['arm']] or c['day_relation_weight'] != .025
            or c['algorithm_id'] != 'pamf_tri_single_rx_day_v1' or c['paper_row']['row_id'] != original_row(row)):
        raise ValueError('Unregistered single-RX source-day continuation')
    proxy = dict(c, run_id=paper.RUN, row_id=original_row(row))
    proxy.pop('shared_prefix', None)
    proxy.pop('parent_source', None)
    paper.validate_config(proxy)
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only the registered single-RX rows are allowed')
    source = Path(project) / 'runs' / paper.RUN / 'F' / original_row(row) / 'source'
    done = read(source / 'complete.json')
    if (done['status'] != 'SOURCE_FROZEN' or done['target_access'] or done['U_truth_read']
            or done['epoch'] != 50 or done['updates'] != 11100):
        raise ValueError('Completed exact original Tri source required')
    c = dict(done['config'])
    paper.validate_config(c)
    for key in ('parent_source', 'segment_epochs', 'epoch_offset', 'step_offset'):
        c.pop(key, None)
    c.update(run_id=RUN, row_id=row['row_id'], code_commit=commit,
             algorithm_id='pamf_tri_single_rx_day_v1', practical_gate=RECIPES[row['arm']],
             day_relation_weight=.025, shared_prefix=str(source / 'segment20'),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    if read(c['source_contract']) != read(context['handles'][PROTOCOL]['source_contract']):
        raise ValueError('Exact source data contract required')
    validate(c)
    return c


def parent(c):
    from experiments.pamf_branch_training.source import checked_e20_parent
    row = validate(c)
    root = Path(c['output_root'])
    expected = root.parents[3] / paper.RUN / 'F' / original_row(row) / 'source' / 'segment20'
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source') or Path(c['shared_prefix']) != expected:
        raise ValueError('Same-setting same-seed Tri E20 and dedicated output root required')
    result = checked_e20_parent(c)
    original = result['config']
    if (original['row_id'] != original_row(row) or original['run_id'] != paper.RUN
            or original['paper_row'] != c['paper_row'] or original['protocol_id'] != PROTOCOL
            or any(original.get(k) != c.get(k) for k in paper.SSL_KEYS)):
        raise ValueError('Original Tri graph, source, and complete recipe must match')
    return result


def check_graph(model, c):
    from experiments.pamf_repl_view_shuffle.source import assert_structure
    return assert_structure(model, dict(c, view_comparison={'view_count': 3}))
