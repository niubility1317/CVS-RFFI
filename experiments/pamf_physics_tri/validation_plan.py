"""Freeze the F0-discovered taper across two RX splits and a label fraction."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .plan import SEEDS


RUN = '20261111-phase1-pamf-tri-taper-validation-m9-r01'
SETTINGS = {'receiver_split_1': 'F1_rx_repartition_r01',
            'receiver_split_2': 'F2_rx_repartition_r01', 'labeled_5pct': 'label05'}
RECIPE = dict(mode='branch', strength=.25, schedule='taper')


def candidate_rows():
    return [dict(row_id=f'{setting}-tri-taper-s{seed}', setting=setting, protocol_id=pid,
                 model_seed=seed, views=list(VIEWS))
            for setting, pid in SETTINGS.items() for seed in SEEDS]


def original_row(row):
    return f'{row["setting"]}-factorial-ssl-full-permuted-s{row["model_seed"]}'


def validate(c):
    if c['run_id'] != RUN:
        raise ValueError('Validation run identity differs')
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['model_seed'] != row['model_seed'] or c['protocol_id'] != row['protocol_id']
            or c['practical_gate'] != RECIPE or c['paper_row']['row_id'] != original_row(row)
            or c['algorithm_id'] != 'pamf_tri_taper_validation_v1'):
        raise ValueError('Unregistered frozen taper validation row')
    original = original_row(row)
    proxy = dict(c, run_id=paper.RUN, row_id=original)
    proxy.pop('shared_prefix', None)
    proxy.pop('parent_source', None)
    paper.validate_config(proxy)
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only preregistered settings and seeds allowed')
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
             algorithm_id='pamf_tri_taper_validation_v1', practical_gate=RECIPE,
             shared_prefix=str(source / 'segment20'),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    if read(c['source_contract']) != read(context['handles'][row['protocol_id']]['source_contract']):
        raise ValueError('Exact source L/U/physical data contract must match')
    validate(c)
    return c


def parent(c):
    from experiments.pamf_branch_training.source import checked_e20_parent

    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Validation source output root differs')
    expected = root.parents[3] / paper.RUN / 'F' / original_row(row) / 'source' / 'segment20'
    if Path(c['shared_prefix']) != expected:
        raise ValueError('Only the same-setting same-seed original Tri E20 is permitted')
    result = checked_e20_parent(c)
    original = result['config']
    if (original['row_id'] != original_row(row) or original['run_id'] != paper.RUN
            or original['paper_row'] != c['paper_row'] or original['protocol_id'] != c['protocol_id']
            or any(original.get(k) != c.get(k) for k in paper.SSL_KEYS)):
        raise ValueError('Original Tri graph, source and complete REPL recipe must match')
    return result


def check_graph(model, c):
    from experiments.pamf_repl_view_shuffle.source import assert_structure

    if c['paper_row']['signature_views'] != 3 or not c['paper_row']['permuted']:
        raise ValueError('Original three-view permuted paper graph required')
    return assert_structure(model, dict(c, view_comparison={'view_count': 3}))
