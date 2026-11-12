"""Both fixed multibranch SSL methods on every remaining paper data setting."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from experiments.pamf_single_view_paper.plan import PROTOCOLS
from .plan import SEEDS, continuation

RUN = '20261112-phase1-pamf-tri-two-methods-full-m60-r01'
METHODS = ('hybrid', 'branch_prototype')
SETTINGS = {name: value[0] for name, value in PROTOCOLS.items() if name != 'source_rx_1'}


def candidate_rows():
    return [dict(row_id=f'{setting}-tri-{method}-s{seed}', setting=setting, protocol_id=pid,
                 model_seed=seed, arm=method, views=list(VIEWS))
            for setting, pid in SETTINGS.items() for method in METHODS for seed in SEEDS]


def original_row(row):
    return f'{row["setting"]}-factorial-ssl-full-permuted-s{row["model_seed"]}'


def source_root(c):
    root = Path(c['output_root'])
    return root.parent if root.name in ('segment30', 'segment50') else root


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c.get('row_id')), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != row['protocol_id'] or c['algorithm_id'] != 'pamf_tri_two_method_matrix_v1'
            or c['paper_row']['row_id'] != original_row(row)
            or c['practical_gate'] != dict(mode='branch', strength=.25, schedule='taper')
            or c['day_relation_weight'] != .025 or c['hybrid_contrast'] != dict(mode='fusion_branch', temperature=.2)
            or c['fused_physics'] != dict(weight='original_joint', raw_detached=True)):
        raise ValueError('Registered identical physical fusion-and-branch method required')
    expected = dict(mode='branch', temperature=.2, coefficient='original_joint') if row['arm'] == 'branch_prototype' else None
    if c.get('source_prototype') != expected:
        raise ValueError('Only the registered optional five-branch source anchor is allowed')
    root = source_root(c)
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Dedicated matrix source root required')
    if row['protocol_id'] != 'F0':
        proxy = dict(c, run_id=paper.RUN, row_id=original_row(row))
        proxy.pop('shared_prefix', None)
        proxy.pop('parent_source', None)
        paper.validate_config(proxy)
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only 60 registered remaining data/method/seed rows are allowed')
    if row['protocol_id'] == 'F0':
        c = continuation(context, row, project, commit, RUN)
    else:
        source = Path(project) / 'runs' / paper.RUN / 'F' / original_row(row) / 'source'
        done = read(source / 'complete.json')
        if (done['status'] != 'SOURCE_FROZEN' or done['target_access'] or done['U_truth_read']
                or done['epoch'] != 50 or done['updates'] != 11100):
            raise ValueError('Exact original same-contract Tri source must be complete')
        c = dict(done['config'])
        paper.validate_config(c)
        for key in ('parent_source', 'segment_epochs', 'epoch_offset', 'step_offset'):
            c.pop(key, None)
        c['shared_prefix'] = str(source / 'segment20')
    c.update(run_id=RUN, row_id=row['row_id'], code_commit=commit,
             algorithm_id='pamf_tri_two_method_matrix_v1',
             practical_gate=dict(mode='branch', strength=.25, schedule='taper'), day_relation_weight=.025,
             fused_physics=dict(weight='original_joint', raw_detached=True),
             hybrid_contrast=dict(mode='fusion_branch', temperature=.2),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    if row['arm'] == 'branch_prototype':
        c['source_prototype'] = dict(mode='branch', temperature=.2, coefficient='original_joint')
    if read(c['source_contract']) != read(context['handles'][row['protocol_id']]['source_contract']):
        raise ValueError('Exact current source data roles and physical IDs required')
    validate(c)
    return c


def parent(c):
    from experiments.pamf_branch_training.source import checked_e20_parent
    from .source import original_parent
    row = validate(c)
    source = source_root(c)
    normalized = dict(c, output_root=str(source))
    if row['protocol_id'] == 'F0':
        return original_parent(normalized)
    expected = source.parents[3] / paper.RUN / 'F' / original_row(row) / 'source' / 'segment20'
    if Path(c['shared_prefix']) != expected:
        raise ValueError('Matching original Tri E20 parent required')
    evidence = checked_e20_parent(normalized)
    original = evidence['config']
    if (original['row_id'] != original_row(row) or original['run_id'] != paper.RUN
            or original['paper_row'] != c['paper_row'] or original['protocol_id'] != row['protocol_id']
            or any(original.get(key) != c.get(key) for key in paper.SSL_KEYS)):
        raise ValueError('Same-setting and seed Tri graph, source and complete REPL required')
    return evidence


def check_graph(model, c):
    ref = model.id_backbone.encoder
    pool = ref.sequence_pool
    count = sum(p.numel() for p in ref.parameters())
    expected = 232993 + 160 * (c['num_classes'] - 6)
    order = [view.window_order.tolist() for view in pool.views]
    if (count != expected or len(pool.views) != 3 or tuple(pool.output_projection.weight.shape) != (160, 336)
            or any(value != [0, 5, 3, 6, 2, 1, 4, 7] for value in order)):
        raise ValueError('Exact full three-view Tri classifier graph required')
    return dict(view_count=3, parameters=count, order=order[0], projection=[160, 336])
