"""Three fixed same-source PL-teacher preservation rules."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import hybrid_contrast_plan as base
from .plan import SEEDS

RUN = '20261112-phase1-pamf-tri-teacher-anchor-m9-r01'
MODES = ('agree', 'teacher', 'teacher_physical')


def candidate_rows():
    return [dict(row_id=f'source_rx_1-tri-teacher-{mode}-s{seed}', setting='source_rx_1',
                 protocol_id='m1-natural', model_seed=seed, arm=mode, views=list(VIEWS))
            for mode in MODES for seed in SEEDS]


def old_row(row):
    return next(r for r in base.candidate_rows() if r['arm'] == 'fusion_branch' and r['model_seed'] == row['model_seed'])


def teacher_source(project, seed):
    row = f'source_rx_1-factorial-pl-full-ordered-s{seed}'
    entry = read(Path(project) / 'runs' / paper.RUN / 'sources.json')[row]
    source = Path(entry['source_output'])
    done = read(source / 'complete.json')
    if (done['status'] != 'SOURCE_FROZEN' or done['target_access'] or done['U_truth_read']
            or done['epoch'] != 50 or done['updates'] != 11100 or done['config']['model_seed'] != seed
            or done['config']['source_rxs'] != [1] or done['config']['source_days'] != [1, 2, 3]):
        raise ValueError('Matched source-only fixed-final ordinary PL teacher required')
    return source


def proxy(c, row):
    original = old_row(row)
    root = Path(c['output_root'])
    return dict(c, run_id=base.RUN, row_id=original['row_id'], algorithm_id='pamf_tri_hybrid_contrast_v1',
                output_root=str(root.parents[3] / base.RUN / 'F' / original['row_id'] / 'source'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != 'm1-natural' or c['algorithm_id'] != 'pamf_tri_teacher_anchor_v1'
            or c['teacher_anchor'] != dict(mode=row['arm'], threshold=.95, coefficient='original_joint')):
        raise ValueError('Unregistered source-only PL teacher anchor')
    root = Path(c['output_root'])
    expected_teacher = teacher_source(root.parents[4], row['model_seed'])
    if Path(c['teacher_source']) != expected_teacher:
        raise ValueError('Teacher must be the same-setting, same-seed source-only PL')
    teacher_cfg = read(expected_teacher / 'complete.json')['config']
    if (teacher_cfg['source_cache'] != c['source_cache']
            or read(teacher_cfg['source_contract']) != read(c['source_contract'])):
        raise ValueError('Teacher and student source data roles and physical identities must match')
    base.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only nine registered teacher preservation rows allowed')
    c = base.configuration(context, old_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_teacher_anchor_v1',
             teacher_source=str(teacher_source(project, row['model_seed'])),
             teacher_anchor=dict(mode=row['arm'], threshold=.95, coefficient='original_joint'),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Dedicated teacher-anchor output root required')
    return base.parent(proxy(c, row))


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    runs = Path(rt['project']) / 'runs'
    seed = row['model_seed']
    controls = list(base.extra_controls(rt, old_row(row)))
    parent = runs / base.RUN / 'readouts' / old_row(row)['row_id']
    controls.append((f'hybrid-fusion-branch-m1-natural-s{seed}', parent / 'predictions', parent / 'scores'))
    return controls
