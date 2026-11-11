"""Natural-order same-source E20 followed by fixed-order Tri SSL."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import single_rx_plan as base
from .plan import SEEDS

RUN = '20261111-phase1-pamf-tri-natural-warmup-m6-r01'
ARMS = ('day_physics', 'branch_certificate')


def candidate_rows():
    return [dict(row_id=f'{base.SETTING}-tri-curriculum-{arm}-s{seed}', setting=base.SETTING,
                 protocol_id=base.PROTOCOL, model_seed=seed, arm=arm, views=list(VIEWS))
            for arm in ARMS for seed in SEEDS]


def old_row(row):
    return next(r for r in base.candidate_rows() if r['arm'] == 'day_physics' and r['model_seed'] == row['model_seed'])


def proxy(c, row):
    original = old_row(row)
    root = Path(c['output_root'])
    return dict(c, run_id=base.RUN, row_id=original['row_id'], algorithm_id='pamf_tri_single_rx_day_v1',
                output_root=str(root.parents[3] / base.RUN / 'F' / original['row_id'] / 'source'))


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    wanted = 'branch' if row and row['arm'] == 'branch_certificate' else 'none'
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != base.PROTOCOL or c['relation_certificate'] != wanted
            or c['algorithm_id'] != 'pamf_tri_natural_warmup_v1'):
        raise ValueError('Unregistered source-only curriculum row')
    base.validate(proxy(c, row))
    return row


def natural_prefix(project, seed):
    name = f'{base.SETTING}-factorial-pl-full-ordered-s{seed}'
    entry = read(Path(project) / 'runs' / paper.RUN / 'sources.json')[name]
    completed = read(Path(entry['source_output']) / 'complete.json')
    if (completed['status'] != 'SOURCE_FROZEN' or completed['target_access'] or completed['U_truth_read']
            or completed['epoch'] != 50 or completed['updates'] != 11100 or len(completed['segments']) != 3):
        raise ValueError('Same-setting source-only ordinary-PL trajectory required')
    return Path(completed['segments'][0]['folder'])


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only the six preregistered curriculum rows are allowed')
    c = base.configuration(context, old_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_natural_warmup_v1',
             relation_certificate='branch' if row['arm'] == 'branch_certificate' else 'none',
             shared_prefix=str(natural_prefix(project, row['model_seed'])),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    from experiments.pamf_branch_training.source import checked_e20_parent
    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Unique curriculum output root required')
    expected = natural_prefix(root.parents[4], row['model_seed'])
    if Path(c['shared_prefix']) != expected:
        raise ValueError('Only same-setting, same-seed original natural-order PL E20 is permitted')
    evidence = checked_e20_parent(c)
    if evidence['config']['source_cache'] != c['source_cache'] or evidence['config']['source_rxs'] != [1]:
        raise ValueError('Natural-order parent data, physical source and roles must match')
    return evidence


def check_graph(model, c):
    return base.check_graph(model, c)


def extra_controls(rt, row):
    from .certified_day_plan import RUN as CERT_RUN
    runs = Path(rt['project']) / 'runs'
    seed = row['model_seed']
    day = runs / base.RUN / 'readouts' / old_row(row)['row_id']
    cert = runs / CERT_RUN / 'readouts' / f'{base.SETTING}-tri-certified-branch-s{seed}'
    return [(f'day-physics-{base.PROTOCOL}-s{seed}', day / 'predictions', day / 'scores'),
            (f'branch-certified-{base.PROTOCOL}-s{seed}', cert / 'predictions', cert / 'scores')]
