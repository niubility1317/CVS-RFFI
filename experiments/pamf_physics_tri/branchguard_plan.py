"""Evaluate one branch-specific guard over four matched source settings."""
from pathlib import Path

from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import validation_plan
from .plan import SEEDS, continuation


RUN = '20261111-phase1-pamf-tri-branchguard-m12-r01'
SETTINGS = {'main_split': 'F0', **validation_plan.SETTINGS}
RECIPE = dict(mode='branch', strength=.25, schedule='taper')


def candidate_rows():
    return [dict(row_id=f'{setting}-tri-branchguard-s{seed}', setting=setting, protocol_id=pid,
                 model_seed=seed, views=list(VIEWS))
            for setting, pid in SETTINGS.items() for seed in SEEDS]


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c['row_id']), None)
    if (row is None or c['run_id'] != RUN or c['model_seed'] != row['model_seed']
            or c['protocol_id'] != row['protocol_id'] or c['practical_gate'] != RECIPE
            or c['algorithm_id'] != 'pamf_tri_taper_branchguard_v1'):
        raise ValueError('Unregistered branch-gradient guard row')
    if row['protocol_id'] != 'F0':
        proxy = dict(c, run_id=validation_plan.RUN, row_id=f'{row["setting"]}-tri-taper-s{row["model_seed"]}',
                     algorithm_id='pamf_tri_taper_validation_v1')
        validation_plan.validate(proxy)
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only registered branch guard settings are allowed')
    if row['protocol_id'] == 'F0':
        c = continuation(context, row, project, commit, run_id)
    else:
        original = next(r for r in validation_plan.candidate_rows()
                        if r['setting'] == row['setting'] and r['model_seed'] == row['model_seed'])
        c = validation_plan.configuration(context, original, project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_taper_branchguard_v1',
             practical_gate=RECIPE, output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def parent(c):
    row = validate(c)
    root = Path(c['output_root'])
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Branch guard source root differs')
    if row['protocol_id'] == 'F0':
        from .source import original_parent
        return original_parent(c)
    proxy_row = f'{row["setting"]}-tri-taper-s{row["model_seed"]}'
    proxy = dict(c, run_id=validation_plan.RUN, row_id=proxy_row, algorithm_id='pamf_tri_taper_validation_v1',
                 output_root=str(root.parents[3] / validation_plan.RUN / 'F' / proxy_row / 'source'))
    return validation_plan.parent(proxy)


def check_graph(model, c):
    if c['protocol_id'] == 'F0':
        from experiments.pamf_repl_view_shuffle.source import assert_structure
        return assert_structure(model, c)
    return validation_plan.check_graph(model, c)


def extra_controls(rt, row):
    from .scheduled_plan import RUN as F0_RUN
    setting, seed, pid = row['setting'], row['model_seed'], row['protocol_id']
    run, candidate = ((F0_RUN, f'tri-weak-taper-s{seed}') if pid == 'F0' else
                      (validation_plan.RUN, f'{setting}-tri-taper-s{seed}'))
    root = Path(rt['project']) / 'runs' / run / 'readouts' / candidate
    return [(f'taper-control-{pid}-s{seed}', root / 'predictions', root / 'scores')]
