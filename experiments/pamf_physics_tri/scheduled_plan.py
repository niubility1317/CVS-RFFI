"""Three fixed time schedules for the same weak branch-gated source U view."""
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .plan import SEEDS, continuation


RUN = '20261111-phase1-pamf-tri-weak-schedule-f0-m9-r01'
ARMS = {'tri-weak-early': 'early', 'tri-weak-late': 'late', 'tri-weak-taper': 'taper'}


def recipe(arm):
    return dict(mode='branch', strength=.25, schedule=ARMS[arm])


def candidate_rows():
    return [dict(row_id=f'{arm}-s{seed}', model_seed=seed, views=list(VIEWS), arm=arm)
            for seed in SEEDS for arm in ARMS]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only registered weak-view schedules are allowed')
    c = continuation(context, row, project, commit, run_id)
    c.update(algorithm_id=row['arm'], practical_gate=recipe(row['arm']))
    return c


def parent(c):
    from .source import original_parent

    seed, arm = c['model_seed'], c['algorithm_id']
    if (c['run_id'] != RUN or seed not in SEEDS or arm not in ARMS or c['row_id'] != f'{arm}-s{seed}'
            or c['practical_gate'] != recipe(arm)
            or any(k in c for k in ('physical_relation', 'receiver_pairing', 'rejected_u_recovery'))):
        raise ValueError('Unregistered weak-view schedule continuation')
    return original_parent(c)
