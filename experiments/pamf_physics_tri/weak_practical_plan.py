"""Matched weak source-channel U dose with random and branch selection."""
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .plan import SEEDS, continuation


RUN = '20261111-phase1-pamf-tri-weak-practical-f0-m9-r01'
ARMS = {'tri-weak-all': 'all', 'tri-weak-random': 'random', 'tri-weak-branch': 'branch'}


def candidate_rows():
    return [dict(row_id=f'{arm}-s{seed}', model_seed=seed, views=list(VIEWS), arm=arm)
            for seed in SEEDS for arm in ARMS]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only registered weak practical-U rows are allowed')
    c = continuation(context, row, project, commit, run_id)
    c.update(algorithm_id=row['arm'], practical_gate=dict(mode=ARMS[row['arm']], strength=.25))
    return c


def parent(c):
    from .source import original_parent

    seed, arm = c['model_seed'], c['algorithm_id']
    if (c['run_id'] != RUN or seed not in SEEDS or arm not in ARMS or c['row_id'] != f'{arm}-s{seed}'
            or c['practical_gate'] != dict(mode=ARMS[arm], strength=.25)
            or any(k in c for k in ('physical_relation', 'receiver_pairing', 'rejected_u_recovery'))):
        raise ValueError('Unregistered weak practical-U continuation')
    return original_parent(c)
