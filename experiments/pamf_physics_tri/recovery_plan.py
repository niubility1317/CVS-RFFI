"""Paired three-seed tests of recovery with and without echo corroboration."""
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .plan import SEEDS, continuation


RUN = '20261110-phase1-pamf-tri-u-recovery-f0-m6-r01'
ARMS = {'tri-branch-recover': False, 'tri-echo-recover': True}


def recipe(arm):
    return dict(require_echo=ARMS[arm], minimum_confidence=.5, required_votes=4, weight=1.)


def candidate_rows():
    return [dict(row_id=f'{arm}-s{seed}', model_seed=seed, views=list(VIEWS), arm=arm)
            for seed in SEEDS for arm in ARMS]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only registered rejected-U recovery rows are allowed')
    c = continuation(context, row, project, commit, run_id)
    c.update(algorithm_id=row['arm'], rejected_u_recovery=recipe(row['arm']))
    return c


def parent(c):
    from .source import original_parent

    seed, arm = c['model_seed'], c['algorithm_id']
    if (c['run_id'] != RUN or seed not in SEEDS or arm not in ARMS or c['row_id'] != f'{arm}-s{seed}'
            or c['rejected_u_recovery'] != recipe(arm) or 'physical_relation' in c or 'receiver_pairing' in c):
        raise ValueError('Unregistered Tri-SIG-TIME rejected-U recovery continuation')
    return original_parent(c)
