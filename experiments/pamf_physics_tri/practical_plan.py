"""Three paired seeds for source practical U augmentation with and without gating."""
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .plan import SEEDS, continuation


RUN = '20261111-phase1-pamf-tri-practical-gate-f0-m6-r01'
ARMS = {'tri-practical-u': False, 'tri-branch-practical-u': True}


def candidate_rows():
    return [dict(row_id=f'{arm}-s{seed}', model_seed=seed, views=list(VIEWS), arm=arm)
            for seed in SEEDS for arm in ARMS]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only registered practical-U rows are allowed')
    c = continuation(context, row, project, commit, run_id)
    c.update(algorithm_id=row['arm'], practical_gate=dict(branch_gate=ARMS[row['arm']]))
    return c


def parent(c):
    from .source import original_parent

    seed, arm = c['model_seed'], c['algorithm_id']
    if (c['run_id'] != RUN or seed not in SEEDS or arm not in ARMS or c['row_id'] != f'{arm}-s{seed}'
            or c['practical_gate'] != dict(branch_gate=ARMS[arm])
            or any(k in c for k in ('physical_relation', 'receiver_pairing', 'rejected_u_recovery'))):
        raise ValueError('Unregistered Tri-SIG-TIME practical-U continuation')
    return original_parent(c)
