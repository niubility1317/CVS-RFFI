"""Two fixed three-seed REPL partner-selection variants on Main split."""
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from .plan import SEEDS, continuation


RUN = '20261110-phase1-pamf-tri-rxpair-f0-m6-r01'
ARMS = {'tri-branch-pair': 0., 'tri-echo-pair': .5}


def candidate_rows():
    return [dict(row_id=f'{arm}-s{seed}', model_seed=seed, views=list(VIEWS), arm=arm)
            for seed in SEEDS for arm in ARMS]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only registered same-seed receiver pairing continuations are allowed')
    c = continuation(context, row, project, commit, run_id)
    c.update(algorithm_id=row['arm'], receiver_pairing=dict(beta=ARMS[row['arm']]))
    return c


def parent(c):
    from .source import original_parent

    seed, arm = c['model_seed'], c['algorithm_id']
    if (c['run_id'] != RUN or seed not in SEEDS or arm not in ARMS or c['row_id'] != f'{arm}-s{seed}'
            or c['receiver_pairing'] != dict(beta=ARMS[arm]) or 'physical_relation' in c):
        raise ValueError('Unregistered Tri-SIG-TIME receiver pairing continuation')
    return original_parent(c)
