from pathlib import Path
from experiments.pamf_tri_u_phase import plan as phase

RUN = '20261111-phase1-pamf-tri-u-channels-f0-m12-r01'
SEEDS = phase.SEEDS
ARMS = {
    'tri-u-cfo': dict(kind='cfo', delay=0, echo_amplitude=0.0),
    'tri-u-echo1': dict(kind='echo', delay=1, echo_amplitude=0.15),
    'tri-u-echo3': dict(kind='echo', delay=3, echo_amplitude=0.15),
    'tri-u-cfo-echo3': dict(kind='cfo_echo', delay=3, echo_amplitude=0.15),
}


def candidate_rows():
    return [dict(row_id=f'{arm}-s{seed}', model_seed=seed, views=list(r['views']), arm=arm)
            for seed in SEEDS for arm in ARMS for r in phase.candidate_rows() if r['model_seed'] == seed]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only registered source-U channel rows allowed')
    base = next(r for r in phase.candidate_rows() if r['model_seed'] == row['model_seed'])
    c = phase.configuration(context, base, str(project), commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id=row['arm'],
             channel_perturbation=dict(ARMS[row['arm']]),
             output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c
