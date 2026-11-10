"""Fixed F0 reference routing with matched three-seed evaluation."""
from copy import deepcopy
from experiments.pamf_physics_branch import plan as physical

RUN = '20261110-phase1-pamf-ptbr-route-f0-s2026092701-m1-r01'
REPLICATION_RUN = '20261110-phase1-pamf-ptbr-route-f0-s2026092702-03-m2-r01'


def candidate_rows(run_id=RUN):
    seeds = {RUN: (2026092701,), REPLICATION_RUN: (2026092702, 2026092703)}[run_id]
    rows = []
    for seed in seeds:
        row = deepcopy(physical.candidate_rows(seed)[1])
        row.pop('weak_views', None)
        row.update(row_id=f'ptbr-route-s{seed}', route_rule='reference_delay_stable_only')
        rows.append(row)
    return rows


def validate_candidate(row, run_id=RUN):
    if row not in candidate_rows(run_id):
        raise ValueError('Only the fixed reference-route candidate is permitted')
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    validate_candidate(row, run_id)
    seed = row['model_seed']
    c = physical.configuration(context, physical.candidate_rows(seed)[1], project, commit, seed=seed)
    c.update(run_id=run_id, row_id=row['row_id'], algorithm_id='pamf_ptbr_route_v1',
             physical_row=row, reference_route_rule=row['route_rule'],
             output_root=project + '/runs/' + run_id + '/F/' + row['row_id'] + '/source')
    return c
