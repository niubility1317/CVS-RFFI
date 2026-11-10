"""One F0 seed, preserving the full old PAMF and ordinary PL."""
from copy import deepcopy
from experiments.pamf_physics_branch import plan as physical

RUN = '20261110-phase1-pamf-ptbr-route-f0-s2026092701-m1-r01'


def candidate_rows():
    row = deepcopy(physical.candidate_rows()[1])
    row.pop('weak_views', None)
    row.update(row_id='ptbr-route-s2026092701', route_rule='reference_delay_stable_only')
    return [row]


def validate_candidate(row):
    if row not in candidate_rows():
        raise ValueError('Only the fixed reference-route candidate is permitted')
    return row


def configuration(context, row, project, commit):
    validate_candidate(row)
    c = physical.configuration(context, physical.candidate_rows()[1], project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_ptbr_route_v1',
             physical_row=row, reference_route_rule=row['route_rule'],
             output_root=project + '/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    return c
