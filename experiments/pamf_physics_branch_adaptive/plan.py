"""Fixed adaptive source-gradient-budget candidate on the original F0 contract."""
from copy import deepcopy
from experiments.pamf_physics_branch import plan as physical

RUN = '20261110-phase1-pamf-ptbr-adaptive-f0-s2026092701-m1-r01'


def candidate_rows():
    row = deepcopy(physical.candidate_rows()[1])
    row.update(row_id='ptbr-adaptive-s2026092701', adaptive_budget=dict(floor=.1, eps=1e-12))
    return [row]


def validate_candidate(row):
    if row not in candidate_rows():
        raise ValueError('Only fixed current-batch adaptive reference budget allowed')
    return row


def configuration(context, row, project, commit):
    validate_candidate(row)
    c = physical.configuration(context, physical.candidate_rows()[1], project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_ptbr_adaptive_v1',
             physical_row=row, adaptive_reference_budget=dict(floor=.1, eps=1e-12),
             output_root=project + '/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    return c
