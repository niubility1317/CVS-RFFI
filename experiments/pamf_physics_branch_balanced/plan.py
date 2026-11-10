"""One fixed reference-budget candidate and its matched F0 contract."""
from copy import deepcopy
from experiments.pamf_physics_branch import plan as physical

RUN = '20261110-phase1-pamf-ptbr-refbudget-f0-s2026092701-m1-r01'


def candidate_rows():
    row = deepcopy(physical.candidate_rows()[1])
    row.update(row_id='ptbr-refbudget-s2026092701', reference_scale=.1)
    return [row]


def validate_candidate(row):
    if row not in candidate_rows():
        raise ValueError('Only fixed PTBR reference-budget candidate allowed')
    return row


def configuration(context, row, project, commit):
    validate_candidate(row)
    c = physical.configuration(context, physical.candidate_rows()[1], project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_ptbr_refbudget_v1',
             physical_row=row, reference_scale=.1,
             output_root=project + '/runs/' + RUN + '/F/' + row['row_id'] + '/source')
    return c
