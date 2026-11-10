"""Validate the fixed source-only candidate without changing its PTBR parent."""
from experiments.pamf_physics_branch import source as physical
from experiments.pamf_physics_branch import plan as physical_plan
from .plan import RUN, candidate_rows, validate_candidate

_physical_validate = physical.validate_source


def validate_source(c):
    row = validate_candidate(c['physical_row'])
    suffix = '/runs/' + RUN + '/F/' + row['row_id'] + '/source'
    if (c['run_id'] != RUN or c['row_id'] != row['row_id'] or c.get('reference_scale') != .1
            or c.get('algorithm_id') != 'pamf_ptbr_refbudget_v1' or not c['output_root'].endswith(suffix)):
        raise ValueError('Fixed reference budget or run ownership differs')
    proxy = dict(c, run_id=physical_plan.RUN, row_id=physical_plan.candidate_rows()[1]['row_id'],
                 physical_row=physical_plan.candidate_rows()[1])
    proxy.pop('reference_scale')
    _physical_validate(proxy)
    return row
