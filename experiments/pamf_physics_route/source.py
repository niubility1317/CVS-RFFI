"""Keep the original same-contract, no-target E20 parent checks."""
from experiments.pamf_physics_branch import source as physical
from experiments.pamf_physics_branch import plan as original
from .plan import RUN, validate_candidate

_validate = physical.validate_source


def validate_source(c):
    row = validate_candidate(c['physical_row'])
    suffix = '/runs/' + RUN + '/F/' + row['row_id'] + '/source'
    if (c['run_id'] != RUN or c['row_id'] != row['row_id']
            or c.get('algorithm_id') != 'pamf_ptbr_route_v1'
            or c.get('reference_route_rule') != 'reference_delay_stable_only'
            or not c['output_root'].endswith(suffix)):
        raise ValueError('Reference route, source contract or run ownership differs')
    proxy = dict(c, run_id=original.RUN, row_id=original.candidate_rows()[1]['row_id'],
                 physical_row=original.candidate_rows()[1])
    _validate(proxy)
    return row
