"""Keep the original same-contract, no-target E20 parent checks."""
from experiments.pamf_branch_training.source import checked_e20_parent
from experiments.pamf_physics_branch.source import validate_source_config
from .plan import validate_candidate


def validate_source(c):
    run_id = c['run_id']
    row = validate_candidate(c['physical_row'], run_id)
    suffix = '/runs/' + run_id + '/F/' + row['row_id'] + '/source'
    if (c.get('algorithm_id') != 'pamf_ptbr_route_v1'
            or c.get('reference_route_rule') != 'reference_delay_stable_only'
            or not c['output_root'].endswith(suffix)):
        raise ValueError('Reference route, source contract or run ownership differs')
    return validate_source_config(c, row, run_id)


def checked_parent(c):
    validate_source(c)
    return checked_e20_parent(c)
