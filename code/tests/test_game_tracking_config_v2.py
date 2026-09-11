import pytest
from cvsrffi.game_tracking.config import parse_args


@pytest.mark.parametrize('options',[
    ['--game_control','catchup'],
    ['--game_control','correction'],
    ['--game_control','both'],
    ['--game_control','replay','--game_actions_replay','source_schedule.jsonl'],
    ['--game_curriculum','capability'],
    ['--game_response_tracking'],
    ['--game_jacobian_interval','500'],
])
def test_no_audit_cannot_silently_disable_requested_v2_mechanism(options):
    with pytest.raises(ValueError,match='requires source audits'):
        parse_args(['--output_dir','unused','--game_no_audit']+options)


def test_no_audit_keeps_independent_b8_solver_available():
    args=parse_args(['--output_dir','unused','--game_no_audit',
                    '--game_solver','head_lookahead','--game_b8_impl','head_grad_only'])
    assert args.game_no_audit and args.game_solver=='head_lookahead'


def test_negative_extra_head_budget_is_invalid_at_parse_time():
    with pytest.raises(ValueError,match='budget'):
        parse_args(['--output_dir','unused','--game_max_extra_head','-1'])
