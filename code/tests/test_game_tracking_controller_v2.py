from copy import deepcopy
from cvsrffi.game_tracking import controller


def evidence(step,quality=True):
    return dict(schema='game_audit_v2',observation_id=f'obs{step}',step=step,encoder_version=step,
                data_valid=True,coverage_valid=True,
                lag=dict(status='RELIABLE_LOW_GAP' if quality else 'OPTIMIZATION_FAILURE',
                         quality_pass=quality,control_ready=quality,gap_normalized=0. if quality else None,
                         readability=.5),
                gradient=dict(valid=True,representative=True,scope='full_current_training_objective',direction_imbalance=.9),
                capability=dict(schema='game_capability_v2',valid=True,identity_valid=True,identity=.95,margin=.4,collapsed=False,
                                step=step,encoder_version=step))


def test_failed_probe_cannot_correct_or_reduce_cadence():
    assert hasattr(controller,'GameControllerV2')
    c=controller.GameControllerV2(controller.ControllerConfig(confirmation_windows=1,audit_interval=250,sparse_audit_interval=500,max_version_lag=10))
    m=evidence(0,False)
    for step in range(11):
        d=c.decide(m,step=step,encoder_version=step)
        assert d['action']=='NORMAL' and d['next_audit_interval']==250
        assert d['reason']=='untrusted_lag_evidence'
        assert not d['hold_curriculum']


def test_single_observation_one_action_survives_resume():
    assert hasattr(controller,'GameControllerV2')
    c=controller.GameControllerV2(controller.ControllerConfig(confirmation_windows=1))
    m=evidence(0)
    d=c.decide(m,step=0,encoder_version=0)
    assert d['action']=='CORRECT'
    c.record_outcome(m['observation_id'],accepted=True,committed_head_steps=0)
    other=controller.GameControllerV2(c.config);other.load_state_dict(deepcopy(c.state_dict()))
    for _ in range(100):
        assert other.decide(m,step=0,encoder_version=0)['action']=='NORMAL'


def test_future_stale_or_clean_subset_gradients_do_not_correct():
    assert hasattr(controller,'GameControllerV2')
    for kwargs in [dict(step=-1,encoder_version=-1),dict(step=20,encoder_version=20)]:
        c=controller.GameControllerV2(controller.ControllerConfig(confirmation_windows=1))
        assert c.decide(evidence(0),**kwargs)['action']=='NORMAL'
    c=controller.GameControllerV2(controller.ControllerConfig(confirmation_windows=1))
    m=evidence(0);m['gradient']['scope']='labeled_clean_subset'
    assert c.decide(m,step=0,encoder_version=0)['action']=='NORMAL'


def test_budget_denial_does_not_claim_committed_action():
    assert hasattr(controller,'GameControllerV2')
    from cvsrffi.game_tracking.budget import BudgetConfig,ComputeBudget
    b=ComputeBudget(BudgetConfig(max_field_evaluations=0))
    c=controller.GameControllerV2(controller.ControllerConfig(confirmation_windows=1))
    assert c.decide(evidence(0),step=0,encoder_version=0,budget=b)['action']=='NORMAL'
    assert c.state_dict()['accepted_events']==0


def test_legacy_capability_cannot_be_wrapped_as_v2_control_evidence():
    c=controller.GameControllerV2(controller.ControllerConfig(confirmation_windows=1))
    m=evidence(0);m['capability'].pop('schema')
    assert c.decide(m,step=0,encoder_version=0)['action']=='NORMAL'


def test_source_calibration_cannot_relabel_reliable_high_gap_as_low():
    from types import SimpleNamespace
    from cvsrffi.game_tracking.runtime_control import calibrate_game_v2
    history = [evidence(step) for step in (0, 250, 500)]
    for m in history:
        m['lag'].update(status='RELIABLE_HIGH_GAP',gap_normalized=1.,
                        stationary=False,plateau=False)
        m['capability'].update(next_identity=.9,next_margin=.3,next_worst_tx=.8)
    c = calibrate_game_v2(history,[m['capability'] for m in history],
        SimpleNamespace(num_classes=6,game_max_extra_head=3,game_audit_interval=250))
    assert c.config.lag_exit == .5
    for step in (750, 1000):
        m = evidence(step)
        m['lag'].update(status='RELIABLE_HIGH_GAP',gap_normalized=.2,
                        stationary=False,plateau=False)
        assert c.decide(m,step=step,encoder_version=step)['action'] == 'NORMAL'
    assert not c.requested_observations
    # The same calibrated controller can still accept genuinely reliable LOW evidence.
    assert c.decide(evidence(1250),step=1250,encoder_version=1250)['action']=='NORMAL'
    assert c.decide(evidence(1500),step=1500,encoder_version=1500)['action']=='CORRECT'


def test_zero_extra_head_budget_does_not_disable_correction_calibration():
    from cvsrffi.game_tracking.config import parse_args
    from cvsrffi.game_tracking.runtime_control import calibrate_game_v2
    args = parse_args(['--output_dir','unused','--game_control','correction',
                       '--game_max_extra_head','0'])
    history = [evidence(step) for step in (0,250,500)]
    for m in history:
        m['capability'].update(next_identity=.9,next_margin=.3,next_worst_tx=.8)
    c = calibrate_game_v2(history,[m['capability'] for m in history],args)
    assert c is not None
    assert c.config.catchup_steps == 0
    assert c.decide(evidence(750),step=750,encoder_version=750)['action']=='NORMAL'
    assert c.decide(evidence(1000),step=1000,encoder_version=1000)['action']=='CORRECT'
    c.record_outcome('obs1000',accepted=True)
    for step in (1250,1500):
        high = evidence(step)
        high['lag'].update(status='RELIABLE_HIGH_GAP',gap_normalized=.8)
        assert c.decide(high,step=step,encoder_version=step)['action']=='NORMAL'
    assert not c.requested_observations
