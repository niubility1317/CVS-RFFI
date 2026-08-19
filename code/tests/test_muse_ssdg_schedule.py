from dataclasses import FrozenInstanceError, fields

import pytest

from cvsrffi.muse_ssdg import MUSEConfig, MUSEScheduleState, muse_schedule_for_epoch


def test_muse_schedule_matches_five_training_segments():
    cfg = MUSEConfig()
    assert muse_schedule_for_epoch(1, cfg).stage == "S1"
    assert not muse_schedule_for_epoch(16, cfg).pseudo_enabled
    assert muse_schedule_for_epoch(17, cfg).stage == "S2A"
    assert muse_schedule_for_epoch(40, cfg).p_sat == 0.25
    assert muse_schedule_for_epoch(41, cfg).candidate_enabled
    assert muse_schedule_for_epoch(69, cfg).ema_decay == 0.999
    assert muse_schedule_for_epoch(161, cfg).stage == "S3B"
    assert muse_schedule_for_epoch(181, cfg).freeze_statistics
    assert muse_schedule_for_epoch(200, cfg).lambda_u == 0.25


def test_muse_schedule_ramps_include_both_endpoints():
    cfg = MUSEConfig()

    assert muse_schedule_for_epoch(17, cfg).lambda_u == 0.0
    assert muse_schedule_for_epoch(40, cfg).lambda_u == 0.2
    assert muse_schedule_for_epoch(41, cfg).lambda_u == 0.2
    assert muse_schedule_for_epoch(68, cfg).lambda_u == 0.5
    assert muse_schedule_for_epoch(17, cfg).p_sat == 0.0
    assert muse_schedule_for_epoch(40, cfg).p_sat == 0.25
    assert muse_schedule_for_epoch(1, cfg).grl_lambda == 0.02
    assert muse_schedule_for_epoch(200, cfg).grl_lambda == 0.10


def test_muse_schedule_state_and_config_are_immutable_with_fixed_fields():
    cfg = MUSEConfig()
    state = muse_schedule_for_epoch(1, cfg)

    assert tuple(field.name for field in fields(MUSEScheduleState)) == (
        "stage",
        "ema_decay",
        "lambda_u",
        "p_sat",
        "grl_lambda",
        "proto_momentum",
        "pseudo_enabled",
        "candidate_enabled",
        "freeze_statistics",
    )
    with pytest.raises(FrozenInstanceError):
        cfg.s2a_start = 18
    with pytest.raises(FrozenInstanceError):
        state.stage = "S2A"


@pytest.mark.parametrize("epoch", [0, -1, 201, 10_000])
def test_muse_schedule_rejects_epochs_outside_training_horizon(epoch):
    with pytest.raises(ValueError):
        muse_schedule_for_epoch(epoch, MUSEConfig())


def test_muse_schedule_keeps_probability_outputs_bounded():
    cfg = MUSEConfig()
    for epoch in range(1, cfg.final_epoch + 1):
        state = muse_schedule_for_epoch(epoch, cfg)
        assert 0.0 <= state.p_sat <= 1.0
        assert 0.0 <= state.grl_lambda <= 1.0
