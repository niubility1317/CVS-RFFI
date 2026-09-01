from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from cvsrffi.phase1_fcr_schedule import FCRLambdaConfig, stage_for_epoch
from cvsrffi.schedule import build_fcr_stage_state, build_stage_state
from train import add_fcr_training_args, resolve_fcr_training_options


BOUNDARIES = (1, 40, 41, 90, 91, 150, 151, 200)


def test_four_stage_boundaries_are_exact_and_out_of_range_is_rejected() -> None:
    states = {epoch: stage_for_epoch(epoch, optimizer_step=0) for epoch in BOUNDARIES}

    assert states[1].name == states[40].name == "E1_40_reconstruction"
    assert states[1].active == frozenset({"id", "self", "eta"})
    assert states[41].name == states[90].name == "E41_90_cross_ramp"
    assert {"swap", "shared", "latent_cycle"} <= states[41].active
    assert states[91].name == states[150].name == "E91_150_intervention"
    assert {"intervention", "transplant"} <= states[91].active
    assert states[151].name == states[200].name == "E151_200_identity_refine"
    assert "id" in states[200].active

    with pytest.raises(ValueError, match="1..200"):
        stage_for_epoch(0)
    with pytest.raises(ValueError, match="1..200"):
        stage_for_epoch(201)
    with pytest.raises(ValueError, match="exactly 200"):
        stage_for_epoch(1, total_epochs=199)


def test_cross_ramp_starts_at_zero_reaches_configured_scale_and_late_raw_iq_is_lower() -> None:
    configured = FCRLambdaConfig(
        self_reconstruction=2.0,
        swap=3.0,
        shared=4.0,
        latent_cycle=5.0,
        eta=6.0,
        factor=7.0,
        transplant_necessity=8.0,
        physical_features=9.0,
    )

    e41 = stage_for_epoch(41, optimizer_step=0, configured=configured)
    e90 = stage_for_epoch(90, optimizer_step=0, configured=configured)
    e151 = stage_for_epoch(151, optimizer_step=0, configured=configured)

    assert e41.scales["swap"] == e41.scales["shared"] == e41.scales["latent_cycle"] == 0.0
    assert e90.scales["swap"] == 3.0
    assert e90.scales["shared"] == 4.0
    assert e90.scales["latent_cycle"] == 5.0
    assert e151.scales["self"] < e90.scales["self"]
    assert e151.scales["swap"] < e90.scales["swap"]


def test_intervention_stage_alternates_normal_and_necessity_by_optimizer_step() -> None:
    normal = stage_for_epoch(91, optimizer_step=0)
    necessity = stage_for_epoch(91, optimizer_step=1)
    normal_again = stage_for_epoch(91, optimizer_step=2)

    assert normal.freeze_decoder_for_necessity is False
    assert necessity.freeze_decoder_for_necessity is True
    assert normal_again.freeze_decoder_for_necessity is False
    assert "necessity" not in normal.active
    assert necessity.active == frozenset({"transplant", "necessity"})


def _legacy_schedule_args() -> SimpleNamespace:
    return SimpleNamespace(
        epochs=200,
        stage1_epochs=35,
        stage2_epochs=80,
        stage3_ramp_epochs=30,
        late_stable_start=0,
        lambda_dom=0.8,
        lambda_adv=0.2,
        lambda_orth=0.1,
        lambda_cons=0.0,
        lambda_cls_pa=0.3,
        lambda_cls_dac=0.4,
        lambda_pa_joint_inv=0.5,
        lambda_pa_kl=0.6,
        lambda_dac_reg=0.7,
        lambda_pa_reg=0.8,
        lambda_group_ce=0.9,
        lambda_sat_cls=0.68,
        phase1_method="adv3b02",
        use_fcr=False,
    )


def test_fcr_schedule_is_additive_and_does_not_mutate_existing_e80_satellite_schedule() -> None:
    args = _legacy_schedule_args()
    before = build_stage_state(80, args)

    assert build_fcr_stage_state(80, args, optimizer_step=3) is None
    after = build_stage_state(80, args)

    assert before == after
    assert args.lambda_sat_cls == 0.68


def test_cli_requires_explicit_candidate_route_and_ordinary_adv3b02_has_effective_zeroes() -> None:
    parser = argparse.ArgumentParser()
    add_fcr_training_args(parser)

    ordinary = resolve_fcr_training_options(parser.parse_args(["--lambda_fcr_self", "9"]))
    assert ordinary.phase1_method == "adv3b02"
    assert ordinary.use_fcr is False
    assert all(value == 0.0 for value in ordinary.effective_fcr_lambdas.values())

    candidate = resolve_fcr_training_options(
        parser.parse_args(["--phase1_method", "adv3b02_fcr", "--use_fcr"])
    )
    assert candidate.use_fcr is True
    assert set(candidate.effective_fcr_lambdas) == {
        "self", "swap", "shared", "latent_cycle", "eta", "factor", "need", "phys"
    }
    assert all(value > 0.0 for value in candidate.effective_fcr_lambdas.values())

    with pytest.raises(ValueError, match="adv3b02_fcr"):
        resolve_fcr_training_options(parser.parse_args(["--use_fcr"]))
    with pytest.raises(ValueError, match="--use_fcr"):
        resolve_fcr_training_options(parser.parse_args(["--phase1_method", "adv3b02_fcr"]))
