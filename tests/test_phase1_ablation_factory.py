from __future__ import annotations

from argparse import Namespace

import pytest

from cvsrffi.phase1_ablation_factory import (
    PHASE1_ABLATION_IDS,
    Phase1AblationConfigError,
    apply_phase1_ablation,
    enabled_objectives,
    phase1_ablation_config,
    phase1_ablation_config_hash,
    phase1_ablation_diff,
)
from SSDG.train_ssdg import build_arg_parser, train


def test_all_six_phase1_t1_arms_have_unique_frozen_hashes() -> None:
    hashes = {
        arm_id: phase1_ablation_config_hash(arm_id)
        for arm_id in PHASE1_ABLATION_IDS
    }
    assert len(PHASE1_ABLATION_IDS) == 6
    assert len(set(hashes.values())) == 6
    assert all(len(value) == 64 for value in hashes.values())


def test_common_protocol_and_budget_never_drift() -> None:
    configs = [phase1_ablation_config(arm_id) for arm_id in PHASE1_ABLATION_IDS]
    assert {config["split_mode"] for config in configs} == {"tx_rx_day_1_7_2"}
    assert {
        (
            config["labeled_ratio"],
            config["unlabeled_ratio"],
            config["source_val_ratio"],
        )
        for config in configs
    } == {(0.07, 0.63, 0.30)}
    assert {config["epochs"] for config in configs} == {200}
    assert {config["checkpoint_selection"] for config in configs} == {
        "final_only"
    }
    assert all(config["phase1_source_val_selection_only"] for config in configs)


def test_p1_b0_changes_only_pseudo_ce_and_entropy() -> None:
    assert set(phase1_ablation_diff("P1-B0")) == {"lambda_u", "lambda_ent"}


def test_p1_c0_turns_off_all_and_only_geometry_group() -> None:
    assert set(phase1_ablation_diff("P1-C0")) == {
        "use_proto_memory",
        "lambda_proto",
        "lambda_open_world_feat",
        "lambda_zid_compact",
        "lambda_proxy_unknown",
        "lambda_soft_unknown_mixup",
    }
    assert not {
        "prototype",
        "open_world_geometry",
        "zid_geometry",
        "coretail_proxy",
        "soft_mix_boundary",
    } & set(enabled_objectives("P1-C0"))


def test_p1_d0_makes_all_extrapolation_routes_unreachable() -> None:
    config = phase1_ablation_config("P1-D0")
    assert set(phase1_ablation_diff("P1-D0")) == {
        "use_mixstyle",
        "lambda_source_episode",
        "use_sat_consistency",
        "use_concat_sat_channel_aug",
        "concat_sat_ce_only",
        "lambda_sat_cls",
    }
    assert config["use_mixstyle"] is False
    assert config["use_concat_sat_channel_aug"] is False
    assert config["lambda_source_episode"] == 0.0
    assert config["lambda_sat_cls"] == 0.0


def test_p1_sup_has_only_tx_objective() -> None:
    assert enabled_objectives("P1-SUP") == ("tx_cosface",)
    config = phase1_ablation_config("P1-SUP")
    assert config["use_unlabeled"] is False
    assert config["label_epochs"] == 200
    assert config["pseudo_epochs"] == 0


def test_p1_a0_is_single_embedding_and_removes_disentanglement_losses() -> None:
    config = phase1_ablation_config("P1-A0")
    assert config["representation_mode"] == "single_parameter_matched"
    assert {
        key for key, value in phase1_ablation_diff("P1-A0").items()
        if value[1] == 0.0
    } == {"lambda_domain", "lambda_adv", "lambda_orth", "lambda_cons"}


def test_apply_factory_requires_full_commit_and_parser_coverage() -> None:
    config = phase1_ablation_config("P1-B0")
    namespace = Namespace(
        ablation_id="P1-B0",
        git_commit="a" * 40,
        **{key: None for key in config},
    )
    manifest = apply_phase1_ablation(namespace)
    assert manifest["ablation_id"] == "P1-B0"
    assert namespace.ablation_config_hash == manifest["config_hash"]
    namespace.git_commit = "short"
    with pytest.raises(Phase1AblationConfigError, match="40-character"):
        apply_phase1_ablation(namespace)


def test_unknown_arm_fails_closed() -> None:
    with pytest.raises(Phase1AblationConfigError, match="unknown"):
        phase1_ablation_config("P1-UNKNOWN")


@pytest.mark.parametrize("ablation_id", PHASE1_ABLATION_IDS)
def test_real_training_parser_accepts_each_frozen_arm_in_dry_run(
    ablation_id: str,
    tmp_path,
) -> None:
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path / ablation_id),
            "--formal_ablation",
            "true",
            "--ablation_id",
            ablation_id,
            "--candidate_id",
            ablation_id,
            "--run_id",
            f"dryrun_{ablation_id.lower()}",
            "--git_commit",
            "a" * 40,
            "--dry_run",
        ]
    )
    assert train(args) == 0
    assert args.ablation_id == ablation_id
    assert args.ablation_config_hash == phase1_ablation_config_hash(ablation_id)
    assert (args.labeled_ratio, args.unlabeled_ratio, args.source_val_ratio) == (
        0.07,
        0.63,
        0.30,
    )
