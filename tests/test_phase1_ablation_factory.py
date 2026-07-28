from __future__ import annotations

from argparse import Namespace
from pathlib import Path

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
from SSDG.train_ssdg import (
    _build_source_split_receipt,
    _formal_ablation_terminal_flags,
    _validate_phase1_checkpoint_payload,
    build_arg_parser,
    train,
)


def test_all_six_phase1_t1_arms_have_unique_frozen_hashes() -> None:
    hashes = {
        arm_id: phase1_ablation_config_hash(arm_id)
        for arm_id in PHASE1_ABLATION_IDS
    }
    assert len(PHASE1_ABLATION_IDS) == 6
    assert len(set(hashes.values())) == 6
    assert all(len(value) == 64 for value in hashes.values())


def test_full_arm_matches_frozen_dual_branch_architecture() -> None:
    config = phase1_ablation_config("P1-FULL")
    assert config["branch_ablation"] == "no_dac"
    assert config["domain_branch_ablation"] == "none"
    assert config["domain_enhancer"] == "rcn_stats"
    assert config["representation_mode"] == "dual"


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
        "source_validation_only"
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
            "--seed",
            "42",
            "--row_key",
            f"{ablation_id}__train_seed_42",
            "--sealed_plan_sha256",
            "b" * 64,
            "--seed_registry_sha256",
            "c" * 64,
            "--wisig_pkl_sha256",
            "d" * 64,
            "--dry_run",
        ]
    )
    assert train(args) == 0
    assert args.ablation_id == ablation_id
    assert args.ablation_method_config_hash == phase1_ablation_config_hash(
        ablation_id
    )
    assert (args.labeled_ratio, args.unlabeled_ratio, args.source_val_ratio) == (
        0.07,
        0.63,
        0.30,
    )


def test_formal_factory_locks_protocol_and_optimizer_cli_drift(tmp_path) -> None:
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path / "locked"),
            "--formal_ablation",
            "true",
            "--ablation_id",
            "P1-FULL",
            "--candidate_id",
            "P1-FULL",
            "--run_id",
            "locked",
            "--git_commit",
            "a" * 40,
            "--seed",
            "42",
            "--row_key",
            "P1-FULL__train_seed_42",
            "--sealed_plan_sha256",
            "b" * 64,
            "--seed_registry_sha256",
            "c" * 64,
            "--wisig_pkl_sha256",
            "d" * 64,
            "--wisig_train_rxs",
            "0,1,2,3,4,5,6,7,8,9,10,11",
            "--lr",
            "9.9",
            "--dry_run",
        ]
    )
    assert train(args) == 0
    assert args.wisig_train_rxs == "0,1,2,3,4,5,6"
    assert args.wisig_test_rxs == "7,8,9,10,11"
    assert args.lr == 0.0002


def test_unfrozen_parser_field_changes_resolved_hash(tmp_path) -> None:
    def parsed(max_grad_norm: str):
        return build_arg_parser().parse_args(
            [
                "--output_dir",
                str(tmp_path / max_grad_norm),
                "--formal_ablation",
                "true",
                "--ablation_id",
                "P1-FULL",
                "--candidate_id",
                "P1-FULL",
                "--run_id",
                "hash-check",
                "--git_commit",
                "a" * 40,
                "--seed",
                "42",
                "--row_key",
                "P1-FULL__train_seed_42",
                "--sealed_plan_sha256",
                "b" * 64,
                "--seed_registry_sha256",
                "c" * 64,
                "--wisig_pkl_sha256",
                "d" * 64,
                "--max_grad_norm",
                max_grad_norm,
                "--dry_run",
            ]
        )

    first = parsed("0")
    second = parsed("9.9")
    assert train(first) == 0
    assert train(second) == 0
    assert first.ablation_config_hash != second.ablation_config_hash


def test_dataset_bytes_hash_changes_resolved_hash(tmp_path) -> None:
    def parsed(dataset_hash: str):
        return build_arg_parser().parse_args(
            [
                "--output_dir",
                str(tmp_path / dataset_hash[:4]),
                "--formal_ablation",
                "true",
                "--ablation_id",
                "P1-FULL",
                "--candidate_id",
                "P1-FULL",
                "--run_id",
                "dataset-hash",
                "--git_commit",
                "a" * 40,
                "--seed",
                "42",
                "--row_key",
                "P1-FULL__train_seed_42",
                "--sealed_plan_sha256",
                "b" * 64,
                "--seed_registry_sha256",
                "c" * 64,
                "--wisig_pkl_sha256",
                dataset_hash,
                "--dry_run",
            ]
        )

    first = parsed("d" * 64)
    second = parsed("e" * 64)
    assert train(first) == 0
    assert train(second) == 0
    assert first.ablation_config_hash != second.ablation_config_hash


@pytest.mark.parametrize("ablation_id", PHASE1_ABLATION_IDS)
def test_arm_aware_terminal_contract_accepts_intentional_disabled_groups(
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
            f"terminal_{ablation_id.lower()}",
            "--git_commit",
            "a" * 40,
            "--seed",
            "42",
            "--row_key",
            f"{ablation_id}__train_seed_42",
            "--sealed_plan_sha256",
            "b" * 64,
            "--seed_registry_sha256",
            "c" * 64,
            "--wisig_pkl_sha256",
            "d" * 64,
            "--dry_run",
        ]
    )
    assert train(args) == 0
    evidence = {
        "checkpoint_role": "source_validation_selected",
        "args": vars(args),
    }
    p0_flags, p1_flags = _formal_ablation_terminal_flags(
        args,
        selected_checkpoint=Path("best_source_validation_ssdg.pth"),
        selected_checkpoint_evidence=evidence,
        selected_checkpoint_sha256="b" * 64,
        export_status={
            "status": "COMPLETE",
            "source_checkpoint_sha256": "b" * 64,
        },
        source_split_receipt={
            "split_manifest_sha256": "d" * 64,
            "source_target_receiver_overlap_count": 0,
        },
    )
    assert all(p0_flags.values()), p0_flags
    assert all(p1_flags.values()), p1_flags


def test_formal_checkpoint_validation_accepts_source_validation_selection() -> None:
    args = Namespace(
        run_id="run",
        candidate_id="P1-SUP",
        best_metric="source_val_sat_hmean",
        checkpoint_selection="source_validation_only",
    )
    payload = {
        "checkpoint_schema": "ssdg_phase1_training_state_v2",
        "checkpoint_role": "source_validation_selected",
        "checkpoint_selection": "source_validation_only",
        "run_id": "run",
        "candidate_id": "P1-SUP",
        "model": {},
        "args": {
            "best_metric": "source_val_sat_hmean",
            "checkpoint_selection": "source_validation_only",
        },
    }
    assert (
        _validate_phase1_checkpoint_payload(payload, args, "checkpoint.pth")
        is payload
    )


def test_source_split_receipt_hashes_label_masks_and_disjoint_receivers() -> None:
    receipt = _build_source_split_receipt(
        seed=7281101,
        split_mode="tx_rx_day_1_7_2",
        source_days=["d0", "d1"],
        target_days=["d2", "d3"],
        source_receivers=[f"r{i}" for i in range(7)],
        target_receivers=[f"r{i}" for i in range(7, 12)],
        labeled_indices=[3, 7],
        unlabeled_indices=[1, 9, 11],
        source_validation_indices=[2, 4],
        wisig_pkl_sha256="d" * 64,
    )
    assert receipt["source_target_receiver_overlap_count"] == 0
    assert receipt["labeled_indices_sha256"] != receipt[
        "unlabeled_indices_sha256"
    ]
    assert len(receipt["split_manifest_sha256"]) == 64
