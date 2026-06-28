from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import optimizer_validate_matrix as validator  # noqa: E402
import spaceborne_fewshot_da_matrix as matrix_gen  # noqa: E402


def test_h06_oldheadfar48_includes_real_phase1_ground_proto_rows():
    candidates = matrix_gen.make_candidates(plan="OA_MSE_H06_OLDHEADFAR48")
    phase1 = [c for c in candidates if c.command_kind == "phase1_safe_ssdg_ground_train"]
    phase2 = [c for c in candidates if c.command_kind != "phase1_safe_ssdg_ground_train"]

    assert len(candidates) == 56
    assert len(phase1) == 8
    assert len(phase2) == 48
    assert all(c.lane == "phase1_ground_dg" for c in phase1)
    assert all(c.phase1_enable_ground_prototype_stats for c in phase1)
    assert all(c.phase1_enable_feature_distribution_audit for c in phase1)

    payload = matrix_gen.matrix_payload("unit_phase1_ground_proto", candidates)
    assert payload["phase1_rows_expected"] == 8
    assert payload["phase2_rows_expected"] == 48
    assert payload["phase1_ground_dg_policy"]["status"] == "enabled"

    rows = payload["candidates"]
    phase1_rows = [row for row in rows if row["lane"] == "phase1_ground_dg"]
    assert len(phase1_rows) == 8
    for row in phase1_rows:
        assert row["protocol"] == "Safe-SSDG-CVS-R01"
        assert row["ground_dg_claim_scope"] == "source_only"
        assert row["no_target_receiver_in_training"] is True
        assert row["CEN51_COMPARABLE"] is True
        assert row["phase1_ground_feature_distribution_objective"] is True
        assert row["source_domain_prototype_outputs_required"] is True
        assert "code/SSDG/train_ssdg.py" in row["parameters"]["entrypoint"]
        assert "code/SSDG/train_ssdg.py" in row["exact_command"]
        assert "--epochs 200" in row["exact_command"]
        assert "--use_safe_ssdg_cvs" not in row["exact_command"]

    validation = validator.validate(rows, payload["expected_count"], matrix_root=payload)
    assert validation["verdict"] == "PASS", validation["issues"]


def test_phase1_ground_rows_do_not_expand_existing_64_row_plans():
    candidates = matrix_gen.make_candidates(plan="OA_MSE_BALANCE64")
    payload = matrix_gen.matrix_payload("unit_no_phase1_on_existing64", candidates)

    assert len(candidates) == 64
    assert payload["phase1_rows_expected"] == 0
    assert payload["phase2_rows_expected"] == 64


def test_validator_rejects_phase1_ground_proto_mask_missing_required_marker():
    candidates = matrix_gen.make_candidates(plan="OA_MSE_H06_OLDHEADFAR48")
    payload = matrix_gen.matrix_payload("unit_phase1_ground_proto_negative", candidates)
    rows = payload["candidates"]
    broken = next(row for row in rows if row["lane"] == "phase1_ground_dg")
    broken["phase1_ground_feature_distribution_objective"] = False
    broken["parameters"]["prototype_mask_modules"] = "phase2_prototypes"

    validation = validator.validate(rows, payload["expected_count"], matrix_root=payload)
    issue_names = {issue["issue"] for issue in validation["issues"]}

    assert validation["verdict"] == "FAIL"
    assert "phase1_ground_proto_mask_requires_phase1_ground_feature_distribution_objective" in issue_names
    assert "phase1_ground_proto_mask_missing_module_token" in issue_names


def test_phase1_gpu0_jointsafe4_matrix_has_guarded_audit_only_rows():
    candidates = matrix_gen.make_candidates(plan="PHASE1_GPU0_JOINTSAFE4")
    payload = matrix_gen.matrix_payload("unit_phase1_gpu0_jointsafe4", candidates)
    rows = payload["candidates"]

    assert len(rows) == 4
    assert payload["phase1_rows_expected"] == 4
    assert payload["phase2_rows_expected"] == 0
    for row in rows:
        params = row["parameters"]
        assert row["lane"] == "phase1_ground_dg"
        assert row["best_metric"] == "joint_safe"
        assert row["joint_checkpoint_policy"] == "joint_safe_guarded"
        assert row["phase1_distribution_audit_only"] is True
        assert params["best_metric"] == "joint_safe"
        assert params["phase1_distribution_audit_only"] is True
        assert params["phase1_prototype_loss_weight"] == 0.0
        assert params["phase1_mask_aux_loss_weight"] == 0.0
        assert params["phase1_geometry_loss_weight"] == 0.0
        assert "--best_metric joint_safe" in row["exact_command"]
        assert "--enable_joint_safe_guard true" in row["exact_command"]
        assert "--paic_guard_enabled true" in row["exact_command"]
        assert "--use_phase2_ground_prototypes true" in row["exact_command"]
        assert "--use_feature_masks true" in row["exact_command"]
        assert "--use_txrx_geometry_losses true" in row["exact_command"]

    validation = validator.validate(rows, payload["expected_count"], matrix_root=payload)
    assert validation["verdict"] == "PASS", validation["issues"]


def test_phase1_gpu0_jointsafe36_matrix_queues_two_per_gpu():
    candidates = matrix_gen.make_candidates(plan="PHASE1_GPU0_JOINTSAFE36")
    payload = matrix_gen.matrix_payload("unit_phase1_gpu0_jointsafe36", candidates)
    rows = payload["candidates"]
    launcher = matrix_gen.render_launcher("unit_phase1_gpu0_jointsafe36", candidates)

    assert len(rows) == 36
    assert payload["phase1_rows_expected"] == 36
    assert payload["phase2_rows_expected"] == 0
    assert payload["phase1_ground_dg_policy"]["queue_max_active_per_gpu"] == 2
    assert set(payload["phase1_ground_dg_policy"]["queued_rows_per_gpu"].values()) == {4, 5}
    assert 'STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"' in launcher
    assert "external_active_for_gpu" in launcher
    assert "[SPACEBORNE-FSDA-WAIT]" in launcher

    by_gpu = {}
    for row in rows:
        gpu = int(row["gpu"])
        by_gpu.setdefault(gpu, []).append(row)
        assert row["stage2_max_active_per_gpu"] == 2
        assert row["best_metric"] == "joint_safe"
        assert row["phase1_distribution_audit_only"] is True
        assert "--best_metric joint_safe" in row["exact_command"]
        assert "--paic_guard_enabled true" in row["exact_command"]
        assert "--lambda_tx_supcon_masked 0" in row["exact_command"]

    assert sorted(by_gpu) == list(range(8))
    assert {gpu: len(gpu_rows) for gpu, gpu_rows in by_gpu.items()} == {
        0: 5,
        1: 5,
        2: 5,
        3: 5,
        4: 4,
        5: 4,
        6: 4,
        7: 4,
    }

    validation = validator.validate(rows, payload["expected_count"], matrix_root=payload, launcher_text=launcher)
    assert validation["verdict"] == "PASS", validation["issues"]


def test_validator_rejects_joint_safe_row_missing_guard_cli():
    candidates = matrix_gen.make_candidates(plan="PHASE1_GPU0_JOINTSAFE4")
    payload = matrix_gen.matrix_payload("unit_phase1_gpu0_jointsafe4_negative", candidates)
    rows = payload["candidates"]
    broken = rows[0]
    broken["exact_command"] = broken["exact_command"].replace("--paic_guard_enabled true ", "")

    validation = validator.validate(rows, payload["expected_count"], matrix_root=payload)
    issue_names = {issue["issue"] for issue in validation["issues"]}

    assert validation["verdict"] == "FAIL"
    assert "phase1_joint_safe_command_missing_required_flag" in issue_names


def test_validator_rejects_unverified_active_proto_mask_loss():
    candidates = matrix_gen.make_candidates(plan="PHASE1_GPU0_JOINTSAFE4")
    payload = matrix_gen.matrix_payload("unit_phase1_gpu0_jointsafe4_active_loss_negative", candidates)
    rows = payload["candidates"]
    broken = rows[0]
    broken["parameters"]["phase1_prototype_loss_weight"] = 0.03
    broken["parameters"]["phase1_distribution_audit_only"] = False
    broken["phase1_prototype_loss_weight"] = 0.03
    broken["phase1_distribution_audit_only"] = False

    validation = validator.validate(rows, payload["expected_count"], matrix_root=payload)
    issue_names = {issue["issue"] for issue in validation["issues"]}

    assert validation["verdict"] == "FAIL"
    assert "phase1_ground_proto_mask_active_loss_requires_verified_training_wiring" in issue_names


def test_validator_rejects_masked_supcon_active_loss_in_audit_only_row():
    candidates = matrix_gen.make_candidates(plan="PHASE1_GPU0_JOINTSAFE4")
    payload = matrix_gen.matrix_payload("unit_phase1_gpu0_jointsafe4_masked_supcon_negative", candidates)
    rows = payload["candidates"]
    broken = rows[0]
    broken["parameters"]["lambda_tx_supcon_masked"] = 0.02
    broken["parameters"]["lambda_rx_supcon_masked"] = 0.01
    broken["lambda_tx_supcon_masked"] = 0.02
    broken["lambda_rx_supcon_masked"] = 0.01

    validation = validator.validate(rows, payload["expected_count"], matrix_root=payload)
    issue_names = {issue["issue"] for issue in validation["issues"]}

    assert validation["verdict"] == "FAIL"
    assert "phase1_ground_proto_mask_active_loss_must_not_be_audit_only" in issue_names
    assert "phase1_ground_proto_mask_active_loss_requires_verified_training_wiring" in issue_names
