from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).parents[1] / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
sys.modules.pop("scripts", None)

from scripts import run_d92_qic_g0
from cvsrffi import stage2_d92_e0d_query_evaluation as e0d


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        outer_key=run_d92_qic_g0.G0_OUTER_KEY,
        reference_arm=run_d92_qic_g0.REFERENCE_ARM,
        candidate_arm=run_d92_qic_g0.CANDIDATE_ARM,
        reference_output_root=str(tmp_path / "reference"),
        candidate_output_root=str(tmp_path / "candidate"),
        g0_validation_path=str(tmp_path / "g0_validation.json"),
        before_enrollment_package_root="before-enrollment",
        before_enrollment_seal_path="before-enrollment.seal",
        before_enrollment_seal_sha256="a" * 64,
        before_apply_package_root="before-apply",
        before_apply_seal_path="before-apply.seal",
        before_apply_seal_sha256="b" * 64,
        after_enrollment_package_root="after-enrollment",
        after_enrollment_seal_path="after-enrollment.seal",
        after_enrollment_seal_sha256="c" * 64,
        after_apply_package_root="after-apply",
        after_apply_seal_path="after-apply.seal",
        after_apply_seal_sha256="d" * 64,
        ground_component_dir="ground",
        ground_manifest_sha256="e" * 64,
        device="cuda:0",
    )


def _row(scene: str, *, candidate: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scene,
        "arm_id": run_d92_qic_g0.CANDIDATE_ARM
        if candidate
        else run_d92_qic_g0.REFERENCE_ARM,
        "after_state_bytes": 8583,
        "query_macs": 3168,
        "after_actual_component_inventory": {
            "actual_component_fit_count": 1,
            "full_component_fit_count": 1,
        },
        "after_registration_resource": {
            "registration_wall_time_ns": 110_000_000 if candidate else 100_000_000,
            "registration_incremental_peak_working_set_bytes": 500_000
            if candidate
            else 100_000,
        },
    }
    if candidate:
        row.update(
            {
                "d92_e0d_qic_active": True,
                "d92_e0d_qic_fallback_active": False,
                "d92_e0d_qic_fallback_reason": None,
                "d92_e0d_qic_e0_state_sha256": "a" * 64,
                "d92_e0d_qic_final_state_sha256": "b" * 64,
                "d92_e0d_qic_modified_state_field_names": ["intercept_fp16"],
                "d92_e0d_qic_intercept_fp16_bit_change_count": 11,
                "d92_e0d_qic_candidate_intercept_fp16_bit_change_count": 11,
                "d92_e0d_qic_e0_residual_l1": 1.0,
                "d92_e0d_qic_candidate_residual_l1": 0.5,
                "d92_e0d_qic_residual_reduction_l1": 0.5,
                "d92_e0d_qic_intercept_byte_exact": False,
                "d92_e0d_qic_coefficient_decode_count": 1,
                "d92_e0d_qic_requantize_call_count": 0,
                "d92_e0d_qic_additional_full_fit_count": 0,
                "d92_e0d_qic_block_fit_count": 0,
                "d92_e0d_qic_loo_fit_count": 0,
                "d92_e0d_qic_fisher_scan_count": 0,
                "d92_e0d_qic_candidate_scan_count": 0,
                "d92_e0d_qic_persistent_state_bytes_delta": 0,
                "d92_e0d_qic_query_macs_delta": 0,
                "d92_e0d_qic_support_only": True,
            }
        )
        for suffix in (
            "coef1_byte_exact",
            "coef2_byte_exact",
            "scale1_byte_exact",
            "scale2_byte_exact",
            "log_diag_byte_exact",
            "coef_fp32_byte_exact",
            "intercept_fp32_byte_exact",
            "class_registry_byte_exact",
            "state_shape_byte_exact",
        ):
            row[f"d92_e0d_qic_{suffix}"] = True
        for suffix in (
            "fit_access",
            "update_access",
            "selection_access",
            "truth_access",
            "role_oracle_access",
            "class_quota_access",
            "global_reassignment",
        ):
            row[f"d92_e0d_qic_query_{suffix}"] = False
        row["d92_e0d_qic_clean_sample_access"] = False
        row["d92_e0d_qic_source_sample_access"] = False
    for suffix in (
        "fit_access",
        "update_access",
        "selection_access",
        "truth_access",
        "role_oracle_access",
        "class_quota_access",
        "global_reassignment",
    ):
        row[f"query_{suffix}"] = False
    return row


def test_parser_freezes_qic_g0_and_exposes_no_truth_or_score_switch() -> None:
    parser = run_d92_qic_g0.parser()
    help_text = parser.format_help().lower()
    assert "truth" not in help_text
    assert "score" not in help_text
    args = parser.parse_args(
        [
            "--before-enrollment-package-root",
            "before-enrollment",
            "--before-enrollment-seal-path",
            "before-enrollment.seal",
            "--before-enrollment-seal-sha256",
            "a" * 64,
            "--before-apply-package-root",
            "before-apply",
            "--before-apply-seal-path",
            "before-apply.seal",
            "--before-apply-seal-sha256",
            "b" * 64,
            "--after-enrollment-package-root",
            "after-enrollment",
            "--after-enrollment-seal-path",
            "after-enrollment.seal",
            "--after-enrollment-seal-sha256",
            "c" * 64,
            "--after-apply-package-root",
            "after-apply",
            "--after-apply-seal-path",
            "after-apply.seal",
            "--after-apply-seal-sha256",
            "d" * 64,
            "--ground-component-dir",
            "ground",
            "--ground-manifest-sha256",
            "e" * 64,
            "--reference-output-root",
            "reference",
            "--candidate-output-root",
            "candidate",
            "--g0-validation-path",
            "validation.json",
            "--device",
            "cuda:0",
        ]
    )
    assert args.outer_key == run_d92_qic_g0.G0_OUTER_KEY
    assert args.reference_arm == run_d92_qic_g0.REFERENCE_ARM
    assert args.candidate_arm == run_d92_qic_g0.CANDIDATE_ARM


def test_g0_reads_persisted_fit_audit_and_writes_pass_marker(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)
    calls: list[str] = []

    def fake_run(*, arm_id: str, output_root: str | Path, **_kwargs: object):
        calls.append(arm_id)
        output = Path(output_root)
        after = output / "after"
        after.mkdir(parents=True)
        rows = [
            _row(scene, candidate=arm_id == run_d92_qic_g0.CANDIDATE_ARM)
            for scene in run_d92_qic_g0.G0_SCENES
        ]
        (after / "fit_audit.json").write_text(
            json.dumps(rows), encoding="utf-8", newline="\n"
        )
        return {"fit_audit": [{"scenario": "wrong-return-value"}]}

    monkeypatch.setattr(e0d, "run_d92_e0d_query_evaluation", fake_run)
    result = run_d92_qic_g0.run(args)

    assert calls == [run_d92_qic_g0.REFERENCE_ARM, run_d92_qic_g0.CANDIDATE_ARM]
    assert result["marker"] == run_d92_qic_g0.G0_MARKER
    persisted = json.loads(Path(args.g0_validation_path).read_text(encoding="utf-8"))
    assert persisted["status"] == run_d92_qic_g0.G0_MARKER
    assert persisted["validation"]["pass"] is True
    assert set(persisted["validation"]["scenes"]) == set(run_d92_qic_g0.G0_SCENES)


def test_existing_output_is_rejected_before_e0d_call(tmp_path) -> None:
    args = _args(tmp_path)
    Path(args.reference_output_root).mkdir()
    with pytest.raises(run_d92_qic_g0.D92QICG0EntryError, match="overwrite"):
        run_d92_qic_g0.run(args)


def test_candidate_gate_rejects_fallback_and_persists_rejection(tmp_path, monkeypatch) -> None:
    args = _args(tmp_path)

    def fake_run(*, arm_id: str, output_root: str | Path, **_kwargs: object):
        output = Path(output_root)
        after = output / "after"
        after.mkdir(parents=True)
        rows = [
            _row(scene, candidate=arm_id == run_d92_qic_g0.CANDIDATE_ARM)
            for scene in run_d92_qic_g0.G0_SCENES
        ]
        if arm_id == run_d92_qic_g0.CANDIDATE_ARM:
            rows[0]["d92_e0d_qic_fallback_active"] = True
        (after / "fit_audit.json").write_text(
            json.dumps(rows), encoding="utf-8", newline="\n"
        )
        return {}

    monkeypatch.setattr(e0d, "run_d92_e0d_query_evaluation", fake_run)
    with pytest.raises(run_d92_qic_g0.D92QICG0EntryError, match="validation failed"):
        run_d92_qic_g0.run(args)
    persisted = json.loads(Path(args.g0_validation_path).read_text(encoding="utf-8"))
    assert persisted["status"] == "D92_QIC_G0_REJECTED"
