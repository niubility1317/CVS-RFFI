from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d42_unified_shrinkage_lda as d42


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "scripts"
    / "probe_d43_structured_covariance.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d43", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_structured_covariance_arms_are_exactly_prelocked() -> None:
    covariance = np.arange(36, dtype=np.float64).reshape(6, 6)
    covariance = covariance + covariance.T + np.eye(6) * 100.0
    blocks = (slice(0, 2), slice(2, 5), slice(5, 6))
    full = probe._structured_covariance(covariance, "full_centered_control", blocks)
    block = probe._structured_covariance(covariance, "block3_centered", blocks)
    diagonal = probe._structured_covariance(covariance, "diagonal_centered", blocks)
    np.testing.assert_array_equal(full, covariance)
    np.testing.assert_array_equal(block[:2, :2], covariance[:2, :2])
    np.testing.assert_array_equal(block[2:5, 2:5], covariance[2:5, 2:5])
    np.testing.assert_array_equal(block[5:, 5:], covariance[5:, 5:])
    assert np.count_nonzero(block[:2, 2:]) == 0
    assert np.count_nonzero(block[2:, :2]) == 0
    np.testing.assert_array_equal(diagonal, np.diag(np.diag(covariance)))


def test_class_common_affine_removal_preserves_argmax_and_margins() -> None:
    rng = np.random.default_rng(43)
    rows = rng.normal(size=(17, 8))
    coefficients = rng.normal(size=(5, 8))
    intercept = rng.normal(size=5)
    centered_coef, centered_intercept = probe._center_affine_scores(
        coefficients, intercept
    )
    original = rows @ coefficients.T + intercept[None, :]
    centered = rows @ centered_coef.T + centered_intercept[None, :]
    np.testing.assert_array_equal(np.argmax(original, axis=1), np.argmax(centered, axis=1))
    np.testing.assert_allclose(
        original[:, :, None] - original[:, None, :],
        centered[:, :, None] - centered[:, None, :],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(centered_coef.mean(axis=0), 0.0, atol=1.0e-15)
    assert abs(float(centered_intercept.mean())) < 1.0e-15


def test_full_centered_control_matches_d42_fp32_support_predictions() -> None:
    rng = np.random.default_rng(4301)
    class_count = 4
    k_shot = 10
    rows = rng.normal(size=(class_count * k_shot, d42.FEATURE_DIM)).astype(np.float32)
    labels = np.repeat(np.arange(class_count), k_shot)
    original_coef, original_intercept, _ = d42._fit_equal_prior_lda(
        rows, labels, class_count, k_shot
    )
    fitted_coef, fitted_intercept, audit = probe.build_structured_fit(
        d42, "full_centered_control"
    )(rows, labels, class_count, k_shot)
    original_scores = rows @ original_coef.T + original_intercept[None, :]
    fitted_scores = rows @ fitted_coef.T + fitted_intercept[None, :]
    np.testing.assert_array_equal(
        np.argmax(original_scores, axis=1), np.argmax(fitted_scores, axis=1)
    )
    np.testing.assert_allclose(
        original_scores[:, :, None] - original_scores[:, None, :],
        fitted_scores[:, :, None] - fitted_scores[:, None, :],
        rtol=1.0e-5,
        atol=1.0e-4,
    )
    assert audit["d43_class_common_affine_omitted"] is True
    assert audit["sklearn_prediction_equivalent"] is True
    assert audit["d43_centered_support_fp32_argmax_equivalent"] is True
    assert audit["d43_centered_support_fp32_argmax_changed_count"] == 0
    assert audit["d43_centered_support_fp32_argmax_drift_allowed"] is False
    assert np.isfinite(audit["d43_centered_support_fp32_pairwise_drift_max"])


def test_probe_guards_force_identity_and_disable_full_refit() -> None:
    runner = SimpleNamespace(
        IDENTITY_CANDIDATE="Z0_SUPPORT_ONLY",
        _canonical_bytes=lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        _candidate_lock=lambda _candidates, _candidate_set="d25_v4": {
            "source_closure": {},
            "sha256": "old",
        },
        _select_d42_candidate=lambda _folds: (
            "D42-USLDA-INT8",
            [
                {
                    "candidate_id": "D42-USLDA-INT8",
                    "eligible_positive_route": True,
                }
            ],
        ),
        _full_state_refit_required=lambda *_args, **_kwargs: True,
    )
    script_sha = "a" * 64
    probe._install_runner_probe_guards(
        runner,
        arm="block3_centered",
        probe_script_sha256=script_sha,
        extra_source_closure={"extra_helper_sha256": "d" * 64},
    )
    selected, decisions = runner._select_d42_candidate({})
    assert selected == "Z0_SUPPORT_ONLY"
    assert decisions[0]["eligible_positive_route"] is False
    assert decisions[0]["d43_probe_pre_guard_eligible_positive_route"] is True
    assert decisions[0]["d43_probe_forced_nonpromotable"] is True
    assert runner._full_state_refit_required("d42_v1", "D42-USLDA-INT8", selected) is False
    lock = runner._candidate_lock({}, "d42_v1")
    assert lock["source_closure"]["d43_probe_script_sha256"] == script_sha
    assert lock["source_closure"]["d43_runtime_legacy_sha256"] == probe.RUNTIME_LEGACY_SHA256
    assert lock["source_closure"]["d43_preloaded_runtime_module_sha256"] == probe.RUNTIME_MODULE_SHA256
    assert lock["source_closure"]["extra_helper_sha256"] == "d" * 64
    assert lock["d43_probe_lock"]["arm"] == "block3_centered"
    assert len(lock["sha256"]) == 64


def test_probe_guard_rejects_extra_source_closure_reserved_key_collision() -> None:
    runner = SimpleNamespace(
        IDENTITY_CANDIDATE="Z0_SUPPORT_ONLY",
        _candidate_lock=lambda _candidates, _candidate_set="d25_v4": {
            "source_closure": {"existing_sha256": "a" * 64},
            "sha256": "old",
        },
        _select_d42_candidate=lambda _folds: ("Z0_SUPPORT_ONLY", []),
        _full_state_refit_required=lambda *_args, **_kwargs: True,
    )
    probe._install_runner_probe_guards(
        runner,
        arm="block3_centered",
        probe_script_sha256="b" * 64,
        extra_source_closure={"existing_sha256": "c" * 64},
    )
    with pytest.raises(probe.D43ProbeError, match="reserved keys"):
        runner._candidate_lock({}, "d42_v1")


def test_runner_argument_lock_requires_exact_d42_development_mode() -> None:
    valid = [
        "--output",
        "probe",
        "--candidate-set",
        "d42_v1",
        "--mode",
        "development_select_unverified_component",
    ]
    probe._require_locked_runner_arguments(valid)
    with pytest.raises(probe.D43ProbeError):
        probe._require_locked_runner_arguments(
            [value if value != "d42_v1" else "d25_v4" for value in valid]
        )
    with pytest.raises(probe.D43ProbeError):
        probe._require_locked_runner_arguments(valid + ["--candidate-set", "d42_v1"])


def test_probe_output_verification_binds_arm_receipt_and_artifacts(tmp_path: Path) -> None:
    arm = "block3_centered"
    structure = "three_block_z160_fft96_rf32"
    script_sha = "b" * 64
    training_rows = []
    for index in range(105):
        if index < 15:
            candidate = "D42-USLDA-INT8"
        elif index < 30:
            candidate = "D42-USLDA-FP32-MATCHED"
        else:
            candidate = "B3_SINGLE_IQ_DIAG_FFTRF"
        row = {"candidate_id": candidate, "query_opened": False}
        if index < 30:
            audit = {
                "d43_probe_arm": arm,
                "d43_covariance_structure": structure,
                "d43_class_common_affine_omitted": True,
            }
            row["geometry_summary"] = {
                "before_covariance_audit": dict(audit),
                "final_covariance_audit": dict(audit),
            }
        training_rows.append(row)
    (tmp_path / "training_log.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in training_rows),
        encoding="utf-8",
    )
    candidate_lock_payload = {
        "source_closure": {
            "ciaf_sha256": probe.RUNTIME_MODULE_SHA256["cvsrffi.stage2_ciaf"],
            "d19_control_helper_sha256": probe.RUNTIME_LEGACY_SHA256,
            "diag_cosine_feature_operator_sha256": probe.RUNTIME_MODULE_SHA256[
                "cvsrffi.stage2_diag_cosine_exploration"
            ],
            "d43_probe_script_sha256": script_sha,
            "d43_runtime_legacy_sha256": probe.RUNTIME_LEGACY_SHA256,
            "d43_preloaded_runtime_module_sha256": dict(
                probe.RUNTIME_MODULE_SHA256
            ),
        },
        "d43_probe_lock": {
            "arm": arm,
            "formal_candidate": False,
            "forced_nonpromotable": True,
            "selected_only_full_k10_refit_allowed": False,
        },
    }
    candidate_lock = {
        **candidate_lock_payload,
        "sha256": hashlib.sha256(
            probe._canonical_bytes(candidate_lock_payload)
        ).hexdigest(),
    }
    (tmp_path / "support_audit.json").write_text(
        json.dumps(
            {
                "query_opened": False,
                "query_rows_opened": 0,
                "query_labels_opened": 0,
                "formal_metric_claim_allowed": False,
                "performance_claim_allowed": False,
                "candidate_lock": candidate_lock,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "selected_candidate_id": "Z0_SUPPORT_ONLY",
                "pre_full_k10_selected_candidate_id": "Z0_SUPPORT_ONLY",
                "selected_positive_route": False,
                "candidate_lock_sha256": candidate_lock["sha256"],
                "candidate_decisions": [
                    {
                        "candidate_id": "D42-USLDA-INT8",
                        "eligible_positive_route": False,
                        "d43_probe_forced_nonpromotable": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for file_name in ("resource_audit.json", "geometry_audit.json"):
        (tmp_path / file_name).write_text("{}", encoding="utf-8")
    receipt = {
        "candidate_set": "d42_v1",
        "candidate_count": 7,
        "folds_per_candidate": 15,
        "mode": "development_select_unverified_component",
        "query_opened": False,
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "selected_candidate_id": "Z0_SUPPORT_ONLY",
        "pre_full_k10_selected_candidate_id": "Z0_SUPPORT_ONLY",
        "selected_positive_route": False,
        "training_log_row_count": 105,
        "status": "DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE",
        "source_closure_unchanged_after_support": True,
        "support_query_disjointness_status": "SUPPORT_ONLY_NO_QUERY_CLAIM",
        "d19_control_helper_sha256": probe.RUNTIME_LEGACY_SHA256,
        "ciaf_sha256": probe.RUNTIME_MODULE_SHA256["cvsrffi.stage2_ciaf"],
        "diag_cosine_feature_operator_sha256": probe.RUNTIME_MODULE_SHA256[
            "cvsrffi.stage2_diag_cosine_exploration"
        ],
        "candidate_lock_sha256": candidate_lock["sha256"],
    }
    for file_name, field in {
        "training_log.jsonl": "training_log_sha256",
        "support_audit.json": "support_audit_sha256",
        "selection.json": "selection_sha256",
        "resource_audit.json": "resource_audit_sha256",
        "geometry_audit.json": "geometry_audit_sha256",
    }.items():
        receipt[field] = probe._sha256(tmp_path / file_name)
    (tmp_path / "RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")
    evidence = probe._verify_probe_output(tmp_path, arm, script_sha)
    assert evidence["verified_training_row_count"] == 105
    assert evidence["verified_d43_fit_row_count"] == 30
    assert evidence["verified_covariance_structure"] == structure
    training_rows[0]["geometry_summary"]["final_covariance_audit"][
        "d43_probe_arm"
    ] = "diagonal_centered"
    (tmp_path / "training_log.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in training_rows),
        encoding="utf-8",
    )
    with pytest.raises(probe.D43ProbeError):
        probe._verify_probe_output(tmp_path, arm, script_sha)
