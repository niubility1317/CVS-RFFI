from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.stage2_m23_overlay_builder import compose_m23_overlay_payloads
from cvsrffi.stage2_m24_row_executor import (
    DA0_REG0,
    DA0_REG1,
    DA1_REG0,
    DA1_REG1,
    M24RowExecutionError,
    d1_overlay_from_base_cache,
    execute_m24_row,
)
from cvsrffi.stage2_m24_safe_residual import D1, D1_REFIT, M24_ARMS
from scripts import preflight_m24_safe_residual as preflight
from scripts import run_m24_safe_residual_suite as suite_runner
from scripts import score_m24_safe_residual_suite as suite_scorer
from scripts import run_m24_d1_expanded_matrix as expanded_runner
from test_stage2_m23_integration import NEW, OLD, _inputs


def _caches() -> tuple[dict, dict]:
    base, support, query = _inputs()
    payloads = compose_m23_overlay_payloads(
        base, support, query, old_classes=OLD, new_classes=NEW, k_shot=2
    )
    rng = np.random.default_rng(2420)
    ground = {
        "core_q": rng.integers(-30, 31, size=(6, 160), dtype=np.int8),
        "core_scale": np.full(6, 0.01, dtype=np.float16),
        "residual_basis_q": rng.integers(-20, 21, size=(6, 2, 160), dtype=np.int8),
        "residual_basis_scale": np.full((6, 2), 0.005, dtype=np.float16),
        "residual_coeff_q": rng.integers(-20, 21, size=(4, 6, 2), dtype=np.int8),
        "residual_coeff_scale": np.full((4, 6), 0.004, dtype=np.float16),
        "domain_registry": np.asarray([f"domain_{index}" for index in range(5)]),
        "residual_domain_registry": np.asarray([f"domain_{index}" for index in range(1, 5)]),
        "class_registry": np.asarray(OLD),
        "center_domain_handle": np.asarray("domain_0"),
    }
    basis, _ = np.linalg.qr(rng.normal(size=(160, 3)))
    base.update({
        "manifest": {
            "receiver": "3-19",
            "k_shot": 2,
            "method_seed": 7282101,
            "capsule_id": "capsule-fixed",
            "split_id": "split-fixed",
            "phase2_data_status": "VALIDATED_ONCE",
        },
        "old_classes": OLD,
        "new_classes": NEW,
        "ground_basis": basis,
        "ground_spectral_weights": np.asarray([0.5, 0.3, 0.2]),
        "ground_audit": {
            "d81_basis_sha256": "a" * 64,
            "d81_spectral_weight_sha256": "b" * 64,
            "d81_participation_ratio_effective_rank": 2.6,
            "d81_retained_rank": 3,
            "d81_rank_policy": "ceil_participation_ratio_effective_rank",
            "ground_component_input_count": 84,
            "ground_statistic_semantics": "class_centered_cross_domain_centroid_drift_eigenspectrum",
        },
    })
    overlay = {
        "manifest": {
            "receiver": "3-19",
            "k_shot": 2,
            "method_seed": 7282101,
            "capsule_id": "capsule-fixed",
            "split_id": "split-fixed",
            "predictor_package_root_sha256": "3" * 64,
            "predictor_package_seal_sha256": "4" * 64,
        },
        "old_classes": OLD,
        "new_classes": NEW,
        "scenario_payloads": payloads,
        "ground_component": ground,
    }
    return base, overlay


def test_suite_freezes_one_seed_across_all_twelve_arms() -> None:
    assert suite_runner._arm_seed_plan(7282101, M24_ARMS) == {
        arm: 7282101 for arm in M24_ARMS
    }


def test_expanded_d1_matrix_is_five_receivers_three_seeds_four_conditions() -> None:
    assert len(expanded_runner.DEFAULT_RECEIVERS) == 5
    assert len(expanded_runner.DEFAULT_SEEDS) == 3
    assert expanded_runner.DEFAULT_CONDITIONS == ((1, 20), (5, 20), (10, 20), (10, 5))
    assert len(expanded_runner.DEFAULT_RECEIVERS) * len(expanded_runner.DEFAULT_SEEDS) * len(expanded_runner.DEFAULT_CONDITIONS) == 60


def test_preflight_accepts_legacy_base_protocol_bound_by_overlay_and_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = {
        "capsule_id": "capsule-fixed",
        "split_id": "p2_min_v1-rx3-19-m7282101-k1-new20",
        "receiver": "3-19",
        "k_shot": 1,
        "method_seed": 7282101,
    }
    base = {**common, "phase2_data_status": "VALIDATED_ONCE"}
    overlay = {**common, "phase2_data_status": "VALIDATED_ONCE", "protocol_schema": "p2_min_v1"}
    scoring = {"schema": "cvs.phase2.scoring_sidecar_manifest.v2", "truth_sidecar_json": "truth.json"}
    paths = []
    for name, value in (("base", base), ("overlay", overlay), ("scoring", scoring)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda *args, **kwargs: "abc123\n")
    receipt = preflight.run_preflight(
        base_manifest=paths[0],
        overlay_manifest=paths[1],
        scoring_manifest=paths[2],
        output_root=tmp_path / "absent-output",
        expected_commit="abc123",
        repository=tmp_path,
    )
    assert receipt["status"] == "PASS"


def test_d1_row_publishes_truth_unopened_four_state_and_nonoverwriting_output(tmp_path: Path) -> None:
    base, overlay = _caches()
    receipt = execute_m24_row(
        arm=D1,
        row_id="synthetic_m24_d1",
        receiver="3-19",
        base_cache=base,
        overlay_cache=overlay,
        output_root=tmp_path / "row",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["query_truth_opened"] is False
    assert receipt["fit_query_rows_used"] == 0
    assert receipt["per_query_independent_all_class_argmax"] is True
    assert receipt["four_state_prediction_columns"] == {
        DA0_REG0: "identity_before",
        DA1_REG0: "candidate_before",
        DA0_REG1: "identity_after",
        DA1_REG1: "candidate_after",
    }
    assert Path(receipt["prediction"]["path"]).is_file()
    assert receipt["resource"]["persistent_update_state_bytes"] == 0
    assert receipt["resource"]["registration_timing_scope"] == "compile_only_existing_p2_a1_head"
    assert receipt["resource"]["prerequisite_p2_a1_fit_included"] is False
    assert receipt["d1_historical_parity"]["before_prediction_disagreements"] == 0
    assert "r_p99" in receipt["quantization"]["margin_normalized"]
    with pytest.raises(FileExistsError):
        execute_m24_row(
            arm=D1,
            row_id="synthetic_m24_d1_repeat",
            receiver="3-19",
            base_cache=base,
            overlay_cache=overlay,
            output_root=tmp_path / "row",
            seed=7282101,
        )


def test_d1_base_only_view_preserves_physical_blocks_without_rf_state(tmp_path: Path) -> None:
    base, _overlay = _caches()
    base["manifest"].update({
        "package_root_sha256": "3" * 64,
        "package_seal_sha256": "4" * 64,
    })
    compact = d1_overlay_from_base_cache(base)
    assert compact["manifest"]["d1_base_only"] is True
    receipt = execute_m24_row(
        arm=D1,
        row_id="synthetic_m24_d1_base_only",
        receiver="3-19",
        base_cache=base,
        overlay_cache=compact,
        output_root=tmp_path / "base_only",
        seed=7282101,
    )
    assert receipt["d1_historical_parity"]["prediction_disagreements"] == 0
    assert receipt["resource"]["deployment_state_bytes"] == 0


def test_d1_refit_does_not_consume_historical_head_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, _overlay = _caches()
    base["manifest"].update({
        "package_root_sha256": "3" * 64,
        "package_seal_sha256": "4" * 64,
    })
    compact = d1_overlay_from_base_cache(base)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("D1-REFIT consumed a historical fitted head")

    monkeypatch.setattr(
        "cvsrffi.stage2_m24_row_executor._f1_reference_head", forbidden
    )
    receipt = execute_m24_row(
        arm=D1_REFIT,
        row_id="synthetic_m24_d1_refit",
        receiver="3-19",
        base_cache=base,
        overlay_cache=compact,
        output_root=tmp_path / "refit",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["resource"]["registration_timing_scope"] == "support_to_compiled_head"
    assert receipt["resource"]["prerequisite_p2_a1_fit_included"] is True
    assert all(
        item["fresh_support_refit"] is True
        for item in receipt["scenario_audit"].values()
    )


def test_row_fails_closed_on_seed_drift(tmp_path: Path) -> None:
    base, overlay = _caches()
    with pytest.raises(M24RowExecutionError, match="row identity"):
        execute_m24_row(
            arm=D1,
            row_id="synthetic_m24_wrong_seed",
            receiver="3-19",
            base_cache=base,
            overlay_cache=overlay,
            output_root=tmp_path / "wrong",
            seed=7282102,
        )


def test_scorer_adapter_preserves_extended_m24_audits_outside_legacy_contract() -> None:
    quantization = {
        "schema": "q",
        "max_logit_abs_error": 0.2,
        "mean_logit_abs_error": 0.1,
        "argmax_flip_rate": 0.0,
        "prediction_agreement_rate": 1.0,
        "margin_normalized": {"r_p99": 0.3},
    }
    resource = {key: 0 for key in suite_scorer._SCORER_RESOURCE_KEYS}
    resource.update({
        "schema": "r",
        "candidate_peak_memory_isolated": False,
        "end_to_end_query_latency_available": False,
        "end_to_end_query_latency_ms": None,
        "batch1_head_resource": None,
        "auxiliary_state_cost_in_candidate_resource": False,
        "auxiliary_prediction_cost_in_candidate_latency": False,
        "compiled_inference_state_bytes": 123,
        "persistent_update_state_bytes": 0,
        "transient_registration_workspace_peak_bytes": 456,
    })
    scorer_quantization, scorer_resource = suite_scorer._legacy_scorer_receipts(
        quantization, resource
    )
    assert set(scorer_quantization) == set(suite_scorer._SCORER_QUANTIZATION_KEYS)
    assert set(scorer_resource) == set(suite_scorer._SCORER_RESOURCE_KEYS)
    assert quantization["margin_normalized"]["r_p99"] == 0.3
    assert resource["compiled_inference_state_bytes"] == 123
