from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from cvsrffi import stage2_m23_overlay_builder as overlay_builder
from cvsrffi.stage2_m23_overlay_builder import (
    M23OverlayBuilderError,
    compose_m23_overlay_payloads,
)
from cvsrffi.stage2_m23_row_executor import (
    DA0_REG0,
    DA0_REG1,
    DA1_REG0,
    DA1_REG1,
    M23_ARMS,
    M23_F0_FULL,
    M23_F1_IF,
    M23_F2_RF32_LOW,
    M23_F3_RF_QUALITY,
    M23_F4_RF_LITE_DIAG,
    M23_F5_RF_LITE_GATED,
    M23RowExecutionError,
    execute_m23_row,
    legacy_low_rf32,
    m23_arm_config_hash,
)
from cvsrffi.stage2_m23_rfguard import IF_DIM
from cvsrffi.stage2_m23_truth_diagnostics import (
    four_state_summary_from_predictions,
    paired_flip_summary,
)
from scripts import run_m23_rfguard_suite as suite_runner


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
OLD = tuple(f"cls_{index:032x}" for index in range(6))
NEW = tuple(f"cls_{index + 100:032x}" for index in range(5))


def _unit(rng: np.random.Generator, rows: int, dim: int) -> np.ndarray:
    value = rng.normal(size=(rows, dim))
    return (value / np.linalg.norm(value, axis=1, keepdims=True)).astype(np.float32)


def _legacy(rng: np.random.Generator, rows: int) -> np.ndarray:
    identity = _unit(rng, rows, 160)
    fft = _unit(rng, rows, 96)
    rf = _unit(rng, rows, 32)
    return np.concatenate(
        [identity / np.sqrt(17.0), 4.0 * fft / np.sqrt(34.0), 4.0 * rf / np.sqrt(34.0)],
        axis=1,
    ).astype(np.float32)


def _iq(rng: np.random.Generator, rows: int) -> np.ndarray:
    value = rng.normal(size=(rows, 512)) + 1j * rng.normal(size=(rows, 512))
    return np.stack([value.real, value.imag], axis=1).astype(np.float32)


def _inputs() -> tuple[dict, dict, dict]:
    rng = np.random.default_rng(212)
    base = {"scenario_payloads": {}}
    support = {}
    query = {}
    all_classes = OLD + NEW
    indices = np.repeat(np.arange(len(all_classes)), 2)
    ranks = np.tile(np.arange(2), len(all_classes))
    for scenario_index, scenario in enumerate(SCENARIOS):
        tokens = np.asarray(
            [f"qid_{scenario_index:02x}{index:062x}" for index in range(7)]
        )
        old_count = 2 * len(OLD)
        new_count = 2 * len(NEW)
        base["scenario_payloads"][scenario] = {
            "old_support_features": _legacy(rng, old_count),
            "old_support_labels": np.repeat(np.asarray(OLD), 2),
            "new_support_features": _legacy(rng, new_count),
            "new_support_labels": np.repeat(np.asarray(NEW), 2),
            "query_features": _legacy(rng, len(tokens)),
            "query_tokens": tokens,
        }
        support[scenario] = {
            "support_pool_rank_within_class": ranks,
            "support_pool_class_indices": indices,
            "support_pool_leo_weak_iq": _iq(rng, len(indices)),
        }
        query[scenario] = {
            "query_leo_weak_iq": _iq(rng, len(tokens)),
            "query_tokens": tokens,
        }
    return base, support, query


def test_overlay_composition_uses_same_rows_and_never_adds_query_truth() -> None:
    base, support, query = _inputs()
    payloads = compose_m23_overlay_payloads(
        base,
        support,
        query,
        old_classes=OLD,
        new_classes=NEW,
        k_shot=2,
    )
    assert set(payloads) == set(SCENARIOS)
    for scenario in SCENARIOS:
        payload = payloads[scenario]
        assert set(payload) == {
            "old_support_blocks",
            "old_support_labels",
            "old_support_quality",
            "new_support_blocks",
            "new_support_labels",
            "new_support_quality",
            "query_blocks",
            "query_tokens",
        }
        assert payload["old_support_blocks"].shape == (12, 266)
        assert payload["new_support_blocks"].shape == (10, 266)
        assert payload["query_blocks"].shape == (7, 266)
        assert "query_labels" not in payload


def test_overlay_composition_fails_on_query_token_drift() -> None:
    base, support, query = _inputs()
    query[SCENARIOS[0]]["query_tokens"] = np.array(
        query[SCENARIOS[0]]["query_tokens"], copy=True
    )
    query[SCENARIOS[0]]["query_tokens"][0] = "qid_" + "f" * 64
    with pytest.raises(M23OverlayBuilderError, match="token"):
        compose_m23_overlay_payloads(
            base,
            support,
            query,
            old_classes=OLD,
            new_classes=NEW,
            k_shot=2,
        )


def test_f0_f5_catalog_and_four_state_names_are_explicit() -> None:
    assert M23_ARMS == (
        M23_F0_FULL,
        M23_F1_IF,
        M23_F2_RF32_LOW,
        M23_F3_RF_QUALITY,
        M23_F4_RF_LITE_DIAG,
        M23_F5_RF_LITE_GATED,
    )
    assert (DA0_REG0, DA1_REG0, DA0_REG1, DA1_REG1) == (
        "DA0_REG0",
        "DA1_REG0",
        "DA0_REG1",
        "DA1_REG1",
    )
    hashes = [m23_arm_config_hash(arm) for arm in M23_ARMS]
    assert len(set(hashes)) == 6
    assert all(len(value) == 64 for value in hashes)


def test_suite_uses_one_method_seed_for_every_same_row_arm() -> None:
    plan = suite_runner._arm_seed_plan(7282101, M23_ARMS)
    assert plan == {arm: 7282101 for arm in M23_ARMS}


def test_component_binding_does_not_require_historical_outer_seal_flag() -> None:
    class Component:
        class_registry = OLD

    base = {
        "old_classes": OLD,
        "ground_audit": {"ground_component_outer_joint_seal_verified": False},
    }
    overlay_builder._validate_component_registry_binding(Component(), base)


def test_low_rf32_arm_has_independent_small_rf_energy() -> None:
    rng = np.random.default_rng(214)
    rows = _legacy(rng, 9)
    projected = legacy_low_rf32(rows, rf_weight=0.5)
    assert projected.shape == rows.shape
    assert np.allclose(np.linalg.norm(projected, axis=1), 1.0)
    identity_energy = np.linalg.norm(projected[:, :160], axis=1)
    fft_energy = np.linalg.norm(projected[:, 160:256], axis=1)
    rf_energy = np.linalg.norm(projected[:, 256:], axis=1)
    assert np.all(fft_energy > identity_energy)
    assert np.all(rf_energy < identity_energy)


def test_paired_flip_summary_reports_help_harm_error_migrations_and_strata() -> None:
    truth = np.asarray(["a", "b", "a", "c", "b", "c", "a", "b"])
    reference = np.asarray(["a", "a", "b", "c", "b", "a", "c", "b"])
    candidate = np.asarray(["a", "b", "b", "a", "b", "c", "c", "b"])
    scenarios = np.asarray([SCENARIOS[index % 3] for index in range(len(truth))])
    roles = np.asarray(
        ["target_new" if value == "b" else "target_old" for value in truth]
    )
    result = paired_flip_summary(
        reference,
        candidate,
        truth,
        scenarios=scenarios,
        true_roles=roles,
        true_classes=truth,
        receiver="3-19",
        k_shot=10,
        bootstrap_repeats=200,
        bootstrap_seed=99,
    )
    assert result["N_help"] == 2
    assert result["N_harm"] == 1
    assert sum(result["error_transition_counts"].values()) == len(truth)
    assert set(result["error_transition_counts"]) == {
        "correct_to_correct",
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_to_same_wrong",
        "wrong_to_different_wrong",
    }
    assert 0.0 <= result["mcnemar_exact_pvalue"] <= 1.0
    assert result["cluster_bootstrap"]["cluster_key"] == "true_class"
    assert set(result["by_scenario"]) == set(SCENARIOS)
    assert set(result["by_role"]) == {"target_old", "target_new"}
    assert result["receiver"] == "3-19" and result["k_shot"] == 10


def test_four_state_summary_marks_reg0_new_metrics_na_and_computes_did() -> None:
    rows = []
    truth = ("old_a", "old_b", "new_a", "new_b")
    roles = ("target_old", "target_old", "target_new", "target_new")
    columns = {
        "identity_before": ("old_a", "old_a", "old_a", "old_b"),
        "candidate_before": ("old_a", "old_b", "old_a", "old_b"),
        "identity_after": ("old_a", "old_a", "new_a", "old_b"),
        "candidate_after": ("old_a", "old_b", "new_a", "new_b"),
    }
    for index, (handle, role) in enumerate(zip(truth, roles)):
        rows.append(
            {
                "scenario": SCENARIOS[0],
                "evaluation_role": role,
                "true_class_handle": handle,
                **{name: values[index] for name, values in columns.items()},
            }
        )
    result = four_state_summary_from_predictions(rows)
    scenario = result["scenario_rows"][0]
    assert scenario["states"]["DA0_REG0"]["new_accuracy"] is None
    assert scenario["states"]["DA1_REG0"]["new_class_metric_status"] == "N/A_UNREGISTERED"
    assert scenario["states"]["DA1_REG1"]["new_accuracy"] == 1.0
    assert scenario["effects"]["difference_in_differences"]["old_accuracy"] == 0.0


def test_compact_row_executor_publishes_four_state_truth_unopened_artifact(
    tmp_path: Path,
) -> None:
    base, support, query = _inputs()
    payloads = compose_m23_overlay_payloads(
        base,
        support,
        query,
        old_classes=OLD,
        new_classes=NEW,
        k_shot=2,
    )
    rng = np.random.default_rng(219)
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
    base.update(
        {
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
            "ground_basis": np.empty((160, 0), dtype=np.float64),
            "ground_spectral_weights": np.empty(0, dtype=np.float64),
            "ground_audit": {},
        }
    )
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
    with pytest.raises(M23RowExecutionError, match="row identity"):
        execute_m23_row(
            arm=M23_F5_RF_LITE_GATED,
            row_id="synthetic_m23_wrong_seed",
            receiver="3-19",
            base_cache=base,
            overlay_cache=overlay,
            output_root=tmp_path / "wrong_seed",
            seed=7282102,
        )
    receipt = execute_m23_row(
        arm=M23_F5_RF_LITE_GATED,
        row_id="synthetic_m23_f5",
        receiver="3-19",
        base_cache=base,
        overlay_cache=overlay,
        output_root=tmp_path / "row",
        seed=7282101,
    )
    assert receipt["status"] == "PREDICTIONS_COMPLETE_TRUTH_UNOPENED"
    assert receipt["query_truth_opened"] is False
    assert receipt["four_state_prediction_columns"] == {
        DA0_REG0: "identity_before",
        DA1_REG0: "candidate_before",
        DA0_REG1: "identity_after",
        DA1_REG1: "candidate_after",
    }
    artifact = Path(receipt["prediction"]["path"])
    assert artifact.is_file() and receipt["prediction"]["readonly"] is True
    assert receipt["resource"]["query_head_mac"] == 11 * IF_DIM
