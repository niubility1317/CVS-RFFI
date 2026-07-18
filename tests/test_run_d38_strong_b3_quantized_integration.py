from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
SCRIPTS = CODE / "scripts"
for value in (CODE, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_d25_support_only_concat as runner  # noqa: E402


def _cell_manifest(
    *, receiver: str = "20-1", seed: int = 713101, new_count: int = 5
) -> tuple[dict[str, object], dict[str, object]]:
    old = [
        {"class_index": index, "class_handle": f"old-{index}"}
        for index in range(6)
    ]
    new = [
        {"class_index": 6 + index, "class_handle": f"new-{index}"}
        for index in range(new_count)
    ]
    common: dict[str, object] = {
        "receiver": receiver,
        "seed": seed,
        "k_shot": 10,
    }
    return (
        {**common, "registered_classes": old},
        {**common, "registered_classes": old + new},
    )


def test_d38_development_cell_is_fail_closed_before_support_open() -> None:
    runner._require_d38_development_cell(*_cell_manifest())
    for manifests in (
        _cell_manifest(receiver="3-19"),
        _cell_manifest(seed=713102),
        _cell_manifest(new_count=2),
        _cell_manifest(new_count=10),
    ):
        with pytest.raises(runner.D25RunnerError):
            runner._require_d38_development_cell(*manifests)


def _support_blocks() -> tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(380718)
    classes = ("old_left", "old_right", "new_left", "new_right")
    labels: list[str] = []
    ranks: list[int] = []
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    direct_rows: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        z_center = np.zeros(160, dtype=np.float32)
        fft_center = np.zeros(96, dtype=np.float32)
        rf_center = np.zeros(32, dtype=np.float32)
        z_center[class_index] = 1.0
        fft_center[class_index] = 1.0
        rf_center[class_index] = 1.0
        for rank in range(10):
            z_rows.append(z_center + 0.003 * rng.normal(size=160).astype(np.float32))
            fft_rows.append(
                fft_center + 0.003 * rng.normal(size=96).astype(np.float32)
            )
            rf_rows.append(rf_center + 0.003 * rng.normal(size=32).astype(np.float32))
            direct_rows.append(
                np.asarray(
                    [2.0, -2.0] if class_index % 2 == 0 else [-2.0, 2.0],
                    dtype=np.float32,
                )
            )
            labels.append(label)
            ranks.append(rank)
    return (
        {
            "labels": np.asarray(labels),
            "ranks": np.asarray(ranks, dtype=np.int64),
            "tokens": np.asarray(
                [f"physical-{index:03d}" for index in range(len(labels))]
            ),
        },
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
        np.asarray(direct_rows, dtype=np.float32),
    )


def test_d38_candidate_lock_is_exact_six_by_three_by_five() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D38_V1)
    assert tuple(candidates) == runner.D38_CANDIDATES
    assert runner.D38_CANDIDATES == (
        runner.IDENTITY_CANDIDATE,
        runner.D38_PROTONET_CDA,
        runner.DIAG_CANDIDATE,
        runner.D38_A_INT8,
        runner.D38_B_INT8,
        runner.D38_B_FP32,
    )
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D38_V1) == (
        runner.D38_B_INT8,
    )
    assert len(candidates) * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 90
    assert candidates[runner.D38_A_INT8].core.stage2c_steps == 0
    assert candidates[runner.D38_A_INT8].deploy_precision == "int8"
    assert candidates[runner.D38_B_INT8].core.stage2c_steps == 10
    assert candidates[runner.D38_B_INT8].deploy_precision == "int8"
    assert candidates[runner.D38_B_FP32].core.stage2c_steps == 10
    assert candidates[runner.D38_B_FP32].deploy_precision == "fp32"
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D38_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v16"
    assert lock["selection_baseline"] == runner.IDENTITY_CANDIDATE
    assert "d38_strong_b3_quantized_core_sha256" in lock["source_closure"]
    assert sum(row["eligible_positive_route"] for row in lock["candidates"]) == 1


def test_d38_feature_geometry_matches_locked_joint_auxiliary_normalization() -> None:
    rng = np.random.default_rng(38)
    z_rows = rng.normal(size=(3, 160)).astype(np.float32)
    fft_rows = rng.normal(size=(3, 96)).astype(np.float32)
    rf_rows = rng.normal(size=(3, 32)).astype(np.float32)
    observed = runner._d1_feature_from_blocks(z_rows, fft_rows, rf_rows)
    z_unit = runner.legacy._normalize_matrix(z_rows)
    auxiliary_unit = runner.legacy._normalize_matrix(
        np.concatenate([fft_rows, rf_rows], axis=1)
    )
    expected = runner.legacy._normalize_matrix(
        np.concatenate([z_unit, np.float32(4.0) * auxiliary_unit], axis=1)
    )

    assert np.array_equal(observed, expected)
    assert np.allclose(np.sum(observed[:, :160] ** 2, axis=1), 1.0 / 17.0)
    assert np.allclose(np.sum(observed[:, 160:] ** 2, axis=1), 16.0 / 17.0)


def test_d38_cli_has_no_query_truth_role_quota_source_or_clean_surface() -> None:
    parser = runner.build_parser()
    action = next(item for item in parser._actions if item.dest == "candidate_set")
    assert runner.CANDIDATE_SET_D38_V1 in action.choices
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    surfaces = (
        {item.dest.lower() for item in parser._actions},
        {name.lower() for name in inspect.signature(runner.run).parameters},
    )
    for names in surfaces:
        assert not any(token in name for name in names for token in forbidden)


def test_d38_outer_fold_emits_pairwise_matched_precision_and_direct_anchor() -> None:
    rows, z_rows, fft_rows, rf_rows, direct_logits = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D38_V1)[
        runner.D38_B_INT8
    ]
    result = runner._evaluate_d38_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        held_ranks=(0, 1),
        candidate_id=runner.D38_B_INT8,
        config=config,
        seed=713101,
        device="cpu",
        scenario="leo_clear_weak",
        outer_fold=0,
    )
    pairwise = result["pairwise_support_diagnostics"]
    assert len(pairwise) == 4
    assert all(
        {
            "scenario",
            "outer_fold",
            "physical_rank",
            "physical_sample_id",
            "true_new_handle",
            "top_competing_new_handle",
            "true_new_score",
            "top_competing_new_score",
            "new_new_margin",
            "top_old_score",
            "new_old_margin",
        }
        <= set(row)
        for row in pairwise
    )
    assert all(row["scenario"] == "leo_clear_weak" for row in pairwise)
    assert result["fit_k_shot"] == 8
    assert result["deployment_precision"] == "int8"
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["resource"]["adaptation_epochs"] == 30
    assert result["resource"]["total_optimizer_steps"] == 30
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["resource"]["dense_query_graph_bytes"] == 0
    assert result["resource"]["persistent_state_cap_pass"] is True
    assert result["matched_fp32_outer_argmax_change_count"] >= 0

    anchor = runner._d38_direct_old_anchor(
        rows,
        direct_logits,
        old_classes=("old_left", "old_right"),
        held_ranks=(0, 1),
    )
    assert anchor["candidate_row"] is False
    assert anchor["support_rows_used"] == 0
    assert anchor["old_only"] is True
    assert anchor["query_rows_used"] == 0


def _metric(value: float, names: tuple[str, ...]) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {name: value for name in names},
    }


def _candidate_rows(
    candidate_id: str,
    *,
    value: float,
    margin: float,
    confusions: int,
    int8: bool,
) -> list[dict[str, object]]:
    old_names = ("old_alpha", "old_beta")
    new_names = ("new_alpha", "new_beta")
    rows: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for fold_index, _ in enumerate(runner.HELD_RANKS):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "fold_index": fold_index,
                    "before_old": _metric(value, old_names),
                    "after_old": _metric(value, old_names),
                    "after_new": _metric(value, new_names),
                    "H_old_new": value,
                    "forgetting": 0.0,
                    "joint_floor": value,
                    "old_score_columns_bitwise_unchanged": True,
                    "outer_held_new_intrusion_count": 0,
                    "new_new_margin_mean": margin,
                    "new_new_confusion_count": confusions,
                    "matched_fp32_outer_argmax_change_count": 0,
                    "target_old_int8_prototypes_used_for_prediction": int8,
                    "target_new_int8_prototypes_used_for_prediction": int8,
                    "resource": {
                        "peak_trainable_parameters": 1_000,
                        "adaptation_epochs": 30,
                        "total_optimizer_steps": 30,
                        "persistent_state_cap_pass": True,
                        "dense_query_graph_bytes": 0,
                        "query_rows_used_for_fit": 0,
                    },
                }
            )
    return rows


def test_d38_selector_is_global_matched_and_only_b_int8_can_promote() -> None:
    values = {
        runner.IDENTITY_CANDIDATE: (0.55, 0.0, 2, False),
        runner.D38_PROTONET_CDA: (0.55, 0.0, 2, False),
        runner.DIAG_CANDIDATE: (0.70, 0.1, 1, False),
        runner.D38_A_INT8: (0.75, 0.2, 1, True),
        runner.D38_B_INT8: (0.90, 0.6, 0, True),
        runner.D38_B_FP32: (0.90, 0.6, 0, False),
    }
    folds = {
        candidate_id: _candidate_rows(
            candidate_id,
            value=value,
            margin=margin,
            confusions=confusions,
            int8=int8,
        )
        for candidate_id, (value, margin, confusions, int8) in values.items()
    }
    selected, decisions = runner._select_d38_candidate(folds)
    assert selected == runner.D38_B_INT8
    selected_decision = next(
        row for row in decisions if row["candidate_id"] == runner.D38_B_INT8
    )
    assert selected_decision["d38_b_improves_a_gate_pass"] is True
    assert selected_decision["d38_matched_comparator_gate_pass"] is True
    assert selected_decision["d38_int8_fp32_outer_argmax_invariance_gate_pass"] is True
    assert [row["candidate_id"] for row in decisions if row["eligible_positive_route"]] == [
        runner.D38_B_INT8
    ]

    folds[runner.D38_B_INT8][0]["matched_fp32_outer_argmax_change_count"] = 1
    selected, _ = runner._select_d38_candidate(folds)
    assert selected == runner.IDENTITY_CANDIDATE


def test_d38_full_k10_audit_and_gate_close_precision_resource_protocol() -> None:
    rows, z_rows, fft_rows, rf_rows, _ = _support_blocks()
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D38_V1)[
        runner.D38_B_INT8
    ]
    resource, geometry = runner._full_d38_state_audit(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_left", "old_right"),
        new_classes=("new_left", "new_right"),
        config=config,
        seed=713101,
        device="cpu",
        scenario="leo_clear_weak",
    )
    assert resource["deployment_k_shot"] == 10
    assert resource["full_k10_refit_only_no_candidate_change"] is True
    assert resource["pairwise_support_diagnostic_row_count"] == 20
    assert resource["target_old_int8_prototypes_used_for_prediction"] is True
    assert resource["target_new_int8_prototypes_used_for_prediction"] is True
    assert resource["resident_fp32_target_prototype_count"] == 0
    assert resource["old_prefix_bitwise_unchanged"] is True
    assert resource["query_rows_used_for_fit"] == 0
    assert geometry["query_rows_used"] == 0

    decisions = [
        {
            "candidate_id": candidate_id,
            "eligible_positive_route": candidate_id == runner.D38_B_INT8,
        }
        for candidate_id in runner.D38_CANDIDATES
    ]
    resources = {
        runner.D38_B_INT8: {
            scenario: dict(resource)
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
    }
    selected, reason = runner._apply_full_k10_d38_gate(
        runner.D38_B_INT8, decisions, resources
    )
    assert selected == runner.D38_B_INT8
    assert reason is None
    resources[runner.D38_B_INT8]["leo_rain_weak"][
        "matched_fp32_full_k10_argmax_change_count"
    ] = 1
    selected, reason = runner._apply_full_k10_d38_gate(
        runner.D38_B_INT8, decisions, resources
    )
    assert selected == runner.IDENTITY_CANDIDATE
    assert reason == "FULL_K10_D38_PRECISION_RESOURCE_OR_PROTOCOL_GATE_FAILED"
