from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
SCRIPTS = CODE / "scripts"
for value in (CODE, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_d25_support_only_concat as runner  # noqa: E402
from cvsrffi.stage2_ciaf import Int8DomainClassComponent  # noqa: E402
from cvsrffi.stage2_multimodal_compact_diag import (  # noqa: E402
    D26CompactDiagConfig,
)


def test_d26_candidate_lock_matrix_and_historical_sets_are_stable() -> None:
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D26_V1)
    assert tuple(candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D26_A,
        runner.D26_B,
        runner.D26_C,
    )
    assert [
        (candidates[name].stage2b_steps, candidates[name].stage2c_steps)
        for name in runner.D26_CANDIDATES
    ] == [(15, 0), (15, 10), (15, 15)]
    assert 6 * len(runner.legacy.FORMAL_LEO_WEAK_SCENARIOS) * len(
        runner.HELD_RANKS
    ) == 90
    lock = runner._candidate_lock(candidates, runner.CANDIDATE_SET_D26_V1)
    assert lock["schema"] == "cvs.phase2.d25.candidate_lock.v3"
    assert lock["candidate_set"] == runner.CANDIDATE_SET_D26_V1
    assert lock["selection_baseline"] == runner.D25_C0
    assert lock["d26_core_git_commit"] == runner.D26_CORE_GIT_COMMIT
    assert "d26_compact_diag_core_sha256" in lock["source_closure"]
    assert "diag_cosine_feature_operator_sha256" in lock["source_closure"]
    assert "d25_c3_core_sha256" not in lock["source_closure"]
    assert all(
        candidates[name].bias_guard_mode == "joint_bias0"
        for name in runner.D26_CANDIDATES
    )

    strict_candidates = runner.preregistered_candidates(
        runner.CANDIDATE_SET_D26_V2
    )
    assert tuple(strict_candidates) == tuple(candidates)
    assert all(
        strict_candidates[name].bias_guard_mode
        == "pre_registration_old_only"
        for name in runner.D26_CANDIDATES
    )
    assert all(
        min(strict_candidates[name].new_group_bias_grid) == -12.0
        for name in runner.D26_CANDIDATES
    )
    strict_lock = runner._candidate_lock(
        strict_candidates, runner.CANDIDATE_SET_D26_V2
    )
    assert strict_lock["schema"] == "cvs.phase2.d25.candidate_lock.v4"
    assert strict_lock["candidate_set"] == runner.CANDIDATE_SET_D26_V2

    d27_candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D27_V1)
    assert tuple(d27_candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D27_A,
        runner.D27_B,
        runner.D27_C,
    )
    assert [
        (d27_candidates[name].stage2b_steps, d27_candidates[name].stage2c_steps)
        for name in runner.D27_CANDIDATES
    ] == [(15, 0), (15, 10), (15, 15)]
    assert all(
        d27_candidates[name].bias_guard_mode
        == "per_new_class_pre_registration_old_only"
        for name in runner.D27_CANDIDATES
    )
    assert all(
        d27_candidates[name].new_class_bias_offsets
        == (0.0, -0.5, -1.0, -2.0, -4.0)
        for name in runner.D27_CANDIDATES
    )
    d27_lock = runner._candidate_lock(
        d27_candidates, runner.CANDIDATE_SET_D27_V1
    )
    assert d27_lock["schema"] == "cvs.phase2.d25.candidate_lock.v5"
    assert d27_lock["candidate_set"] == runner.CANDIDATE_SET_D27_V1
    assert d27_lock["selection_baseline"] == runner.D25_C0
    assert all(
        row["family"] == "d27_per_new_class_bias"
        for row in d27_lock["candidates"]
        if row["candidate_id"] in runner.D27_CANDIDATES
    )

    d28_candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D28_V1)
    assert tuple(d28_candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D28_A,
        runner.D28_B,
        runner.D28_C,
    )
    assert isinstance(d28_candidates[runner.D28_A], D26CompactDiagConfig)
    assert d28_candidates[runner.D28_A].stage2c_steps == 10
    assert d28_candidates[runner.D28_B].gate.delta == 1.0
    assert d28_candidates[runner.D28_C].gate.delta == 2.0
    d28_lock = runner._candidate_lock(
        d28_candidates, runner.CANDIDATE_SET_D28_V1
    )
    assert d28_lock["schema"] == "cvs.phase2.d25.candidate_lock.v6"
    assert d28_lock["candidate_set"] == runner.CANDIDATE_SET_D28_V1
    assert "d28_support_evidence_gate_core_sha256" in d28_lock["source_closure"]
    assert all(
        row["family"] in (
            "d27_per_new_class_bias",
            "d28_support_evidence_gate",
        )
        for row in d28_lock["candidates"]
        if row["candidate_id"] in runner.D28_CANDIDATES
    )

    d29_candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D29_V1)
    assert tuple(d29_candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D29_A,
        runner.D29_B,
        runner.D29_C,
    )
    assert d29_candidates[runner.D29_A].release.safety_budget == 0.25
    assert d29_candidates[runner.D29_B].release.objective == "balance_first"
    assert d29_candidates[runner.D29_C].release.safety_budget == 1.0
    d29_lock = runner._candidate_lock(
        d29_candidates, runner.CANDIDATE_SET_D29_V1
    )
    assert d29_lock["schema"] == "cvs.phase2.d25.candidate_lock.v7"
    assert d29_lock["candidate_set"] == runner.CANDIDATE_SET_D29_V1
    assert "d29_classwise_safe_release_core_sha256" in d29_lock["source_closure"]
    assert all(
        row["family"] == "d29_per_class_safe_release"
        for row in d29_lock["candidates"]
        if row["candidate_id"] in runner.D29_CANDIDATES
    )
    d29_config = next(
        row["config"]
        for row in d29_lock["candidates"]
        if row["candidate_id"] == runner.D29_B
    )
    assert d29_config["base"]["learning_rate"] > 0.0
    assert "new_group_bias_grid" in d29_config["base"]

    d30_candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D30_V1)
    assert tuple(d30_candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D30_A,
        runner.D30_B,
        runner.D30_C,
    )
    assert [
        d30_candidates[name].dali.ground_weight
        for name in runner.D30_CANDIDATES
    ] == [0.025, 0.05, 0.10]
    assert [
        d30_candidates[name].envelope_objective
        for name in runner.D30_CANDIDATES
    ] == ["overall_first", "balance_first", "floor_first"]
    d30_lock = runner._candidate_lock(
        d30_candidates, runner.CANDIDATE_SET_D30_V1
    )
    assert d30_lock["schema"] == "cvs.phase2.d25.candidate_lock.v8"
    assert d30_lock["candidate_set"] == runner.CANDIDATE_SET_D30_V1
    assert "d20_dali_core_sha256" in d30_lock["source_closure"]
    assert "d30_max_envelope_core_sha256" in d30_lock["source_closure"]
    assert all(
        row["family"] == "d30_b3_dali_dual_envelope"
        for row in d30_lock["candidates"]
        if row["candidate_id"] in runner.D30_CANDIDATES
    )

    d31_candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D31_V1)
    assert tuple(d31_candidates) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D31_A,
        runner.D31_B,
        runner.D31_C,
    )
    assert [
        d31_candidates[name].stage2c.method_id
        for name in runner.D31_CANDIDATES
    ] == list(runner.D31_CANDIDATES)
    assert all(
        d31_candidates[name].base.stage2b_steps == 15
        and d31_candidates[name].base.stage2c_steps == 0
        for name in runner.D31_CANDIDATES
    )
    d31_lock = runner._candidate_lock(
        d31_candidates, runner.CANDIDATE_SET_D31_V1
    )
    assert d31_lock["schema"] == "cvs.phase2.d25.candidate_lock.v9"
    assert runner._positive_route_candidates(runner.CANDIDATE_SET_D31_V1) == (
        runner.D31_CANDIDATES
    )
    assert runner.D25_C0 not in runner._positive_route_candidates(
        runner.CANDIDATE_SET_D31_V1
    )
    assert "d31_all_registered_suffix_core_sha256" in d31_lock["source_closure"]
    assert all(
        row["family"] == "d31_all_registered_suffix_with_dali"
        for row in d31_lock["candidates"]
        if row["candidate_id"] in runner.D31_CANDIDATES
    )

    assert tuple(runner.preregistered_candidates()) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.D25_C1,
        runner.D25_C2,
    )
    assert tuple(runner.preregistered_candidates(runner.CANDIDATE_SET_C3_V1)) == (
        runner.IDENTITY_CANDIDATE,
        runner.DIAG_CANDIDATE,
        runner.D25_C0,
        runner.C3_A,
        runner.C3_B,
        runner.C3_C,
    )
    assert runner._candidate_lock(runner.preregistered_candidates())["schema"] == (
        "cvs.phase2.d25.candidate_lock.v1"
    )
    assert runner._candidate_lock(
        runner.preregistered_candidates(runner.CANDIDATE_SET_C3_V1),
        runner.CANDIDATE_SET_C3_V1,
    )["schema"] == "cvs.phase2.d25.candidate_lock.v2"


def test_d26_cli_has_no_query_source_or_clean_surface() -> None:
    parser = runner.build_parser()
    candidate_action = next(
        action for action in parser._actions if action.dest == "candidate_set"
    )
    assert candidate_action.default == runner.CANDIDATE_SET_D25_V4
    assert tuple(candidate_action.choices) == (
        runner.CANDIDATE_SET_D25_V4,
        runner.CANDIDATE_SET_C3_V1,
        runner.CANDIDATE_SET_D26_V1,
        runner.CANDIDATE_SET_D26_V2,
        runner.CANDIDATE_SET_D27_V1,
        runner.CANDIDATE_SET_D28_V1,
        runner.CANDIDATE_SET_D29_V1,
        runner.CANDIDATE_SET_D30_V1,
        runner.CANDIDATE_SET_D31_V1,
    )
    forbidden = ("query", "truth", "scorer", "role", "quota", "source", "clean")
    destinations = {action.dest.lower() for action in parser._actions}
    parameters = {name.lower() for name in inspect.signature(runner.run).parameters}
    for names in (destinations, parameters):
        assert not any(token in name for name in names for token in forbidden)


def _synthetic_blocks() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(260718)
    classes = tuple(f"old_{index}" for index in range(3)) + tuple(
        f"new_{index}" for index in range(2)
    )
    labels: list[str] = []
    ranks: list[int] = []
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        z_center = rng.normal(size=160)
        fft_center = rng.normal(size=96)
        rf_center = rng.normal(size=32)
        for rank in range(10):
            labels.append(label)
            ranks.append(rank)
            z_rows.append(z_center + 0.02 * rng.normal(size=160))
            fft_rows.append(fft_center + 0.02 * rng.normal(size=96))
            rf_rows.append(rf_center + 0.02 * rng.normal(size=32))
    return (
        {"labels": np.asarray(labels), "ranks": np.asarray(ranks, dtype=np.int64)},
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
    )


def _synthetic_d28_blocks(
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(280718)
    classes = tuple(f"old_{index}" for index in range(3)) + tuple(
        f"new_{index}" for index in range(3)
    )
    labels: list[str] = []
    ranks: list[int] = []
    z_rows: list[np.ndarray] = []
    fft_rows: list[np.ndarray] = []
    rf_rows: list[np.ndarray] = []
    for class_index, label in enumerate(classes):
        # Keep group evidence learnable while retaining class-specific identity.
        group_shift = -0.25 if class_index < 3 else 0.25
        z_center = rng.normal(size=160) + group_shift
        fft_center = rng.normal(size=96) + group_shift
        rf_center = rng.normal(size=32) + group_shift
        for rank in range(10):
            labels.append(label)
            ranks.append(rank)
            z_rows.append(z_center + 0.02 * rng.normal(size=160))
            fft_rows.append(fft_center + 0.02 * rng.normal(size=96))
            rf_rows.append(rf_center + 0.02 * rng.normal(size=32))
    return (
        {"labels": np.asarray(labels), "ranks": np.asarray(ranks, dtype=np.int64)},
        np.asarray(z_rows, dtype=np.float32),
        np.asarray(fft_rows, dtype=np.float32),
        np.asarray(rf_rows, dtype=np.float32),
    )


def test_real_d26_fold_freezes_old_scores_and_logs_support_only_bias() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_blocks()
    result = runner._evaluate_d26_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        held_ranks=(0, 1),
        candidate_id=runner.D26_A,
        config=D26CompactDiagConfig(stage2b_steps=1, stage2c_steps=0),
    )
    assert result["old_prefix_sha256_before"] == result["old_prefix_sha256_after"]
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["fit_k_shot"] == 8
    assert result["resource"]["total_optimizer_steps"] == 1
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["new_group_bias_support_only_audit"]["query_rows_used"] == 0
    assert result["geometry_summary"]["bias_applied_to_new_suffix_only"] is True
    assert len(result["training_trace"]) == 3  # B step0+1, C closed-form step0.


def test_full_d26_resource_and_geometry_are_deployment_bounded() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_blocks()
    resource, geometry = runner._full_d26_state_audit(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        config=D26CompactDiagConfig(stage2b_steps=1, stage2c_steps=0),
    )

    assert resource["total_optimizer_steps"] == 1
    assert resource["total_adaptation_epochs"] == 1
    assert resource["peak_trainable_parameters"] == 4 * 288
    assert resource["persistent_state_cap_pass"] is True
    assert resource["trainable_parameter_cap_pass"] is True
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["estimated_score_mac_ratio_vs_identity_single_qknn"] < 1.0
    assert resource["batch1_head_latency_sample_count"] == len(rows["labels"])
    assert len(resource["complete_loss_trace"]) == 3
    assert geometry["bias_applied_to_new_suffix_only"] is True
    assert "weights" not in geometry
    assert "features" not in geometry
    assert "prototypes" not in geometry


def test_real_d27_fold_records_per_new_class_biases_and_old_guard() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_blocks()
    result = runner._evaluate_d26_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1"),
        held_ranks=(0, 1),
        candidate_id=runner.D27_A,
        config=D26CompactDiagConfig(
            stage2b_steps=1,
            stage2c_steps=0,
            bias_guard_mode="per_new_class_pre_registration_old_only",
        ),
    )
    assert result["old_support_non_degradation_pass"] is True
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert len(result["new_class_biases"]) == 2
    audit = result["new_group_bias_support_only_audit"]
    assert audit["bias_guard_mode"] == (
        "per_new_class_pre_registration_old_only"
    )
    assert len(audit["selected_biases"]) == 2
    assert result["resource"]["new_class_bias_scalar_count"] == 2


def test_real_d28_fold_is_row_local_support_only_and_resource_bounded() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_d28_blocks()
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D28_V1)
    result = runner._evaluate_d28_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1", "new_2"),
        held_ranks=(0, 1),
        candidate_id=runner.D28_B,
        config=candidates[runner.D28_B],
    )
    assert result["fit_k_shot"] == 8
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["resource"]["total_optimizer_steps"] == 25
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["resource"]["query_batch_global_assignment"] is False
    assert result["resource"]["dense_query_graph_bytes"] == 0
    assert result["resource"]["persistent_state_cap_pass"] is True
    assert result["resource"]["active_adaptation_parameter_count"] <= 80_000
    assert result["gate_fit_audit"]["query_rows_used"] == 0


def test_real_d29_fold_is_row_local_support_only_and_resource_bounded() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_d28_blocks()
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D29_V1)
    result = runner._evaluate_d29_fold(
        rows,
        z_rows,
        fft_rows,
        rf_rows,
        old_classes=("old_0", "old_1", "old_2"),
        new_classes=("new_0", "new_1", "new_2"),
        held_ranks=(0, 1),
        candidate_id=runner.D29_B,
        config=candidates[runner.D29_B],
    )
    assert result["fit_k_shot"] == 8
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["resource"]["total_optimizer_steps"] == 25
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["resource"]["query_batch_global_assignment"] is False
    assert result["resource"]["dense_query_graph_bytes"] == 0
    assert result["resource"]["persistent_state_cap_pass"] is True
    assert result["resource"]["active_adaptation_parameter_count"] <= 80_000
    assert result["release_fit_audit"]["query_rows_used"] == 0
    assert result["resource"]["estimated_row_local_scalar_ops_per_query"] <= 12


def _synthetic_int8_component(
    old_classes: tuple[str, ...],
) -> Int8DomainClassComponent:
    q = np.zeros((3, len(old_classes), 160), dtype=np.int8)
    for domain_index in range(3):
        for class_index in range(len(old_classes)):
            q[domain_index, class_index, class_index] = 127
            q[domain_index, class_index, 20 + domain_index] = domain_index + 1
    return Int8DomainClassComponent(
        q,
        np.full((3, len(old_classes)), 1.0 / 127.0, dtype=np.float16),
        np.ones((3, len(old_classes)), dtype=np.uint8),
        old_classes,
    )


def test_real_d30_fold_uses_b3_geometry_dual_envelopes_and_int8_audit() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_d28_blocks()
    old_classes = ("old_0", "old_1", "old_2")
    new_classes = ("new_0", "new_1", "new_2")
    candidates = runner.preregistered_candidates(runner.CANDIDATE_SET_D30_V1)
    result = runner._evaluate_d30_fold(
        _synthetic_int8_component(old_classes),
        rows,
        z_rows,
        np.zeros((len(z_rows), len(old_classes)), dtype=np.float32),
        fft_rows,
        rf_rows,
        old_classes=old_classes,
        new_classes=new_classes,
        held_ranks=(0, 1),
        candidate_id=runner.D30_B,
        config=candidates[runner.D30_B],
    )
    assert result["fit_k_shot"] == 8
    assert result["old_score_columns_bitwise_unchanged"] is True
    assert result["resource"]["total_optimizer_steps"] == 25
    assert result["resource"]["query_rows_used_for_fit"] == 0
    assert result["resource"]["query_batch_global_assignment"] is False
    assert result["resource"]["dense_query_graph_bytes"] == 0
    assert result["resource"]["persistent_state_cap_pass"] is True
    assert result["resource"]["active_adaptation_parameter_count"] <= 80_000
    assert result["resource"]["feature_geometry"] == (
        "b3_auxiliary_dominant_z160_fft96_rf32_v1"
    )
    assert result["resource"]["int8_component_loaded_and_audited"] is True
    assert result["resource"]["int8_component_state_bytes"] > 0
    assert result["dali_old_support_gate"]["held_rows_used"] == 0
    assert result["max_envelope_fit_audit"]["query_rows_used"] == 0
    before_confusion = result["new_confusion_before_envelope"]
    after_confusion = result["new_confusion_after_envelope"]
    assert before_confusion["new_aggregate"]["old_win"] == (
        after_confusion["new_aggregate"]["old_win"]
    )


def test_d30_k1_forces_exact_base_head_passthrough() -> None:
    assert runner._d30_enable_dali(1, True) is False
    assert runner._d30_enable_dali(5, False) is False
    assert runner._d30_enable_dali(5, True) is True


def test_full_d30_resource_is_bounded_and_records_dual_confusions() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_d28_blocks()
    old_classes = ("old_0", "old_1", "old_2")
    new_classes = ("new_0", "new_1", "new_2")
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D30_V1)[
        runner.D30_B
    ]
    resource, geometry = runner._full_d30_state_audit(
        _synthetic_int8_component(old_classes),
        rows,
        z_rows,
        np.zeros((len(z_rows), len(old_classes)), dtype=np.float32),
        fft_rows,
        rf_rows,
        old_classes=old_classes,
        new_classes=new_classes,
        config=config,
    )
    assert resource["total_optimizer_steps"] == 25
    assert resource["total_adaptation_epochs"] == 25
    assert resource["persistent_state_cap_pass"] is True
    assert resource["active_adaptation_parameter_count"] <= 80_000
    assert resource["int8_component_loaded_and_audited"] is True
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["estimated_score_mac_ratio_vs_identity_single_qknn"] < 1.0
    before = geometry["support_confusion_before_envelope"]
    after = geometry["support_confusion_after_envelope"]
    assert before["new_aggregate"]["old_win"] == after["new_aggregate"]["old_win"]


def test_real_d31_fold_uses_all_registered_suffix_and_dual_state_accounting() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_d28_blocks()
    old_classes = ("old_0", "old_1", "old_2")
    new_classes = ("new_0", "new_1", "new_2")
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D31_V1)[
        runner.D31_B
    ]
    result = runner._evaluate_d31_fold(
        _synthetic_int8_component(old_classes),
        rows,
        z_rows,
        np.zeros((len(z_rows), len(old_classes)), dtype=np.float32),
        fft_rows,
        rf_rows,
        old_classes=old_classes,
        new_classes=new_classes,
        held_ranks=(0, 1),
        candidate_id=runner.D31_B,
        config=config,
    )
    resource = result["resource"]
    assert resource["total_optimizer_steps"] == 25
    assert resource["peak_trainable_parameters"] <= 2_016
    assert resource["query_rows_used_for_fit"] == 0
    assert resource["dense_query_graph_bytes"] == 0
    assert resource["authorized_full_bundle_state_bytes"] > 0
    assert resource["actual_current_dali_state_bytes"] > (
        resource["projected_slim_dali_runtime_bytes"]
    )
    old_count = len(old_classes)
    assert resource["selected_medoid_int8_anchor_bytes"] == old_count * 160
    assert resource["selected_medoid_fp32_scale_bytes"] == old_count * 4
    assert resource["selected_medoid_fp32_radius_bytes"] == old_count * 4
    assert resource["selected_medoid_u16_column_index_bytes"] == old_count * 2
    assert resource["selected_medoid_class_digest_bytes"] == old_count * 32
    assert resource["selected_medoid_header_hash_bytes"] == 128
    assert resource["selected_medoid_int8_view_bytes"] == (
        old_count * (160 + 4 + 4 + 2 + 32) + 128
    )
    assert resource["slim_runtime_projection_only"] is True
    assert result["raw_confusion"]["sample_count"] == result["final_confusion"][
        "sample_count"
    ]
    assert len(result["training_trace"]) == 27  # B: 0..15, C: 0..10.


def test_full_d31_resource_is_bounded_and_keeps_bundle_residency_explicit() -> None:
    rows, z_rows, fft_rows, rf_rows = _synthetic_d28_blocks()
    old_classes = ("old_0", "old_1", "old_2")
    new_classes = ("new_0", "new_1", "new_2")
    config = runner.preregistered_candidates(runner.CANDIDATE_SET_D31_V1)[
        runner.D31_C
    ]
    resource, geometry = runner._full_d31_state_audit(
        _synthetic_int8_component(old_classes),
        rows,
        z_rows,
        np.zeros((len(z_rows), len(old_classes)), dtype=np.float32),
        fft_rows,
        rf_rows,
        old_classes=old_classes,
        new_classes=new_classes,
        config=config,
    )
    assert resource["total_optimizer_steps"] == 30
    assert resource["estimated_adaptation_macs"] > (
        resource["estimated_stage2c_adaptation_macs"]
    )
    assert resource["batch1_head_latency_mean_ms"] > 0.0
    assert resource["batch1_head_latency_p95_ms"] > 0.0
    assert resource["batch1_head_latency_sample_count"] == len(rows["labels"])
    assert resource["persistent_state_cap_pass"] is True
    assert resource["full_bundle_resident_combined_state_bytes"] > (
        resource["projected_slim_active_predictor_state_bytes"]
    )
    assert resource["full_authorized_bundle_must_remain_resident_or_sealed_accessible"] is True
    assert resource["estimated_score_mac_ratio_vs_identity_single_qknn"] < 1.0
    assert "old_to_new" in geometry["final_confusion"]
    assert "new_to_old" in geometry["final_confusion"]
    assert "new_to_wrong_new" in geometry["final_confusion"]


def _metric(value: float, labels: tuple[str, ...]) -> dict[str, object]:
    return {
        "overall_accuracy": value,
        "class_floor_accuracy": value,
        "per_class_accuracy": {label: value for label in labels},
    }


def _candidate_rows(
    candidate_id: str, value: float, *, support_pass: bool, steps: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS:
        for _ in runner.HELD_RANKS:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "before_old": _metric(value, ("old-a", "old-b")),
                    "after_old": _metric(value, ("old-a", "old-b")),
                    "after_new": _metric(value, ("new-a", "new-b")),
                    "H_old_new": value,
                    "forgetting": 0.0,
                    "joint_floor": value,
                    "old_score_columns_bitwise_unchanged": True,
                    "old_support_non_degradation_pass": support_pass,
                    "resource": {"total_optimizer_steps": steps},
                }
            )
    return rows


def test_d26_selector_uses_c0_gates_and_b3_as_reference_only() -> None:
    folds = {
        runner.IDENTITY_CANDIDATE: _candidate_rows(
            runner.IDENTITY_CANDIDATE, 0.40, support_pass=True, steps=0
        ),
        runner.DIAG_CANDIDATE: _candidate_rows(
            runner.DIAG_CANDIDATE, 0.70, support_pass=True, steps=60
        ),
        runner.D25_C0: _candidate_rows(
            runner.D25_C0, 0.50, support_pass=True, steps=0
        ),
        runner.D26_A: _candidate_rows(runner.D26_A, 0.65, support_pass=True, steps=15),
        runner.D26_B: _candidate_rows(runner.D26_B, 0.65, support_pass=True, steps=25),
        runner.D26_C: _candidate_rows(runner.D26_C, 0.65, support_pass=True, steps=30),
    }
    selected, decisions = runner._select_d26_candidate(folds)
    assert selected == runner.D26_A
    d26 = next(row for row in decisions if row["candidate_id"] == runner.D26_A)
    b3 = next(row for row in decisions if row["candidate_id"] == runner.DIAG_CANDIDATE)
    assert d26["eligible_positive_route"] is True
    assert d26["B3_performance_reference_only"] is True
    assert d26["mean_H_delta_vs_B3"] < 0.0
    assert b3["diagnostic_only"] is True
    assert b3["eligible_positive_route"] is False


def test_d27_selector_uses_explicit_per_class_bias_candidate_registry() -> None:
    folds = {
        runner.IDENTITY_CANDIDATE: _candidate_rows(
            runner.IDENTITY_CANDIDATE, 0.40, support_pass=True, steps=0
        ),
        runner.DIAG_CANDIDATE: _candidate_rows(
            runner.DIAG_CANDIDATE, 0.70, support_pass=True, steps=60
        ),
        runner.D25_C0: _candidate_rows(
            runner.D25_C0, 0.50, support_pass=True, steps=0
        ),
        runner.D27_A: _candidate_rows(runner.D27_A, 0.65, support_pass=True, steps=15),
        runner.D27_B: _candidate_rows(runner.D27_B, 0.65, support_pass=True, steps=25),
        runner.D27_C: _candidate_rows(runner.D27_C, 0.65, support_pass=True, steps=30),
    }
    selected, decisions = runner._select_d26_candidate(
        folds, runner.D27_CANDIDATES
    )
    assert selected == runner.D27_A
    d27 = next(row for row in decisions if row["candidate_id"] == runner.D27_A)
    assert d27["family"] == "d27_per_new_class_bias"
    assert d27["eligible_positive_route"] is True


def test_d29_selector_uses_explicit_pcsr_candidate_registry() -> None:
    folds = {
        runner.IDENTITY_CANDIDATE: _candidate_rows(
            runner.IDENTITY_CANDIDATE, 0.40, support_pass=True, steps=0
        ),
        runner.DIAG_CANDIDATE: _candidate_rows(
            runner.DIAG_CANDIDATE, 0.70, support_pass=True, steps=60
        ),
        runner.D25_C0: _candidate_rows(
            runner.D25_C0, 0.50, support_pass=True, steps=0
        ),
        runner.D29_A: _candidate_rows(runner.D29_A, 0.65, support_pass=True, steps=25),
        runner.D29_B: _candidate_rows(runner.D29_B, 0.65, support_pass=True, steps=25),
        runner.D29_C: _candidate_rows(runner.D29_C, 0.65, support_pass=True, steps=25),
    }
    selected, decisions = runner._select_d26_candidate(
        folds, runner.D29_CANDIDATES
    )
    assert selected in runner.D29_CANDIDATES
    d29 = next(row for row in decisions if row["candidate_id"] == selected)
    assert d29["family"] == "d29_per_class_safe_release"
    assert d29["eligible_positive_route"] is True


def test_d26_full_k10_failure_revokes_selected_route() -> None:
    decisions = [{"candidate_id": runner.D26_A, "eligible_positive_route": True}]
    resources = {
        runner.D26_A: {
            scenario: {"old_support_non_degradation_pass": scenario != "leo_rain_weak"}
            for scenario in runner.legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
    }
    selected, reason = runner._apply_full_k10_d26_old_support_gate(
        runner.D26_A, decisions, resources
    )
    assert selected == runner.D25_C0
    assert reason == "FULL_K10_OLD_SUPPORT_NON_DEGRADATION_FAILED"
    assert decisions[0]["eligible_positive_route"] is False
