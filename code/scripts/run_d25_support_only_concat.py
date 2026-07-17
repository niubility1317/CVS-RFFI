#!/usr/bin/env python3
"""D25 development support-only screen over sealed LEO_weak enrollment rows.

This runner deliberately reuses the already-audited D19 sealed-support helpers
without changing the historical D19/D20 runner.  Its CLI has no query, truth,
score-table, or scorer input.  FFT96 and RF32 are each extracted exactly once
per physical received-IQ row and shared by the B3 diagnostic and D25 routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_d19_support_only_ciaf as legacy  # noqa: E402
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    rf_statistics,
    spectral_logmag_sketch,
)
from cvsrffi.stage2_multimodal_concat_fusion import (  # noqa: E402
    SCORE_COSINE,
    SCORE_RADIUS,
    MultimodalConcatConfig,
    append_new_classes_concat,
    build_concat288,
    fit_old_concat,
    predict_one as predict_one_concat,
    score_one as score_one_concat,
)


MODE = legacy.MODE
SUPPORT_QUERY_DISJOINTNESS_STATUS = legacy.SUPPORT_QUERY_DISJOINTNESS_STATUS
HELD_RANKS = legacy.HELD_RANKS
IDENTITY_CANDIDATE = legacy.IDENTITY_CANDIDATE
DIAG_CANDIDATE = legacy.DIAG_CANDIDATE
D25_C0 = "D25-C0-DIM-CONCAT"
D25_C1 = "D25-C1-UF-GROUNDZ"
D25_C2 = "D25-C2-BLOCK-RADIUS"
D25_CANDIDATES = (D25_C0, D25_C1, D25_C2)
CORE_COMMIT = "f349850dbd94841ae2ef8105ac76bd7a9912c128"


class D25RunnerError(ValueError):
    """Raised when the D25 support-only screen must fail closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_hashes(value: np.ndarray) -> list[str]:
    rows = np.asarray(value, dtype=np.float32)
    return [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in rows
    ]


def preregistered_candidates() -> dict[str, object]:
    """Return the five candidates fixed before any support materialization."""

    controls = legacy.preregistered_candidates()
    return {
        IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
        DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
        D25_C0: MultimodalConcatConfig(
            score_mode=SCORE_COSINE,
            use_ground_identity_fusion=False,
        ),
        D25_C1: MultimodalConcatConfig(
            score_mode=SCORE_COSINE,
            use_ground_identity_fusion=True,
        ),
        D25_C2: MultimodalConcatConfig(
            score_mode=SCORE_RADIUS,
            use_ground_identity_fusion=True,
        ),
    }


def _candidate_lock(candidates: Mapping[str, object]) -> dict[str, Any]:
    source_closure = {
        "d25_core_sha256": _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_concat_fusion.py"
        ),
        "d24_uncertainty_fusion_sha256": _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_uncertainty_proto_fusion.py"
        ),
        "ciaf_sha256": _sha256_file(CODE_ROOT / "cvsrffi" / "stage2_ciaf.py"),
        "d19_control_helper_sha256": _sha256_file(
            SCRIPT_DIR / "run_d19_support_only_ciaf.py"
        ),
        "d25_runner_sha256": _sha256_file(Path(__file__).resolve()),
    }
    rows: list[dict[str, Any]] = []
    for candidate_id, config in candidates.items():
        if isinstance(config, MultimodalConcatConfig):
            config_row: dict[str, Any] = {
                "block_energy": list(config.block_energy),
                "r0_by_block": list(config.r0_by_block),
                "r_min": config.r_min,
                "separation_margin": config.separation_margin,
                "score_mode": config.score_mode,
                "use_ground_identity_fusion": config.use_ground_identity_fusion,
            }
            family = "d25"
        else:
            config_row = {
                "ground_weight": float(config.ground_weight),
                "direct_weight": float(config.direct_weight),
            }
            family = "control"
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": family,
                "config": config_row,
                "eligible_positive_route": candidate_id in D25_CANDIDATES,
            }
        )
    lock = {
        "schema": "cvs.phase2.d25.candidate_lock.v1",
        "core_commit": CORE_COMMIT,
        "held_ranks": [list(value) for value in HELD_RANKS],
        "candidates": rows,
        "selection_baseline": IDENTITY_CANDIDATE,
        "diagnostic_comparator": DIAG_CANDIDATE,
        "source_closure": source_closure,
    }
    return {**lock, "sha256": hashlib.sha256(_canonical_bytes(lock)).hexdigest()}


def _d1_feature_from_blocks(
    z_id160: np.ndarray, fft96: np.ndarray, rf32: np.ndarray
) -> np.ndarray:
    """Rebuild the historical B3 288-D feature without recomputing FFT/RF."""

    z_rows = legacy._normalize_matrix(np.asarray(z_id160, dtype=np.float32))
    auxiliary = np.concatenate(
        [np.asarray(fft96, dtype=np.float32), np.asarray(rf32, dtype=np.float32)],
        axis=1,
    )
    auxiliary = legacy._normalize_matrix(auxiliary)
    return legacy._normalize_matrix(
        np.concatenate([z_rows, np.float32(4.0) * auxiliary], axis=1)
    )


def _operator_lineage(rows: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    tokens = np.asarray(rows["tokens"]).astype(str)
    hashes = np.asarray(rows["hashes"]).astype(str)
    if (
        len(tokens) != len(hashes)
        or len(set(tokens.tolist())) != len(tokens)
        or any(len(value) != 64 for value in hashes)
    ):
        raise D25RunnerError("D25 feature-operator lineage parent drift")
    operators = (
        "adv3b02_zid160_base_v1",
        "same_received_iq_fft96_v1",
        "same_received_iq_rf32_v1",
    )
    return [
        {
            "physical_sample_id": token,
            "parent_received_iq_sha256": parent,
            "feature_operator_ids": list(operators),
            "support_row_multiplicity": 1,
            "derived_support_rows": 0,
            "additional_physical_sample_count": 0,
            "additional_leo_overlay_count": 0,
        }
        for token, parent in zip(tokens.tolist(), hashes.tolist())
    ]


def _geometry_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    pairs = list(value["pairs"])
    worst = min(pairs, key=lambda row: float(row["gap"])) if pairs else None
    return {
        "schema": "cvs.phase2.d25.geometry_summary.v1",
        "pair_count": int(value["pair_count"]),
        "collision_count": int(value["collision_count"]),
        "pass": bool(value["pass"]),
        "worst_pair": worst,
        "query_rows_used": 0,
    }


def _evaluate_d25_fold(
    component: object,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: MultimodalConcatConfig,
) -> dict[str, Any]:
    labels = np.asarray(rows["labels"]).astype(str)
    ranks = np.asarray(rows["ranks"], dtype=np.int64)
    held = np.isin(ranks, np.asarray(held_ranks, dtype=np.int64))
    train = ~held
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    if (
        int(np.sum(train & old)) != 8 * len(old_classes)
        or int(np.sum(train & new)) != 8 * len(new_classes)
        or int(np.sum(held & old)) != 2 * len(old_classes)
        or int(np.sum(held & new)) != 2 * len(new_classes)
    ):
        raise D25RunnerError("D25 leave-two-out class symmetry drift")

    ground_component = component if config.use_ground_identity_fusion else None
    before = fit_old_concat(
        ground_component,
        z_id160[train & old],
        fft96[train & old],
        rf32[train & old],
        labels[train & old],
        registered_classes=old_classes,
        config=config,
    )
    after = append_new_classes_concat(
        before,
        z_id160[train & new],
        fft96[train & new],
        rf32[train & new],
        labels[train & new],
        registered_classes=new_classes,
    )
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("D25 registered class order drift")
    if before.old_prefix_sha256 != after.old_prefix_sha256:
        raise D25RunnerError("D25 old prefix changed after registration")

    held_old_feature = build_concat288(
        z_id160[held & old],
        fft96[held & old],
        rf32[held & old],
        block_energy=config.block_energy,
    )
    held_new_feature = build_concat288(
        z_id160[held & new],
        fft96[held & new],
        rf32[held & new],
        block_energy=config.block_energy,
    )
    before_predictions = [
        predict_one_concat(before, row)[0] for row in held_old_feature
    ]
    after_old_predictions = [
        predict_one_concat(after, row)[0] for row in held_old_feature
    ]
    after_new_predictions = [
        predict_one_concat(after, row)[0] for row in held_new_feature
    ]
    old_scores_unchanged = all(
        np.array_equal(
            score_one_concat(before, row),
            score_one_concat(after, row)[: len(old_classes)],
        )
        for row in held_old_feature
    )
    if not old_scores_unchanged:
        raise D25RunnerError("D25 old score columns changed after registration")

    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    h_value = legacy._harmonic(
        float(after_old["overall_accuracy"]),
        float(after_new["overall_accuracy"]),
    )
    forgetting = float(
        before_old["overall_accuracy"] - after_old["overall_accuracy"]
    )
    geometry = after.geometry_audit()
    resource = dict(after.resource_audit())
    resource.update(
        {
            "int8_component_used_for_prediction": config.use_ground_identity_fusion,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "query_features_used_for_fit": False,
            "query_labels_used_for_fit": False,
            "source_sample_access": False,
            "source_cache_access": False,
            "source_derived_signal_access": False,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "complete_loss_trace": [],
        }
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": h_value,
        "forgetting": forgetting,
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_prefix_sha256,
        "old_prefix_sha256_after": after.old_prefix_sha256,
        "geometry_summary": _geometry_summary(geometry),
        "resource": resource,
    }


def _fold_guard(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    tolerance = 1.0e-12
    old_classwise = all(
        float(row["after_old"]["per_class_accuracy"][label]) + tolerance
        >= float(baseline["after_old"]["per_class_accuracy"][label])
        for label in row["after_old"]["per_class_accuracy"]
    )
    new_classwise = all(
        float(row["after_new"]["per_class_accuracy"][label]) + tolerance
        >= float(baseline["after_new"]["per_class_accuracy"][label])
        for label in row["after_new"]["per_class_accuracy"]
    )
    return bool(
        float(row["before_old"]["class_floor_accuracy"]) + tolerance
        >= float(baseline["before_old"]["class_floor_accuracy"])
        and float(row["after_old"]["class_floor_accuracy"]) + tolerance
        >= float(baseline["after_old"]["class_floor_accuracy"])
        and float(row["after_new"]["class_floor_accuracy"]) + tolerance
        >= float(baseline["after_new"]["class_floor_accuracy"])
        and float(row["H_old_new"]) + tolerance
        >= float(baseline["H_old_new"])
        and float(row["forgetting"])
        <= float(baseline["forgetting"]) + tolerance
        and old_classwise
        and new_classwise
    )


def _select_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    baseline = list(folds_by_candidate[IDENTITY_CANDIDATE])
    baseline_aggregate = legacy._aggregate_candidate(baseline)
    diagnostic = list(folds_by_candidate[DIAG_CANDIDATE])
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float]] = []
    tolerance = 1.0e-12
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        guards = [
            _fold_guard(row, zero) for row, zero in zip(rows, baseline)
        ]
        diagnostic_guards = [
            _fold_guard(row, diag) for row, diag in zip(rows, diagnostic)
        ]
        strict_old_floor = bool(
            float(aggregate["worst_after_old_floor"])
            > float(baseline_aggregate["worst_after_old_floor"]) + tolerance
        )
        strict_joint_floor = bool(
            float(aggregate["worst_joint_floor"])
            > float(baseline_aggregate["worst_joint_floor"]) + tolerance
        )
        is_d25 = candidate_id in D25_CANDIDATES
        eligible_positive = bool(
            is_d25 and all(guards) and (strict_old_floor or strict_joint_floor)
        )
        decision = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": "d25" if is_d25 else "control",
            "atomic_noninferiority_vs_Z0": bool(all(guards)),
            "noninferior_fold_count": int(sum(guards)),
            "noninferior_vs_B3_fold_count": int(sum(diagnostic_guards)),
            "strict_worst_old_floor_improvement_vs_Z0": strict_old_floor,
            "strict_worst_joint_floor_improvement_vs_Z0": strict_joint_floor,
            "eligible_positive_route": eligible_positive,
            "fallback": candidate_id == IDENTITY_CANDIDATE,
            "diagnostic_only": candidate_id == DIAG_CANDIDATE,
        }
        decisions.append(decision)
        if eligible_positive:
            eligible.append(
                (
                    candidate_id,
                    float(aggregate["worst_joint_floor"]),
                    float(aggregate["worst_after_old_floor"]),
                    float(aggregate["mean_H_old_new"]),
                )
            )
    selected = (
        max(eligible, key=lambda value: (value[1], value[2], value[3], value[0]))[0]
        if eligible
        else IDENTITY_CANDIDATE
    )
    return selected, decisions


def _full_d25_state_audit(
    component: object,
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: MultimodalConcatConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    fit_started = time.perf_counter()
    before = fit_old_concat(
        component if config.use_ground_identity_fusion else None,
        z_id160[old],
        fft96[old],
        rf32[old],
        labels[old],
        registered_classes=old_classes,
        config=config,
    )
    after = append_new_classes_concat(
        before,
        z_id160[new],
        fft96[new],
        rf32[new],
        labels[new],
        registered_classes=new_classes,
    )
    if before.old_prefix_sha256 != after.old_prefix_sha256:
        raise D25RunnerError("D25 deployment old prefix drift")
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    registered_feature = build_concat288(
        z_id160,
        fft96,
        rf32,
        block_energy=config.block_energy,
    )
    score_elapsed_ms: list[float] = []
    for feature in registered_feature:
        score_started = time.perf_counter()
        score_one_concat(after, feature)
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = dict(after.resource_audit())
    registered_count = len(old_classes) + len(new_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    identity_qknn_fp32_state_bytes = registered_count * 10 * 160 * 4
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": registered_count,
            "old_prefix_sha256": after.old_prefix_sha256,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
            "identity_single_qknn_fp16_sample_state_bytes": identity_qknn_fp16_state_bytes,
            "identity_single_qknn_fp32_sample_state_bytes": identity_qknn_fp32_state_bytes,
            "estimated_score_mac_ratio_vs_identity_single_qknn": float(
                resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "estimated_score_mac_reduction_vs_identity_single_qknn": float(
                1.0
                - resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
                resource["persistent_state_bytes"] / identity_qknn_fp16_state_bytes
            ),
            "closed_form_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu",
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return resource, after.geometry_audit()


def run(
    *,
    before_root: Path,
    before_seal: Path,
    expected_before_seal_sha256: str,
    before_formal_policy: Path,
    before_formal_policy_authorization: Path,
    before_signed_policy_authorization_envelope: Path,
    expected_before_signed_policy_authorization_envelope_sha256: str,
    after_root: Path,
    after_seal: Path,
    expected_after_seal_sha256: str,
    after_formal_policy: Path,
    after_formal_policy_authorization: Path,
    after_signed_policy_authorization_envelope: Path,
    expected_after_signed_policy_authorization_envelope_sha256: str,
    component_dir: Path,
    expected_component_manifest_sha256: str,
    class_binding_path: Path,
    expected_class_binding_sha256: str,
    output: Path,
    device_name: str = "auto",
    mode: str = MODE,
) -> dict[str, Any]:
    if mode != MODE:
        raise D25RunnerError("D25 runner is development support-only")
    if output.exists():
        raise D25RunnerError("output path already exists")
    candidates = preregistered_candidates()
    candidate_lock = _candidate_lock(candidates)

    before_preopen_manifest = legacy._preopen_manifest(
        before_root,
        before_seal,
        expected_seal_sha256=expected_before_seal_sha256,
    )
    after_preopen_manifest = legacy._preopen_manifest(
        after_root,
        after_seal,
        expected_seal_sha256=expected_after_seal_sha256,
    )
    legacy._manifest_binding(before_preopen_manifest, after_preopen_manifest)
    preopen_old_classes = legacy._registered_handles(before_preopen_manifest)
    component, component_audit = legacy._load_component(
        component_dir,
        expected_manifest_sha256=expected_component_manifest_sha256,
        expected_checkpoint_sha256=str(
            before_preopen_manifest["phase1_checkpoint_sha256"]
        ),
        bound_old_handles=preopen_old_classes,
        class_binding_path=class_binding_path,
        expected_class_binding_sha256=expected_class_binding_sha256,
    )
    device = torch.device(
        "cuda:0"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    model = legacy.load_torchscript_backbone_same_fd(
        before_root,
        legacy._member(before_preopen_manifest, "feature_runtime"),
        device=device,
    )
    runtime_direct_logit_binding_audit = legacy._verify_runtime_direct_logit_binding(
        model,
        before_preopen_manifest,
        component_audit["column_binding"],
    )

    before_evidence = legacy.materialize_somph_enrollment_with_signed_authority(
        before_root,
        detached_seal_path=before_seal,
        expected_seal_sha256=expected_before_seal_sha256,
        formal_policy_path=before_formal_policy,
        formal_policy_authorization_path=before_formal_policy_authorization,
        signed_policy_authorization_envelope_path=(
            before_signed_policy_authorization_envelope
        ),
        expected_signed_policy_authorization_envelope_sha256=(
            expected_before_signed_policy_authorization_envelope_sha256
        ),
    )
    after_evidence = legacy.materialize_somph_enrollment_with_signed_authority(
        after_root,
        detached_seal_path=after_seal,
        expected_seal_sha256=expected_after_seal_sha256,
        formal_policy_path=after_formal_policy,
        formal_policy_authorization_path=after_formal_policy_authorization,
        signed_policy_authorization_envelope_path=(
            after_signed_policy_authorization_envelope
        ),
        expected_signed_policy_authorization_envelope_sha256=(
            expected_after_signed_policy_authorization_envelope_sha256
        ),
    )
    before_authority = (
        legacy.finalize_somph_enrollment_authority_after_materialization(
            before_evidence
        )
    )
    after_authority = (
        legacy.finalize_somph_enrollment_authority_after_materialization(
            after_evidence
        )
    )
    legacy._require_post_materialization_authority(before_authority, after_authority)
    before_manifest = before_evidence.manifest
    after_manifest = after_evidence.manifest
    legacy._manifest_binding(before_manifest, after_manifest)
    old_classes = legacy._registered_handles(before_manifest)
    all_classes = legacy._registered_handles(after_manifest)
    if all_classes[: len(old_classes)] != old_classes:
        raise D25RunnerError("after registry does not append new classes")
    new_classes = all_classes[len(old_classes) :]
    if old_classes != preopen_old_classes:
        raise D25RunnerError("post-materialization registry differs from pre-open binding")

    before_overlay, before_overlay_audit = legacy._overlay_index(
        before_root, before_manifest
    )
    after_overlay, after_overlay_audit = legacy._overlay_index(
        after_root, after_manifest
    )
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    scene_rows: dict[str, dict[str, np.ndarray]] = {}
    scene_z: dict[str, np.ndarray] = {}
    scene_logits: dict[str, np.ndarray] = {}
    scene_fft: dict[str, np.ndarray] = {}
    scene_rf: dict[str, np.ndarray] = {}
    scene_b3: dict[str, np.ndarray] = {}
    extraction_audits: dict[str, Any] = {}
    old_reuse_audits: dict[str, Any] = {}
    for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
        before_rows = legacy._rows_with_overlay(
            before_evidence.materialized_payloads[scenario],
            before_manifest,
            before_overlay,
            scenario=scenario,
        )
        after_rows = legacy._rows_with_overlay(
            after_evidence.materialized_payloads[scenario],
            after_manifest,
            after_overlay,
            scenario=scenario,
        )
        legacy._old_reuse(before_rows, after_rows)
        backbone_started = time.perf_counter()
        z_id160, direct_logits, extraction = legacy._extract_scene_signals(
            model,
            device,
            after_rows,
            component_audit["column_binding"]["direct_logit_indices"],
        )
        backbone_elapsed_ms = (time.perf_counter() - backbone_started) * 1000.0
        fft_started = time.perf_counter()
        fft96 = spectral_logmag_sketch(after_rows["iq"])
        fft_elapsed_ms = (time.perf_counter() - fft_started) * 1000.0
        rf_started = time.perf_counter()
        rf32 = rf_statistics(after_rows["iq"])
        rf_elapsed_ms = (time.perf_counter() - rf_started) * 1000.0
        if not len(z_id160) == len(fft96) == len(rf32) == len(after_rows["iq"]):
            raise D25RunnerError("D25 feature block row alignment drift")
        b3_features = _d1_feature_from_blocks(z_id160, fft96, rf32)
        scene_rows[scenario] = after_rows
        scene_z[scenario] = z_id160
        scene_logits[scenario] = direct_logits
        scene_fft[scenario] = fft96
        scene_rf[scenario] = rf32
        scene_b3[scenario] = b3_features
        extraction.update(
            {
                "feature_operator_count": 3,
                "feature_operator_ids": [
                    "adv3b02_zid160_base_v1",
                    "same_received_iq_fft96_v1",
                    "same_received_iq_rf32_v1",
                ],
                "support_view_count": 1,
                "support_row_multiplicity": 1,
                "derived_support_rows": 0,
                "additional_physical_sample_count": 0,
                "additional_leo_overlay_count": 0,
                "same_received_iq_fft96_extractions": int(len(fft96)),
                "same_received_iq_rf32_extractions": int(len(rf32)),
                "fft96_sha256": _row_hashes(fft96),
                "rf32_sha256": _row_hashes(rf32),
                "d25_block_dimensions": [160, 96, 32],
                "d25_concatenated_feature_dimension": 288,
                "b3_registered_feature_dimension": int(b3_features.shape[1]),
                "additional_backbone_forwards_for_fft_rf": 0,
                "backbone_elapsed_ms": backbone_elapsed_ms,
                "backbone_mean_ms_per_physical_sample": float(
                    backbone_elapsed_ms / len(after_rows["iq"])
                ),
                "fft96_elapsed_ms": fft_elapsed_ms,
                "fft96_mean_ms_per_physical_sample": float(
                    fft_elapsed_ms / len(after_rows["iq"])
                ),
                "rf32_elapsed_ms": rf_elapsed_ms,
                "rf32_mean_ms_per_physical_sample": float(
                    rf_elapsed_ms / len(after_rows["iq"])
                ),
                "feature_operator_lineage": _operator_lineage(after_rows),
            }
        )
        extraction_audits[scenario] = extraction
        old_reuse_audits[scenario] = {
            "old_support_exact_reuse": True,
            "before_old_rows": int(len(before_rows["labels"])),
            "after_total_rows": int(len(after_rows["labels"])),
        }
    cross_scene = legacy._cross_scene_disjointness(scene_rows)

    training_log: list[dict[str, Any]] = []
    folds_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidates
    }
    diag_caches: dict[str, dict[tuple[str, tuple[int, int]], dict[str, Any]]] = {
        scenario: {} for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
    }
    for candidate_id, config in candidates.items():
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            for fold_index, held_ranks in enumerate(HELD_RANKS):
                if candidate_id in D25_CANDIDATES:
                    if not isinstance(config, MultimodalConcatConfig):
                        raise D25RunnerError("D25 candidate config drift")
                    row = _evaluate_d25_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_fft[scenario],
                        scene_rf[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                    )
                else:
                    row = legacy._evaluate_fold(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_b3[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        held_ranks=held_ranks,
                        candidate_id=candidate_id,
                        config=config,
                        fit_seed=int(before_manifest["seed"]) + fold_index,
                        device=device,
                        diag_cache=diag_caches[scenario],
                    )
                row.update(
                    {
                        "schema": "cvs.phase2.d25.support_fold.v1",
                        "scenario": scenario,
                        "fold_index": fold_index,
                        "query_opened": False,
                        "formal_metric_claim_allowed": False,
                        "performance_claim_allowed": False,
                    }
                )
                folds_by_candidate[candidate_id].append(row)
                training_log.append(row)
    expected_rows = (
        len(candidates)
        * len(legacy.FORMAL_LEO_WEAK_SCENARIOS)
        * len(HELD_RANKS)
    )
    if len(training_log) != expected_rows or expected_rows != 75:
        raise D25RunnerError("D25 training-log cardinality drift")
    selected_id, candidate_decisions = _select_candidate(folds_by_candidate)

    deployment_resources: dict[str, dict[str, Any]] = {
        candidate_id: {} for candidate_id in candidates
    }
    geometry_matrix: dict[str, dict[str, Any]] = {
        candidate_id: {} for candidate_id in D25_CANDIDATES
    }
    for candidate_id, config in candidates.items():
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            if candidate_id in D25_CANDIDATES:
                resource, geometry = _full_d25_state_audit(
                    component,
                    scene_rows[scenario],
                    scene_z[scenario],
                    scene_fft[scenario],
                    scene_rf[scenario],
                    old_classes=old_classes,
                    new_classes=new_classes,
                    config=config,
                )
                deployment_resources[candidate_id][scenario] = resource
                geometry_matrix[candidate_id][scenario] = geometry
            else:
                deployment_resources[candidate_id][scenario] = (
                    legacy._deployment_state_audit(
                        component,
                        scene_rows[scenario],
                        scene_z[scenario],
                        scene_logits[scenario],
                        scene_b3[scenario],
                        old_classes=old_classes,
                        new_classes=new_classes,
                        candidate_id=candidate_id,
                        config=config,
                        fit_seed=int(before_manifest["seed"]),
                        device=device,
                    )
                )

    elapsed = time.perf_counter() - start
    support_audit = {
        "schema": "cvs.phase2.d25.support_audit.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_USER_AUTHORIZED_PREBUNDLE_INT8_SCREEN",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "query_opened": False,
        "query_rows_opened": 0,
        "query_labels_opened": 0,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "sample_level_source_feature_access": False,
        "authorized_int8_phase1_aggregate_component_access": True,
        "int8_component_update_access": False,
        "one_physical_support_one_leo_channel_observation": True,
        "support_view_count": 1,
        "support_row_multiplicity": 1,
        "feature_operator_count": 3,
        "derived_support_rows": 0,
        "additional_physical_sample_count": 0,
        "additional_leo_overlay_count": 0,
        "feature_operators_count_toward_k": False,
        "candidate_lock": candidate_lock,
        "old_reuse_by_scenario": old_reuse_audits,
        "cross_scene_disjointness": cross_scene,
        "before_overlay_audit": before_overlay_audit,
        "after_overlay_audit": after_overlay_audit,
        "feature_extraction": extraction_audits,
        "runtime_direct_logit_binding": runtime_direct_logit_binding_audit,
        "component": component_audit,
        "before_post_materialization_audit_sha256": before_authority[
            "post_materialization_audit_sha256"
        ],
        "after_post_materialization_audit_sha256": after_authority[
            "post_materialization_audit_sha256"
        ],
    }
    training_log_sha256 = legacy._write_jsonl(
        output / "training_log.jsonl", training_log
    )
    support_audit_sha256 = legacy._write_json(
        output / "support_audit.json", support_audit
    )
    selection = {
        "schema": "cvs.phase2.d25.selection.v1",
        "selected_candidate_id": selected_id,
        "selected_positive_route": selected_id in D25_CANDIDATES,
        "fallback_to_identity": selected_id == IDENTITY_CANDIDATE,
        "selection_baseline": IDENTITY_CANDIDATE,
        "diagnostic_comparator": DIAG_CANDIDATE,
        "eligible_candidate_ids": list(D25_CANDIDATES),
        "selection_rule": (
            "all_15_folds_classwise_noninferior_vs_Z0_and_strict_worst_"
            "after_old_or_joint_floor_improvement;_B3_is_diagnostic_only"
        ),
        "candidate_lock_sha256": candidate_lock["sha256"],
        "candidate_decisions": candidate_decisions,
    }
    selection_sha256 = legacy._write_json(output / "selection.json", selection)
    resource_sha256 = legacy._write_json(
        output / "resource_audit.json",
        {
            "schema": "cvs.phase2.d25.resource_matrix.v1",
            "selected_candidate_id": selected_id,
            "by_candidate_by_scenario": deployment_resources,
        },
    )
    geometry_sha256 = legacy._write_json(
        output / "geometry_audit.json",
        {
            "schema": "cvs.phase2.d25.geometry_matrix.v1",
            "query_rows_used": 0,
            "by_candidate_by_scenario": geometry_matrix,
        },
    )

    current_source_closure = _candidate_lock(candidates)["source_closure"]
    if current_source_closure != candidate_lock["source_closure"]:
        raise D25RunnerError("D25 source closure changed after support opening")
    receipt = {
        "schema": "cvs.phase2.d25.receipt.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_COMPLETE",
        "mode": mode,
        "core_commit": CORE_COMMIT,
        **candidate_lock["source_closure"],
        "source_closure_unchanged_after_support": True,
        "candidate_lock_sha256": candidate_lock["sha256"],
        "selected_candidate_id": selected_id,
        "selected_positive_route": selected_id in D25_CANDIDATES,
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "performance_claim_allowed": False,
        "query_opened": False,
        "support_query_disjointness_status": SUPPORT_QUERY_DISJOINTNESS_STATUS,
        "receiver": str(before_manifest["receiver"]),
        "seed": int(before_manifest["seed"]),
        "k_shot": int(before_manifest["k_shot"]),
        "old_class_count": len(old_classes),
        "new_class_count": len(new_classes),
        "scenarios": list(legacy.FORMAL_LEO_WEAK_SCENARIOS),
        "candidate_count": len(candidates),
        "folds_per_candidate": len(legacy.FORMAL_LEO_WEAK_SCENARIOS)
        * len(HELD_RANKS),
        "training_log_row_count": len(training_log),
        "elapsed_seconds": elapsed,
        "training_log_sha256": training_log_sha256,
        "support_audit_sha256": support_audit_sha256,
        "selection_sha256": selection_sha256,
        "resource_audit_sha256": resource_sha256,
        "geometry_audit_sha256": geometry_sha256,
        "component_manifest_sha256": expected_component_manifest_sha256,
        "component_npz_sha256": component_audit["manifest"][
            "component_npz_sha256"
        ],
        "component_provenance_status": component_audit["manifest"][
            "provenance_status"
        ],
    }
    receipt_sha256 = legacy._write_json(output / "RECEIPT.json", receipt)
    return {"receipt_sha256": receipt_sha256, **receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--before-seal", type=Path, required=True)
    parser.add_argument("--before-seal-sha256", required=True)
    parser.add_argument("--before-formal-policy", type=Path, required=True)
    parser.add_argument("--before-formal-policy-authorization", type=Path, required=True)
    parser.add_argument(
        "--before-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--before-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--after-seal", type=Path, required=True)
    parser.add_argument("--after-seal-sha256", required=True)
    parser.add_argument("--after-formal-policy", type=Path, required=True)
    parser.add_argument("--after-formal-policy-authorization", type=Path, required=True)
    parser.add_argument(
        "--after-signed-policy-authorization-envelope", type=Path, required=True
    )
    parser.add_argument(
        "--after-signed-policy-authorization-envelope-sha256", required=True
    )
    parser.add_argument("--component-dir", type=Path, required=True)
    parser.add_argument("--component-manifest-sha256", required=True)
    parser.add_argument("--class-binding", type=Path, required=True)
    parser.add_argument("--class-binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=(MODE,), required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        before_root=args.before_root,
        before_seal=args.before_seal,
        expected_before_seal_sha256=args.before_seal_sha256,
        before_formal_policy=args.before_formal_policy,
        before_formal_policy_authorization=args.before_formal_policy_authorization,
        before_signed_policy_authorization_envelope=(
            args.before_signed_policy_authorization_envelope
        ),
        expected_before_signed_policy_authorization_envelope_sha256=(
            args.before_signed_policy_authorization_envelope_sha256
        ),
        after_root=args.after_root,
        after_seal=args.after_seal,
        expected_after_seal_sha256=args.after_seal_sha256,
        after_formal_policy=args.after_formal_policy,
        after_formal_policy_authorization=args.after_formal_policy_authorization,
        after_signed_policy_authorization_envelope=(
            args.after_signed_policy_authorization_envelope
        ),
        expected_after_signed_policy_authorization_envelope_sha256=(
            args.after_signed_policy_authorization_envelope_sha256
        ),
        component_dir=args.component_dir,
        expected_component_manifest_sha256=args.component_manifest_sha256,
        class_binding_path=args.class_binding,
        expected_class_binding_sha256=args.class_binding_sha256,
        output=args.output,
        device_name=args.device,
        mode=args.mode,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
