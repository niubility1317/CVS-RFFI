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
from cvsrffi.stage2_multimodal_diag_floor_adapter import (  # noqa: E402
    D25C3Config,
    D25C3LossWeights,
    D25C3State,
    append_stage2c_new_suffix,
    fit_stage2b_diag_floor,
    predict_one as predict_one_c3,
    score_one as score_one_c3,
)
from cvsrffi.stage2_multimodal_compact_diag import (  # noqa: E402
    D26CompactDiagConfig,
    D26CompactDiagState,
    append_stage2c_new_suffix as append_stage2c_d26,
    fit_stage2b_compact_diag,
    predict_all_registered as predict_all_d26,
    score_all_registered as score_all_d26,
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
CANDIDATE_SET_D25_V4 = "d25_v4"
CANDIDATE_SET_C3_V1 = "c3_v1"
CANDIDATE_SET_D26_V1 = "d26_v1"
C3_A = "D25-C3A-DIAG-CE-CLOSEDREG"
C3_B = "D25-C3B-DIAG-CE-NEWFIT"
C3_C = "D25-C3C-DIAG-STRONGFLOOR-NEWFIT"
C3_CANDIDATES = (C3_A, C3_B, C3_C)
D26_A = "D26-A-COMPACT-DIAG-CLOSEDREG"
D26_B = "D26-B-COMPACT-DIAG-NEWFIT10"
D26_C = "D26-C-COMPACT-DIAG-NEWFIT15"
D26_CANDIDATES = (D26_A, D26_B, D26_C)
CORE_COMMIT = "f349850dbd94841ae2ef8105ac76bd7a9912c128"
D26_CORE_GIT_COMMIT = "0a9fbb20e58f1f77c7f9ccc350cc826351ce0d79"


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


def preregistered_candidates(
    candidate_set: str = CANDIDATE_SET_D25_V4,
) -> dict[str, object]:
    """Return the candidate set fixed before any support materialization."""

    controls = legacy.preregistered_candidates()
    historical = {
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
    if candidate_set == CANDIDATE_SET_D25_V4:
        return historical
    if candidate_set == CANDIDATE_SET_D26_V1:
        return {
            IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
            DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
            D25_C0: historical[D25_C0],
            D26_A: D26CompactDiagConfig(stage2b_steps=15, stage2c_steps=0),
            D26_B: D26CompactDiagConfig(stage2b_steps=15, stage2c_steps=10),
            D26_C: D26CompactDiagConfig(stage2b_steps=15, stage2c_steps=15),
        }
    if candidate_set != CANDIDATE_SET_C3_V1:
        raise D25RunnerError("unknown D25 candidate set")
    ce_weights = D25C3LossWeights(
        equal_class_ce=1.0,
        tail_cvar=0.0,
        hard_negative_margin=0.0,
        proximity=0.01,
    )
    strong_weights = D25C3LossWeights(
        equal_class_ce=1.0,
        tail_cvar=0.20,
        hard_negative_margin=0.10,
        proximity=0.01,
    )
    return {
        IDENTITY_CANDIDATE: controls[IDENTITY_CANDIDATE],
        DIAG_CANDIDATE: controls[DIAG_CANDIDATE],
        D25_C0: historical[D25_C0],
        C3_A: D25C3Config(
            loss_weights=ce_weights,
            stage2b_steps=20,
            stage2c_steps=0,
        ),
        C3_B: D25C3Config(
            loss_weights=ce_weights,
            stage2b_steps=20,
            stage2c_steps=10,
        ),
        C3_C: D25C3Config(
            loss_weights=strong_weights,
            stage2b_steps=15,
            stage2c_steps=15,
        ),
    }


def _candidate_lock(
    candidates: Mapping[str, object],
    candidate_set: str = CANDIDATE_SET_D25_V4,
) -> dict[str, Any]:
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
        "diag_cosine_feature_operator_sha256": _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_diag_cosine_exploration.py"
        ),
        "d25_runner_sha256": _sha256_file(Path(__file__).resolve()),
    }
    if any(isinstance(value, D25C3Config) for value in candidates.values()):
        source_closure["d25_c3_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_diag_floor_adapter.py"
        )
    if any(isinstance(value, D26CompactDiagConfig) for value in candidates.values()):
        source_closure["d26_compact_diag_core_sha256"] = _sha256_file(
            CODE_ROOT / "cvsrffi" / "stage2_multimodal_compact_diag.py"
        )
    rows: list[dict[str, Any]] = []
    for candidate_id, config in candidates.items():
        if isinstance(config, D25C3Config):
            config_row = config.lock_payload()
            family = "d25_c3"
        elif isinstance(config, D26CompactDiagConfig):
            config_row = {
                "stage2b_steps": int(config.stage2b_steps),
                "stage2c_steps": int(config.stage2c_steps),
                "learning_rate": float(config.learning_rate),
                "weight_decay": float(config.weight_decay),
                "prototype_anchor_weight": float(config.prototype_anchor_weight),
                "diagonal_proximity_weight": float(
                    config.diagonal_proximity_weight
                ),
                "new_group_bias_grid": list(config.new_group_bias_grid),
            }
            family = "d26_compact_diag"
        elif isinstance(config, MultimodalConcatConfig):
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
                "eligible_positive_route": candidate_id
                in (
                    C3_CANDIDATES
                    if candidate_set == CANDIDATE_SET_C3_V1
                    else D26_CANDIDATES
                    if candidate_set == CANDIDATE_SET_D26_V1
                    else D25_CANDIDATES
                ),
            }
        )
    lock = {
        "schema": (
            "cvs.phase2.d25.candidate_lock.v2"
            if candidate_set == CANDIDATE_SET_C3_V1
            else "cvs.phase2.d25.candidate_lock.v3"
            if candidate_set == CANDIDATE_SET_D26_V1
            else "cvs.phase2.d25.candidate_lock.v1"
        ),
        "core_commit": CORE_COMMIT,
        "held_ranks": [list(value) for value in HELD_RANKS],
        "candidates": rows,
        "selection_baseline": (
            D25_C0
            if candidate_set in (CANDIDATE_SET_C3_V1, CANDIDATE_SET_D26_V1)
            else IDENTITY_CANDIDATE
        ),
        "diagnostic_comparator": DIAG_CANDIDATE,
        "source_closure": source_closure,
    }
    if candidate_set in (CANDIDATE_SET_C3_V1, CANDIDATE_SET_D26_V1):
        lock["candidate_set"] = candidate_set
    if candidate_set == CANDIDATE_SET_D26_V1:
        # CORE_COMMIT above identifies the sealed Phase1 model lineage.  Keep
        # the D26 implementation commit separate so the receipt cannot imply
        # that the new adapter was already present in that older model commit.
        lock["d26_core_git_commit"] = D26_CORE_GIT_COMMIT
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


def _c3_geometry(state: D25C3State) -> dict[str, Any]:
    prototypes = np.asarray(state.prototypes, dtype=np.float32)
    pairs: list[dict[str, Any]] = []
    for left in range(len(state.classes)):
        for right in range(left + 1, len(state.classes)):
            distance = float(
                1.0
                - np.dot(
                    prototypes[left].astype(np.float32),
                    prototypes[right].astype(np.float32),
                )
            )
            role = (
                "old_old"
                if right < state.old_class_count
                else ("new_new" if left >= state.old_class_count else "old_new")
            )
            pairs.append(
                {
                    "left": state.classes[left],
                    "right": state.classes[right],
                    "role": role,
                    "cosine_distance": distance,
                    "collision_below_0p05": distance < 0.05,
                }
            )
    distances = [float(row["cosine_distance"]) for row in pairs]
    return {
        "schema": "cvs.phase2.d25_c3.prototype_geometry.v1",
        "class_count": len(state.classes),
        "old_class_count": state.old_class_count,
        "pair_count": len(pairs),
        "minimum_cosine_distance": min(distances) if distances else None,
        "collision_count_below_0p05": sum(
            int(bool(row["collision_below_0p05"])) for row in pairs
        ),
        "pairs": pairs,
    }


def _d26_geometry(state: D26CompactDiagState) -> dict[str, Any]:
    weights = np.asarray(state.weights, dtype=np.float32)
    pairs: list[dict[str, Any]] = []
    for left in range(len(state.classes)):
        for right in range(left + 1, len(state.classes)):
            distance = float(1.0 - np.dot(weights[left], weights[right]))
            role = (
                "old_old"
                if right < state.old_class_count
                else ("new_new" if left >= state.old_class_count else "old_new")
            )
            pairs.append(
                {
                    "left": state.classes[left],
                    "right": state.classes[right],
                    "role": role,
                    "cosine_distance": distance,
                    "collision_below_0p05": distance < 0.05,
                }
            )
    distances = [float(row["cosine_distance"]) for row in pairs]
    bias_audit = json.loads(state.bias_audit_json)
    return {
        "schema": "cvs.phase2.d26_compact_diag_geometry.v1",
        "class_count": len(state.classes),
        "old_class_count": state.old_class_count,
        "pair_count": len(pairs),
        "minimum_cosine_distance": min(distances) if distances else None,
        "collision_count_below_0p05": sum(
            int(bool(row["collision_below_0p05"])) for row in pairs
        ),
        "new_group_bias": float(state.new_group_bias),
        "bias_applied_to_new_suffix_only": True,
        "bias_support_only_audit": bias_audit,
        "pairs": pairs,
    }


def _evaluate_c3_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D25C3Config,
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
        raise D25RunnerError("C3 leave-two-out class symmetry drift")
    features = build_concat288(z_id160, fft96, rf32)
    before_fit = fit_stage2b_diag_floor(
        features[train & old],
        labels[train & old],
        old_classes,
        config=config,
    )
    before = before_fit.state
    after_fit = append_stage2c_new_suffix(
        before,
        features[train & new],
        labels[train & new],
        new_classes,
    )
    after = after_fit.state
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("C3 registered class order drift")
    if (
        before.old_prefix_sha256 != after.old_prefix_sha256
        or before.shared_sha256 != after.shared_sha256
    ):
        raise D25RunnerError("C3 shared or old prefix changed after registration")

    held_old = features[held & old]
    held_new = features[held & new]
    before_predictions = [predict_one_c3(before, row)[0] for row in held_old]
    after_old_predictions = [predict_one_c3(after, row)[0] for row in held_old]
    after_new_predictions = [predict_one_c3(after, row)[0] for row in held_new]
    old_scores_unchanged = all(
        np.array_equal(
            score_one_c3(before, row),
            score_one_c3(after, row)[: len(old_classes)],
        )
        for row in held_old
    )
    if not old_scores_unchanged:
        raise D25RunnerError("C3 old raw score prefix changed after registration")

    fit_old_features = features[train & old]
    fit_old_labels = labels[train & old]
    fit_before_predictions = [
        predict_one_c3(before, row)[0] for row in fit_old_features
    ]
    fit_after_predictions = [
        predict_one_c3(after, row)[0] for row in fit_old_features
    ]
    fit_before = legacy._metric_block(
        fit_old_labels, fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        fit_old_labels, fit_after_predictions, old_classes
    )
    tolerance = 1.0e-12
    fit_classwise_non_degradation = all(
        float(fit_after["per_class_accuracy"][label]) + tolerance
        >= float(fit_before["per_class_accuracy"][label])
        for label in old_classes
    )
    fit_floor_non_degradation = (
        float(fit_after["class_floor_accuracy"]) + tolerance
        >= float(fit_before["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(
        fit_classwise_non_degradation and fit_floor_non_degradation
    )

    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    resource = dict(after.resource_audit())
    resource.update(
        {
            "old_support_non_degradation_pass": old_support_non_degradation,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "complete_loss_trace": list(before_fit.training_trace)
            + list(after_fit.training_trace),
            "query_features_used_for_fit": False,
            "query_labels_used_for_fit": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_prefix_sha256,
        "old_prefix_sha256_after": after.old_prefix_sha256,
        "shared_sha256_before": before.shared_sha256,
        "shared_sha256_after": after.shared_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": fit_classwise_non_degradation,
        "old_support_floor_non_degradation": fit_floor_non_degradation,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "training_trace": list(before_fit.training_trace)
        + list(after_fit.training_trace),
        "geometry_summary": _c3_geometry(after),
        "resource": resource,
    }


def _evaluate_d26_fold(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    held_ranks: tuple[int, int],
    candidate_id: str,
    config: D26CompactDiagConfig,
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
        raise D25RunnerError("D26 leave-two-out class symmetry drift")
    features = build_concat288(z_id160, fft96, rf32)
    fit_old_features = features[train & old]
    fit_old_labels = labels[train & old]
    before_fit = fit_stage2b_compact_diag(
        fit_old_features,
        fit_old_labels,
        old_classes,
        config=config,
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[train & new],
        labels[train & new],
        new_classes,
        fit_old_features,
        fit_old_labels,
    )
    after = after_fit.state
    if before.classes != old_classes or after.classes != old_classes + new_classes:
        raise D25RunnerError("D26 registered class order drift")
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D26 shared diagonal or old weight prefix changed")

    held_old = features[held & old]
    held_new = features[held & new]
    before_old_scores = score_all_d26(before, held_old)
    after_old_scores = score_all_d26(after, held_old)
    if not np.array_equal(
        before_old_scores, after_old_scores[:, : len(old_classes)]
    ):
        raise D25RunnerError("D26 old raw score prefix changed after registration")
    before_predictions = predict_all_d26(before, held_old).astype(str).tolist()
    after_old_predictions = predict_all_d26(after, held_old).astype(str).tolist()
    after_new_predictions = predict_all_d26(after, held_new).astype(str).tolist()

    fit_before_predictions = (
        predict_all_d26(before, fit_old_features).astype(str).tolist()
    )
    fit_after_predictions = (
        predict_all_d26(after, fit_old_features).astype(str).tolist()
    )
    fit_before = legacy._metric_block(
        fit_old_labels, fit_before_predictions, old_classes
    )
    fit_after = legacy._metric_block(
        fit_old_labels, fit_after_predictions, old_classes
    )
    tolerance = 1.0e-12
    fit_classwise_non_degradation = all(
        float(fit_after["per_class_accuracy"][label]) + tolerance
        >= float(fit_before["per_class_accuracy"][label])
        for label in old_classes
    )
    fit_floor_non_degradation = (
        float(fit_after["class_floor_accuracy"]) + tolerance
        >= float(fit_before["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(
        fit_classwise_non_degradation and fit_floor_non_degradation
    )
    before_old = legacy._metric_block(
        labels[held & old], before_predictions, old_classes
    )
    after_old = legacy._metric_block(
        labels[held & old], after_old_predictions, old_classes
    )
    after_new = legacy._metric_block(
        labels[held & new], after_new_predictions, new_classes
    )
    training_trace = list(before_fit.loss_trace) + list(after_fit.loss_trace)
    bias_audit = json.loads(after.bias_audit_json)
    resource = dict(after.resource_audit())
    resource.update(
        {
            "old_support_non_degradation_pass": old_support_non_degradation,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "complete_loss_trace": training_trace,
            "new_group_bias": float(after.new_group_bias),
            "new_group_bias_support_only_audit": bias_audit,
            "query_features_used_for_fit": False,
            "query_labels_used_for_fit": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return {
        "candidate_id": candidate_id,
        "held_ranks": list(held_ranks),
        "fit_k_shot": 8,
        "before_old": before_old,
        "after_old": after_old,
        "after_new": after_new,
        "H_old_new": legacy._harmonic(
            float(after_old["overall_accuracy"]),
            float(after_new["overall_accuracy"]),
        ),
        "forgetting": float(
            before_old["overall_accuracy"] - after_old["overall_accuracy"]
        ),
        "joint_floor": float(
            min(
                float(after_old["class_floor_accuracy"]),
                float(after_new["class_floor_accuracy"]),
            )
        ),
        "old_score_columns_bitwise_unchanged": True,
        "old_prefix_sha256_before": before.old_lock_sha256,
        "old_prefix_sha256_after": after.old_lock_sha256,
        "fit_old_before_registration": fit_before,
        "fit_old_after_registration": fit_after,
        "old_support_classwise_non_degradation": fit_classwise_non_degradation,
        "old_support_floor_non_degradation": fit_floor_non_degradation,
        "old_support_non_degradation_pass": old_support_non_degradation,
        "new_group_bias": float(after.new_group_bias),
        "new_group_bias_support_only_audit": bias_audit,
        "training_trace": training_trace,
        "geometry_summary": _d26_geometry(after),
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


def _pooled_scenario_classwise(
    rows: Sequence[Mapping[str, Any]], metric_key: str
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
        selected = [row for row in rows if row.get("scenario") == scenario]
        if len(selected) != len(HELD_RANKS):
            raise D25RunnerError("C3 pooled scenario fold cardinality drift")
        labels = tuple(selected[0][metric_key]["per_class_accuracy"])
        result[scenario] = {
            label: float(
                np.mean(
                    [
                        float(row[metric_key]["per_class_accuracy"][label])
                        for row in selected
                    ]
                )
            )
            for label in labels
        }
    return result


def _select_c3_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    baseline_rows = list(folds_by_candidate[D25_C0])
    baseline_old = _pooled_scenario_classwise(baseline_rows, "after_old")
    baseline_new = _pooled_scenario_classwise(baseline_rows, "after_new")
    baseline_h = float(np.mean([float(row["H_old_new"]) for row in baseline_rows]))
    baseline_forgetting = float(
        np.mean([float(row["forgetting"]) for row in baseline_rows])
    )
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, float, int]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d25_c3"
                if candidate_id in C3_CANDIDATES
                else ("d25" if candidate_id == D25_C0 else "control")
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id == DIAG_CANDIDATE,
            "eligible_positive_route": False,
        }
        if candidate_id not in C3_CANDIDATES:
            decisions.append(decision)
            continue
        candidate_old = _pooled_scenario_classwise(rows, "after_old")
        candidate_new = _pooled_scenario_classwise(rows, "after_new")
        safety = True
        old_floor_gain: list[float] = []
        new_floor_gain: list[float] = []
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            for label, baseline_value in baseline_old[scenario].items():
                safety = safety and (
                    candidate_old[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            for label, baseline_value in baseline_new[scenario].items():
                safety = safety and (
                    candidate_new[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            old_floor_gain.append(
                min(candidate_old[scenario].values())
                - min(baseline_old[scenario].values())
            )
            new_floor_gain.append(
                min(candidate_new[scenario].values())
                - min(baseline_new[scenario].values())
            )
        old_support_pass = all(
            bool(row["old_support_non_degradation_pass"]) for row in rows
        )
        mean_h = float(np.mean([float(row["H_old_new"]) for row in rows]))
        mean_forgetting = float(
            np.mean([float(row["forgetting"]) for row in rows])
        )
        floor_pass = bool(
            all(value >= 0.10 - 1.0e-12 for value in old_floor_gain)
            and all(value >= 0.10 - 1.0e-12 for value in new_floor_gain)
        )
        balance_pass = bool(
            mean_h + 1.0e-12 >= baseline_h
            and mean_forgetting <= baseline_forgetting + 1.0e-12
        )
        eligible_positive = bool(
            safety and floor_pass and balance_pass and old_support_pass
        )
        decision.update(
            {
                "pooled_per_class_safety_vs_C0_pass": safety,
                "pooled_old_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, old_floor_gain)
                ),
                "pooled_new_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, new_floor_gain)
                ),
                "pooled_floor_gate_pass": floor_pass,
                "old_support_non_degradation_all_folds": old_support_pass,
                "mean_H_noninferior_vs_C0": mean_h + 1.0e-12 >= baseline_h,
                "mean_forgetting_noninferior_vs_C0": mean_forgetting
                <= baseline_forgetting + 1.0e-12,
                "eligible_positive_route": eligible_positive,
            }
        )
        decisions.append(decision)
        if eligible_positive:
            steps = int(rows[0]["resource"]["total_optimizer_steps"])
            eligible.append(
                (
                    candidate_id,
                    min(min(old_floor_gain), min(new_floor_gain)),
                    float(aggregate["worst_joint_floor"]),
                    mean_h,
                    -mean_forgetting,
                    -steps,
                )
            )
    selected = (
        max(eligible, key=lambda value: value[1:])[0] if eligible else D25_C0
    )
    return selected, decisions


def _select_d26_candidate(
    folds_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, list[dict[str, Any]]]:
    baseline_rows = list(folds_by_candidate[D25_C0])
    diagnostic_rows = list(folds_by_candidate[DIAG_CANDIDATE])
    baseline_old = _pooled_scenario_classwise(baseline_rows, "after_old")
    baseline_new = _pooled_scenario_classwise(baseline_rows, "after_new")
    diagnostic_old = _pooled_scenario_classwise(diagnostic_rows, "after_old")
    diagnostic_new = _pooled_scenario_classwise(diagnostic_rows, "after_new")
    baseline_h = float(np.mean([float(row["H_old_new"]) for row in baseline_rows]))
    baseline_forgetting = float(
        np.mean([float(row["forgetting"]) for row in baseline_rows])
    )
    diagnostic_h = float(
        np.mean([float(row["H_old_new"]) for row in diagnostic_rows])
    )
    diagnostic_forgetting = float(
        np.mean([float(row["forgetting"]) for row in diagnostic_rows])
    )
    decisions: list[dict[str, Any]] = []
    eligible: list[tuple[str, float, float, float, float, int]] = []
    for candidate_id, raw_rows in folds_by_candidate.items():
        rows = list(raw_rows)
        aggregate = legacy._aggregate_candidate(rows)
        decision: dict[str, Any] = {
            **aggregate,
            "candidate_id": candidate_id,
            "family": (
                "d26_compact_diag"
                if candidate_id in D26_CANDIDATES
                else ("d25" if candidate_id == D25_C0 else "control")
            ),
            "fallback": candidate_id == D25_C0,
            "diagnostic_only": candidate_id == DIAG_CANDIDATE,
            "eligible_positive_route": False,
        }
        if candidate_id not in D26_CANDIDATES:
            decisions.append(decision)
            continue
        candidate_old = _pooled_scenario_classwise(rows, "after_old")
        candidate_new = _pooled_scenario_classwise(rows, "after_new")
        safety = True
        old_floor_gain: list[float] = []
        new_floor_gain: list[float] = []
        old_floor_delta_vs_b3: list[float] = []
        new_floor_delta_vs_b3: list[float] = []
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            for label, baseline_value in baseline_old[scenario].items():
                safety = safety and (
                    candidate_old[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            for label, baseline_value in baseline_new[scenario].items():
                safety = safety and (
                    candidate_new[scenario][label] + 1.0e-12
                    >= baseline_value - 0.10
                )
            old_floor_gain.append(
                min(candidate_old[scenario].values())
                - min(baseline_old[scenario].values())
            )
            new_floor_gain.append(
                min(candidate_new[scenario].values())
                - min(baseline_new[scenario].values())
            )
            old_floor_delta_vs_b3.append(
                min(candidate_old[scenario].values())
                - min(diagnostic_old[scenario].values())
            )
            new_floor_delta_vs_b3.append(
                min(candidate_new[scenario].values())
                - min(diagnostic_new[scenario].values())
            )
        old_support_pass = all(
            bool(row["old_support_non_degradation_pass"]) for row in rows
        )
        mean_h = float(np.mean([float(row["H_old_new"]) for row in rows]))
        mean_forgetting = float(
            np.mean([float(row["forgetting"]) for row in rows])
        )
        floor_pass = bool(
            all(value >= 0.10 - 1.0e-12 for value in old_floor_gain)
            and all(value >= 0.10 - 1.0e-12 for value in new_floor_gain)
        )
        balance_pass = bool(
            mean_h + 1.0e-12 >= baseline_h
            and mean_forgetting <= baseline_forgetting + 1.0e-12
        )
        eligible_positive = bool(
            safety and floor_pass and balance_pass and old_support_pass
        )
        decision.update(
            {
                "pooled_per_class_safety_vs_C0_pass": safety,
                "pooled_old_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, old_floor_gain)
                ),
                "pooled_new_floor_gain_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, new_floor_gain)
                ),
                "pooled_floor_gate_pass": floor_pass,
                "old_support_non_degradation_all_folds": old_support_pass,
                "mean_H_noninferior_vs_C0": mean_h + 1.0e-12 >= baseline_h,
                "mean_forgetting_noninferior_vs_C0": mean_forgetting
                <= baseline_forgetting + 1.0e-12,
                "B3_performance_reference_only": True,
                "mean_H_delta_vs_B3": mean_h - diagnostic_h,
                "mean_forgetting_delta_vs_B3": (
                    mean_forgetting - diagnostic_forgetting
                ),
                "pooled_old_floor_delta_vs_B3_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, old_floor_delta_vs_b3)
                ),
                "pooled_new_floor_delta_vs_B3_by_scenario": dict(
                    zip(legacy.FORMAL_LEO_WEAK_SCENARIOS, new_floor_delta_vs_b3)
                ),
                "eligible_positive_route": eligible_positive,
            }
        )
        decisions.append(decision)
        if eligible_positive:
            steps = int(rows[0]["resource"]["total_optimizer_steps"])
            eligible.append(
                (
                    candidate_id,
                    min(min(old_floor_gain), min(new_floor_gain)),
                    float(aggregate["worst_joint_floor"]),
                    mean_h,
                    -mean_forgetting,
                    -steps,
                )
            )
    selected = (
        max(eligible, key=lambda value: value[1:])[0] if eligible else D25_C0
    )
    return selected, decisions


def _apply_full_k10_c3_old_support_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in C3_CANDIDATES:
            continue
        by_scenario = {
            scenario: bool(
                deployment_resources[candidate_id][scenario][
                    "old_support_non_degradation_pass"
                ]
            )
            for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        full_pass = all(by_scenario.values())
        decision["full_k10_old_support_non_degradation_by_scenario"] = by_scenario
        decision["full_k10_old_support_non_degradation_pass"] = full_pass
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in C3_CANDIDATES:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_old_support_non_degradation_pass"]):
        return selected_id, None
    return D25_C0, "FULL_K10_OLD_SUPPORT_NON_DEGRADATION_FAILED"


def _apply_full_k10_d26_old_support_gate(
    selected_id: str,
    candidate_decisions: list[dict[str, Any]],
    deployment_resources: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[str, str | None]:
    for decision in candidate_decisions:
        candidate_id = str(decision["candidate_id"])
        if candidate_id not in D26_CANDIDATES:
            continue
        by_scenario = {
            scenario: bool(
                deployment_resources[candidate_id][scenario][
                    "old_support_non_degradation_pass"
                ]
            )
            for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS
        }
        full_pass = all(by_scenario.values())
        decision["full_k10_old_support_non_degradation_by_scenario"] = by_scenario
        decision["full_k10_old_support_non_degradation_pass"] = full_pass
        decision["eligible_positive_route"] = bool(
            decision.get("eligible_positive_route", False) and full_pass
        )
    if selected_id not in D26_CANDIDATES:
        return selected_id, None
    selected_decision = next(
        row for row in candidate_decisions if row["candidate_id"] == selected_id
    )
    if bool(selected_decision["full_k10_old_support_non_degradation_pass"]):
        return selected_id, None
    return D25_C0, "FULL_K10_OLD_SUPPORT_NON_DEGRADATION_FAILED"


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


def _full_c3_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D25C3Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = build_concat288(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    before_fit = fit_stage2b_diag_floor(
        features[old], labels[old], old_classes, config=config
    )
    before = before_fit.state
    after_fit = append_stage2c_new_suffix(
        before, features[new], labels[new], new_classes
    )
    after = after_fit.state
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    if (
        before.old_prefix_sha256 != after.old_prefix_sha256
        or before.shared_sha256 != after.shared_sha256
    ):
        raise D25RunnerError("C3 deployment frozen state drift")
    before_old_predictions = [predict_one_c3(before, row)[0] for row in features[old]]
    after_old_predictions = [predict_one_c3(after, row)[0] for row in features[old]]
    before_old_metric = legacy._metric_block(
        labels[old], before_old_predictions, old_classes
    )
    after_old_metric = legacy._metric_block(
        labels[old], after_old_predictions, old_classes
    )
    old_support_non_degradation = all(
        float(after_old_metric["per_class_accuracy"][label]) + 1.0e-12
        >= float(before_old_metric["per_class_accuracy"][label])
        for label in old_classes
    )
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        score_one_c3(after, feature)
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = dict(after.resource_audit())
    registered_count = len(old_classes) + len(new_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": registered_count,
            "old_prefix_sha256": after.old_prefix_sha256,
            "shared_sha256": after.shared_sha256,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "old_support_before_registration": before_old_metric,
            "old_support_after_registration": after_old_metric,
            "old_support_non_degradation_pass": old_support_non_degradation,
            "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
            "identity_single_qknn_fp16_sample_state_bytes": identity_qknn_fp16_state_bytes,
            "estimated_score_mac_ratio_vs_identity_single_qknn": float(
                resource["estimated_head_macs_per_query"] / identity_qknn_macs
            ),
            "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
                resource["persistent_state_bytes"]
                / identity_qknn_fp16_state_bytes
            ),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32",
            "complete_loss_trace": list(before_fit.training_trace)
            + list(after_fit.training_trace),
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return resource, _c3_geometry(after)


def _full_d26_state_audit(
    rows: Mapping[str, np.ndarray],
    z_id160: np.ndarray,
    fft96: np.ndarray,
    rf32: np.ndarray,
    *,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
    config: D26CompactDiagConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = np.asarray(rows["labels"]).astype(str)
    old = np.isin(labels, np.asarray(old_classes))
    new = np.isin(labels, np.asarray(new_classes))
    features = build_concat288(z_id160, fft96, rf32)
    fit_started = time.perf_counter()
    before_fit = fit_stage2b_compact_diag(
        features[old], labels[old], old_classes, config=config
    )
    before = before_fit.state
    after_fit = append_stage2c_d26(
        before,
        features[new],
        labels[new],
        new_classes,
        features[old],
        labels[old],
    )
    after = after_fit.state
    fit_elapsed_ms = (time.perf_counter() - fit_started) * 1000.0
    if (
        before.old_lock_sha256 != after.old_lock_sha256
        or before.log_diag.tobytes() != after.log_diag.tobytes()
        or before.weights.tobytes()
        != after.weights[: len(old_classes)].tobytes()
    ):
        raise D25RunnerError("D26 deployment frozen state drift")
    before_old_scores = score_all_d26(before, features[old])
    after_old_scores = score_all_d26(after, features[old])
    if not np.array_equal(
        before_old_scores, after_old_scores[:, : len(old_classes)]
    ):
        raise D25RunnerError("D26 deployment old raw score prefix drift")
    before_old_predictions = (
        predict_all_d26(before, features[old]).astype(str).tolist()
    )
    after_old_predictions = (
        predict_all_d26(after, features[old]).astype(str).tolist()
    )
    before_old_metric = legacy._metric_block(
        labels[old], before_old_predictions, old_classes
    )
    after_old_metric = legacy._metric_block(
        labels[old], after_old_predictions, old_classes
    )
    classwise_pass = all(
        float(after_old_metric["per_class_accuracy"][label]) + 1.0e-12
        >= float(before_old_metric["per_class_accuracy"][label])
        for label in old_classes
    )
    floor_pass = (
        float(after_old_metric["class_floor_accuracy"]) + 1.0e-12
        >= float(before_old_metric["class_floor_accuracy"])
    )
    old_support_non_degradation = bool(classwise_pass and floor_pass)
    score_elapsed_ms: list[float] = []
    for feature in features:
        score_started = time.perf_counter()
        score_all_d26(after, feature[None, :])
        score_elapsed_ms.append((time.perf_counter() - score_started) * 1000.0)
    resource = dict(after.resource_audit())
    registered_count = len(old_classes) + len(new_classes)
    identity_qknn_macs = registered_count * 10 * 160
    identity_qknn_fp16_state_bytes = registered_count * 10 * 160 * 2
    resource.update(
        {
            "deployment_k_shot": 10,
            "registered_class_count": registered_count,
            "old_prefix_sha256": after.old_lock_sha256,
            "old_score_columns_bitwise_unchanged_after_registration": True,
            "old_support_before_registration": before_old_metric,
            "old_support_after_registration": after_old_metric,
            "old_support_classwise_non_degradation_pass": classwise_pass,
            "old_support_floor_non_degradation_pass": floor_pass,
            "old_support_non_degradation_pass": old_support_non_degradation,
            "new_group_bias": float(after.new_group_bias),
            "new_group_bias_support_only_audit": json.loads(
                after.bias_audit_json
            ),
            "identity_single_qknn_estimated_score_macs_per_query": identity_qknn_macs,
            "identity_single_qknn_fp16_sample_state_bytes": identity_qknn_fp16_state_bytes,
            "estimated_score_mac_ratio_vs_identity_single_qknn": float(
                resource["estimated_macs_per_query"] / identity_qknn_macs
            ),
            "persistent_state_ratio_vs_identity_single_qknn_fp16": float(
                resource["persistent_state_bytes"]
                / identity_qknn_fp16_state_bytes
            ),
            "support_adaptation_and_registration_elapsed_ms": fit_elapsed_ms,
            "batch1_head_latency_mean_ms": float(np.mean(score_elapsed_ms)),
            "batch1_head_latency_p95_ms": float(
                np.quantile(np.asarray(score_elapsed_ms, dtype=np.float64), 0.95)
            ),
            "batch1_head_latency_sample_count": len(score_elapsed_ms),
            "head_peak_cuda_memory_bytes": 0,
            "head_runtime": "numpy_cpu_fp32",
            "complete_loss_trace": list(before_fit.loss_trace)
            + list(after_fit.loss_trace),
            "query_role_oracle_access": False,
            "query_true_batch_class_count_access": False,
            "query_class_quota_access": False,
            "query_batch_global_assignment": False,
            "source_sample_access": False,
            "clean_sample_access": False,
        }
    )
    return resource, _d26_geometry(after)


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
    candidate_set: str = CANDIDATE_SET_D25_V4,
) -> dict[str, Any]:
    if mode != MODE:
        raise D25RunnerError("D25 runner is development support-only")
    if output.exists():
        raise D25RunnerError("output path already exists")
    candidates = preregistered_candidates(candidate_set)
    candidate_lock = _candidate_lock(candidates, candidate_set)

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
                if isinstance(config, D25C3Config):
                    row = _evaluate_c3_fold(
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
                elif isinstance(config, D26CompactDiagConfig):
                    row = _evaluate_d26_fold(
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
                elif candidate_id in D25_CANDIDATES:
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
    if len(training_log) != expected_rows:
        raise D25RunnerError("D25 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D25_V4 and expected_rows != 75:
        raise D25RunnerError("D25 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_C3_V1 and expected_rows != 90:
        raise D25RunnerError("D25 training-log cardinality drift")
    if candidate_set == CANDIDATE_SET_D26_V1 and expected_rows != 90:
        raise D25RunnerError("D26 training-log cardinality drift")
    selected_id, candidate_decisions = (
        _select_c3_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_C3_V1
        else _select_d26_candidate(folds_by_candidate)
        if candidate_set == CANDIDATE_SET_D26_V1
        else _select_candidate(folds_by_candidate)
    )

    deployment_resources: dict[str, dict[str, Any]] = {
        candidate_id: {} for candidate_id in candidates
    }
    geometry_ids = (
        (D25_C0,) + C3_CANDIDATES
        if candidate_set == CANDIDATE_SET_C3_V1
        else (D25_C0,) + D26_CANDIDATES
        if candidate_set == CANDIDATE_SET_D26_V1
        else D25_CANDIDATES
    )
    geometry_matrix: dict[str, dict[str, Any]] = {
        candidate_id: {} for candidate_id in geometry_ids
    }
    for candidate_id, config in candidates.items():
        for scenario in legacy.FORMAL_LEO_WEAK_SCENARIOS:
            if isinstance(config, D25C3Config):
                resource, geometry = _full_c3_state_audit(
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
            elif isinstance(config, D26CompactDiagConfig):
                resource, geometry = _full_d26_state_audit(
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
            elif candidate_id in D25_CANDIDATES:
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

    pre_full_k10_selected_id = selected_id
    full_k10_fallback_reason: str | None = None
    if candidate_set == CANDIDATE_SET_C3_V1:
        selected_id, full_k10_fallback_reason = (
            _apply_full_k10_c3_old_support_gate(
                selected_id, candidate_decisions, deployment_resources
            )
        )
    elif candidate_set == CANDIDATE_SET_D26_V1:
        selected_id, full_k10_fallback_reason = (
            _apply_full_k10_d26_old_support_gate(
                selected_id, candidate_decisions, deployment_resources
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
        "pre_full_k10_selected_candidate_id": pre_full_k10_selected_id,
        "full_k10_fallback_reason": full_k10_fallback_reason,
        "selected_positive_route": selected_id
        in (
            C3_CANDIDATES
            if candidate_set == CANDIDATE_SET_C3_V1
            else D26_CANDIDATES
            if candidate_set == CANDIDATE_SET_D26_V1
            else D25_CANDIDATES
        ),
        "fallback_to_identity": selected_id == IDENTITY_CANDIDATE,
        "selection_baseline": (
            D25_C0
            if candidate_set in (CANDIDATE_SET_C3_V1, CANDIDATE_SET_D26_V1)
            else IDENTITY_CANDIDATE
        ),
        "diagnostic_comparator": DIAG_CANDIDATE,
        "eligible_candidate_ids": list(
            C3_CANDIDATES
            if candidate_set == CANDIDATE_SET_C3_V1
            else D26_CANDIDATES
            if candidate_set == CANDIDATE_SET_D26_V1
            else D25_CANDIDATES
        ),
        "selection_rule": (
            "C3:_all_fold_old_support_non_degradation,_per_scenario_pooled_"
            "old_and_new_floor_gain>=0.10,_per_class_drop<=0.10,_H_and_"
            "forgetting_noninferior_vs_C0;_B3_diagnostic_only"
            if candidate_set == CANDIDATE_SET_C3_V1
            else "D26:_all_fold_old_support_non_degradation,_per_scenario_"
            "pooled_old_and_new_floor_gain>=0.10,_per_class_drop<=0.10,_H_"
            "and_forgetting_noninferior_vs_C0;_B3_performance_reference_only"
            if candidate_set == CANDIDATE_SET_D26_V1
            else "all_15_folds_classwise_noninferior_vs_Z0_and_strict_worst_"
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

    current_source_closure = _candidate_lock(candidates, candidate_set)[
        "source_closure"
    ]
    if current_source_closure != candidate_lock["source_closure"]:
        raise D25RunnerError("D25 source closure changed after support opening")
    receipt = {
        "schema": "cvs.phase2.d25.receipt.v1",
        "status": "DEVELOPMENT_SUPPORT_ONLY_COMPLETE",
        "mode": mode,
        "core_commit": CORE_COMMIT,
        **(
            {
                "phase1_core_commit": CORE_COMMIT,
                "d26_core_git_commit": D26_CORE_GIT_COMMIT,
            }
            if candidate_set == CANDIDATE_SET_D26_V1
            else {}
        ),
        **candidate_lock["source_closure"],
        "source_closure_unchanged_after_support": True,
        "candidate_lock_sha256": candidate_lock["sha256"],
        "selected_candidate_id": selected_id,
        "pre_full_k10_selected_candidate_id": pre_full_k10_selected_id,
        "full_k10_fallback_reason": full_k10_fallback_reason,
        "selected_positive_route": selected_id
        in (
            C3_CANDIDATES
            if candidate_set == CANDIDATE_SET_C3_V1
            else D26_CANDIDATES
            if candidate_set == CANDIDATE_SET_D26_V1
            else D25_CANDIDATES
        ),
        "candidate_set": candidate_set,
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
    parser.add_argument(
        "--candidate-set",
        choices=(
            CANDIDATE_SET_D25_V4,
            CANDIDATE_SET_C3_V1,
            CANDIDATE_SET_D26_V1,
        ),
        default=CANDIDATE_SET_D25_V4,
    )
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
        candidate_set=args.candidate_set,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
