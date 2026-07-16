"""D15 CL-CDR conservative class-local diagonal density-ratio envelope."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_joint_residual_logit_head import (
    RuntimeAuthorizedFeatureArtifact,
)


EPS = 1.0e-8
SCHEMA = "cvs.phase2.cl_cdr_envelope.v2"
MAX_STATE_BYTES = 256 * 1024


class ClCdrEnvelopeError(ValueError):
    """Raised when the CL-CDR support-only contract fails closed."""


@dataclass(frozen=True)
class ClCdrHyperparameters:
    candidate_id: str
    rank: int
    shrink: float
    ridge: float
    gamma: float
    min_stability: float
    own_llr_quantile: float = 0.20
    rest_llr_quantile: float = 0.80
    min_llr_gap: float = 0.0
    margin_band: float = 1.0
    operator_id: str = "base"
    force_zero: bool = False


@dataclass(frozen=True)
class ClCdrEnvelopeState:
    schema: str
    candidate_id: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    selected_dims: np.ndarray
    class_mean: np.ndarray
    class_var: np.ndarray
    rest_mean: np.ndarray
    rest_var: np.ndarray
    enabled: np.ndarray
    stability: np.ndarray
    llr_mid: np.ndarray
    llr_half_gap: np.ndarray
    hyperparameters: ClCdrHyperparameters
    feature_dim: int
    k_shot: int
    old_class_count: int
    registration_generation: int
    resource: Mapping[str, Any]
    support_feature_artifact_sha256: str
    support_selection_sha256: str
    sealed_runtime_sha256: str
    feature_code_sha256: str
    sealed_phase1_checkpoint_sha256: str
    operator_id: str
    view_seed: int
    state_content_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "prototypes",
            "class_mean",
            "class_var",
            "rest_mean",
            "rest_var",
            "stability",
            "llr_mid",
            "llr_half_gap",
        ):
            source = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            object.__setattr__(
                self,
                name,
                np.frombuffer(source.tobytes(), dtype=np.float32).reshape(
                    source.shape
                ),
            )
        dims = np.ascontiguousarray(self.selected_dims, dtype=np.int64)
        object.__setattr__(
            self,
            "selected_dims",
            np.frombuffer(dims.tobytes(), dtype=np.int64).reshape(dims.shape),
        )
        enabled = np.ascontiguousarray(self.enabled, dtype=np.bool_)
        object.__setattr__(
            self,
            "enabled",
            np.frombuffer(enabled.tobytes(), dtype=np.bool_).reshape(enabled.shape),
        )
        computed = _state_sha(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise ClCdrEnvelopeError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)
        _validate_state(self)


@dataclass(frozen=True)
class BeforeAfterClCdrFit:
    before_state: ClCdrEnvelopeState
    after_state: ClCdrEnvelopeState
    trace: tuple[dict[str, Any], ...]


def _validate_hp(hp: ClCdrHyperparameters) -> None:
    if (
        not hp.candidate_id
        or hp.operator_id != "base"
        or not 0 <= int(hp.rank) <= 16
        or not np.isfinite(hp.shrink)
        or not 0.0 <= hp.shrink <= 1.0
        or not np.isfinite(hp.ridge)
        or hp.ridge <= 0.0
        or not np.isfinite(hp.gamma)
        or hp.gamma < 0.0
        or not np.isfinite(hp.min_stability)
        or not 0.0 <= hp.min_stability <= 1.0
        or not 0.0 <= hp.own_llr_quantile <= 1.0
        or not 0.0 <= hp.rest_llr_quantile <= 1.0
        or not np.isfinite(hp.min_llr_gap)
        or hp.min_llr_gap < 0.0
        or not np.isfinite(hp.margin_band)
        or hp.margin_band < 0.0
        or (
            hp.force_zero
            and (hp.rank != 0 or hp.gamma != 0.0)
        )
    ):
        raise ClCdrEnvelopeError("hyperparameter drift")


def _rows(artifact: RuntimeAuthorizedFeatureArtifact) -> np.ndarray:
    if not isinstance(artifact, RuntimeAuthorizedFeatureArtifact):
        raise ClCdrEnvelopeError("runtime-authorized feature artifact required")
    return artifact.features


def _validate_support(artifact, labels, ranks, k):
    rows = _rows(artifact)
    labels = np.asarray(labels).astype(str)
    ranks = np.asarray(ranks, dtype=np.int64)
    if len(rows) != len(labels) or len(rows) != len(ranks):
        raise ClCdrEnvelopeError("support alignment drift")
    classes, counts = np.unique(labels, return_counts=True)
    if (
        set(counts.tolist()) != {int(k)}
        or any(set(ranks[labels == label]) != set(range(int(k))) for label in classes)
        or int(k) < 1
    ):
        raise ClCdrEnvelopeError("strict physical K-shot drift")
    return rows, labels, ranks, tuple(sorted(classes.tolist()))


def _validate_binding(before, after):
    if (
        before.sealed_runtime_sha256 != after.sealed_runtime_sha256
        or before.feature_code_sha256 != after.feature_code_sha256
        or before.sealed_phase1_checkpoint_sha256
        != after.sealed_phase1_checkpoint_sha256
        or before.operator_id != after.operator_id
        or before.view_seed != after.view_seed
        or before.operator_id != "base"
    ):
        raise ClCdrEnvelopeError("runtime/operator binding drift")


def _validate_old_reuse(before, bl, br, after, al, ar, old_classes):
    allowed = set(old_classes)

    def keyed(artifact, labels, ranks):
        return {
            (str(labels[i]), int(ranks[i])): (
                artifact.physical_sample_ids[i],
                artifact.parent_received_iq_sha256[i],
                artifact.per_row_feature_sha256[i],
            )
            for i in range(len(labels))
            if str(labels[i]) in allowed
        }

    if keyed(before, bl, br) != keyed(after, al, ar):
        raise ClCdrEnvelopeError("old exact-reuse lock failed")


def _normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), EPS)


def _prototypes(rows, labels, classes):
    z = _normalize(rows)
    means = np.stack([np.mean(z[labels == label], axis=0) for label in classes])
    return _normalize(means).astype(np.float32)


def _select_stats(
    rows: np.ndarray,
    labels: np.ndarray,
    label: str,
    hp: ClCdrHyperparameters,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]] | None:
    z = _normalize(rows)
    own = z[labels == label]
    rest = z[labels != label]
    if len(own) < 2 or not len(rest):
        return None
    mu_c = np.mean(own, axis=0)
    mu_r = np.mean(rest, axis=0)
    raw_c = np.var(own, axis=0, ddof=0)
    raw_r = np.var(rest, axis=0, ddof=0)
    pooled = (
        max(len(own) - 1, 1) * raw_c + max(len(rest) - 1, 1) * raw_r
    ) / max(len(own) + len(rest) - 2, 1)
    var_c = (1.0 - hp.shrink) * raw_c + hp.shrink * pooled + hp.ridge
    var_r = raw_r + hp.ridge
    fisher = np.square(mu_c - mu_r) / (var_c + var_r)
    rank = min(int(hp.rank), len(fisher))
    dims = np.argsort(-fisher, kind="stable")[:rank].astype(np.int64)
    return dims, (
        mu_c[dims].astype(np.float32),
        var_c[dims].astype(np.float32),
        mu_r[dims].astype(np.float32),
        var_r[dims].astype(np.float32),
    )


def _llr_from_stats(
    row: np.ndarray,
    dims: np.ndarray,
    stats: tuple[np.ndarray, ...],
) -> float:
    mu_c, var_c, mu_r, var_r = stats
    current = row[dims]
    return float(
        -0.5
        * np.mean(
            np.square(current - mu_c) / var_c
            - np.square(current - mu_r) / var_r
            + np.log(var_c / var_r)
        )
    )


def _cross_fitted_class_safety(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    class_index: int,
    hp: ClCdrHyperparameters,
) -> dict[str, Any]:
    z = _normalize(rows)
    label = classes[class_index]
    own_llr: list[float] = []
    rest_llr: list[float] = []
    records: list[tuple[int, np.ndarray, int, float]] = []
    for held_index, held in enumerate(z):
        keep = np.ones(len(rows), dtype=bool)
        keep[held_index] = False
        current = _select_stats(rows[keep], labels[keep], label, hp)
        if current is None:
            return {
                "safety_pass": False,
                "reason": "inner_loo_stats_unavailable",
            }
        dims, stats = current
        loo_prototypes = _prototypes(rows[keep], labels[keep], classes)
        base_scores = held @ loo_prototypes.T
        base_prediction = int(np.argmax(base_scores))
        llr = _llr_from_stats(held, dims, stats)
        truth = classes.index(str(labels[held_index]))
        if truth == class_index:
            own_llr.append(llr)
        else:
            rest_llr.append(llr)
        records.append((truth, base_scores, base_prediction, llr))
    if not own_llr or not rest_llr:
        return {"safety_pass": False, "reason": "empty_own_or_rest_llr"}
    q_pos = float(
        np.quantile(own_llr, hp.own_llr_quantile, method="linear")
    )
    q_neg = float(
        np.quantile(rest_llr, hp.rest_llr_quantile, method="linear")
    )
    gap = q_pos - q_neg
    gap_pass = gap > hp.min_llr_gap
    mid = 0.5 * (q_pos + q_neg)
    half_gap = 0.5 * gap
    own_base = sum(
        int(base_prediction == class_index)
        for truth, _, base_prediction, _ in records
        if truth == class_index
    )
    own_candidate = 0
    added_capture = {value: 0 for value in classes if value != label}
    nested_calibration = []
    nested_stability = []
    for held_index, (truth, base_scores, base_prediction, llr) in enumerate(records):
        calibration = _nested_l2o_calibration(
            rows, labels, classes, class_index, held_index, hp
        )
        if calibration is None:
            return {
                "safety_pass": False,
                "reason": "nested_l2o_calibration_unavailable",
            }
        nested_calibration.append(calibration)
        nested_stability.append(calibration["stability"])
        candidate_scores = np.array(base_scores, dtype=np.float32, copy=True)
        rival = float(np.max(np.delete(base_scores, class_index)))
        active = float(base_scores[class_index]) - rival >= -hp.margin_band
        if active:
            conformal = np.clip(
                (llr - calibration["mid"])
                / (calibration["half_gap"] + EPS),
                -1.0,
                1.0,
            )
            candidate_scores[class_index] += (
                np.float32(hp.gamma) * np.float32(conformal)
            )
        candidate_prediction = int(np.argmax(candidate_scores))
        if truth == class_index:
            own_candidate += int(candidate_prediction == class_index)
        elif (
            base_prediction != class_index
            and candidate_prediction == class_index
        ):
            added_capture[classes[truth]] += 1
    own_non_degrade = own_candidate >= own_base
    zero_capture = all(value == 0 for value in added_capture.values())
    nested_gap_pass = all(row["gap_pass"] for row in nested_calibration)
    nested_stability_pass = all(
        value + 1.0e-12 >= hp.min_stability for value in nested_stability
    )
    safety_pass = bool(
        own_non_degrade
        and zero_capture
        and gap_pass
        and nested_gap_pass
        and nested_stability_pass
    )
    return {
        "safety_pass": safety_pass,
        "reason": "pass" if safety_pass else "class_safety_gate_failed",
        "evaluated_inner_held": len(records),
        "own_correct_base": own_base,
        "own_correct_candidate": own_candidate,
        "own_non_degrade": own_non_degrade,
        "added_capture_by_other_truth_class": added_capture,
        "zero_added_capture": zero_capture,
        "own_llr_low_quantile": q_pos,
        "rest_llr_high_quantile": q_neg,
        "llr_gap": gap,
        "llr_gap_pass": gap_pass,
        "nested_l2o_gap_pass": nested_gap_pass,
        "nested_l2o_stability_pass": nested_stability_pass,
        "minimum_nested_stability": min(nested_stability),
        "nested_l2o_calibration": tuple(nested_calibration),
        "llr_mid": mid,
        "llr_half_gap": half_gap,
        "conformal_clip": [-1.0, 1.0],
        "own_llr_quantile": hp.own_llr_quantile,
        "rest_llr_quantile": hp.rest_llr_quantile,
        "min_llr_gap": hp.min_llr_gap,
        "margin_band": hp.margin_band,
    }


def _nested_l2o_calibration(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    class_index: int,
    evaluated_index: int,
    hp: ClCdrHyperparameters,
) -> dict[str, Any] | None:
    z = _normalize(rows)
    label = classes[class_index]
    own: list[float] = []
    rest: list[float] = []
    nested_dims: list[list[int]] = []
    for calibration_index in range(len(rows)):
        if calibration_index == evaluated_index:
            continue
        keep = np.ones(len(rows), dtype=bool)
        keep[[evaluated_index, calibration_index]] = False
        current = _select_stats(rows[keep], labels[keep], label, hp)
        if current is None:
            return None
        dims, stats = current
        nested_dims.append(dims.tolist())
        llr = _llr_from_stats(z[calibration_index], dims, stats)
        if classes.index(str(labels[calibration_index])) == class_index:
            own.append(llr)
        else:
            rest.append(llr)
    if not own or not rest:
        return None
    q_pos = float(np.quantile(own, hp.own_llr_quantile, method="linear"))
    q_neg = float(np.quantile(rest, hp.rest_llr_quantile, method="linear"))
    gap = q_pos - q_neg
    overlaps = [
        len(set(nested_dims[left]) & set(nested_dims[right]))
        / max(int(hp.rank), 1)
        for left in range(len(nested_dims))
        for right in range(left + 1, len(nested_dims))
    ]
    stability = float(np.mean(overlaps)) if overlaps else 0.0
    return {
        "evaluated_record_index": evaluated_index,
        "calibration_excludes_evaluated_record": True,
        "every_calibration_model_excludes_evaluated_record": True,
        "own_calibration_count": len(own),
        "rest_calibration_count": len(rest),
        "q_pos": q_pos,
        "q_neg": q_neg,
        "gap": gap,
        "gap_pass": gap > hp.min_llr_gap,
        "mid": 0.5 * (q_pos + q_neg),
        "half_gap": 0.5 * gap,
        "stability": stability,
        "stability_policy": "nested_l2o_pairwise_consensus",
        "nested_reselected_dims": tuple(nested_dims),
    }


def _fit_envelopes(rows, labels, classes, targets, hp):
    rank = int(hp.rank)
    count = len(targets)
    dims = np.full((count, rank), -1, dtype=np.int64)
    class_mean = np.zeros((count, rank), dtype=np.float32)
    class_var = np.ones((count, rank), dtype=np.float32)
    rest_mean = np.zeros((count, rank), dtype=np.float32)
    rest_var = np.ones((count, rank), dtype=np.float32)
    enabled = np.zeros(count, dtype=np.bool_)
    stability = np.zeros(count, dtype=np.float32)
    llr_mid = np.zeros(count, dtype=np.float32)
    llr_half_gap = np.zeros(count, dtype=np.float32)
    diagnostics = []
    if hp.force_zero or rank == 0:
        return (
            dims, class_mean, class_var, rest_mean, rest_var, enabled,
            stability, llr_mid, llr_half_gap, (),
        )
    for output_index, class_index in enumerate(targets):
        label = classes[class_index]
        full = _select_stats(rows, labels, label, hp)
        if full is None:
            diagnostics.append({"class_handle": label, "enabled": False, "reason": "insufficient_rows"})
            continue
        full_dims, stats = full
        loo_dims = []
        for held in range(len(rows)):
            keep = np.ones(len(rows), dtype=bool)
            keep[held] = False
            current = _select_stats(rows[keep], labels[keep], label, hp)
            current_dims = np.empty((0,), dtype=np.int64) if current is None else current[0]
            loo_dims.append(current_dims.tolist())
        safety = _cross_fitted_class_safety(
            rows, labels, classes, class_index, hp
        )
        stable = float(safety.get("minimum_nested_stability", 0.0))
        stability[output_index] = np.float32(stable)
        if not bool(safety["safety_pass"]):
            diagnostics.append(
                {
                    "class_handle": label,
                    "enabled": False,
                    "reason": "cross_fitted_class_safety_failed",
                    "stability": stable,
                    "stability_policy": "minimum_nested_l2o_pairwise_consensus",
                    "full_dims": full_dims.tolist(),
                    "inner_loo_reselected_dims": loo_dims,
                    "class_safety": safety,
                }
            )
            continue
        enabled[output_index] = True
        dims[output_index] = full_dims
        class_mean[output_index], class_var[output_index], rest_mean[output_index], rest_var[output_index] = stats
        llr_mid[output_index] = np.float32(safety["llr_mid"])
        llr_half_gap[output_index] = np.float32(safety["llr_half_gap"])
        diagnostics.append(
            {
                "class_handle": label,
                "enabled": True,
                "stability": stable,
                "stability_policy": "minimum_nested_l2o_pairwise_consensus",
                "full_dims": full_dims.tolist(),
                "inner_loo_reselected_dims": loo_dims,
                "class_safety": safety,
            }
        )
    return (
        dims, class_mean, class_var, rest_mean, rest_var, enabled,
        stability, llr_mid, llr_half_gap, tuple(diagnostics),
    )


def _selection_sha(artifact, labels, ranks, selection):
    records = [
        (
            str(labels[i]),
            int(ranks[i]),
            artifact.physical_sample_ids[i],
            artifact.parent_received_iq_sha256[i],
            artifact.per_row_feature_sha256[i],
        )
        for i in np.flatnonzero(selection)
    ]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _array_bytes(state):
    return sum(
        value.nbytes
        for value in (
            state.prototypes, state.selected_dims, state.class_mean,
            state.class_var, state.rest_mean, state.rest_var,
            state.enabled, state.stability, state.llr_mid, state.llr_half_gap,
        )
    )


def _make_state(
    artifact, classes, prototypes, arrays, hp, *,
    k_shot, old_count, generation, selection_sha
):
    dims, cm, cv, rm, rv, enabled, stability, llr_mid, llr_half_gap = arrays
    rank = int(hp.rank)
    array_bytes = int(
        prototypes.nbytes + dims.nbytes + cm.nbytes + cv.nbytes
        + rm.nbytes + rv.nbytes + enabled.nbytes + stability.nbytes
        + llr_mid.nbytes + llr_half_gap.nbytes
    )
    return ClCdrEnvelopeState(
        schema=SCHEMA, candidate_id=hp.candidate_id, classes=classes,
        prototypes=prototypes, selected_dims=dims, class_mean=cm,
        class_var=cv, rest_mean=rm, rest_var=rv, enabled=enabled,
        stability=stability, llr_mid=llr_mid, llr_half_gap=llr_half_gap,
        hyperparameters=hp,
        feature_dim=int(prototypes.shape[1]), k_shot=int(k_shot),
        old_class_count=int(old_count), registration_generation=int(generation),
        resource={
            "trainable_parameters": 0, "adapt_epochs": 0,
            "persistent_array_state_bytes": array_bytes,
            "enabled_envelope_count": int(np.sum(enabled)),
            "rank": rank,
            "prototype_cosine_mac_per_sample": int(len(classes) * prototypes.shape[1]),
            "envelope_scalar_ops_per_sample_upper_bound": int(np.sum(enabled) * rank * 8),
            "backbone_forwards_per_physical_sample": 1,
            "fft_branches_per_physical_sample": 0,
            "dense_query_graph": False,
        },
        support_feature_artifact_sha256=artifact.artifact_sha256,
        support_selection_sha256=selection_sha,
        sealed_runtime_sha256=artifact.sealed_runtime_sha256,
        feature_code_sha256=artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=artifact.sealed_phase1_checkpoint_sha256,
        operator_id=artifact.operator_id, view_seed=artifact.view_seed,
    )


def _fit_selected(
    before_artifact, old_rows, old_labels, old_ranks, old_classes, old_selection,
    after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes, joint_selection, hp
):
    old_rows_s, old_labels_s = old_rows[old_selection], old_labels[old_selection]
    joint_rows_s, joint_labels_s = joint_rows[joint_selection], joint_labels[joint_selection]
    old_proto = _prototypes(old_rows_s, old_labels_s, old_classes)
    joint_proto = _prototypes(joint_rows_s, joint_labels_s, joint_classes)
    if not np.array_equal(old_proto, joint_proto[:len(old_classes)]):
        raise ClCdrEnvelopeError("old prototype lock failed")
    old_fit = _fit_envelopes(
        old_rows_s, old_labels_s, old_classes, tuple(range(len(old_classes))), hp
    )
    old_arrays, old_diag = old_fit[:-1], old_fit[-1]
    new_targets = tuple(range(len(old_classes), len(joint_classes)))
    new_fit = _fit_envelopes(
        joint_rows_s, joint_labels_s, joint_classes, new_targets, hp
    )
    new_arrays, new_diag = new_fit[:-1], new_fit[-1]
    combined = tuple(
        np.concatenate([old_arrays[index], new_arrays[index]], axis=0)
        for index in range(len(old_arrays))
    )
    before = _make_state(
        before_artifact, old_classes, old_proto, old_arrays, hp,
        k_shot=int(np.sum(old_selection & (old_labels == old_classes[0]))),
        old_count=len(old_classes), generation=0,
        selection_sha=_selection_sha(before_artifact, old_labels, old_ranks, old_selection)
    )
    after = _make_state(
        after_artifact, joint_classes, joint_proto, combined, hp,
        k_shot=int(np.sum(joint_selection & (joint_labels == joint_classes[0]))),
        old_count=len(old_classes), generation=1,
        selection_sha=_selection_sha(after_artifact, joint_labels, joint_ranks, joint_selection)
    )
    for name in (
        "selected_dims", "class_mean", "class_var", "rest_mean", "rest_var",
        "enabled", "stability", "llr_mid", "llr_half_gap",
    ):
        if not np.array_equal(getattr(before, name), getattr(after, name)[:len(old_classes)]):
            raise ClCdrEnvelopeError("old envelope lock failed")
    return BeforeAfterClCdrFit(
        before_state=before, after_state=after,
        trace=(
            {"phase": "before_cl_cdr_fit", "diagnostics": [dict(x) for x in old_diag],
             "support_selection_sha256": before.support_selection_sha256},
            {"phase": "after_cl_cdr_fit", "diagnostics": [dict(x) for x in new_diag],
             "support_selection_sha256": after.support_selection_sha256},
        )
    )


def fit_before_after_locked(
    before_artifact, before_labels, before_ranks,
    after_artifact, after_labels, after_ranks, *,
    k_shot, hyperparameters
):
    _validate_hp(hyperparameters)
    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot
    )
    joint_rows, joint_labels, joint_ranks, found = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot
    )
    if not set(old_classes) < set(found):
        raise ClCdrEnvelopeError("registration class drift")
    _validate_binding(before_artifact, after_artifact)
    _validate_old_reuse(
        before_artifact, old_labels, old_ranks,
        after_artifact, joint_labels, joint_ranks, old_classes
    )
    joint_classes = old_classes + tuple(sorted(set(found) - set(old_classes)))
    effective_hp = (
        _canonical_k1_zero_hyperparameters()
        if int(k_shot) == 1
        else hyperparameters
    )
    return _fit_selected(
        before_artifact, old_rows, old_labels, old_ranks, old_classes,
        np.ones(len(old_labels), dtype=bool),
        after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes,
        np.ones(len(joint_labels), dtype=bool), effective_hp
    )


def _canonical_k1_zero_hyperparameters() -> ClCdrHyperparameters:
    return ClCdrHyperparameters(
        candidate_id="d15_clcdr_k1_canonical_z0",
        rank=0,
        shrink=1.0,
        ridge=1.0e-8,
        gamma=0.0,
        min_stability=1.0,
        own_llr_quantile=0.20,
        rest_llr_quantile=0.80,
        min_llr_gap=0.0,
        margin_band=0.0,
        operator_id="base",
        force_zero=True,
    )


def _score_numpy(rows, state):
    _validate_state(state)
    z = _normalize(rows)
    old_base = z @ state.prototypes[: state.old_class_count].T
    if len(state.classes) > state.old_class_count:
        new_base = z @ state.prototypes[state.old_class_count :].T
        base = np.concatenate([old_base, new_base], axis=1)
    else:
        base = old_base
    result = np.array(base, dtype=np.float32, copy=True)
    for index in np.flatnonzero(state.enabled):
        dims = state.selected_dims[index]
        current = z[:, dims]
        llr = -0.5 * np.mean(
            np.square(current - state.class_mean[index]) / state.class_var[index]
            - np.square(current - state.rest_mean[index]) / state.rest_var[index]
            + np.log(state.class_var[index] / state.rest_var[index]),
            axis=1,
        )
        rival_pool = (
            base[:, : state.old_class_count]
            if index < state.old_class_count
            else base
        )
        local_index = index
        rival = np.max(
            np.concatenate(
                [rival_pool[:, :local_index], rival_pool[:, local_index + 1 :]],
                axis=1,
            ),
            axis=1,
        )
        active = (
            base[:, index] - rival >= -state.hyperparameters.margin_band
        )
        conformal = np.clip(
            (llr[active] - state.llr_mid[index])
            / (state.llr_half_gap[index] + EPS),
            -1.0,
            1.0,
        )
        result[active, index] += (
            np.float32(state.hyperparameters.gamma)
            * conformal.astype(np.float32)
        )
    return result


def _state_sha(state):
    digest = hashlib.sha256()
    for value in (
        state.prototypes, state.selected_dims, state.class_mean, state.class_var,
        state.rest_mean, state.rest_var, state.enabled, state.stability,
        state.llr_mid, state.llr_half_gap,
    ):
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps({
        "schema": state.schema, "candidate_id": state.candidate_id,
        "classes": state.classes, "feature_dim": state.feature_dim,
        "k_shot": state.k_shot, "old_class_count": state.old_class_count,
        "generation": state.registration_generation, "resource": dict(state.resource),
        "artifact": state.support_feature_artifact_sha256,
        "selection": state.support_selection_sha256,
        "runtime": state.sealed_runtime_sha256, "code": state.feature_code_sha256,
        "checkpoint": state.sealed_phase1_checkpoint_sha256,
        "operator": state.operator_id, "view_seed": state.view_seed,
        "hp": state.hyperparameters.__dict__,
    }, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cl_cdr_state(
    state: ClCdrEnvelopeState,
    npz_path: Path,
    metadata_path: Path,
) -> dict[str, str]:
    """Persist a minimal array payload plus JSON metadata for sealed loading."""

    _validate_state(state)
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    if npz_path.suffix.lower() != ".npz" or metadata_path.suffix.lower() != ".json":
        raise ClCdrEnvelopeError("state path suffix drift")
    if npz_path.exists() or metadata_path.exists():
        raise ClCdrEnvelopeError("state target already exists")
    arrays = {
        "prototypes": state.prototypes,
        "selected_dims": state.selected_dims,
        "class_mean": state.class_mean,
        "class_var": state.class_var,
        "rest_mean": state.rest_mean,
        "rest_var": state.rest_var,
        "enabled": state.enabled,
        "stability": state.stability,
        "llr_mid": state.llr_mid,
        "llr_half_gap": state.llr_half_gap,
    }
    with npz_path.open("wb") as handle:
        np.savez(handle, **arrays)
    metadata = {
        "schema": state.schema,
        "candidate_id": state.candidate_id,
        "classes": list(state.classes),
        "hyperparameters": dict(state.hyperparameters.__dict__),
        "feature_dim": state.feature_dim,
        "k_shot": state.k_shot,
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "resource": dict(state.resource),
        "support_feature_artifact_sha256": state.support_feature_artifact_sha256,
        "support_selection_sha256": state.support_selection_sha256,
        "sealed_runtime_sha256": state.sealed_runtime_sha256,
        "feature_code_sha256": state.feature_code_sha256,
        "sealed_phase1_checkpoint_sha256": (
            state.sealed_phase1_checkpoint_sha256
        ),
        "operator_id": state.operator_id,
        "view_seed": state.view_seed,
        "state_content_sha256": state.state_content_sha256,
    }
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "npz_sha256": _sha256_path(npz_path),
        "metadata_sha256": _sha256_path(metadata_path),
    }


def load_cl_cdr_state(
    npz_path: Path,
    metadata_path: Path,
    *,
    expected_npz_sha256: str,
    expected_metadata_sha256: str,
) -> ClCdrEnvelopeState:
    """Load only an externally hash-pinned NPZ+JSON CL-CDR state."""

    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    try:
        external_hash_ok = (
            len(expected_npz_sha256) == 64
            and len(expected_metadata_sha256) == 64
            and _sha256_path(npz_path) == expected_npz_sha256
            and _sha256_path(metadata_path) == expected_metadata_sha256
        )
    except OSError as error:
        raise ClCdrEnvelopeError("sealed state file unavailable") from error
    if not external_hash_ok:
        raise ClCdrEnvelopeError("sealed state external hash mismatch")
    required_arrays = {
        "prototypes",
        "selected_dims",
        "class_mean",
        "class_var",
        "rest_mean",
        "rest_var",
        "enabled",
        "stability",
        "llr_mid",
        "llr_half_gap",
    }
    required_metadata = {
        "schema",
        "candidate_id",
        "classes",
        "hyperparameters",
        "feature_dim",
        "k_shot",
        "old_class_count",
        "registration_generation",
        "resource",
        "support_feature_artifact_sha256",
        "support_selection_sha256",
        "sealed_runtime_sha256",
        "feature_code_sha256",
        "sealed_phase1_checkpoint_sha256",
        "operator_id",
        "view_seed",
        "state_content_sha256",
    }
    required_hp = {
        "candidate_id",
        "rank",
        "shrink",
        "ridge",
        "gamma",
        "min_stability",
        "own_llr_quantile",
        "rest_llr_quantile",
        "min_llr_gap",
        "margin_band",
        "operator_id",
        "force_zero",
    }
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if set(metadata) != required_metadata:
            raise ClCdrEnvelopeError("sealed state metadata key drift")
        hp = metadata["hyperparameters"]
        if not isinstance(hp, dict) or set(hp) != required_hp:
            raise ClCdrEnvelopeError("sealed state hyperparameter key drift")
        with np.load(npz_path, allow_pickle=False) as loaded:
            if set(loaded.files) != required_arrays:
                raise ClCdrEnvelopeError("sealed state member drift")
            arrays = {
                name: np.array(loaded[name], copy=True)
                for name in required_arrays
            }
        return ClCdrEnvelopeState(
            schema=str(metadata["schema"]),
            candidate_id=str(metadata["candidate_id"]),
            classes=tuple(str(value) for value in metadata["classes"]),
            prototypes=arrays["prototypes"],
            selected_dims=arrays["selected_dims"],
            class_mean=arrays["class_mean"],
            class_var=arrays["class_var"],
            rest_mean=arrays["rest_mean"],
            rest_var=arrays["rest_var"],
            enabled=arrays["enabled"],
            stability=arrays["stability"],
            llr_mid=arrays["llr_mid"],
            llr_half_gap=arrays["llr_half_gap"],
            hyperparameters=ClCdrHyperparameters(
                candidate_id=str(hp["candidate_id"]),
                rank=int(hp["rank"]),
                shrink=float(hp["shrink"]),
                ridge=float(hp["ridge"]),
                gamma=float(hp["gamma"]),
                min_stability=float(hp["min_stability"]),
                own_llr_quantile=float(hp["own_llr_quantile"]),
                rest_llr_quantile=float(hp["rest_llr_quantile"]),
                min_llr_gap=float(hp["min_llr_gap"]),
                margin_band=float(hp["margin_band"]),
                operator_id=str(hp["operator_id"]),
                force_zero=bool(hp["force_zero"]),
            ),
            feature_dim=int(metadata["feature_dim"]),
            k_shot=int(metadata["k_shot"]),
            old_class_count=int(metadata["old_class_count"]),
            registration_generation=int(
                metadata["registration_generation"]
            ),
            resource=dict(metadata["resource"]),
            support_feature_artifact_sha256=str(
                metadata["support_feature_artifact_sha256"]
            ),
            support_selection_sha256=str(
                metadata["support_selection_sha256"]
            ),
            sealed_runtime_sha256=str(metadata["sealed_runtime_sha256"]),
            feature_code_sha256=str(metadata["feature_code_sha256"]),
            sealed_phase1_checkpoint_sha256=str(
                metadata["sealed_phase1_checkpoint_sha256"]
            ),
            operator_id=str(metadata["operator_id"]),
            view_seed=int(metadata["view_seed"]),
            state_content_sha256=str(metadata["state_content_sha256"]),
        )
    except ClCdrEnvelopeError:
        raise
    except OSError as error:
        raise ClCdrEnvelopeError("sealed state file unavailable") from error
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ClCdrEnvelopeError("sealed state metadata drift") from error


def _calibration_sha(state):
    digest = hashlib.sha256()
    for value in (
        state.prototypes, state.selected_dims, state.class_mean, state.class_var,
        state.rest_mean, state.rest_var, state.enabled, state.stability,
        state.llr_mid, state.llr_half_gap,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _validate_state(state):
    _validate_hp(state.hyperparameters)
    rank = int(state.hyperparameters.rank)
    c = len(state.classes)
    arrays = (
        state.prototypes, state.class_mean, state.class_var, state.rest_mean,
        state.rest_var, state.stability, state.llr_mid, state.llr_half_gap,
    )
    if (
        state.schema != SCHEMA
        or state.prototypes.shape != (c, state.feature_dim)
        or state.selected_dims.shape != (c, rank)
        or any(value.shape != (c, rank) for value in (state.class_mean, state.class_var, state.rest_mean, state.rest_var))
        or state.enabled.shape != (c,)
        or state.stability.shape != (c,)
        or state.llr_mid.shape != (c,)
        or state.llr_half_gap.shape != (c,)
        or not all(np.isfinite(value).all() for value in arrays)
        or np.any(state.class_var <= 0.0) or np.any(state.rest_var <= 0.0)
        or np.any(state.stability < 0.0) or np.any(state.stability > 1.0)
        or np.any(state.selected_dims[state.enabled] < 0)
        or np.any(state.selected_dims[state.enabled] >= state.feature_dim)
        or np.any(state.selected_dims[~state.enabled] != -1)
        or np.any(state.llr_half_gap[state.enabled] <= 0.0)
        or np.any(state.llr_mid[~state.enabled] != 0.0)
        or np.any(state.llr_half_gap[~state.enabled] != 0.0)
        or _array_bytes(state) > MAX_STATE_BYTES
        or state.resource.get("trainable_parameters") != 0
        or state.resource.get("adapt_epochs") != 0
        or state.resource.get("persistent_array_state_bytes") != _array_bytes(state)
        or bool(state.resource.get("dense_query_graph", True))
        or state.state_content_sha256 != _state_sha(state)
    ):
        raise ClCdrEnvelopeError("state drift")
    if state.hyperparameters.force_zero and (np.any(state.enabled) or rank != 0):
        raise ClCdrEnvelopeError("true zero drift")


def _masks(labels, ranks):
    if set(np.unique(ranks)) != set(range(10)):
        raise ClCdrEnvelopeError("joint L2O requires strict K10")
    return tuple(np.isin(ranks, (first, first + 1)) for first in range(0, 10, 2))


def _metrics(truth, pred, classes):
    pc = {x: float(np.mean(pred[truth == x] == x)) for x in classes}
    return {"overall_accuracy": float(np.mean(pred == truth)), "min_class_accuracy": min(pc.values()), "per_class_accuracy": pc}


def _aggregate(folds, key, classes):
    pc = {x: float(np.mean([f[key]["per_class_accuracy"][x] for f in folds])) for x in classes}
    return {"overall_accuracy": float(np.mean([f[key]["overall_accuracy"] for f in folds])), "min_class_accuracy": min(pc.values()), "per_class_accuracy": pc}


def evaluate_joint_leave_two_out(
    before_artifact, before_labels, before_ranks,
    after_artifact, after_labels, after_ranks, *, hyperparameters
):
    old_rows, old_labels, old_ranks, old_classes = _validate_support(before_artifact, before_labels, before_ranks, 10)
    joint_rows, joint_labels, joint_ranks, found = _validate_support(after_artifact, after_labels, after_ranks, 10)
    _validate_binding(before_artifact, after_artifact)
    _validate_old_reuse(before_artifact, old_labels, old_ranks, after_artifact, joint_labels, joint_ranks, old_classes)
    joint_classes = old_classes + tuple(sorted(set(found) - set(old_classes)))
    new_classes = joint_classes[len(old_classes):]
    old_masks, joint_masks = _masks(old_labels, old_ranks), _masks(joint_labels, joint_ranks)
    old_in_joint = np.isin(joint_labels, old_classes)
    zero = ClCdrHyperparameters(
        candidate_id="d15_clcdr_z0",
        rank=0,
        shrink=1.0,
        ridge=hyperparameters.ridge,
        gamma=0.0,
        min_stability=1.0,
        force_zero=True,
    )
    folds, trace = [], []
    for fold, (ho, hj) in enumerate(zip(old_masks, joint_masks)):
        fit = _fit_selected(before_artifact, old_rows, old_labels, old_ranks, old_classes, ~ho, after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes, ~hj, hyperparameters)
        base = _fit_selected(before_artifact, old_rows, old_labels, old_ranks, old_classes, ~ho, after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes, ~hj, zero)
        hoj, hnj = hj & old_in_joint, hj & ~old_in_joint
        scores = {
            "before": _score_numpy(old_rows[ho], fit.before_state),
            "base_before": _score_numpy(old_rows[ho], base.before_state),
            "old": _score_numpy(joint_rows[hoj], fit.after_state),
            "new": _score_numpy(joint_rows[hnj], fit.after_state),
            "base_old": _score_numpy(joint_rows[hoj], base.after_state),
            "base_new": _score_numpy(joint_rows[hnj], base.after_state),
            "locked_old_old": _score_numpy(joint_rows[hoj], fit.before_state),
            "locked_old_new": _score_numpy(joint_rows[hnj], fit.before_state),
        }
        pred = lambda value, classes: np.asarray(classes)[np.argmax(value, axis=1)]
        row = {
            "fold": fold,
            "before_old": _metrics(old_labels[ho], pred(scores["before"], old_classes), old_classes),
            "base_before_old": _metrics(old_labels[ho], pred(scores["base_before"], old_classes), old_classes),
            "after_old": _metrics(joint_labels[hoj], pred(scores["old"], joint_classes), old_classes),
            "after_new": _metrics(joint_labels[hnj], pred(scores["new"], joint_classes), new_classes),
            "base_after_old": _metrics(joint_labels[hoj], pred(scores["base_old"], joint_classes), old_classes),
            "base_after_new": _metrics(joint_labels[hnj], pred(scores["base_new"], joint_classes), new_classes),
            "old_score_bitwise_locked": bool(
                np.array_equal(scores["old"][:, :len(old_classes)], scores["locked_old_old"])
                and np.array_equal(scores["new"][:, :len(old_classes)], scores["locked_old_new"])
            ),
            "enabled": fit.after_state.enabled.tolist(),
            "stability": fit.after_state.stability.tolist(),
            "before_selection_sha": fit.before_state.support_selection_sha256,
            "after_selection_sha": fit.after_state.support_selection_sha256,
            "calibration_sha": _calibration_sha(fit.after_state),
        }
        old_pred, new_pred = pred(scores["old"], joint_classes), pred(scores["new"], joint_classes)
        truth = np.concatenate([joint_labels[hoj], joint_labels[hnj]])
        row["joint_accuracy"] = float(np.mean(np.concatenate([old_pred, new_pred]) == truth))
        folds.append(row)
        trace.extend({"fold": fold, **x} for x in fit.trace)
        trace.append({"phase": "joint_l2o_fold", **row})
    result = {key: _aggregate(folds, key, classes) for key, classes in (
        ("before_old", old_classes), ("base_before_old", old_classes),
        ("after_old", old_classes), ("after_new", new_classes),
        ("base_after_old", old_classes), ("base_after_new", new_classes)
    )}
    result["joint_accuracy"] = float(np.mean([x["joint_accuracy"] for x in folds]))
    old, new = result["after_old"]["overall_accuracy"], result["after_new"]["overall_accuracy"]
    result["h_old_new"] = 0.0 if old + new == 0 else 2 * old * new / (old + new)
    result["old_forgetting"] = result["before_old"]["overall_accuracy"] - old
    result["old_score_bitwise_locked"] = all(x["old_score_bitwise_locked"] for x in folds)
    result["folds"] = folds
    return result, tuple(trace)


def predict_all_registered(state, query_artifact):
    rows = _rows(query_artifact)
    if len(rows) != 1:
        raise ClCdrEnvelopeError("formal prediction requires exactly one query")
    if (
        query_artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or query_artifact.feature_code_sha256 != state.feature_code_sha256
        or query_artifact.sealed_phase1_checkpoint_sha256 != state.sealed_phase1_checkpoint_sha256
        or query_artifact.operator_id != state.operator_id
        or query_artifact.view_seed != state.view_seed
    ):
        raise ClCdrEnvelopeError("query binding mismatch")
    scores = _score_numpy(rows, state)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)], scores
