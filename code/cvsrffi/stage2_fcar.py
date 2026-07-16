"""D16 efficient support-only floor-conditioned asymmetric registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cvsrffi.stage2_cl_cdr_envelope import (
    EPS,
    ClCdrHyperparameters,
    _llr_from_stats,
    _normalize,
    _prototypes,
    _select_stats,
)
from cvsrffi.stage2_joint_residual_logit_head import RuntimeAuthorizedFeatureArtifact


SCHEMA = "cvs.phase2.fcar.v1"
AMPLITUDE_GRID = (0.0, 0.005, 0.01, 0.02)
MAX_STATE_BYTES = 256 * 1024


class FcarError(ValueError):
    """Raised when the D16 support-only contract fails closed."""


@dataclass(frozen=True)
class FcarHyperparameters:
    candidate_id: str
    rank: int = 8
    shrink: float = 0.5
    ridge: float = 0.01
    own_llr_quantile: float = 0.20
    rest_llr_quantile: float = 0.80
    min_llr_gap: float = 0.0
    activation_threshold: float = 0.5
    margin_band: float = 0.20
    operator_id: str = "base"
    force_zero: bool = False


@dataclass(frozen=True)
class FcarState:
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
    llr_mid: np.ndarray
    llr_half_gap: np.ndarray
    a_plus: np.ndarray
    a_minus: np.ndarray
    hyperparameters: FcarHyperparameters
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
            "prototypes", "class_mean", "class_var", "rest_mean", "rest_var",
            "llr_mid", "llr_half_gap", "a_plus", "a_minus",
        ):
            source = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            object.__setattr__(
                self, name,
                np.frombuffer(source.tobytes(), dtype=np.float32).reshape(source.shape),
            )
        dims = np.ascontiguousarray(self.selected_dims, dtype=np.int64)
        object.__setattr__(
            self, "selected_dims",
            np.frombuffer(dims.tobytes(), dtype=np.int64).reshape(dims.shape),
        )
        enabled = np.ascontiguousarray(self.enabled, dtype=np.bool_)
        object.__setattr__(
            self, "enabled",
            np.frombuffer(enabled.tobytes(), dtype=np.bool_).reshape(enabled.shape),
        )
        computed = _state_sha(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise FcarError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)
        _validate_state(self)


@dataclass(frozen=True)
class BeforeAfterFcarFit:
    before_state: FcarState
    after_state: FcarState
    trace: tuple[dict[str, Any], ...]


def _validate_hp(hp: FcarHyperparameters) -> None:
    if (
        not hp.candidate_id
        or hp.operator_id != "base"
        or not 0 <= int(hp.rank) <= 16
        or not np.isfinite(hp.shrink)
        or not 0.0 <= hp.shrink <= 1.0
        or not np.isfinite(hp.ridge)
        or hp.ridge <= 0.0
        or not 0.0 <= hp.own_llr_quantile <= 1.0
        or not 0.0 <= hp.rest_llr_quantile <= 1.0
        or not np.isfinite(hp.min_llr_gap)
        or hp.min_llr_gap < 0.0
        or hp.activation_threshold != 0.5
        or not np.isfinite(hp.margin_band)
        or hp.margin_band < 0.0
        or (hp.force_zero and hp.rank != 0)
    ):
        raise FcarError("hyperparameter drift")


def _rows(artifact):
    if not isinstance(artifact, RuntimeAuthorizedFeatureArtifact):
        raise FcarError("runtime-authorized feature artifact required")
    return artifact.features


def _validate_support(artifact, labels, ranks, k):
    rows = _rows(artifact)
    labels = np.asarray(labels).astype(str)
    ranks = np.asarray(ranks, dtype=np.int64)
    if len(rows) != len(labels) or len(rows) != len(ranks):
        raise FcarError("support alignment drift")
    classes, counts = np.unique(labels, return_counts=True)
    if (
        int(k) < 1
        or set(counts.tolist()) != {int(k)}
        or any(set(ranks[labels == value]) != set(range(int(k))) for value in classes)
    ):
        raise FcarError("strict physical K-shot drift")
    return rows, labels, ranks, tuple(sorted(classes.tolist()))


def _validate_binding(before, after):
    if (
        before.sealed_runtime_sha256 != after.sealed_runtime_sha256
        or before.feature_code_sha256 != after.feature_code_sha256
        or before.sealed_phase1_checkpoint_sha256
        != after.sealed_phase1_checkpoint_sha256
        or before.operator_id != "base"
        or after.operator_id != "base"
        or before.view_seed != after.view_seed
    ):
        raise FcarError("runtime/operator binding drift")


def _validate_old_reuse(before, bl, br, after, al, ar, old_classes):
    allowed = set(old_classes)

    def keyed(artifact, labels, ranks):
        return {
            (str(labels[index]), int(ranks[index])): (
                artifact.physical_sample_ids[index],
                artifact.parent_received_iq_sha256[index],
                artifact.per_row_feature_sha256[index],
            )
            for index in range(len(labels))
            if str(labels[index]) in allowed
        }

    if keyed(before, bl, br) != keyed(after, al, ar):
        raise FcarError("old exact-reuse lock failed")


def _clhp(hp):
    return ClCdrHyperparameters(
        candidate_id=hp.candidate_id, rank=hp.rank, shrink=hp.shrink,
        ridge=hp.ridge, gamma=0.0, min_stability=0.0,
        own_llr_quantile=hp.own_llr_quantile,
        rest_llr_quantile=hp.rest_llr_quantile,
        min_llr_gap=hp.min_llr_gap, margin_band=hp.margin_band,
        operator_id=hp.operator_id, force_zero=hp.force_zero,
    )


def _delta(h, plus, minus, threshold):
    positive = max(0.0, (h - threshold) / (1.0 - threshold))
    negative = max(0.0, (-h - threshold) / (1.0 - threshold))
    return plus * positive - minus * negative


def _margin(scores, truth):
    return float(scores[truth] - np.max(np.delete(scores, truth)))


def _q20(values):
    return float(np.quantile(values, 0.20, method="linear"))


def _fold_masks(ranks):
    return (ranks % 2 == 0, ranks % 2 == 1)


def _build_oof(rows, labels, ranks, classes, class_index, hp):
    z = _normalize(rows)
    records = []
    fold_stats = []
    for eval_fold, eval_mask in enumerate(_fold_masks(ranks)):
        train_mask = ~eval_mask
        selected = _select_stats(
            rows[train_mask], labels[train_mask], classes[class_index], _clhp(hp)
        )
        if selected is None:
            return None
        dims, stats = selected
        prototypes = _prototypes(rows[train_mask], labels[train_mask], classes)
        fold_records = []
        for index in np.flatnonzero(eval_mask):
            fold_records.append({
                "row_index": int(index),
                "eval_fold": eval_fold,
                "truth": classes.index(str(labels[index])),
                "base_scores": z[index] @ prototypes.T,
                "llr": _llr_from_stats(z[index], dims, stats),
                "model_train_row_indices": tuple(np.flatnonzero(train_mask).tolist()),
            })
        fold_stats.append((dims, stats))
        records.extend(fold_records)
    calibrated = []
    for record in records:
        peers = [
            value for value in records
            if value["eval_fold"] == record["eval_fold"]
            and value["row_index"] != record["row_index"]
        ]
        own = [value["llr"] for value in peers if value["truth"] == class_index]
        rest = [value["llr"] for value in peers if value["truth"] != class_index]
        if not own or not rest:
            return None
        q_pos = float(np.quantile(own, hp.own_llr_quantile, method="linear"))
        q_neg = float(np.quantile(rest, hp.rest_llr_quantile, method="linear"))
        gap = q_pos - q_neg
        if gap <= hp.min_llr_gap:
            return None
        calibrated.append({
            **record,
            "threshold_peer_row_indices": tuple(value["row_index"] for value in peers),
            "threshold_models_exclude_self": all(
                record["row_index"] not in value["model_train_row_indices"]
                for value in peers
            ),
            "q_pos": q_pos, "q_neg": q_neg,
            "mid": 0.5 * (q_pos + q_neg),
            "half_gap": 0.5 * gap,
            "h": float(np.clip(
                (record["llr"] - 0.5 * (q_pos + q_neg)) / (0.5 * gap + EPS),
                -1.0, 1.0,
            )),
        })
    return tuple(calibrated), tuple(fold_stats)


def _baseline_floor(records, classes):
    class_count = len(classes)
    correct = np.zeros(class_count, dtype=np.int64)
    total = np.zeros(class_count, dtype=np.int64)
    for record in records:
        truth = record["truth"]
        total[truth] += 1
        correct[truth] += int(np.argmax(record["base_scores"]) == truth)
    accuracy = correct / np.maximum(total, 1)
    floor_count = max(1, int(np.ceil(class_count * 0.25)))
    ordered = sorted(
        range(class_count), key=lambda index: (accuracy[index], classes[index])
    )
    return set(ordered[:floor_count]), accuracy


def _evaluate(records, class_index, plus, minus, hp, *, combined=None):
    class_count = len(records[0]["base_scores"])
    base_correct = np.zeros(class_count, dtype=np.int64)
    candidate_correct = np.zeros(class_count, dtype=np.int64)
    base_margins = [[] for _ in range(class_count)]
    candidate_margins = [[] for _ in range(class_count)]
    new_capture = np.zeros(class_count, dtype=np.int64)
    lost_baseline_correct = np.zeros(class_count, dtype=np.int64)
    for record in records:
        truth = record["truth"]
        base = record["base_scores"]
        candidate = np.array(base, copy=True)
        if combined:
            for target, pair in combined.items():
                margin = _margin(base, target)
                if abs(margin) <= hp.margin_band:
                    candidate[target] += np.float32(
                        _delta(record["h_by_class"][target], pair[0], pair[1], hp.activation_threshold)
                    )
        else:
            margin = _margin(base, class_index)
            if abs(margin) <= hp.margin_band:
                candidate[class_index] += np.float32(
                    _delta(record["h"], plus, minus, hp.activation_threshold)
                )
        base_prediction = int(np.argmax(base))
        prediction = int(np.argmax(candidate))
        base_correct[truth] += int(base_prediction == truth)
        candidate_correct[truth] += int(prediction == truth)
        base_margins[truth].append(_margin(base, truth))
        candidate_margins[truth].append(_margin(candidate, truth))
        if combined:
            if (
                prediction != truth
                and prediction in combined
                and base_prediction != prediction
            ):
                new_capture[truth] += 1
        elif (
            truth != class_index
            and base_prediction != class_index
            and prediction == class_index
        ):
            new_capture[truth] += 1
        if base_prediction == truth and prediction != truth:
            lost_baseline_correct[truth] += 1
    return {
        "base_correct": base_correct,
        "candidate_correct": candidate_correct,
        "new_capture": new_capture,
        "lost_baseline_correct": lost_baseline_correct,
        "base_q20": np.asarray([_q20(value) for value in base_margins]),
        "candidate_q20": np.asarray([_q20(value) for value in candidate_margins]),
    }


def _passes(metrics, class_index, floor_classes, *, strict_floor):
    if np.any(metrics["candidate_correct"] < metrics["base_correct"]):
        return False
    if metrics["candidate_q20"][class_index] + 1.0e-12 < metrics["base_q20"][class_index]:
        return False
    if np.any(metrics["new_capture"]):
        return False
    if np.any(metrics["lost_baseline_correct"]):
        return False
    any_strict_benefit = bool(
        np.sum(metrics["candidate_correct"]) > np.sum(metrics["base_correct"])
        or metrics["candidate_q20"][class_index]
        > metrics["base_q20"][class_index] + 1.0e-12
        or np.any(
            metrics["candidate_q20"]
            > metrics["base_q20"] + 1.0e-12
        )
    )
    if not any_strict_benefit:
        return False
    if strict_floor and class_index in floor_classes:
        return bool(
            metrics["candidate_correct"][class_index]
            > metrics["base_correct"][class_index]
            or metrics["candidate_q20"][class_index]
            > metrics["base_q20"][class_index] + 1.0e-12
        )
    return True


def _rollback_key(index, targets, floor_classes, benefit, classes):
    return (
        targets[index] in floor_classes,
        benefit.get(index, (0, 0.0))[0],
        benefit.get(index, (0, 0.0))[1],
        classes[targets[index]],
    )


def _deployment_consistency_records(
    rows,
    reference_records,
    full_prototypes,
    targets,
    active,
    dims,
    cm,
    cv,
    rm,
    rv,
    mid,
    half_gap,
):
    """Build a support-inclusive deployment-state consistency veto surface."""

    z = _normalize(rows)
    result = []
    for record in reference_records:
        row_index = record["row_index"]
        base_scores = z[row_index] @ full_prototypes.T
        h_by_class = {}
        for index in active:
            stats = (cm[index], cv[index], rm[index], rv[index])
            llr = _llr_from_stats(z[row_index], dims[index], stats)
            h_by_class[targets[index]] = float(np.clip(
                (llr - mid[index]) / (half_gap[index] + EPS),
                -1.0, 1.0,
            ))
        result.append({
            **record,
            "base_scores": base_scores,
            "h_by_class": h_by_class,
            "deployment_state_base_and_h": True,
        })
    return tuple(result)


def _select_amplitudes(records, class_index, floor_classes, hp):
    choices = []
    for plus in AMPLITUDE_GRID:
        for minus in AMPLITUDE_GRID:
            if plus == 0.0 and minus == 0.0:
                continue
            metrics = _evaluate(records, class_index, plus, minus, hp)
            if _passes(metrics, class_index, floor_classes, strict_floor=True):
                choices.append((
                    int(metrics["candidate_correct"][class_index] - metrics["base_correct"][class_index]),
                    float(metrics["candidate_q20"][class_index] - metrics["base_q20"][class_index]),
                    -(plus + minus), -plus, -minus, plus, minus,
                ))
    if not choices:
        return 0.0, 0.0, (0, 0.0)
    chosen = max(choices)
    return chosen[-2], chosen[-1], (chosen[0], chosen[1])


def _fit_envelopes(rows, labels, ranks, classes, targets, hp):
    count, rank = len(targets), int(hp.rank)
    dims = np.full((count, rank), -1, dtype=np.int64)
    cm = np.zeros((count, rank), dtype=np.float32)
    cv = np.ones((count, rank), dtype=np.float32)
    rm = np.zeros((count, rank), dtype=np.float32)
    rv = np.ones((count, rank), dtype=np.float32)
    enabled = np.zeros(count, dtype=np.bool_)
    mid = np.zeros(count, dtype=np.float32)
    half_gap = np.zeros(count, dtype=np.float32)
    plus = np.zeros(count, dtype=np.float32)
    minus = np.zeros(count, dtype=np.float32)
    diagnostics = []
    if hp.force_zero or rank == 0:
        return (dims, cm, cv, rm, rv, enabled, mid, half_gap, plus, minus), ()
    per_class_records = {}
    benefit_by_index = {}
    full_prototypes = _prototypes(rows, labels, classes)
    baseline_records = None
    for output_index, class_index in enumerate(targets):
        built = _build_oof(rows, labels, ranks, classes, class_index, hp)
        if built is None:
            diagnostics.append({
                "class_handle": classes[class_index], "enabled": False,
                "reason": "two_fold_oof_unavailable",
            })
            continue
        records, fold_stats = built
        if baseline_records is None:
            baseline_records = records
        floor_classes, baseline_accuracy = _baseline_floor(records, classes)
        selected_plus, selected_minus, benefit = _select_amplitudes(
            records, class_index, floor_classes, hp
        )
        full = _select_stats(rows, labels, classes[class_index], _clhp(hp))
        own = [value["llr"] for value in records if value["truth"] == class_index]
        rest = [value["llr"] for value in records if value["truth"] != class_index]
        if full is None or not own or not rest:
            continue
        q_pos = float(np.quantile(own, hp.own_llr_quantile, method="linear"))
        q_neg = float(np.quantile(rest, hp.rest_llr_quantile, method="linear"))
        if q_pos - q_neg <= hp.min_llr_gap or (selected_plus == selected_minus == 0.0):
            diagnostics.append({
                "class_handle": classes[class_index], "enabled": False,
                "reason": "safe_nonzero_amplitude_unavailable",
                "floor_class": class_index in floor_classes,
            })
            continue
        full_dims, stats = full
        enabled[output_index] = True
        dims[output_index] = full_dims
        cm[output_index], cv[output_index], rm[output_index], rv[output_index] = stats
        mid[output_index] = np.float32(0.5 * (q_pos + q_neg))
        half_gap[output_index] = np.float32(0.5 * (q_pos - q_neg))
        plus[output_index] = np.float32(selected_plus)
        minus[output_index] = np.float32(selected_minus)
        per_class_records[output_index] = records
        benefit_by_index[output_index] = benefit
        diagnostics.append({
            "class_handle": classes[class_index], "enabled": True,
            "floor_class": class_index in floor_classes,
            "baseline_accuracy": baseline_accuracy.tolist(),
            "a_plus": selected_plus, "a_minus": selected_minus,
            "benefit": benefit,
            "all_threshold_models_exclude_self": all(
                value["threshold_models_exclude_self"] for value in records
            ),
            "fold_train_sizes": [
                len(value["model_train_row_indices"]) for value in records[:2]
            ],
        })
    # Support-inclusive deployment consistency veto; never a performance proof.
    active = list(np.flatnonzero(enabled))
    if active:
        reference = per_class_records[active[0]]
        floor_classes, _ = _baseline_floor(reference, classes)
        while active:
            common = _deployment_consistency_records(
                rows, reference, full_prototypes, targets, active,
                dims, cm, cv, rm, rv, mid, half_gap,
            )
            combined = {
                targets[index]: (float(plus[index]), float(minus[index]))
                for index in active
            }
            joint = _evaluate(common, targets[active[0]], 0.0, 0.0, hp, combined=combined)
            if (
                np.all(joint["candidate_correct"] >= joint["base_correct"])
                and np.all(joint["candidate_q20"] + 1.0e-12 >= joint["base_q20"])
                and not np.any(joint["new_capture"])
                and not np.any(joint["lost_baseline_correct"])
                and all(
                    (
                        targets[index] not in floor_classes
                        or joint["candidate_correct"][targets[index]]
                        > joint["base_correct"][targets[index]]
                        or joint["candidate_q20"][targets[index]]
                        > joint["base_q20"][targets[index]] + 1.0e-12
                    )
                    for index in active
                )
            ):
                break
            remove = min(
                active,
                key=lambda index: _rollback_key(
                    index, targets, floor_classes, benefit_by_index, classes
                ),
            )
            active.remove(remove)
            enabled[remove] = False
            dims[remove] = -1
            cm[remove] = rm[remove] = 0.0
            cv[remove] = rv[remove] = 1.0
            mid[remove] = half_gap[remove] = plus[remove] = minus[remove] = 0.0
            diagnostics[remove] = {
                **diagnostics[remove], "enabled": False,
                "reason": "deployment_state_consistency_rollback",
            }
        for index in active:
            diagnostics[index] = {
                **diagnostics[index],
                "deployment_state_consistency_veto": "pass",
            }
    return (dims, cm, cv, rm, rv, enabled, mid, half_gap, plus, minus), tuple(diagnostics)


def _selection_sha(artifact, labels, ranks, selection):
    records = [
        (
            str(labels[index]), int(ranks[index]),
            artifact.physical_sample_ids[index],
            artifact.parent_received_iq_sha256[index],
            artifact.per_row_feature_sha256[index],
        )
        for index in np.flatnonzero(selection)
    ]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _array_bytes(state):
    return sum(getattr(state, name).nbytes for name in (
        "prototypes", "selected_dims", "class_mean", "class_var", "rest_mean",
        "rest_var", "enabled", "llr_mid", "llr_half_gap", "a_plus", "a_minus",
    ))


def _make_state(artifact, classes, prototypes, arrays, hp, *, k, old, generation, selection_sha):
    dims, cm, cv, rm, rv, enabled, mid, half_gap, plus, minus = arrays
    state_bytes = sum(value.nbytes for value in (
        prototypes, dims, cm, cv, rm, rv, enabled, mid, half_gap, plus, minus,
    ))
    return FcarState(
        schema=SCHEMA, candidate_id=hp.candidate_id, classes=classes,
        prototypes=prototypes, selected_dims=dims, class_mean=cm, class_var=cv,
        rest_mean=rm, rest_var=rv, enabled=enabled, llr_mid=mid,
        llr_half_gap=half_gap, a_plus=plus, a_minus=minus, hyperparameters=hp,
        feature_dim=int(prototypes.shape[1]), k_shot=int(k),
        old_class_count=int(old), registration_generation=int(generation),
        resource={
            "trainable_parameters": 0, "adapt_epochs": 0,
            "persistent_array_state_bytes": state_bytes,
            "enabled_class_count": int(np.sum(enabled)),
            "amplitude_grid": list(AMPLITUDE_GRID),
            "two_fold_model_fits_per_class": 0 if hp.rank == 0 else 2,
            "backbone_forwards_per_physical_sample": 1,
            "fft_branches_per_physical_sample": 0,
            "head_scalar_ops_per_sample_upper_bound": int(
                len(classes) * int(hp.rank) * 12
            ),
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
    after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes,
    joint_selection, hp,
):
    old_rows_s, old_labels_s, old_ranks_s = (
        old_rows[old_selection], old_labels[old_selection], old_ranks[old_selection]
    )
    joint_rows_s, joint_labels_s, joint_ranks_s = (
        joint_rows[joint_selection], joint_labels[joint_selection], joint_ranks[joint_selection]
    )
    old_proto = _prototypes(old_rows_s, old_labels_s, old_classes)
    joint_proto = _prototypes(joint_rows_s, joint_labels_s, joint_classes)
    if not np.array_equal(old_proto, joint_proto[:len(old_classes)]):
        raise FcarError("old prototype lock failed")
    old_arrays, old_diag = _fit_envelopes(
        old_rows_s, old_labels_s, old_ranks_s, old_classes,
        tuple(range(len(old_classes))), hp,
    )
    new_arrays, new_diag = _fit_envelopes(
        joint_rows_s, joint_labels_s, joint_ranks_s, joint_classes,
        tuple(range(len(old_classes), len(joint_classes))), hp,
    )
    combined = tuple(
        np.concatenate([old_arrays[index], new_arrays[index]], axis=0)
        for index in range(len(old_arrays))
    )
    before = _make_state(
        before_artifact, old_classes, old_proto, old_arrays, hp,
        k=int(np.sum(old_selection & (old_labels == old_classes[0]))),
        old=len(old_classes), generation=0,
        selection_sha=_selection_sha(before_artifact, old_labels, old_ranks, old_selection),
    )
    after = _make_state(
        after_artifact, joint_classes, joint_proto, combined, hp,
        k=int(np.sum(joint_selection & (joint_labels == joint_classes[0]))),
        old=len(old_classes), generation=1,
        selection_sha=_selection_sha(after_artifact, joint_labels, joint_ranks, joint_selection),
    )
    for name in (
        "selected_dims", "class_mean", "class_var", "rest_mean", "rest_var",
        "enabled", "llr_mid", "llr_half_gap", "a_plus", "a_minus",
    ):
        if not np.array_equal(getattr(before, name), getattr(after, name)[:len(old_classes)]):
            raise FcarError("old state lock failed")
    return BeforeAfterFcarFit(
        before, after,
        (
            {"phase": "before_fcar_fit", "diagnostics": list(old_diag)},
            {"phase": "after_fcar_fit", "diagnostics": list(new_diag)},
        ),
    )


def _canonical_k1_hp():
    return FcarHyperparameters(
        candidate_id="d16_fcar_k1_canonical_z0", rank=0, shrink=1.0,
        ridge=1.0e-8, margin_band=0.0, force_zero=True,
    )


def fit_before_after_locked(
    before_artifact, before_labels, before_ranks,
    after_artifact, after_labels, after_ranks, *, k_shot, hyperparameters,
):
    _validate_hp(hyperparameters)
    if 2 <= int(k_shot) <= 4:
        raise FcarError("K2-K4 unsupported; use canonical Z0 outside FCAR")
    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot
    )
    joint_rows, joint_labels, joint_ranks, found = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot
    )
    if not set(old_classes) < set(found):
        raise FcarError("registration class drift")
    _validate_binding(before_artifact, after_artifact)
    _validate_old_reuse(
        before_artifact, old_labels, old_ranks,
        after_artifact, joint_labels, joint_ranks, old_classes,
    )
    classes = old_classes + tuple(sorted(set(found) - set(old_classes)))
    hp = _canonical_k1_hp() if int(k_shot) == 1 else hyperparameters
    return _fit_selected(
        before_artifact, old_rows, old_labels, old_ranks, old_classes,
        np.ones(len(old_labels), dtype=bool),
        after_artifact, joint_rows, joint_labels, joint_ranks, classes,
        np.ones(len(joint_labels), dtype=bool), hp,
    )


def _base_scores(rows, state):
    z = _normalize(rows)
    old = z @ state.prototypes[:state.old_class_count].T
    if len(state.classes) == state.old_class_count:
        return z, old
    new = z @ state.prototypes[state.old_class_count:].T
    return z, np.concatenate([old, new], axis=1)


def _score_numpy(rows, state):
    _validate_state(state)
    z, base = _base_scores(rows, state)
    result = np.array(base, copy=True)
    for index in np.flatnonzero(state.enabled):
        rival_pool = (
            base[:, :state.old_class_count]
            if index < state.old_class_count
            else base
        )
        margin = base[:, index] - np.max(
            np.concatenate(
                [rival_pool[:, :index], rival_pool[:, index + 1:]], axis=1
            ),
            axis=1,
        )
        for row_index in np.flatnonzero(np.abs(margin) <= state.hyperparameters.margin_band):
            stats = (
                state.class_mean[index], state.class_var[index],
                state.rest_mean[index], state.rest_var[index],
            )
            llr = _llr_from_stats(z[row_index], state.selected_dims[index], stats)
            h = float(np.clip(
                (llr - state.llr_mid[index]) / (state.llr_half_gap[index] + EPS),
                -1.0, 1.0,
            ))
            result[row_index, index] += np.float32(_delta(
                h, state.a_plus[index], state.a_minus[index],
                state.hyperparameters.activation_threshold,
            ))
    return result


def predict_all_registered(state, artifact):
    rows = _rows(artifact)
    if len(rows) != 1:
        raise FcarError("exactly one query required")
    if (
        artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or artifact.feature_code_sha256 != state.feature_code_sha256
        or artifact.sealed_phase1_checkpoint_sha256
        != state.sealed_phase1_checkpoint_sha256
        or artifact.operator_id != state.operator_id
        or artifact.view_seed != state.view_seed
    ):
        raise FcarError("query runtime binding drift")
    scores = _score_numpy(rows, state)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)], scores


def _l2o_masks(ranks):
    if set(np.unique(ranks)) != set(range(10)):
        raise FcarError("joint L2O requires strict K10")
    return tuple(
        np.isin(ranks, (first, first + 1))
        for first in range(0, 10, 2)
    )


def _metrics(truth, prediction, classes):
    per_class = {
        value: float(np.mean(prediction[truth == value] == value))
        for value in classes
    }
    return {
        "overall_accuracy": float(np.mean(prediction == truth)),
        "min_class_accuracy": min(per_class.values()),
        "per_class_accuracy": per_class,
    }


def _aggregate(folds, key, classes):
    per_class = {
        value: float(np.mean([
            fold[key]["per_class_accuracy"][value] for fold in folds
        ]))
        for value in classes
    }
    return {
        "overall_accuracy": float(np.mean([
            fold[key]["overall_accuracy"] for fold in folds
        ])),
        "min_class_accuracy": min(per_class.values()),
        "per_class_accuracy": per_class,
    }


def _harmonic(left, right):
    return (
        0.0
        if left + right <= 0.0
        else float(2.0 * left * right / (left + right))
    )


def _decision_tensor_sha(state):
    digest = hashlib.sha256()
    for name in (
        "prototypes", "selected_dims", "class_mean", "class_var", "rest_mean",
        "rest_var", "enabled", "llr_mid", "llr_half_gap", "a_plus", "a_minus",
    ):
        digest.update(np.ascontiguousarray(getattr(state, name)).tobytes())
    return digest.hexdigest()


def _physical_id_sha(artifact, selection):
    values = [
        artifact.physical_sample_ids[index]
        for index in np.flatnonzero(selection)
    ]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def _floor_handles(fit):
    return sorted({
        row["class_handle"]
        for phase in fit.trace
        for row in phase["diagnostics"]
        if row.get("floor_class", False)
    })


def evaluate_joint_leave_two_out(
    before_artifact,
    before_labels,
    before_ranks,
    after_artifact,
    after_labels,
    after_ranks,
    *,
    hyperparameters,
):
    """Strict K10 support-only five-fold audit; outer held2 never fits state."""

    _validate_hp(hyperparameters)
    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, 10
    )
    joint_rows, joint_labels, joint_ranks, found = _validate_support(
        after_artifact, after_labels, after_ranks, 10
    )
    _validate_binding(before_artifact, after_artifact)
    _validate_old_reuse(
        before_artifact, old_labels, old_ranks,
        after_artifact, joint_labels, joint_ranks, old_classes,
    )
    joint_classes = old_classes + tuple(sorted(set(found) - set(old_classes)))
    new_classes = joint_classes[len(old_classes):]
    old_masks = _l2o_masks(old_ranks)
    joint_masks = _l2o_masks(joint_ranks)
    old_in_joint = np.isin(joint_labels, old_classes)
    zero = FcarHyperparameters(
        candidate_id="d16_fcar_l2o_z0",
        rank=0,
        shrink=1.0,
        ridge=hyperparameters.ridge,
        own_llr_quantile=hyperparameters.own_llr_quantile,
        rest_llr_quantile=hyperparameters.rest_llr_quantile,
        min_llr_gap=hyperparameters.min_llr_gap,
        activation_threshold=0.5,
        margin_band=0.0,
        operator_id="base",
        force_zero=True,
    )
    folds = []
    trace = []
    for fold_index, (held_old, held_joint) in enumerate(
        zip(old_masks, joint_masks)
    ):
        fit = _fit_selected(
            before_artifact, old_rows, old_labels, old_ranks, old_classes,
            ~held_old,
            after_artifact, joint_rows, joint_labels, joint_ranks,
            joint_classes, ~held_joint, hyperparameters,
        )
        base = _fit_selected(
            before_artifact, old_rows, old_labels, old_ranks, old_classes,
            ~held_old,
            after_artifact, joint_rows, joint_labels, joint_ranks,
            joint_classes, ~held_joint, zero,
        )
        held_joint_old = held_joint & old_in_joint
        held_joint_new = held_joint & ~old_in_joint
        score = {
            "before": _score_numpy(old_rows[held_old], fit.before_state),
            "base_before": _score_numpy(old_rows[held_old], base.before_state),
            "after_old": _score_numpy(
                joint_rows[held_joint_old], fit.after_state
            ),
            "base_after_old": _score_numpy(
                joint_rows[held_joint_old], base.after_state
            ),
            "after_new": _score_numpy(
                joint_rows[held_joint_new], fit.after_state
            ),
            "base_after_new": _score_numpy(
                joint_rows[held_joint_new], base.after_state
            ),
            "locked_old_old": _score_numpy(
                joint_rows[held_joint_old], fit.before_state
            ),
            "locked_old_new": _score_numpy(
                joint_rows[held_joint_new], fit.before_state
            ),
        }
        predict = lambda value, choices: np.asarray(choices)[
            np.argmax(value, axis=1)
        ]
        row = {
            "fold": fold_index,
            "train_rows_per_class": 8,
            "held_rows_per_class": 2,
            "before_old": _metrics(
                old_labels[held_old],
                predict(score["before"], old_classes),
                old_classes,
            ),
            "base_before_old": _metrics(
                old_labels[held_old],
                predict(score["base_before"], old_classes),
                old_classes,
            ),
            "after_old": _metrics(
                joint_labels[held_joint_old],
                predict(score["after_old"], joint_classes),
                old_classes,
            ),
            "base_after_old": _metrics(
                joint_labels[held_joint_old],
                predict(score["base_after_old"], joint_classes),
                old_classes,
            ),
            "after_new": _metrics(
                joint_labels[held_joint_new],
                predict(score["after_new"], joint_classes),
                new_classes,
            ),
            "base_after_new": _metrics(
                joint_labels[held_joint_new],
                predict(score["base_after_new"], joint_classes),
                new_classes,
            ),
            "old_score_bitwise_locked": bool(
                np.array_equal(
                    score["after_old"][:, :len(old_classes)],
                    score["locked_old_old"],
                )
                and np.array_equal(
                    score["after_new"][:, :len(old_classes)],
                    score["locked_old_new"],
                )
            ),
            "before_selection_sha": fit.before_state.support_selection_sha256,
            "after_selection_sha": fit.after_state.support_selection_sha256,
            "state_sha": fit.after_state.state_content_sha256,
            "enabled": fit.after_state.enabled.tolist(),
            "before_decision_tensor_sha": _decision_tensor_sha(
                fit.before_state
            ),
            "after_decision_tensor_sha": _decision_tensor_sha(
                fit.after_state
            ),
            "floor_handles": _floor_handles(fit),
            "before_train_physical_id_sha": _physical_id_sha(
                before_artifact, ~held_old
            ),
            "before_held_physical_id_sha": _physical_id_sha(
                before_artifact, held_old
            ),
            "after_train_physical_id_sha": _physical_id_sha(
                after_artifact, ~held_joint
            ),
            "after_held_physical_id_sha": _physical_id_sha(
                after_artifact, held_joint
            ),
            "held_disjoint_from_selection": bool(
                not set(np.flatnonzero(held_old))
                & set(np.flatnonzero(~held_old))
                and not set(np.flatnonzero(held_joint))
                & set(np.flatnonzero(~held_joint))
            ),
        }
        joint_truth = np.concatenate([
            joint_labels[held_joint_old],
            joint_labels[held_joint_new],
        ])
        joint_prediction = np.concatenate([
            predict(score["after_old"], joint_classes),
            predict(score["after_new"], joint_classes),
        ])
        base_joint_prediction = np.concatenate([
            predict(score["base_after_old"], joint_classes),
            predict(score["base_after_new"], joint_classes),
        ])
        row["joint"] = _metrics(
            joint_truth, joint_prediction, joint_classes
        )
        row["base_joint"] = _metrics(
            joint_truth, base_joint_prediction, joint_classes
        )
        row["H_old_new"] = _harmonic(
            row["after_old"]["overall_accuracy"],
            row["after_new"]["overall_accuracy"],
        )
        row["base_H_old_new"] = _harmonic(
            row["base_after_old"]["overall_accuracy"],
            row["base_after_new"]["overall_accuracy"],
        )
        row["old_forgetting"] = (
            row["before_old"]["overall_accuracy"]
            - row["after_old"]["overall_accuracy"]
        )
        row["per_class_old_forgetting"] = {
            value: (
                row["before_old"]["per_class_accuracy"][value]
                - row["after_old"]["per_class_accuracy"][value]
            )
            for value in old_classes
        }
        row["candidate_vs_z0_per_class_non_degraded"] = {
            "before_old": {
                value: (
                    row["before_old"]["per_class_accuracy"][value]
                    + 1.0e-12
                    >= row["base_before_old"]["per_class_accuracy"][value]
                )
                for value in old_classes
            },
            "after_old": {
                value: (
                    row["after_old"]["per_class_accuracy"][value]
                    + 1.0e-12
                    >= row["base_after_old"]["per_class_accuracy"][value]
                )
                for value in old_classes
            },
            "after_new": {
                value: (
                    row["after_new"]["per_class_accuracy"][value]
                    + 1.0e-12
                    >= row["base_after_new"]["per_class_accuracy"][value]
                )
                for value in new_classes
            },
        }
        folds.append(row)
        trace.extend({"fold": fold_index, **value} for value in fit.trace)
        trace.append({"phase": "joint_l2o_fold", **row})
    result = {
        key: _aggregate(folds, key, classes)
        for key, classes in (
            ("before_old", old_classes),
            ("base_before_old", old_classes),
            ("after_old", old_classes),
            ("base_after_old", old_classes),
            ("after_new", new_classes),
            ("base_after_new", new_classes),
            ("joint", joint_classes),
            ("base_joint", joint_classes),
        )
    }
    result["folds"] = folds
    result["old_score_bitwise_locked"] = all(
        row["old_score_bitwise_locked"] for row in folds
    )
    result["H_old_new"] = _harmonic(
        result["after_old"]["overall_accuracy"],
        result["after_new"]["overall_accuracy"],
    )
    result["base_H_old_new"] = _harmonic(
        result["base_after_old"]["overall_accuracy"],
        result["base_after_new"]["overall_accuracy"],
    )
    result["old_forgetting"] = (
        result["before_old"]["overall_accuracy"]
        - result["after_old"]["overall_accuracy"]
    )
    result["per_class_old_forgetting"] = {
        value: (
            result["before_old"]["per_class_accuracy"][value]
            - result["after_old"]["per_class_accuracy"][value]
        )
        for value in old_classes
    }
    result["candidate_vs_z0_per_class_non_degraded"] = {
        "before_old": {
            value: (
                result["before_old"]["per_class_accuracy"][value] + 1.0e-12
                >= result["base_before_old"]["per_class_accuracy"][value]
            )
            for value in old_classes
        },
        "after_old": {
            value: (
                result["after_old"]["per_class_accuracy"][value] + 1.0e-12
                >= result["base_after_old"]["per_class_accuracy"][value]
            )
            for value in old_classes
        },
        "after_new": {
            value: (
                result["after_new"]["per_class_accuracy"][value] + 1.0e-12
                >= result["base_after_new"]["per_class_accuracy"][value]
            )
            for value in new_classes
        },
    }
    return result, tuple(trace)


def _state_sha(state):
    digest = hashlib.sha256()
    for name in (
        "prototypes", "selected_dims", "class_mean", "class_var", "rest_mean",
        "rest_var", "enabled", "llr_mid", "llr_half_gap", "a_plus", "a_minus",
    ):
        value = getattr(state, name)
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps({
        "schema": state.schema, "candidate_id": state.candidate_id,
        "classes": state.classes, "feature_dim": state.feature_dim,
        "k_shot": state.k_shot, "old_class_count": state.old_class_count,
        "generation": state.registration_generation,
        "resource": dict(state.resource),
        "artifact": state.support_feature_artifact_sha256,
        "selection": state.support_selection_sha256,
        "runtime": state.sealed_runtime_sha256,
        "code": state.feature_code_sha256,
        "checkpoint": state.sealed_phase1_checkpoint_sha256,
        "operator": state.operator_id, "view_seed": state.view_seed,
        "hp": state.hyperparameters.__dict__,
    }, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _validate_state(state):
    _validate_hp(state.hyperparameters)
    count, rank = len(state.classes), state.hyperparameters.rank
    canonical_k1 = _canonical_k1_hp()
    floats = (
        state.prototypes, state.class_mean, state.class_var, state.rest_mean,
        state.rest_var, state.llr_mid, state.llr_half_gap, state.a_plus,
        state.a_minus,
    )
    if (
        state.schema != SCHEMA
        or state.prototypes.shape != (count, state.feature_dim)
        or state.selected_dims.shape != (count, rank)
        or any(value.shape != (count, rank) for value in (
            state.class_mean, state.class_var, state.rest_mean, state.rest_var,
        ))
        or any(value.shape != (count,) for value in (
            state.enabled, state.llr_mid, state.llr_half_gap,
            state.a_plus, state.a_minus,
        ))
        or not all(np.isfinite(value).all() for value in floats)
        or np.any(state.class_var <= 0.0)
        or np.any(state.rest_var <= 0.0)
        or np.any(state.llr_half_gap[state.enabled] <= 0.0)
        or np.any(state.selected_dims[state.enabled] < 0)
        or np.any(state.selected_dims[state.enabled] >= state.feature_dim)
        or np.any(state.selected_dims[~state.enabled] != -1)
        or np.any(state.class_mean[~state.enabled] != 0.0)
        or np.any(state.rest_mean[~state.enabled] != 0.0)
        or np.any(state.class_var[~state.enabled] != 1.0)
        or np.any(state.rest_var[~state.enabled] != 1.0)
        or np.any(state.llr_mid[~state.enabled] != 0.0)
        or np.any(state.llr_half_gap[~state.enabled] != 0.0)
        or np.any(state.a_plus[~state.enabled] != 0.0)
        or np.any(state.a_minus[~state.enabled] != 0.0)
        or np.any(
            state.enabled
            & (state.a_plus == 0.0)
            & (state.a_minus == 0.0)
        )
        or not np.all(np.isin(
            state.a_plus, np.asarray(AMPLITUDE_GRID, dtype=np.float32)
        ))
        or not np.all(np.isin(
            state.a_minus, np.asarray(AMPLITUDE_GRID, dtype=np.float32)
        ))
        or _array_bytes(state) > MAX_STATE_BYTES
        or state.k_shot < 1
        or 2 <= state.k_shot <= 4
        or (state.k_shot == 1 and state.hyperparameters != canonical_k1)
        or (state.k_shot == 1 and rank != 0)
        or (rank > 0 and state.k_shot < 5)
        or (
            (rank == 0 or state.hyperparameters.force_zero)
            and (
                np.any(state.enabled)
                or state.selected_dims.shape[1] != 0
                or np.any(state.class_mean)
                or np.any(state.rest_mean)
                or np.any(state.class_var != 1.0)
                or np.any(state.rest_var != 1.0)
                or np.any(state.llr_mid)
                or np.any(state.llr_half_gap)
                or np.any(state.a_plus)
                or np.any(state.a_minus)
            )
        )
        or not 1 <= state.old_class_count <= count
        or (
            state.registration_generation == 0
            and state.old_class_count != count
        )
        or (
            state.registration_generation == 1
            and state.old_class_count >= count
        )
        or state.registration_generation not in (0, 1)
        or state.candidate_id != state.hyperparameters.candidate_id
        or state.operator_id != state.hyperparameters.operator_id
        or state.operator_id != "base"
        or state.resource.get("trainable_parameters") != 0
        or state.resource.get("adapt_epochs") != 0
        or state.resource.get("persistent_array_state_bytes") != _array_bytes(state)
        or state.resource.get("enabled_class_count") != int(np.sum(state.enabled))
        or state.resource.get("amplitude_grid") != list(AMPLITUDE_GRID)
        or state.resource.get("two_fold_model_fits_per_class") != (
            0 if rank == 0 else 2
        )
        or state.resource.get("backbone_forwards_per_physical_sample") != 1
        or state.resource.get("fft_branches_per_physical_sample") != 0
        or not isinstance(
            state.resource.get("head_scalar_ops_per_sample_upper_bound"), int
        )
        or state.resource.get("head_scalar_ops_per_sample_upper_bound")
        != count * rank * 12
        or bool(state.resource.get("dense_query_graph", True))
        or state.state_content_sha256 != _state_sha(state)
    ):
        raise FcarError("state drift")
