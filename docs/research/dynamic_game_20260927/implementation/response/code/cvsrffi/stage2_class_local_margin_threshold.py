"""D15 support-only class-local cosine-margin threshold calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_joint_residual_logit_head import (
    RuntimeAuthorizedFeatureArtifact,
)


EPS = 1.0e-8
SCHEMA = "cvs.phase2.class_local_margin_threshold.v1"
MAX_STATE_BYTES = 256 * 1024


class ClassLocalMarginThresholdError(ValueError):
    """Raised when the D15 support-only contract fails closed."""


@dataclass(frozen=True)
class MarginThresholdHyperparameters:
    candidate_id: str
    cap: float
    select_band_old: float = 0.20
    max_old_pairs: int = 3
    operator_id: str = "base"
    force_zero: bool = False


@dataclass(frozen=True)
class ClassLocalMarginThresholdState:
    schema: str
    candidate_id: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    old_pairs: np.ndarray
    old_thresholds: np.ndarray
    new_thresholds: np.ndarray
    hyperparameters: MarginThresholdHyperparameters
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
        for name in ("prototypes", "old_thresholds", "new_thresholds"):
            source = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            object.__setattr__(
                self,
                name,
                np.frombuffer(source.tobytes(), dtype=np.float32).reshape(
                    source.shape
                ),
            )
        pairs = np.ascontiguousarray(self.old_pairs, dtype=np.int64)
        object.__setattr__(
            self,
            "old_pairs",
            np.frombuffer(pairs.tobytes(), dtype=np.int64).reshape(pairs.shape),
        )
        computed = _state_content_sha256(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise ClassLocalMarginThresholdError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)
        _validate_state(self)


@dataclass(frozen=True)
class BeforeAfterMarginThresholdFit:
    before_state: ClassLocalMarginThresholdState
    after_state: ClassLocalMarginThresholdState
    trace: tuple[dict[str, Any], ...]


def _validate_hp(value: MarginThresholdHyperparameters) -> None:
    if (
        not value.candidate_id
        or value.operator_id != "base"
        or not np.isfinite(value.cap)
        or value.cap < 0.0
        or not np.isfinite(value.select_band_old)
        or value.select_band_old < 0.0
        or not 0 <= int(value.max_old_pairs) <= 3
        or (
            value.force_zero
            and (value.cap != 0.0 or int(value.max_old_pairs) != 0)
        )
    ):
        raise ClassLocalMarginThresholdError("hyperparameter drift")


def _rows(value: RuntimeAuthorizedFeatureArtifact) -> np.ndarray:
    if not isinstance(value, RuntimeAuthorizedFeatureArtifact):
        raise ClassLocalMarginThresholdError(
            "ordinary feature mapping/array forbidden; authorized artifact required"
        )
    return value.features


def _validate_support(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = _rows(artifact)
    label_rows = np.asarray(labels).astype(str)
    rank_rows = np.asarray(ranks, dtype=np.int64)
    if len(rows) != len(label_rows) or len(rows) != len(rank_rows):
        raise ClassLocalMarginThresholdError("support alignment drift")
    classes, counts = np.unique(label_rows, return_counts=True)
    if (
        int(k_shot) < 1
        or len(classes) < 2
        or set(counts.tolist()) != {int(k_shot)}
        or any(
            set(rank_rows[label_rows == label].tolist()) != set(range(int(k_shot)))
            for label in classes
        )
    ):
        raise ClassLocalMarginThresholdError("strict physical K-shot support drift")
    return rows, label_rows, rank_rows, tuple(sorted(classes.tolist()))


def _validate_binding(
    before: RuntimeAuthorizedFeatureArtifact,
    after: RuntimeAuthorizedFeatureArtifact,
) -> None:
    if (
        before.sealed_runtime_sha256 != after.sealed_runtime_sha256
        or before.feature_code_sha256 != after.feature_code_sha256
        or before.sealed_phase1_checkpoint_sha256
        != after.sealed_phase1_checkpoint_sha256
        or before.operator_id != after.operator_id
        or before.view_seed != after.view_seed
        or before.operator_id != "base"
    ):
        raise ClassLocalMarginThresholdError("runtime/operator binding drift")


def _validate_old_reuse(
    before: RuntimeAuthorizedFeatureArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after: RuntimeAuthorizedFeatureArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    old_classes: Sequence[str],
) -> None:
    def keyed(artifact, labels, ranks):
        return {
            (str(labels[index]), int(ranks[index])): (
                artifact.physical_sample_ids[index],
                artifact.parent_received_iq_sha256[index],
                artifact.per_row_feature_sha256[index],
            )
            for index in range(len(labels))
            if str(labels[index]) in set(old_classes)
        }

    if keyed(before, before_labels, before_ranks) != keyed(
        after, after_labels, after_ranks
    ):
        raise ClassLocalMarginThresholdError("old support exact-reuse lock failed")


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)


def _prototypes(
    rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    normalized = _normalize(rows)
    means = np.stack(
        [np.mean(normalized[labels == label], axis=0) for label in classes]
    ).astype(np.float32)
    return _normalize(means).astype(np.float32)


def _maximum_weight_matching(
    weights: Mapping[tuple[int, int], float],
    classes: tuple[str, ...],
    *,
    max_edges: int,
) -> tuple[tuple[int, int], ...]:
    positive = {
        tuple(sorted(pair)): float(value)
        for pair, value in weights.items()
        if np.isfinite(value) and value > 0.0
    }
    candidates = []

    def visit(remaining, selected, total):
        canonical = tuple(
            sorted(
                tuple(sorted((classes[a], classes[b]))) for a, b in selected
            )
        )
        candidates.append((float(total), canonical, tuple(sorted(selected))))
        if not remaining or len(selected) >= max_edges:
            return
        first = remaining[0]
        visit(remaining[1:], selected, total)
        for position, second in enumerate(remaining[1:], start=1):
            pair = tuple(sorted((first, second)))
            if pair in positive:
                visit(
                    remaining[1:position] + remaining[position + 1 :],
                    selected + (pair,),
                    total + positive[pair],
                )

    visit(tuple(range(len(classes))), (), 0.0)
    best = max(value[0] for value in candidates)
    tied = [value for value in candidates if abs(value[0] - best) <= 1e-12]
    tied.sort(key=lambda value: (-len(value[2]), value[1]))
    return tied[0][2]


def _threshold_candidates(values: Sequence[float], cap: float) -> tuple[float, ...]:
    finite = sorted(set(float(value) for value in values if np.isfinite(value)))
    candidates = {0.0, -float(cap), float(cap)}
    candidates.update(float(np.clip(value, -cap, cap)) for value in finite)
    candidates.update(
        float(np.clip((first + second) / 2.0, -cap, cap))
        for first, second in zip(finite, finite[1:])
    )
    return tuple(sorted(candidates))


def _choose_binary_threshold(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    cap: float,
) -> tuple[float, dict[str, Any]]:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not len(pos) or not len(neg) or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        return 0.0, {"calibratable": False, "selected_threshold": 0.0}
    rows = []
    for threshold in _threshold_candidates(np.concatenate([pos, neg]), cap):
        pos_acc = float(np.mean(pos >= threshold))
        neg_acc = float(np.mean(neg < threshold))
        rows.append(
            (
                min(pos_acc, neg_acc),
                (pos_acc + neg_acc) / 2.0,
                -abs(threshold),
                -threshold,
                threshold,
                pos_acc,
                neg_acc,
            )
        )
    selected = max(rows)
    return float(selected[4]), {
        "calibratable": True,
        "selected_threshold": float(selected[4]),
        "positive_accuracy": float(selected[5]),
        "negative_accuracy": float(selected[6]),
        "floor_accuracy": float(selected[0]),
        "balanced_accuracy": float(selected[1]),
        "candidate_count": len(rows),
    }


def _fit_old(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    hp: MarginThresholdHyperparameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    prototypes = _prototypes(rows, labels, classes)
    counts = [int(np.sum(labels == label)) for label in classes]
    if hp.force_zero or hp.max_old_pairs == 0 or min(counts) < 2:
        return (
            prototypes,
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            (),
        )
    lookup = {label: index for index, label in enumerate(classes)}
    weights: dict[tuple[int, int], float] = {}
    margins: dict[tuple[int, int], list[tuple[int, float]]] = {}
    normalized = _normalize(rows)
    for index, held in enumerate(normalized):
        keep = np.ones(len(rows), dtype=bool)
        keep[index] = False
        loo = _prototypes(rows[keep], labels[keep], classes)
        score = held @ loo.T
        top = np.argsort(score, kind="stable")[-2:][::-1]
        truth = lookup[str(labels[index])]
        if int(top[0]) != truth:
            pair = tuple(sorted((truth, int(top[0]))))
            weights[pair] = weights.get(pair, 0.0) + 1.0
        else:
            runner = int(top[1])
            gap = float(score[truth] - score[runner])
            if hp.select_band_old > 0.0 and gap <= hp.select_band_old:
                pair = tuple(sorted((truth, runner)))
                weights[pair] = weights.get(pair, 0.0) + max(
                    0.0, 1.0 - gap / hp.select_band_old
                )
        for other in range(len(classes)):
            if other == truth:
                continue
            pair = tuple(sorted((truth, other)))
            margin = float(score[pair[0]] - score[pair[1]])
            margins.setdefault(pair, []).append((truth, margin))
    chosen = _maximum_weight_matching(
        weights, classes, max_edges=int(hp.max_old_pairs)
    )
    thresholds = []
    diagnostics = []
    for first, second in chosen:
        records = margins.get((first, second), [])
        positive = [margin for truth, margin in records if truth == first]
        negative = [margin for truth, margin in records if truth == second]
        threshold, diagnostic = _choose_binary_threshold(
            positive, negative, cap=hp.cap
        )
        thresholds.append(threshold)
        diagnostics.append(
            {
                "kind": "old_pair_margin_threshold",
                "first_class": classes[first],
                "second_class": classes[second],
                "collision_weight": float(weights[(first, second)]),
                **diagnostic,
            }
        )
    return (
        prototypes,
        np.asarray(chosen, dtype=np.int64).reshape(-1, 2),
        np.asarray(thresholds, dtype=np.float32),
        tuple(diagnostics),
    )


def _old_scores(
    rows: np.ndarray,
    prototypes: np.ndarray,
    pairs: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    base = _normalize(rows) @ prototypes.T
    result = np.array(base, dtype=np.float32, copy=True)
    top2 = np.argsort(base, axis=1, kind="stable")[:, -2:]
    for index, (first, second) in enumerate(pairs):
        first, second = int(first), int(second)
        active = np.all(
            np.sort(top2, axis=1) == np.asarray([first, second]), axis=1
        )
        half = np.float32(thresholds[index] / 2.0)
        result[active, first] -= half
        result[active, second] += half
    return result


def _calibrate_new_thresholds(
    old_rows: np.ndarray,
    old_labels: np.ndarray,
    old_classes: tuple[str, ...],
    joint_rows: np.ndarray,
    joint_labels: np.ndarray,
    joint_classes: tuple[str, ...],
    joint_prototypes: np.ndarray,
    locked_old_prototypes: np.ndarray,
    locked_old_pairs: np.ndarray,
    locked_old_thresholds: np.ndarray,
    hp: MarginThresholdHyperparameters,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    new_classes = joint_classes[len(old_classes) :]
    thresholds = np.zeros(len(new_classes), dtype=np.float32)
    if hp.force_zero:
        return thresholds, ()
    old_negative: list[float] = []
    for index, held in enumerate(_normalize(old_rows)):
        keep = np.ones(len(old_rows), dtype=bool)
        keep[index] = False
        prototypes, pairs, pair_thresholds, _ = _fit_old(
            old_rows[keep], old_labels[keep], old_classes, hp
        )
        scores = _old_scores(
            held[None, :], prototypes, pairs, pair_thresholds
        )[0]
        old_negative.append(float(-np.max(scores)))
    diagnostics = []
    normalized_joint = _normalize(joint_rows)
    for offset, label in enumerate(new_classes):
        class_index = len(old_classes) + offset
        positives = []
        mask = joint_labels == label
        indices = np.flatnonzero(mask)
        for index in indices:
            keep = mask.copy()
            keep[index] = False
            new_prototype = _prototypes(
                joint_rows[keep],
                joint_labels[keep],
                (label,),
            )[0]
            new_score = float(normalized_joint[index] @ new_prototype)
            before_old = _old_scores(
                joint_rows[index : index + 1],
                locked_old_prototypes,
                locked_old_pairs,
                locked_old_thresholds,
            )[0]
            positives.append(new_score - float(np.max(before_old)))
        negatives = [
            value + float(_normalize(old_rows[index : index + 1])[0] @ joint_prototypes[class_index])
            for index, value in enumerate(old_negative)
        ]
        threshold, diagnostic = _choose_binary_threshold(
            positives, negatives, cap=hp.cap
        )
        thresholds[offset] = np.float32(threshold)
        diagnostics.append(
            {
                "kind": "new_dynamic_max_old_margin_threshold",
                "new_class": label,
                "threshold_sign": (
                    "positive_suppress_new"
                    if threshold > 0
                    else ("negative_boost_new" if threshold < 0 else "identity")
                ),
                **diagnostic,
            }
        )
    return thresholds, tuple(diagnostics)


def _support_selection_sha(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    selection: np.ndarray,
) -> str:
    records = [
        {
            "label": str(labels[index]),
            "rank": int(ranks[index]),
            "physical_sample_id": artifact.physical_sample_ids[index],
            "parent_received_iq_sha256": artifact.parent_received_iq_sha256[index],
            "feature_sha256": artifact.per_row_feature_sha256[index],
        }
        for index in np.flatnonzero(selection)
    ]
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _make_state(
    artifact: RuntimeAuthorizedFeatureArtifact,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    pairs: np.ndarray,
    old_thresholds: np.ndarray,
    new_thresholds: np.ndarray,
    hp: MarginThresholdHyperparameters,
    *,
    k_shot: int,
    old_class_count: int,
    generation: int,
    selection_sha: str,
) -> ClassLocalMarginThresholdState:
    array_bytes = int(
        prototypes.nbytes
        + pairs.nbytes
        + old_thresholds.nbytes
        + new_thresholds.nbytes
    )
    return ClassLocalMarginThresholdState(
        schema=SCHEMA,
        candidate_id=hp.candidate_id,
        classes=classes,
        prototypes=prototypes,
        old_pairs=pairs,
        old_thresholds=old_thresholds,
        new_thresholds=new_thresholds,
        hyperparameters=hp,
        feature_dim=int(prototypes.shape[1]),
        k_shot=int(k_shot),
        old_class_count=int(old_class_count),
        registration_generation=int(generation),
        resource={
            "trainable_parameters": 0,
            "adapt_epochs": 0,
            "persistent_array_state_bytes": array_bytes,
            "old_pair_count": int(len(pairs)),
            "new_threshold_count": int(len(new_thresholds)),
            "prototype_cosine_mac_per_sample": int(
                len(classes) * prototypes.shape[1]
            ),
            "threshold_scalar_ops_per_sample_upper_bound": int(
                len(pairs) * 5 + len(new_thresholds)
            ),
            "backbone_forwards_per_physical_sample": 1,
            "fft_branches_per_physical_sample": 0,
            "dense_query_graph": False,
        },
        support_feature_artifact_sha256=artifact.artifact_sha256,
        support_selection_sha256=selection_sha,
        sealed_runtime_sha256=artifact.sealed_runtime_sha256,
        feature_code_sha256=artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=artifact.sealed_phase1_checkpoint_sha256,
        operator_id=artifact.operator_id,
        view_seed=artifact.view_seed,
    )


def _fit_selected(
    before_artifact,
    old_rows,
    old_labels,
    old_ranks,
    old_classes,
    old_selection,
    after_artifact,
    joint_rows,
    joint_labels,
    joint_ranks,
    joint_classes,
    joint_selection,
    hp,
) -> BeforeAfterMarginThresholdFit:
    selected_old_rows = old_rows[old_selection]
    selected_old_labels = old_labels[old_selection]
    selected_joint_rows = joint_rows[joint_selection]
    selected_joint_labels = joint_labels[joint_selection]
    old_prototypes, pairs, old_thresholds, old_diag = _fit_old(
        selected_old_rows, selected_old_labels, old_classes, hp
    )
    joint_prototypes = _prototypes(
        selected_joint_rows, selected_joint_labels, joint_classes
    )
    if not np.array_equal(
        old_prototypes, joint_prototypes[: len(old_classes)]
    ):
        raise ClassLocalMarginThresholdError("old prototype bitwise lock failed")
    new_thresholds, new_diag = _calibrate_new_thresholds(
        selected_old_rows,
        selected_old_labels,
        old_classes,
        selected_joint_rows,
        selected_joint_labels,
        joint_classes,
        joint_prototypes,
        old_prototypes,
        pairs,
        old_thresholds,
        hp,
    )
    before = _make_state(
        before_artifact,
        old_classes,
        old_prototypes,
        pairs,
        old_thresholds,
        np.empty((0,), dtype=np.float32),
        hp,
        k_shot=int(np.sum(old_selection & (old_labels == old_classes[0]))),
        old_class_count=len(old_classes),
        generation=0,
        selection_sha=_support_selection_sha(
            before_artifact, old_labels, old_ranks, old_selection
        ),
    )
    after = _make_state(
        after_artifact,
        joint_classes,
        joint_prototypes,
        np.array(pairs, copy=True),
        np.array(old_thresholds, copy=True),
        new_thresholds,
        hp,
        k_shot=int(np.sum(joint_selection & (joint_labels == joint_classes[0]))),
        old_class_count=len(old_classes),
        generation=1,
        selection_sha=_support_selection_sha(
            after_artifact, joint_labels, joint_ranks, joint_selection
        ),
    )
    return BeforeAfterMarginThresholdFit(
        before_state=before,
        after_state=after,
        trace=(
            {
                "phase": "before_margin_threshold_fit",
                "candidate_id": hp.candidate_id,
                "old_pairs": pairs.tolist(),
                "old_thresholds": old_thresholds.tolist(),
                "diagnostics": [dict(value) for value in old_diag],
                "support_selection_sha256": before.support_selection_sha256,
            },
            {
                "phase": "after_margin_threshold_fit",
                "candidate_id": hp.candidate_id,
                "old_pairs": pairs.tolist(),
                "old_thresholds": old_thresholds.tolist(),
                "new_thresholds": new_thresholds.tolist(),
                "diagnostics": [dict(value) for value in new_diag],
                "support_selection_sha256": after.support_selection_sha256,
            },
        ),
    )


def fit_before_after_locked(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    k_shot: int,
    hyperparameters: MarginThresholdHyperparameters,
) -> BeforeAfterMarginThresholdFit:
    _validate_hp(hyperparameters)
    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=k_shot
    )
    joint_rows, joint_labels, joint_ranks, found = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=k_shot
    )
    if not set(old_classes) < set(found):
        raise ClassLocalMarginThresholdError("registration class drift")
    _validate_binding(before_artifact, after_artifact)
    _validate_old_reuse(
        before_artifact,
        old_labels,
        old_ranks,
        after_artifact,
        joint_labels,
        joint_ranks,
        old_classes,
    )
    joint_classes = old_classes + tuple(sorted(set(found) - set(old_classes)))
    return _fit_selected(
        before_artifact,
        old_rows,
        old_labels,
        old_ranks,
        old_classes,
        np.ones(len(old_labels), dtype=bool),
        after_artifact,
        joint_rows,
        joint_labels,
        joint_ranks,
        joint_classes,
        np.ones(len(joint_labels), dtype=bool),
        hyperparameters,
    )


def _score_numpy(
    rows: np.ndarray, state: ClassLocalMarginThresholdState
) -> np.ndarray:
    _validate_state(state)
    before = _old_scores(
        rows,
        state.prototypes[: state.old_class_count],
        state.old_pairs,
        state.old_thresholds,
    )
    if state.registration_generation == 0:
        return before
    new = _normalize(rows) @ state.prototypes[state.old_class_count :].T
    new = new.astype(np.float32) - state.new_thresholds[None, :]
    result = np.concatenate([before, new], axis=1)
    if not np.array_equal(result[:, : state.old_class_count], before):
        raise ClassLocalMarginThresholdError("After old score freeze violated")
    return result


def _state_content_sha256(state: ClassLocalMarginThresholdState) -> str:
    digest = hashlib.sha256()
    for value in (
        state.prototypes,
        state.old_pairs,
        state.old_thresholds,
        state.new_thresholds,
    ):
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(
        json.dumps(
            {
                "schema": state.schema,
                "candidate_id": state.candidate_id,
                "classes": state.classes,
                "feature_dim": state.feature_dim,
                "k_shot": state.k_shot,
                "old_class_count": state.old_class_count,
                "registration_generation": state.registration_generation,
                "resource": dict(state.resource),
                "support_feature_artifact_sha256": state.support_feature_artifact_sha256,
                "support_selection_sha256": state.support_selection_sha256,
                "sealed_runtime_sha256": state.sealed_runtime_sha256,
                "feature_code_sha256": state.feature_code_sha256,
                "sealed_phase1_checkpoint_sha256": state.sealed_phase1_checkpoint_sha256,
                "operator_id": state.operator_id,
                "view_seed": state.view_seed,
                "hp": {
                    "cap": state.hyperparameters.cap,
                    "select_band_old": state.hyperparameters.select_band_old,
                    "max_old_pairs": state.hyperparameters.max_old_pairs,
                    "operator_id": state.hyperparameters.operator_id,
                    "force_zero": state.hyperparameters.force_zero,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()


def _calibration_tensor_sha256(
    state: ClassLocalMarginThresholdState,
) -> str:
    digest = hashlib.sha256()
    for value in (
        state.prototypes,
        state.old_pairs,
        state.old_thresholds,
        state.new_thresholds,
    ):
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _validate_state(state: ClassLocalMarginThresholdState) -> None:
    _validate_hp(state.hyperparameters)
    endpoints = state.old_pairs.reshape(-1).tolist()
    arrays = (
        state.prototypes,
        state.old_thresholds,
        state.new_thresholds,
    )
    array_bytes = sum(
        value.nbytes
        for value in (
            state.prototypes,
            state.old_pairs,
            state.old_thresholds,
            state.new_thresholds,
        )
    )
    new_count = len(state.classes) - state.old_class_count
    if (
        state.schema != SCHEMA
        or state.operator_id != "base"
        or state.hyperparameters.operator_id != state.operator_id
        or state.prototypes.shape != (len(state.classes), state.feature_dim)
        or state.old_pairs.ndim != 2
        or state.old_pairs.shape[1:] != (2,)
        or len(state.old_pairs) > 3
        or len(endpoints) != len(set(endpoints))
        or any(
            first < 0
            or second <= first
            or second >= state.old_class_count
            for first, second in state.old_pairs
        )
        or state.old_thresholds.shape != (len(state.old_pairs),)
        or state.new_thresholds.shape != (new_count,)
        or not all(np.isfinite(value).all() for value in arrays)
        or np.any(np.abs(state.old_thresholds) > state.hyperparameters.cap + 1e-7)
        or np.any(np.abs(state.new_thresholds) > state.hyperparameters.cap + 1e-7)
        or array_bytes > MAX_STATE_BYTES
        or int(state.resource.get("trainable_parameters", -1)) != 0
        or int(state.resource.get("adapt_epochs", -1)) != 0
        or int(state.resource.get("persistent_array_state_bytes", -1))
        != array_bytes
        or bool(state.resource.get("dense_query_graph", True))
        or state.state_content_sha256 != _state_content_sha256(state)
    ):
        raise ClassLocalMarginThresholdError("state drift")
    if state.hyperparameters.force_zero and (
        len(state.old_pairs)
        or np.any(state.old_thresholds != 0.0)
        or np.any(state.new_thresholds != 0.0)
    ):
        raise ClassLocalMarginThresholdError("true zero drift")


def _leave_two_out_masks(labels: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, ...]:
    if set(np.unique(ranks).tolist()) != set(range(10)):
        raise ClassLocalMarginThresholdError("joint L2O requires strict K10")
    masks = []
    for first in range(0, 10, 2):
        held = np.isin(ranks, (first, first + 1))
        if any(np.sum(held & (labels == label)) != 2 for label in np.unique(labels)):
            raise ClassLocalMarginThresholdError("physical held2 drift")
        masks.append(held)
    return tuple(masks)


def _metrics(truth, prediction, classes):
    per_class = {
        label: float(np.mean(prediction[truth == label] == label))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean(prediction == truth)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _aggregate(folds, key, classes):
    per_class = {
        label: float(np.mean([row[key]["per_class_accuracy"][label] for row in folds]))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean([row[key]["overall_accuracy"] for row in folds])),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def evaluate_joint_leave_two_out(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    hyperparameters: MarginThresholdHyperparameters,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=10
    )
    joint_rows, joint_labels, joint_ranks, found = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=10
    )
    _validate_binding(before_artifact, after_artifact)
    _validate_old_reuse(
        before_artifact, old_labels, old_ranks,
        after_artifact, joint_labels, joint_ranks, old_classes
    )
    joint_classes = old_classes + tuple(sorted(set(found) - set(old_classes)))
    new_classes = joint_classes[len(old_classes):]
    old_masks = _leave_two_out_masks(old_labels, old_ranks)
    joint_masks = _leave_two_out_masks(joint_labels, joint_ranks)
    old_in_joint = np.isin(joint_labels, old_classes)
    zero_hp = MarginThresholdHyperparameters(
        candidate_id="d15_z0", cap=0.0, max_old_pairs=0, force_zero=True
    )
    folds, trace = [], []
    for fold, (held_old, held_joint) in enumerate(zip(old_masks, joint_masks)):
        fitted = _fit_selected(
            before_artifact, old_rows, old_labels, old_ranks, old_classes, ~held_old,
            after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes,
            ~held_joint, hyperparameters
        )
        base = _fit_selected(
            before_artifact, old_rows, old_labels, old_ranks, old_classes, ~held_old,
            after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes,
            ~held_joint, zero_hp
        )
        held_old_joint = held_joint & old_in_joint
        held_new_joint = held_joint & ~old_in_joint
        before_scores = _score_numpy(old_rows[held_old], fitted.before_state)
        base_before_scores = _score_numpy(old_rows[held_old], base.before_state)
        after_old_scores = _score_numpy(joint_rows[held_old_joint], fitted.after_state)
        after_new_scores = _score_numpy(joint_rows[held_new_joint], fitted.after_state)
        base_old_scores = _score_numpy(joint_rows[held_old_joint], base.after_state)
        base_new_scores = _score_numpy(joint_rows[held_new_joint], base.after_state)
        locked_old_old = _score_numpy(joint_rows[held_old_joint], fitted.before_state)
        locked_old_new = _score_numpy(joint_rows[held_new_joint], fitted.before_state)
        pred = lambda scores, classes: np.asarray(classes)[np.argmax(scores, axis=1)]
        row = {
            "fold": fold,
            "before_old": _metrics(old_labels[held_old], pred(before_scores, old_classes), old_classes),
            "base_before_old": _metrics(old_labels[held_old], pred(base_before_scores, old_classes), old_classes),
            "after_old": _metrics(joint_labels[held_old_joint], pred(after_old_scores, joint_classes), old_classes),
            "after_new": _metrics(joint_labels[held_new_joint], pred(after_new_scores, joint_classes), new_classes),
            "base_after_old": _metrics(joint_labels[held_old_joint], pred(base_old_scores, joint_classes), old_classes),
            "base_after_new": _metrics(joint_labels[held_new_joint], pred(base_new_scores, joint_classes), new_classes),
            "joint_accuracy": float(np.mean(
                np.concatenate([pred(after_old_scores, joint_classes), pred(after_new_scores, joint_classes)])
                == np.concatenate([joint_labels[held_old_joint], joint_labels[held_new_joint]])
            )),
            "base_joint_accuracy": float(np.mean(
                np.concatenate([pred(base_old_scores, joint_classes), pred(base_new_scores, joint_classes)])
                == np.concatenate([joint_labels[held_old_joint], joint_labels[held_new_joint]])
            )),
            "old_score_bitwise_locked": bool(
                np.array_equal(after_old_scores[:, :len(old_classes)], locked_old_old)
                and np.array_equal(after_new_scores[:, :len(old_classes)], locked_old_new)
            ),
            "old_pairs": fitted.before_state.old_pairs.tolist(),
            "old_thresholds": fitted.before_state.old_thresholds.tolist(),
            "new_thresholds": fitted.after_state.new_thresholds.tolist(),
            "before_selection_sha": fitted.before_state.support_selection_sha256,
            "after_selection_sha": fitted.after_state.support_selection_sha256,
            "calibration_tensor_sha": _calibration_tensor_sha256(
                fitted.after_state
            ),
            "old_train_rows_per_class": 8,
            "new_train_rows_per_class": 8,
        }
        row["h_old_new"] = (
            0.0 if row["after_old"]["overall_accuracy"] + row["after_new"]["overall_accuracy"] == 0
            else 2 * row["after_old"]["overall_accuracy"] * row["after_new"]["overall_accuracy"]
            / (row["after_old"]["overall_accuracy"] + row["after_new"]["overall_accuracy"])
        )
        folds.append(row)
        trace.extend({"fold": fold, **value} for value in fitted.trace)
        trace.append({"phase": "joint_l2o_fold", **row})
    result = {
        key: _aggregate(folds, key, classes)
        for key, classes in (
            ("before_old", old_classes), ("base_before_old", old_classes),
            ("after_old", old_classes), ("after_new", new_classes),
            ("base_after_old", old_classes), ("base_after_new", new_classes),
        )
    }
    result["joint_accuracy"] = float(np.mean([row["joint_accuracy"] for row in folds]))
    result["base_joint_accuracy"] = float(np.mean([row["base_joint_accuracy"] for row in folds]))
    old, new = result["after_old"]["overall_accuracy"], result["after_new"]["overall_accuracy"]
    base_old, base_new = result["base_after_old"]["overall_accuracy"], result["base_after_new"]["overall_accuracy"]
    result["h_old_new"] = 0.0 if old + new == 0 else 2 * old * new / (old + new)
    result["base_h_old_new"] = 0.0 if base_old + base_new == 0 else 2 * base_old * base_new / (base_old + base_new)
    result["old_forgetting"] = result["before_old"]["overall_accuracy"] - old
    result["before_old_per_class_non_degraded_vs_base"] = all(
        result["before_old"]["per_class_accuracy"][label] + 1e-12
        >= result["base_before_old"]["per_class_accuracy"][label]
        for label in old_classes
    )
    result["after_old_per_class_non_degraded_vs_before"] = all(
        result["after_old"]["per_class_accuracy"][label] + 1e-12
        >= result["before_old"]["per_class_accuracy"][label]
        for label in old_classes
    )
    result["after_new_per_class_non_degraded_vs_base"] = all(
        result["after_new"]["per_class_accuracy"][label] + 1e-12
        >= result["base_after_new"]["per_class_accuracy"][label]
        for label in new_classes
    )
    result["old_score_bitwise_locked"] = all(row["old_score_bitwise_locked"] for row in folds)
    result["folds"] = folds
    return result, tuple(trace)


def predict_all_registered(
    state: ClassLocalMarginThresholdState,
    query_artifact: RuntimeAuthorizedFeatureArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    rows = _rows(query_artifact)
    if len(rows) != 1:
        raise ClassLocalMarginThresholdError("formal prediction requires exactly one query")
    if (
        query_artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or query_artifact.feature_code_sha256 != state.feature_code_sha256
        or query_artifact.sealed_phase1_checkpoint_sha256 != state.sealed_phase1_checkpoint_sha256
        or query_artifact.operator_id != state.operator_id
        or query_artifact.view_seed != state.view_seed
    ):
        raise ClassLocalMarginThresholdError("query binding mismatch")
    scores = _score_numpy(rows, state)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)], scores
