"""D17 sparse pair-specific robust Student-t density-ratio registration.

Only runtime-authorized base features extracted from one fixed received
LEO_weak IQ per physical sample are accepted.  Before registration, at most
three endpoint-disjoint old--old edges apply zero-sum score corrections.
After registration the complete old score function is bitwise frozen; at most
two old rivals per new class may change only the highest new score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_joint_residual_logit_head import (
    RuntimeAuthorizedFeatureArtifact,
)


EPS = 1.0e-8
NU = 3.0
THRESHOLD = 0.5
ALPHA_GRID = (0.0, 0.005, 0.01)
SCHEMA = "cvs.phase2.sprtdr.v1"
MAX_STATE_BYTES = 50 * 1024


class SprtdrError(ValueError):
    """Raised when the D17 support-only contract fails closed."""


@dataclass(frozen=True)
class SprtdrHyperparameters:
    candidate_id: str
    rank: int = 8
    margin_band: float = 0.02
    max_old_edges: int = 3
    max_new_rivals: int = 2
    operator_id: str = "base"
    force_zero: bool = False


@dataclass(frozen=True)
class SprtdrState:
    schema: str
    candidate_id: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    old_pairs: np.ndarray
    old_dims: np.ndarray
    old_mu_a: np.ndarray
    old_var_a: np.ndarray
    old_mu_b: np.ndarray
    old_var_b: np.ndarray
    old_mid: np.ndarray
    old_gap: np.ndarray
    old_alpha_pos: np.ndarray
    old_alpha_neg: np.ndarray
    new_rivals: np.ndarray
    new_dims: np.ndarray
    new_mu: np.ndarray
    new_var: np.ndarray
    rival_mu: np.ndarray
    rival_var: np.ndarray
    new_mid: np.ndarray
    new_gap: np.ndarray
    new_alpha_pos: np.ndarray
    new_alpha_neg: np.ndarray
    old_floor: np.ndarray
    new_floor: np.ndarray
    hyperparameters: SprtdrHyperparameters
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
        floats = (
            "prototypes", "old_mu_a", "old_var_a", "old_mu_b", "old_var_b",
            "old_mid", "old_gap", "old_alpha_pos", "old_alpha_neg",
            "new_mu", "new_var", "rival_mu", "rival_var", "new_mid",
            "new_gap", "new_alpha_pos", "new_alpha_neg",
        )
        integers = ("old_pairs", "old_dims", "new_rivals", "new_dims")
        booleans = ("old_floor", "new_floor")
        for name in floats:
            value = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            object.__setattr__(
                self, name,
                np.frombuffer(value.tobytes(), dtype=np.float32).reshape(value.shape),
            )
        for name in integers:
            value = np.ascontiguousarray(getattr(self, name), dtype=np.int64)
            object.__setattr__(
                self, name,
                np.frombuffer(value.tobytes(), dtype=np.int64).reshape(value.shape),
            )
        for name in booleans:
            value = np.ascontiguousarray(getattr(self, name), dtype=np.bool_)
            object.__setattr__(
                self, name,
                np.frombuffer(value.tobytes(), dtype=np.bool_).reshape(value.shape),
            )
        computed = _state_sha(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise SprtdrError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)
        _validate_state(self)


@dataclass(frozen=True)
class BeforeAfterSprtdrFit:
    before_state: SprtdrState
    after_state: SprtdrState
    trace: tuple[dict[str, Any], ...]


def _validate_hp(hp: SprtdrHyperparameters) -> None:
    if (
        not hp.candidate_id
        or hp.operator_id != "base"
        or int(hp.rank) < 0
        or int(hp.rank) > 8
        or not np.isfinite(hp.margin_band)
        or hp.margin_band not in (0.0, 0.02, 0.04)
        or not 0 <= int(hp.max_old_edges) <= 3
        or not 0 <= int(hp.max_new_rivals) <= 2
        or (
            hp.force_zero
            and (
                hp.rank != 0
                or hp.margin_band != 0.0
                or hp.max_old_edges != 0
                or hp.max_new_rivals != 0
            )
        )
    ):
        raise SprtdrError("hyperparameter drift")


def _canonical_k1_hp(candidate_id: str) -> SprtdrHyperparameters:
    return SprtdrHyperparameters(
        candidate_id=candidate_id,
        rank=0,
        margin_band=0.0,
        max_old_edges=0,
        max_new_rivals=0,
        operator_id="base",
        force_zero=True,
    )


def _rows(artifact: RuntimeAuthorizedFeatureArtifact) -> np.ndarray:
    if not isinstance(artifact, RuntimeAuthorizedFeatureArtifact):
        raise SprtdrError("runtime-authorized feature artifact required")
    return artifact.features


def _validate_support(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: Sequence[str] | np.ndarray,
    ranks: Sequence[int] | np.ndarray,
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    rows = _rows(artifact)
    labels = np.asarray(labels).astype(str)
    ranks = np.asarray(ranks, dtype=np.int64)
    if len(rows) != len(labels) or len(rows) != len(ranks):
        raise SprtdrError("support alignment drift")
    classes, counts = np.unique(labels, return_counts=True)
    if (
        int(k_shot) < 1
        or len(classes) < 2
        or set(counts.tolist()) != {int(k_shot)}
        or any(
            set(ranks[labels == value].tolist()) != set(range(int(k_shot)))
            for value in classes
        )
    ):
        raise SprtdrError("strict physical K-shot drift")
    return rows, labels, ranks, tuple(sorted(classes.tolist()))


def _validate_binding(
    before: RuntimeAuthorizedFeatureArtifact,
    after: RuntimeAuthorizedFeatureArtifact,
) -> None:
    if (
        before.sealed_runtime_sha256 != after.sealed_runtime_sha256
        or before.feature_code_sha256 != after.feature_code_sha256
        or before.sealed_phase1_checkpoint_sha256
        != after.sealed_phase1_checkpoint_sha256
        or before.operator_id != "base"
        or after.operator_id != "base"
        or before.view_seed != after.view_seed
    ):
        raise SprtdrError("runtime/operator binding drift")


def _validate_old_reuse(
    before: RuntimeAuthorizedFeatureArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after: RuntimeAuthorizedFeatureArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    old_classes: Sequence[str],
) -> None:
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

    if keyed(before, before_labels, before_ranks) != keyed(
        after, after_labels, after_ranks
    ):
        raise SprtdrError("old exact-reuse lock failed")


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), EPS)


def _prototypes(
    rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> np.ndarray:
    means = np.stack([
        np.mean(_normalize(rows[labels == value]), axis=0)
        for value in classes
    ])
    return _normalize(means).astype(np.float32)


def _student_logq(
    rows: np.ndarray, dims: np.ndarray, mu: np.ndarray, variance: np.ndarray
) -> np.ndarray:
    z = _normalize(rows)[:, dims]
    return _student_logq_normalized_selected(z, mu, variance)


def _student_logq_normalized_selected(
    selected_z: np.ndarray, mu: np.ndarray, variance: np.ndarray
) -> np.ndarray:
    sigma = np.sqrt(variance)
    return -np.sum(
        np.log(sigma)
        + 0.5 * (NU + 1.0)
        * np.log1p(np.square(selected_z - mu) / (NU * variance)),
        axis=1,
    )


def _pair_model(
    rows: np.ndarray,
    labels: np.ndarray,
    class_a: str,
    class_b: str,
    rank: int,
) -> tuple[np.ndarray, ...] | None:
    a = _normalize(rows[labels == class_a])
    b = _normalize(rows[labels == class_b])
    if len(a) < 2 or len(b) < 2 or rank <= 0:
        return None
    med_a = np.median(a, axis=0)
    med_b = np.median(b, axis=0)
    pooled = 0.5 * (np.var(a, axis=0) + np.var(b, axis=0))
    score = np.abs(med_a - med_b) / np.sqrt(pooled + 0.01)
    dims = np.argsort(-score, kind="stable")[: min(rank, rows.shape[1])]
    mu_a = med_a[dims]
    mu_b = med_b[dims]
    pooled_dims = pooled[dims]
    mad_a = np.median(np.abs(a[:, dims] - mu_a), axis=0)
    mad_b = np.median(np.abs(b[:, dims] - mu_b), axis=0)
    var_a = 0.75 * pooled_dims + 0.25 * np.square(1.4826 * mad_a) + 0.01
    var_b = 0.75 * pooled_dims + 0.25 * np.square(1.4826 * mad_b) + 0.01
    values_a = (
        _student_logq(a, dims, mu_a, var_a)
        - _student_logq(a, dims, mu_b, var_b)
    )
    values_b = (
        _student_logq(b, dims, mu_a, var_a)
        - _student_logq(b, dims, mu_b, var_b)
    )
    q20 = float(np.quantile(values_a, 0.20, method="linear"))
    q80 = float(np.quantile(values_b, 0.80, method="linear"))
    gap = 0.5 * (q20 - q80)
    if not np.isfinite(gap) or gap <= 0.0:
        return None
    mid = 0.5 * (q20 + q80)
    return (
        np.asarray(dims, dtype=np.int64),
        np.asarray(mu_a, dtype=np.float32),
        np.asarray(var_a, dtype=np.float32),
        np.asarray(mu_b, dtype=np.float32),
        np.asarray(var_b, dtype=np.float32),
        np.asarray(mid, dtype=np.float32),
        np.asarray(gap, dtype=np.float32),
    )


def _llr_h(
    rows: np.ndarray,
    dims: np.ndarray,
    mu_a: np.ndarray,
    var_a: np.ndarray,
    mu_b: np.ndarray,
    var_b: np.ndarray,
    mid: float,
    gap: float,
) -> np.ndarray:
    return _llr_h_normalized(
        _normalize(rows), dims, mu_a, var_a, mu_b, var_b, mid, gap
    )


def _llr_h_normalized(
    normalized_rows: np.ndarray,
    dims: np.ndarray,
    mu_a: np.ndarray,
    var_a: np.ndarray,
    mu_b: np.ndarray,
    var_b: np.ndarray,
    mid: float,
    gap: float,
) -> np.ndarray:
    selected = normalized_rows[:, dims]
    llr = (
        _student_logq_normalized_selected(selected, mu_a, var_a)
        - _student_logq_normalized_selected(selected, mu_b, var_b)
    )
    return np.tanh((llr - float(mid)) / (float(gap) + EPS))


def _phi_positive(h: float) -> float:
    return max(0.0, (h - THRESHOLD) / (1.0 - THRESHOLD))


def _phi_negative(h: float) -> float:
    return max(0.0, (-h - THRESHOLD) / (1.0 - THRESHOLD))


def _maximum_weight_matching(
    weights: Mapping[tuple[int, int], float],
    opaque_handles: Sequence[str],
    max_edges: int,
) -> tuple[tuple[int, int], ...]:
    candidates: list[tuple[float, tuple[tuple[str, str], ...], tuple[tuple[int, int], ...]]] = []
    positive = {
        tuple(sorted(pair)): float(weight)
        for pair, weight in weights.items()
        if np.isfinite(weight) and weight > 0.0
    }

    def visit(remaining, selected, total):
        named = tuple(sorted(
            tuple(sorted((opaque_handles[a], opaque_handles[b])))
            for a, b in selected
        ))
        candidates.append((total, named, tuple(sorted(selected))))
        if not remaining or len(selected) >= max_edges:
            return
        first = remaining[0]
        visit(remaining[1:], selected, total)
        for position, second in enumerate(remaining[1:], start=1):
            pair = tuple(sorted((first, second)))
            if pair in positive:
                visit(
                    remaining[1:position] + remaining[position + 1:],
                    selected + (pair,),
                    total + positive[pair],
                )

    visit(tuple(range(len(opaque_handles))), (), 0.0)
    best = max(value[0] for value in candidates)
    tied = [value for value in candidates if abs(value[0] - best) <= 1e-12]
    tied.sort(key=lambda value: (-len(value[2]), value[1]))
    return tied[0][2]


def _loo_base(
    rows: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]
) -> np.ndarray:
    z = _normalize(rows)
    result = np.empty((len(rows), len(classes)), dtype=np.float32)
    for index in range(len(rows)):
        keep = np.ones(len(rows), dtype=bool)
        keep[index] = False
        result[index] = z[index] @ _prototypes(
            rows[keep], labels[keep], classes
        ).T
    return result


def _floor_mask(
    labels: np.ndarray,
    scores: np.ndarray,
    classes: tuple[str, ...],
    opaque_tie_handles: Sequence[str],
    score_classes: tuple[str, ...] | None = None,
) -> np.ndarray:
    choices = classes if score_classes is None else score_classes
    predictions = np.asarray(choices)[np.argmax(scores, axis=1)]
    accuracy = np.asarray([
        np.mean(predictions[labels == value] == value)
        for value in classes
    ])
    count = max(1, int(np.ceil(len(classes) * 0.25)))
    order = sorted(
        range(len(classes)),
        key=lambda i: (accuracy[i], opaque_tie_handles[i]),
    )
    result = np.zeros(len(classes), dtype=np.bool_)
    result[order[:count]] = True
    return result


def _opaque_class_handles(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: np.ndarray,
    selection: np.ndarray,
    classes: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            json.dumps(
                sorted(
                    artifact.physical_sample_ids[index]
                    for index in np.flatnonzero(
                        selection & (labels == value)
                    )
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for value in classes
    )


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    classes: tuple[str, ...],
    score_classes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    choices = classes if score_classes is None else score_classes
    predictions = np.asarray(choices)[np.argmax(scores, axis=1)]
    per_class = {
        value: float(np.mean(predictions[labels == value] == value))
        for value in classes
    }
    return {
        "overall_accuracy": float(np.mean(predictions == labels)),
        "min_class_accuracy": min(per_class.values()),
        "per_class_accuracy": per_class,
    }


def _harmonic(a: float, b: float) -> float:
    return 0.0 if a + b <= 0.0 else float(2.0 * a * b / (a + b))


def _selection_sha(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    selection: np.ndarray,
) -> str:
    records = [
        (
            str(labels[index]), int(ranks[index]),
            artifact.physical_sample_ids[index],
            artifact.parent_received_iq_sha256[index],
            artifact.per_row_feature_sha256[index],
        )
        for index in np.flatnonzero(selection)
    ]
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode()
    ).hexdigest()


def _array_bytes(state: SprtdrState) -> int:
    names = (
        "prototypes", "old_pairs", "old_dims", "old_mu_a", "old_var_a",
        "old_mu_b", "old_var_b", "old_mid", "old_gap", "old_alpha_pos",
        "old_alpha_neg", "new_rivals", "new_dims", "new_mu", "new_var",
        "rival_mu", "rival_var", "new_mid", "new_gap", "new_alpha_pos",
        "new_alpha_neg", "old_floor", "new_floor",
    )
    return int(sum(getattr(state, name).nbytes for name in names))


def _state_sha(state: SprtdrState) -> str:
    digest = hashlib.sha256()
    for name in (
        "prototypes", "old_pairs", "old_dims", "old_mu_a", "old_var_a",
        "old_mu_b", "old_var_b", "old_mid", "old_gap", "old_alpha_pos",
        "old_alpha_neg", "new_rivals", "new_dims", "new_mu", "new_var",
        "rival_mu", "rival_var", "new_mid", "new_gap", "new_alpha_pos",
        "new_alpha_neg", "old_floor", "new_floor",
    ):
        value = getattr(state, name)
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps({
        "schema": state.schema,
        "candidate_id": state.candidate_id,
        "classes": state.classes,
        "hp": state.hyperparameters.__dict__,
        "feature_dim": state.feature_dim,
        "k_shot": state.k_shot,
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "resource": dict(state.resource),
        "artifact": state.support_feature_artifact_sha256,
        "selection": state.support_selection_sha256,
        "runtime": state.sealed_runtime_sha256,
        "code": state.feature_code_sha256,
        "checkpoint": state.sealed_phase1_checkpoint_sha256,
        "operator": state.operator_id,
        "view_seed": state.view_seed,
    }, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _validate_state(state: SprtdrState) -> None:
    _validate_hp(state.hyperparameters)
    count = len(state.classes)
    rank = state.hyperparameters.rank
    edge_count = len(state.old_pairs)
    new_count = count - state.old_class_count
    rival_slots = new_count * state.hyperparameters.max_new_rivals
    floats = [
        state.prototypes, state.old_mu_a, state.old_var_a, state.old_mu_b,
        state.old_var_b, state.old_mid, state.old_gap, state.old_alpha_pos,
        state.old_alpha_neg, state.new_mu, state.new_var, state.rival_mu,
        state.rival_var, state.new_mid, state.new_gap, state.new_alpha_pos,
        state.new_alpha_neg,
    ]
    if (
        state.schema != SCHEMA
        or state.candidate_id != state.hyperparameters.candidate_id
        or len(set(state.classes)) != count
        or state.operator_id != "base"
        or state.operator_id != state.hyperparameters.operator_id
        or state.prototypes.shape != (count, state.feature_dim)
        or state.old_pairs.shape != (edge_count, 2)
        or state.old_dims.shape != (edge_count, rank)
        or any(value.shape != (edge_count, rank) for value in (
            state.old_mu_a, state.old_var_a, state.old_mu_b, state.old_var_b
        ))
        or any(value.shape != (edge_count,) for value in (
            state.old_mid, state.old_gap, state.old_alpha_pos, state.old_alpha_neg
        ))
        or state.new_rivals.shape != (new_count, state.hyperparameters.max_new_rivals)
        or state.new_dims.shape != (new_count, state.hyperparameters.max_new_rivals, rank)
        or any(value.shape != (new_count, state.hyperparameters.max_new_rivals, rank)
               for value in (state.new_mu, state.new_var, state.rival_mu, state.rival_var))
        or any(value.shape != (new_count, state.hyperparameters.max_new_rivals)
               for value in (
                   state.new_mid, state.new_gap, state.new_alpha_pos,
                   state.new_alpha_neg,
               ))
        or state.old_floor.shape != (state.old_class_count,)
        or state.new_floor.shape != (new_count,)
        or not all(np.isfinite(value).all() for value in floats)
        or np.any(state.old_var_a <= 0.0)
        or np.any(state.old_var_b <= 0.0)
        or np.any(state.new_var <= 0.0)
        or np.any(state.rival_var <= 0.0)
        or np.any(state.old_gap <= 0.0)
        or np.any(state.new_gap[state.new_rivals >= 0] <= 0.0)
        or np.any(state.old_dims < 0)
        or np.any(state.old_dims >= state.feature_dim)
        or np.any(state.new_dims[state.new_rivals >= 0] < 0)
        or np.any(state.new_dims[state.new_rivals >= 0] >= state.feature_dim)
        or np.any(state.new_dims[state.new_rivals < 0] != -1)
        or np.any(state.new_mu[state.new_rivals < 0] != 0.0)
        or np.any(state.rival_mu[state.new_rivals < 0] != 0.0)
        or np.any(state.new_var[state.new_rivals < 0] != 1.0)
        or np.any(state.rival_var[state.new_rivals < 0] != 1.0)
        or np.any(state.new_mid[state.new_rivals < 0] != 0.0)
        or np.any(state.new_gap[state.new_rivals < 0] != 0.0)
        or np.any(state.new_alpha_pos[state.new_rivals < 0] != 0.0)
        or np.any(state.new_alpha_neg[state.new_rivals < 0] != 0.0)
        or np.any(state.new_rivals[state.new_rivals >= 0] >= state.old_class_count)
        or np.any(state.new_rivals < -1)
        or any(
            len([value for value in row if value >= 0])
            != len(set(value for value in row if value >= 0))
            for row in state.new_rivals.tolist()
        )
        or np.any(state.old_pairs[:, 0] >= state.old_pairs[:, 1])
        or np.any(state.old_pairs < 0)
        or np.any(state.old_pairs >= state.old_class_count)
        or len(state.old_pairs.reshape(-1)) != len(set(state.old_pairs.reshape(-1).tolist()))
        or edge_count > state.hyperparameters.max_old_edges
        or rival_slots > 2 * new_count
        or not np.all(np.isin(
            state.old_alpha_pos, np.asarray(ALPHA_GRID, dtype=np.float32)
        ))
        or not np.all(np.isin(
            state.old_alpha_neg, np.asarray(ALPHA_GRID, dtype=np.float32)
        ))
        or not np.all(np.isin(
            state.new_alpha_pos, np.asarray(ALPHA_GRID, dtype=np.float32)
        ))
        or not np.all(np.isin(
            state.new_alpha_neg, np.asarray(ALPHA_GRID, dtype=np.float32)
        ))
        or _array_bytes(state) > MAX_STATE_BYTES
        or state.resource.get("persistent_array_state_bytes") != _array_bytes(state)
        or state.resource.get("trainable_parameters") != 0
        or state.resource.get("adapt_epochs") != 0
        or bool(state.resource.get("dense_query_graph", True))
        or state.resource.get("student_t_nu") != 3
        or state.resource.get("rank") != rank
        or state.resource.get("old_edge_count") != edge_count
        or state.resource.get("new_rival_edge_count") != int(np.sum(state.new_rivals >= 0))
        or state.resource.get("estimated_serialized_state_bytes")
        != _array_bytes(state) + 4096
        or state.resource.get("estimated_serialized_state_bytes")
        > MAX_STATE_BYTES
        or state.resource.get("backbone_forwards_per_physical_sample") != 1
        or state.resource.get("fft_branches_per_physical_sample") != 0
        or state.resource.get("prototype_mac_per_query")
        != count * state.feature_dim
        or state.resource.get("prototype_scorer_passes_per_query") != 1
        or state.resource.get("prototype_scorer")
        != "class_column_independent_einsum_optimize_false"
        or state.resource.get("normalized_feature_passes_per_query") != 1
        or state.resource.get("max_active_old_pair_per_query") != 1
        or state.resource.get("max_active_new_pair_per_query") != 1
        or state.resource.get("pair_density_mac_upper_bound_per_query")
        != 32 * rank
        or state.resource.get("head_mac_upper_bound_per_query")
        != count * state.feature_dim + 32 * rank
        or state.resource.get("student_log1p_ops_upper_bound_per_query")
        != 4 * rank
        or state.resource.get("identity_qknn_mac_per_query")
        != count * state.k_shot * state.feature_dim
        or (
            state.k_shot >= 5
            and state.resource.get("head_mac_upper_bound_per_query")
            >= state.resource.get("identity_qknn_mac_per_query")
        )
        or state.k_shot < 1
        or 2 <= state.k_shot <= 4
        or state.registration_generation not in (0, 1)
        or not 1 <= state.old_class_count <= count
        or (state.registration_generation == 0 and state.old_class_count != count)
        or (state.registration_generation == 1 and state.old_class_count >= count)
        or state.state_content_sha256 != _state_sha(state)
    ):
        raise SprtdrError("state drift")
    if (
        np.any(
            (state.old_alpha_pos == 0.0)
            & (state.old_alpha_neg == 0.0)
        )
        or np.any(
            (state.new_rivals >= 0)
            & (state.new_alpha_pos == 0.0)
            & (state.new_alpha_neg == 0.0)
        )
        or (
            (state.hyperparameters.force_zero or rank == 0)
            and (
                edge_count
                or np.any(state.new_rivals >= 0)
            )
        )
    ):
        raise SprtdrError("state drift")
    if state.k_shot == 1:
        canonical = _canonical_k1_hp(state.candidate_id)
        if (
            state.hyperparameters != canonical
            or edge_count
            or np.any(state.new_rivals >= 0)
            or rank != 0
        ):
            raise SprtdrError("state drift")


def _empty_pair_arrays(rank: int, new_count: int, max_rivals: int):
    return {
        "old_pairs": np.empty((0, 2), dtype=np.int64),
        "old_dims": np.empty((0, rank), dtype=np.int64),
        "old_mu_a": np.empty((0, rank), dtype=np.float32),
        "old_var_a": np.empty((0, rank), dtype=np.float32),
        "old_mu_b": np.empty((0, rank), dtype=np.float32),
        "old_var_b": np.empty((0, rank), dtype=np.float32),
        "old_mid": np.empty((0,), dtype=np.float32),
        "old_gap": np.empty((0,), dtype=np.float32),
        "old_alpha_pos": np.empty((0,), dtype=np.float32),
        "old_alpha_neg": np.empty((0,), dtype=np.float32),
        "new_rivals": np.full((new_count, max_rivals), -1, dtype=np.int64),
        "new_dims": np.full((new_count, max_rivals, rank), -1, dtype=np.int64),
        "new_mu": np.zeros((new_count, max_rivals, rank), dtype=np.float32),
        "new_var": np.ones((new_count, max_rivals, rank), dtype=np.float32),
        "rival_mu": np.zeros((new_count, max_rivals, rank), dtype=np.float32),
        "rival_var": np.ones((new_count, max_rivals, rank), dtype=np.float32),
        "new_mid": np.zeros((new_count, max_rivals), dtype=np.float32),
        "new_gap": np.zeros((new_count, max_rivals), dtype=np.float32),
        "new_alpha_pos": np.zeros((new_count, max_rivals), dtype=np.float32),
        "new_alpha_neg": np.zeros((new_count, max_rivals), dtype=np.float32),
    }


def _make_state(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    selection: np.ndarray,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    hp: SprtdrHyperparameters,
    old_class_count: int,
    generation: int,
    arrays: Mapping[str, np.ndarray],
    old_floor: np.ndarray,
    new_floor: np.ndarray,
) -> SprtdrState:
    k_shot = int(np.sum(selection & (labels == classes[0])))
    prototype_mac = int(len(classes) * prototypes.shape[1])
    pair_mac = int(32 * hp.rank)
    array_bytes = int(
        prototypes.nbytes
        + old_floor.nbytes
        + new_floor.nbytes
        + sum(value.nbytes for value in arrays.values())
    )
    resource = {
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "dense_query_graph": False,
        "student_t_nu": 3,
        "rank": hp.rank,
        "old_edge_count": len(arrays["old_pairs"]),
        "new_rival_edge_count": int(np.sum(arrays["new_rivals"] >= 0)),
        "backbone_forwards_per_physical_sample": 1,
        "fft_branches_per_physical_sample": 0,
        "persistent_array_state_bytes": array_bytes,
        "estimated_serialized_state_bytes": array_bytes + 4096,
        "prototype_mac_per_query": prototype_mac,
        "prototype_scorer_passes_per_query": 1,
        "prototype_scorer": (
            "class_column_independent_einsum_optimize_false"
        ),
        "normalized_feature_passes_per_query": 1,
        "max_active_old_pair_per_query": 1,
        "max_active_new_pair_per_query": 1,
        "pair_density_mac_upper_bound_per_query": pair_mac,
        "head_mac_upper_bound_per_query": prototype_mac + pair_mac,
        "student_log1p_ops_upper_bound_per_query": int(4 * hp.rank),
        "identity_qknn_mac_per_query": int(
            len(classes) * k_shot * prototypes.shape[1]
        ),
    }
    return SprtdrState(
        schema=SCHEMA,
        candidate_id=hp.candidate_id,
        classes=classes,
        prototypes=prototypes,
        **arrays,
        old_floor=old_floor,
        new_floor=new_floor,
        hyperparameters=hp,
        feature_dim=prototypes.shape[1],
        k_shot=k_shot,
        old_class_count=old_class_count,
        registration_generation=generation,
        resource=resource,
        support_feature_artifact_sha256=artifact.artifact_sha256,
        support_selection_sha256=_selection_sha(
            artifact, labels, ranks, selection
        ),
        sealed_runtime_sha256=artifact.sealed_runtime_sha256,
        feature_code_sha256=artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=artifact.sealed_phase1_checkpoint_sha256,
        operator_id=artifact.operator_id,
        view_seed=artifact.view_seed,
    )


def _prototype_scores(
    normalized_rows: np.ndarray, prototypes: np.ndarray
) -> np.ndarray:
    # Class-column-independent reduction: changing C_registered must not alter
    # the floating-point path for any existing class column.  optimize=False
    # also avoids a backend-dependent GEMM contraction that can break the
    # Before/After old-score bitwise lock when new columns are appended.
    return np.einsum(
        "nd,cd->nc", normalized_rows, prototypes, optimize=False
    ).astype(np.float32, copy=False)


def _score_old_from_base(
    normalized_rows: np.ndarray,
    base: np.ndarray,
    state: SprtdrState,
) -> np.ndarray:
    scores = np.array(base, copy=True)
    for edge, (a, b) in enumerate(state.old_pairs):
        top = np.argsort(
            base[:, :state.old_class_count], axis=1, kind="stable"
        )[:, -2:]
        hit = np.asarray([
            set(value.tolist()) == {int(a), int(b)} for value in top
        ])
        margin = np.abs(base[:, a] - base[:, b])
        active = hit & (margin <= state.hyperparameters.margin_band)
        if not np.any(active):
            continue
        h = _llr_h_normalized(
            normalized_rows[active], state.old_dims[edge],
            state.old_mu_a[edge], state.old_var_a[edge],
            state.old_mu_b[edge], state.old_var_b[edge],
            state.old_mid[edge], state.old_gap[edge],
        )
        delta = np.asarray([
            state.old_alpha_pos[edge] * _phi_positive(value)
            - state.old_alpha_neg[edge] * _phi_negative(value)
            for value in h
        ], dtype=np.float32)
        scores[active, a] += delta
        scores[active, b] -= delta
    return scores.astype(np.float32)


def _score_old(rows: np.ndarray, state: SprtdrState) -> np.ndarray:
    normalized = _normalize(rows)
    base = _prototype_scores(
        normalized, state.prototypes[:state.old_class_count]
    )
    return _score_old_from_base(normalized, base, state)


def _score_numpy(rows: np.ndarray, state: SprtdrState) -> np.ndarray:
    normalized = _normalize(rows)
    immutable_base = _prototype_scores(normalized, state.prototypes)
    old_base = immutable_base[:, :state.old_class_count]
    old_scores = _score_old_from_base(normalized, old_base, state)
    if state.registration_generation == 0:
        return old_scores
    new_base = immutable_base[:, state.old_class_count:]
    scores = np.concatenate([old_scores, new_base], axis=1).astype(np.float32)
    if not new_base.shape[1]:
        return scores
    best_new_local = np.argmax(new_base, axis=1)
    best_old = np.argmax(immutable_base[:, :state.old_class_count], axis=1)
    for row_index in range(len(rows)):
        new_local = int(best_new_local[row_index])
        old_index = int(best_old[row_index])
        slots = np.flatnonzero(state.new_rivals[new_local] == old_index)
        if not len(slots):
            continue
        new_index = state.old_class_count + new_local
        global_top2 = np.argsort(
            immutable_base[row_index], kind="stable"
        )[-2:]
        if set(global_top2.tolist()) != {new_index, old_index}:
            continue
        if abs(float(
            immutable_base[row_index, new_index]
            - immutable_base[row_index, old_index]
        )) > (
            state.hyperparameters.margin_band
        ):
            continue
        slot = int(slots[0])
        h = float(_llr_h_normalized(
            normalized[row_index:row_index + 1],
            state.new_dims[new_local, slot],
            state.new_mu[new_local, slot],
            state.new_var[new_local, slot],
            state.rival_mu[new_local, slot],
            state.rival_var[new_local, slot],
            state.new_mid[new_local, slot],
            state.new_gap[new_local, slot],
        )[0])
        scores[row_index, new_index] += np.float32(
            state.new_alpha_pos[new_local, slot] * _phi_positive(h)
            - state.new_alpha_neg[new_local, slot] * _phi_negative(h)
        )
    return scores


def _select_old_pairs(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    hp: SprtdrHyperparameters,
    floor: np.ndarray,
    opaque_handles: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    base = _loo_base(rows, labels, classes)
    lookup = {value: index for index, value in enumerate(classes)}
    weights: dict[tuple[int, int], float] = {}
    for index, value in enumerate(labels):
        truth = lookup[str(value)]
        top = np.argsort(base[index], kind="stable")[-2:][::-1]
        rival = int(top[0] if top[0] != truth else top[1])
        margin = float(base[index, truth] - base[index, rival])
        if int(top[0]) != truth or margin <= hp.margin_band:
            pair = tuple(sorted((truth, rival)))
            weights[pair] = weights.get(pair, 0.0) + (
                1.0 if int(top[0]) != truth else max(
                    0.0, 1.0 - margin / max(hp.margin_band, EPS)
                )
            )
            if floor[pair[0]] or floor[pair[1]]:
                weights[pair] += 0.25
    return _maximum_weight_matching(
        weights, opaque_handles, hp.max_old_edges
    )


def _choose_pair_alpha(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    pair: tuple[int, int],
    hp: SprtdrHyperparameters,
    floor: np.ndarray,
) -> tuple[float, float]:
    base = _loo_base(rows, labels, classes)
    lookup = {value: index for index, value in enumerate(classes)}
    records = []
    for index in range(len(rows)):
        keep = np.ones(len(rows), dtype=bool)
        keep[index] = False
        model = _pair_model(
            rows[keep], labels[keep], classes[pair[0]], classes[pair[1]], hp.rank
        )
        if model is None:
            return 0.0, 0.0
        records.append(float(_llr_h(rows[index:index + 1], *model)[0]))
    baseline = np.asarray([
        lookup[str(value)] == int(np.argmax(base[index]))
        for index, value in enumerate(labels)
    ])
    choices = []
    for pos in ALPHA_GRID:
        for neg in ALPHA_GRID:
            if pos == neg == 0.0:
                continue
            candidate = np.array(base, copy=True)
            for index, h in enumerate(records):
                top = np.argsort(candidate[index], kind="stable")[-2:]
                if set(top.tolist()) != set(pair):
                    continue
                if abs(float(candidate[index, pair[0]] - candidate[index, pair[1]])) > hp.margin_band:
                    continue
                delta = pos * _phi_positive(h) - neg * _phi_negative(h)
                candidate[index, pair[0]] += delta
                candidate[index, pair[1]] -= delta
            correct = np.asarray([
                lookup[str(value)] == int(np.argmax(candidate[index]))
                for index, value in enumerate(labels)
            ])
            endpoints = np.isin(labels, (classes[pair[0]], classes[pair[1]]))
            floor_strict = (
                not (floor[pair[0]] or floor[pair[1]])
                or any(
                    np.sum(correct[labels == classes[index]])
                    > np.sum(baseline[labels == classes[index]])
                    for index in pair
                    if floor[index]
                )
            )
            if (
                np.sum(correct & endpoints) >= np.sum(baseline & endpoints)
                and np.sum(correct) >= np.sum(baseline)
                and np.sum(correct) > np.sum(baseline)
                and floor_strict
            ):
                choices.append((
                    int(np.sum(correct) - np.sum(baseline)),
                    -(pos + neg), -pos, -neg, pos, neg,
                ))
    if not choices:
        return 0.0, 0.0
    chosen = max(choices)
    return chosen[-2], chosen[-1]


def _choose_new_alpha(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    new_index: int,
    rival: int,
    hp: SprtdrHyperparameters,
    old_floor: np.ndarray,
    new_floor: np.ndarray,
    old_class_count: int,
) -> tuple[float, float]:
    base = _loo_base(rows, labels, classes)
    lookup = {value: index for index, value in enumerate(classes)}
    evidence = []
    for index in range(len(rows)):
        keep = np.ones(len(rows), dtype=bool)
        keep[index] = False
        model = _pair_model(
            rows[keep], labels[keep], classes[new_index], classes[rival], hp.rank
        )
        if model is None:
            return 0.0, 0.0
        evidence.append(float(_llr_h(rows[index:index + 1], *model)[0]))
    baseline_prediction = np.argmax(base, axis=1)
    baseline_correct = np.asarray([
        baseline_prediction[index] == lookup[str(value)]
        for index, value in enumerate(labels)
    ])
    choices = []
    for pos in ALPHA_GRID:
        for neg in ALPHA_GRID:
            if pos == neg == 0.0:
                continue
            candidate = np.array(base, copy=True)
            for index, h in enumerate(evidence):
                top2 = np.argsort(base[index], kind="stable")[-2:]
                if set(top2.tolist()) != {new_index, rival}:
                    continue
                if abs(float(base[index, new_index] - base[index, rival])) > hp.margin_band:
                    continue
                candidate[index, new_index] += (
                    pos * _phi_positive(h) - neg * _phi_negative(h)
                )
            prediction = np.argmax(candidate, axis=1)
            correct = np.asarray([
                prediction[index] == lookup[str(value)]
                for index, value in enumerate(labels)
            ])
            per_class_ok = all(
                np.sum(correct[labels == value])
                >= np.sum(baseline_correct[labels == value])
                for value in classes
            )
            floor_targets = []
            if old_floor[rival]:
                floor_targets.append(rival)
            new_local = new_index - old_class_count
            if new_floor[new_local]:
                floor_targets.append(new_index)
            floor_strict = (
                not floor_targets
                or any(
                    np.sum(correct[labels == classes[index]])
                    > np.sum(baseline_correct[labels == classes[index]])
                    for index in floor_targets
                )
            )
            if (
                per_class_ok
                and np.sum(correct) > np.sum(baseline_correct)
                and floor_strict
            ):
                choices.append((
                    int(np.sum(correct) - np.sum(baseline_correct)),
                    -(pos + neg), -pos, -neg, pos, neg,
                ))
    if not choices:
        return 0.0, 0.0
    selected = max(choices)
    return selected[-2], selected[-1]


def _fit_arrays(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    old_class_count: int,
    hp: SprtdrHyperparameters,
    before_arrays: Mapping[str, np.ndarray] | None = None,
    old_floor: np.ndarray | None = None,
    new_floor: np.ndarray | None = None,
    opaque_handles: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], tuple[dict[str, Any], ...]]:
    new_count = len(classes) - old_class_count
    arrays = _empty_pair_arrays(hp.rank, new_count, hp.max_new_rivals)
    diagnostics: list[dict[str, Any]] = []
    if hp.force_zero or hp.rank == 0:
        return arrays, ()
    old_classes = classes[:old_class_count]
    if before_arrays is None:
        accepted = []
        for pair in _select_old_pairs(
            rows[np.isin(labels, old_classes)],
            labels[np.isin(labels, old_classes)],
            old_classes,
            hp,
            np.asarray(old_floor, dtype=np.bool_),
            tuple(opaque_handles[:old_class_count]),
        ):
            model = _pair_model(rows, labels, classes[pair[0]], classes[pair[1]], hp.rank)
            alpha = _choose_pair_alpha(
                rows[np.isin(labels, old_classes)],
                labels[np.isin(labels, old_classes)],
                old_classes,
                pair,
                hp,
                np.asarray(old_floor, dtype=np.bool_),
            )
            if model is not None and alpha != (0.0, 0.0):
                accepted.append((pair, model, alpha))
        if accepted:
            arrays["old_pairs"] = np.asarray([value[0] for value in accepted], dtype=np.int64)
            for name, offset in (
                ("old_dims", 1), ("old_mu_a", 2), ("old_var_a", 3),
                ("old_mu_b", 4), ("old_var_b", 5), ("old_mid", 6),
                ("old_gap", 7),
            ):
                arrays[name] = np.stack([value[1][offset - 1] for value in accepted])
            arrays["old_alpha_pos"] = np.asarray([value[2][0] for value in accepted], dtype=np.float32)
            arrays["old_alpha_neg"] = np.asarray([value[2][1] for value in accepted], dtype=np.float32)
    else:
        for name in (
            "old_pairs", "old_dims", "old_mu_a", "old_var_a", "old_mu_b",
            "old_var_b", "old_mid", "old_gap", "old_alpha_pos", "old_alpha_neg",
        ):
            arrays[name] = np.array(before_arrays[name], copy=True)
    if new_count:
        base = _loo_base(rows, labels, classes)
        for new_local, new_index in enumerate(range(old_class_count, len(classes))):
            new_rows = np.flatnonzero(labels == classes[new_index])
            rival_weight: dict[int, float] = {}
            for index in new_rows:
                old_rival = int(np.argmax(base[index, :old_class_count]))
                margin = float(base[index, new_index] - base[index, old_rival])
                if int(np.argmax(base[index])) != new_index or margin <= hp.margin_band:
                    rival_weight[old_rival] = rival_weight.get(old_rival, 0.0) + (
                        1.0 if int(np.argmax(base[index])) != new_index
                        else max(0.0, 1.0 - margin / max(hp.margin_band, EPS))
                    )
            for index in np.flatnonzero(np.isin(labels, old_classes)):
                truth_old = classes.index(str(labels[index]))
                if int(np.argmax(base[index])) == new_index:
                    rival_weight[truth_old] = rival_weight.get(truth_old, 0.0) + 1.0
                elif (
                    float(base[index, truth_old] - base[index, new_index])
                    <= hp.margin_band
                ):
                    rival_weight[truth_old] = rival_weight.get(truth_old, 0.0) + 0.25
            rivals = sorted(
                rival_weight,
                key=lambda value: (
                    -rival_weight[value], opaque_handles[value]
                ),
            )[:hp.max_new_rivals]
            for slot, rival in enumerate(rivals):
                model = _pair_model(
                    rows, labels, classes[new_index], classes[rival], hp.rank
                )
                if model is None:
                    continue
                alpha = _choose_new_alpha(
                    rows, labels, classes, new_index, rival, hp,
                    np.asarray(old_floor, dtype=np.bool_),
                    np.asarray(new_floor, dtype=np.bool_),
                    old_class_count,
                )
                if alpha == (0.0, 0.0):
                    continue
                arrays["new_rivals"][new_local, slot] = rival
                arrays["new_dims"][new_local, slot] = model[0]
                arrays["new_mu"][new_local, slot] = model[1]
                arrays["new_var"][new_local, slot] = model[2]
                arrays["rival_mu"][new_local, slot] = model[3]
                arrays["rival_var"][new_local, slot] = model[4]
                arrays["new_mid"][new_local, slot] = model[5]
                arrays["new_gap"][new_local, slot] = model[6]
                arrays["new_alpha_pos"][new_local, slot] = alpha[0]
                arrays["new_alpha_neg"][new_local, slot] = alpha[1]
    diagnostics.append({
        "old_pairs": arrays["old_pairs"].tolist(),
        "new_rivals": arrays["new_rivals"].tolist(),
        "pair_dimensions_use_endpoints_only": True,
        "student_t_nu": 3,
    })
    return arrays, tuple(diagnostics)


def _zero_hp(hp: SprtdrHyperparameters) -> SprtdrHyperparameters:
    return SprtdrHyperparameters(
        candidate_id=hp.candidate_id,
        rank=0,
        margin_band=0.0,
        max_old_edges=0,
        max_new_rivals=0,
        operator_id="base",
        force_zero=True,
    )


def _oof_replay_scores(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    state: SprtdrState,
) -> np.ndarray:
    """Replay a selected state with every scored physical row excluded."""

    base = _loo_base(rows, labels, classes)
    scores = np.array(base, copy=True)
    for row_index in range(len(rows)):
        keep = np.ones(len(rows), dtype=bool)
        keep[row_index] = False
        for edge, (a, b) in enumerate(state.old_pairs):
            top2 = np.argsort(
                base[row_index, :state.old_class_count], kind="stable"
            )[-2:]
            if set(top2.tolist()) != {int(a), int(b)}:
                continue
            if abs(float(base[row_index, a] - base[row_index, b])) > (
                state.hyperparameters.margin_band
            ):
                continue
            model = _pair_model(
                rows[keep], labels[keep], classes[a], classes[b],
                state.hyperparameters.rank,
            )
            if model is None:
                continue
            h = float(_llr_h(rows[row_index:row_index + 1], *model)[0])
            delta = (
                state.old_alpha_pos[edge] * _phi_positive(h)
                - state.old_alpha_neg[edge] * _phi_negative(h)
            )
            scores[row_index, a] += delta
            scores[row_index, b] -= delta
        if state.registration_generation == 0:
            continue
        new_base = base[row_index, state.old_class_count:]
        new_local = int(np.argmax(new_base))
        new_index = state.old_class_count + new_local
        old_index = int(np.argmax(base[row_index, :state.old_class_count]))
        slots = np.flatnonzero(state.new_rivals[new_local] == old_index)
        if not len(slots):
            continue
        top2 = np.argsort(base[row_index], kind="stable")[-2:]
        if set(top2.tolist()) != {new_index, old_index}:
            continue
        if abs(float(base[row_index, new_index] - base[row_index, old_index])) > (
            state.hyperparameters.margin_band
        ):
            continue
        model = _pair_model(
            rows[keep], labels[keep], classes[new_index], classes[old_index],
            state.hyperparameters.rank,
        )
        if model is None:
            continue
        slot = int(slots[0])
        h = float(_llr_h(rows[row_index:row_index + 1], *model)[0])
        scores[row_index, new_index] += (
            state.new_alpha_pos[new_local, slot] * _phi_positive(h)
            - state.new_alpha_neg[new_local, slot] * _phi_negative(h)
        )
    return scores.astype(np.float32)


def _fit_selected(
    before_artifact,
    before_rows,
    before_labels,
    before_ranks,
    old_classes,
    before_selection,
    after_artifact,
    after_rows,
    after_labels,
    after_ranks,
    joint_classes,
    after_selection,
    hp,
) -> BeforeAfterSprtdrFit:
    if int(np.sum(before_selection & (before_labels == old_classes[0]))) == 1:
        hp = _canonical_k1_hp(hp.candidate_id)
    selected_old = before_rows[before_selection]
    selected_old_labels = before_labels[before_selection]
    selected_joint = after_rows[after_selection]
    selected_joint_labels = after_labels[after_selection]
    old_prototypes = _prototypes(selected_old, selected_old_labels, old_classes)
    joint_prototypes = _prototypes(selected_joint, selected_joint_labels, joint_classes)
    if not np.array_equal(old_prototypes, joint_prototypes[:len(old_classes)]):
        raise SprtdrError("old prototype bitwise lock failed")
    k_selected = int(np.sum(selected_old_labels == old_classes[0]))
    old_base = (
        _normalize(selected_old) @ old_prototypes.T
        if k_selected == 1
        else _loo_base(selected_old, selected_old_labels, old_classes)
    )
    old_opaque_handles = _opaque_class_handles(
        before_artifact, before_labels, before_selection, old_classes
    )
    old_floor = _floor_mask(
        selected_old_labels,
        old_base,
        old_classes,
        old_opaque_handles,
    )
    new_classes = joint_classes[len(old_classes):]
    joint_opaque_handles = _opaque_class_handles(
        after_artifact, after_labels, after_selection, joint_classes
    )
    joint_base = (
        _normalize(selected_joint) @ joint_prototypes.T
        if k_selected == 1
        else _loo_base(selected_joint, selected_joint_labels, joint_classes)
    )
    new_mask = np.isin(selected_joint_labels, new_classes)
    new_floor = (
        _floor_mask(
            selected_joint_labels[new_mask],
            joint_base[new_mask],
            new_classes,
            _opaque_class_handles(
                after_artifact, after_labels, after_selection, new_classes
            ),
            joint_classes,
        )
        if new_classes else np.empty((0,), dtype=np.bool_)
    )
    before_arrays, before_diag = _fit_arrays(
        selected_old, selected_old_labels, old_classes, len(old_classes), hp,
        old_floor=old_floor, new_floor=np.empty((0,), dtype=np.bool_),
        opaque_handles=old_opaque_handles,
    )
    after_arrays, after_diag = _fit_arrays(
        selected_joint, selected_joint_labels, joint_classes,
        len(old_classes), hp, before_arrays, old_floor=old_floor,
        new_floor=new_floor, opaque_handles=joint_opaque_handles,
    )
    active_after_selection = bool(
        len(before_arrays["old_pairs"])
        or np.any(after_arrays["new_rivals"] >= 0)
    )
    true_z0_reason = None
    if not active_after_selection:
        true_z0_reason = (
            "k1_canonical_z0"
            if k_selected == 1
            else (
                "requested_true_z0"
                if hp.force_zero
                else "no_active_pair_after_self_excluded_selection"
            )
        )
        hp = _zero_hp(hp)
        before_arrays = _empty_pair_arrays(0, 0, 0)
        after_arrays = _empty_pair_arrays(0, len(new_classes), 0)
    before_state = _make_state(
        before_artifact, before_labels, before_ranks, before_selection,
        old_classes, old_prototypes, hp, len(old_classes), 0,
        before_arrays, old_floor, np.empty((0,), dtype=np.bool_),
    )
    after_state = _make_state(
        after_artifact, after_labels, after_ranks, after_selection,
        joint_classes, joint_prototypes, hp, len(old_classes), 1,
        after_arrays, old_floor, new_floor,
    )
    # Support-inclusive deployment-consistency veto.  Per-row pair statistics
    # exclude the scored row, but pair/rival selection uses the whole support;
    # therefore this surface may only revoke and is not a performance proof.
    before_candidate = (
        old_base
        if k_selected == 1
        else _oof_replay_scores(
            selected_old, selected_old_labels, old_classes, before_state
        )
    )
    after_candidate = (
        joint_base
        if k_selected == 1
        else _oof_replay_scores(
            selected_joint, selected_joint_labels, joint_classes, after_state
        )
    )
    before_z0 = old_base
    after_z0 = joint_base
    old_joint_mask = np.isin(selected_joint_labels, old_classes)
    before_metrics = _metrics(selected_old_labels, before_candidate, old_classes)
    before_base_metrics = _metrics(selected_old_labels, before_z0, old_classes)
    after_old_metrics = _metrics(
        selected_joint_labels[old_joint_mask],
        after_candidate[old_joint_mask],
        old_classes,
        joint_classes,
    )
    after_old_base = _metrics(
        selected_joint_labels[old_joint_mask],
        after_z0[old_joint_mask],
        old_classes,
        joint_classes,
    )
    after_new_metrics = _metrics(
        selected_joint_labels[new_mask],
        after_candidate[new_mask],
        new_classes,
        joint_classes,
    )
    after_new_base = _metrics(
        selected_joint_labels[new_mask],
        after_z0[new_mask],
        new_classes,
        joint_classes,
    )
    joint_metrics = _metrics(selected_joint_labels, after_candidate, joint_classes)
    joint_base_metrics = _metrics(selected_joint_labels, after_z0, joint_classes)
    h = _harmonic(after_old_metrics["overall_accuracy"], after_new_metrics["overall_accuracy"])
    h0 = _harmonic(after_old_base["overall_accuracy"], after_new_base["overall_accuracy"])
    nonzero = len(before_state.old_pairs) or np.any(after_state.new_rivals >= 0)
    floor_old_ok = all(
        after_old_metrics["per_class_accuracy"][value] + 1e-12
        >= after_old_base["per_class_accuracy"][value]
        for index, value in enumerate(old_classes) if old_floor[index]
    )
    floor_new_ok = all(
        after_new_metrics["per_class_accuracy"][value] + 1e-12
        >= after_new_base["per_class_accuracy"][value]
        for index, value in enumerate(new_classes) if new_floor[index]
    )
    gate = (
        all(
            before_metrics["per_class_accuracy"][value] + 1e-12
            >= before_base_metrics["per_class_accuracy"][value]
            for value in old_classes
        )
        and all(
            after_old_metrics["per_class_accuracy"][value] + 1e-12
            >= before_metrics["per_class_accuracy"][value]
            and after_old_metrics["per_class_accuracy"][value] + 1e-12
            >= after_old_base["per_class_accuracy"][value]
            for value in old_classes
        )
        and all(
            after_new_metrics["per_class_accuracy"][value] + 1e-12
            >= after_new_base["per_class_accuracy"][value]
            for value in new_classes
        )
        and joint_metrics["overall_accuracy"] + 1e-12
        >= joint_base_metrics["overall_accuracy"]
        and h + 1e-12 >= h0
        and floor_old_ok
        and floor_new_ok
    )
    if nonzero and not gate:
        zero = _zero_hp(hp)
        before_arrays = _empty_pair_arrays(0, 0, 0)
        after_arrays = _empty_pair_arrays(0, len(new_classes), 0)
        before_state = _make_state(
            before_artifact, before_labels, before_ranks, before_selection,
            old_classes, old_prototypes, zero, len(old_classes), 0,
            before_arrays, old_floor, np.empty((0,), dtype=np.bool_),
        )
        after_state = _make_state(
            after_artifact, after_labels, after_ranks, after_selection,
            joint_classes, joint_prototypes, zero, len(old_classes), 1,
            after_arrays, old_floor, new_floor,
        )
        true_z0_reason = "support_inclusive_veto_failed"
    return BeforeAfterSprtdrFit(
        before_state,
        after_state,
        ({
            "phase": "support_inclusive_deployment_consistency_veto",
            "candidate_id": hp.candidate_id,
            "gate_pass": bool(gate),
            "returned_z0": bool(
                before_state.hyperparameters.force_zero
                and after_state.hyperparameters.force_zero
            ),
            "true_z0_reason": true_z0_reason,
            "canonical_true_z0": bool(
                before_state.hyperparameters == _zero_hp(before_state.hyperparameters)
                and after_state.hyperparameters == _zero_hp(after_state.hyperparameters)
            ),
            "old_floor": [old_classes[i] for i in np.flatnonzero(old_floor)],
            "new_floor": [new_classes[i] for i in np.flatnonzero(new_floor)],
            "before_diagnostics": [dict(value) for value in before_diag],
            "after_diagnostics": [dict(value) for value in after_diag],
            "support_only": True,
            "revocation_only_not_performance_certificate": True,
        },),
    )


def fit_before_after_locked(
    before_artifact,
    before_labels,
    before_ranks,
    after_artifact,
    after_labels,
    after_ranks,
    *,
    k_shot: int,
    hyperparameters: SprtdrHyperparameters,
) -> BeforeAfterSprtdrFit:
    _validate_hp(hyperparameters)
    if 2 <= int(k_shot) <= 4:
        raise SprtdrError("K2-K4 unsupported")
    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot
    )
    joint_rows, joint_labels, joint_ranks, found = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot
    )
    if not set(old_classes) < set(found):
        raise SprtdrError("new-class registration set drift")
    _validate_binding(before_artifact, after_artifact)
    _validate_old_reuse(
        before_artifact, old_labels, old_ranks,
        after_artifact, joint_labels, joint_ranks, old_classes,
    )
    joint_classes = old_classes + tuple(sorted(set(found) - set(old_classes)))
    return _fit_selected(
        before_artifact, old_rows, old_labels, old_ranks, old_classes,
        np.ones(len(old_labels), dtype=bool),
        after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes,
        np.ones(len(joint_labels), dtype=bool),
        hyperparameters,
    )


def _l2o_masks(ranks: np.ndarray) -> tuple[np.ndarray, ...]:
    if set(np.unique(ranks).tolist()) != set(range(10)):
        raise SprtdrError("joint L2O requires strict K10")
    return tuple(np.isin(ranks, (first, first + 1)) for first in range(0, 10, 2))


def _decision_sha(state: SprtdrState) -> str:
    digest = hashlib.sha256()
    for name in (
        "prototypes", "old_pairs", "old_dims", "old_mu_a", "old_var_a",
        "old_mu_b", "old_var_b", "old_mid", "old_gap", "old_alpha_pos",
        "old_alpha_neg", "new_rivals", "new_dims", "new_mu", "new_var",
        "rival_mu", "rival_var", "new_mid", "new_gap", "new_alpha_pos",
        "new_alpha_neg", "old_floor", "new_floor",
    ):
        value = getattr(state, name)
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps({
        "classes": state.classes,
        "hp": state.hyperparameters.__dict__,
        "old_class_count": state.old_class_count,
        "registration_generation": state.registration_generation,
        "support_selection_sha256": state.support_selection_sha256,
    }, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def evaluate_joint_leave_two_out(
    before_artifact,
    before_labels,
    before_ranks,
    after_artifact,
    after_labels,
    after_ranks,
    *,
    hyperparameters: SprtdrHyperparameters,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
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
    old_in_joint = np.isin(joint_labels, old_classes)
    folds = []
    trace = []
    for fold, (held_old, held_joint) in enumerate(
        zip(_l2o_masks(old_ranks), _l2o_masks(joint_ranks))
    ):
        fit = _fit_selected(
            before_artifact, old_rows, old_labels, old_ranks, old_classes, ~held_old,
            after_artifact, joint_rows, joint_labels, joint_ranks, joint_classes,
            ~held_joint, hyperparameters,
        )
        base = _fit_selected(
            before_artifact, old_rows, old_labels, old_ranks, old_classes,
            ~held_old,
            after_artifact, joint_rows, joint_labels, joint_ranks,
            joint_classes, ~held_joint, _zero_hp(hyperparameters),
        )
        held_joint_old = held_joint & old_in_joint
        held_joint_new = held_joint & ~old_in_joint
        before_scores = _score_numpy(old_rows[held_old], fit.before_state)
        after_old_scores = _score_numpy(joint_rows[held_joint_old], fit.after_state)
        after_new_scores = _score_numpy(joint_rows[held_joint_new], fit.after_state)
        base_before_scores = _score_numpy(
            old_rows[held_old], base.before_state
        )
        base_after_old_scores = _score_numpy(
            joint_rows[held_joint_old], base.after_state
        )
        base_after_new_scores = _score_numpy(
            joint_rows[held_joint_new], base.after_state
        )
        locked_old_old = _score_numpy(joint_rows[held_joint_old], fit.before_state)
        locked_old_new = _score_numpy(joint_rows[held_joint_new], fit.before_state)
        row = {
            "fold": fold,
            "train_rows_per_class": 8,
            "held_rows_per_class": 2,
            "before_old": _metrics(old_labels[held_old], before_scores, old_classes),
            "base_before_old": _metrics(
                old_labels[held_old], base_before_scores, old_classes
            ),
            "after_old": _metrics(
                joint_labels[held_joint_old], after_old_scores, old_classes,
                joint_classes,
            ),
            "after_new": _metrics(
                joint_labels[held_joint_new], after_new_scores, new_classes,
                joint_classes,
            ),
            "base_after_old": _metrics(
                joint_labels[held_joint_old], base_after_old_scores,
                old_classes, joint_classes,
            ),
            "base_after_new": _metrics(
                joint_labels[held_joint_new], base_after_new_scores,
                new_classes, joint_classes,
            ),
            "old_score_bitwise_locked": bool(
                np.array_equal(
                    after_old_scores[:, :len(old_classes)], locked_old_old
                )
                and np.array_equal(
                    after_new_scores[:, :len(old_classes)], locked_old_new
                )
            ),
            "old_pairs": fit.before_state.old_pairs.tolist(),
            "new_rivals": fit.after_state.new_rivals.tolist(),
            "old_dims": fit.before_state.old_dims.tolist(),
            "new_dims": fit.after_state.new_dims.tolist(),
            "old_stats_sha": hashlib.sha256(
                b"".join(np.ascontiguousarray(value).tobytes() for value in (
                    fit.before_state.old_mu_a, fit.before_state.old_var_a,
                    fit.before_state.old_mu_b, fit.before_state.old_var_b,
                    fit.before_state.old_mid, fit.before_state.old_gap,
                ))
            ).hexdigest(),
            "new_stats_sha": hashlib.sha256(
                b"".join(np.ascontiguousarray(value).tobytes() for value in (
                    fit.after_state.new_mu, fit.after_state.new_var,
                    fit.after_state.rival_mu, fit.after_state.rival_var,
                    fit.after_state.new_mid, fit.after_state.new_gap,
                ))
            ).hexdigest(),
            "amplitude_sha": hashlib.sha256(
                b"".join(np.ascontiguousarray(value).tobytes() for value in (
                    fit.before_state.old_alpha_pos, fit.before_state.old_alpha_neg,
                    fit.after_state.new_alpha_pos, fit.after_state.new_alpha_neg,
                ))
            ).hexdigest(),
            "floor_handles": {
                "old": [old_classes[i] for i in np.flatnonzero(fit.after_state.old_floor)],
                "new": [new_classes[i] for i in np.flatnonzero(fit.after_state.new_floor)],
            },
            "before_decision_sha": _decision_sha(fit.before_state),
            "after_decision_sha": _decision_sha(fit.after_state),
            "before_selection_sha": fit.before_state.support_selection_sha256,
            "after_selection_sha": fit.after_state.support_selection_sha256,
            "before_held_physical_id_sha": hashlib.sha256(
                json.dumps([
                    before_artifact.physical_sample_ids[index]
                    for index in np.flatnonzero(held_old)
                ], separators=(",", ":")).encode()
            ).hexdigest(),
            "after_old_held_physical_id_sha": hashlib.sha256(
                json.dumps([
                    after_artifact.physical_sample_ids[index]
                    for index in np.flatnonzero(held_joint_old)
                ], separators=(",", ":")).encode()
            ).hexdigest(),
        }
        row["H_old_new"] = _harmonic(
            row["after_old"]["overall_accuracy"],
            row["after_new"]["overall_accuracy"],
        )
        row["old_forgetting"] = (
            row["before_old"]["overall_accuracy"]
            - row["after_old"]["overall_accuracy"]
        )
        joint_truth = np.concatenate([
            joint_labels[held_joint_old], joint_labels[held_joint_new]
        ])
        joint_scores = np.concatenate([after_old_scores, after_new_scores])
        base_joint_scores = np.concatenate([
            base_after_old_scores, base_after_new_scores
        ])
        row["joint"] = _metrics(
            joint_truth, joint_scores, joint_classes
        )
        row["base_joint"] = _metrics(
            joint_truth, base_joint_scores, joint_classes
        )
        row["base_H_old_new"] = _harmonic(
            row["base_after_old"]["overall_accuracy"],
            row["base_after_new"]["overall_accuracy"],
        )
        row["candidate_vs_z0_per_class_non_degraded"] = {
            "before_old": {
                value: (
                    row["before_old"]["per_class_accuracy"][value] + 1e-12
                    >= row["base_before_old"]["per_class_accuracy"][value]
                )
                for value in old_classes
            },
            "after_old": {
                value: (
                    row["after_old"]["per_class_accuracy"][value] + 1e-12
                    >= row["base_after_old"]["per_class_accuracy"][value]
                    and row["after_old"]["per_class_accuracy"][value] + 1e-12
                    >= row["before_old"]["per_class_accuracy"][value]
                )
                for value in old_classes
            },
            "after_new": {
                value: (
                    row["after_new"]["per_class_accuracy"][value] + 1e-12
                    >= row["base_after_new"]["per_class_accuracy"][value]
                )
                for value in new_classes
            },
        }
        folds.append(row)
        trace.extend({"fold": fold, **item} for item in fit.trace)
        trace.append({"phase": "sprtdr_outer_l2o_fold", **row})
    def aggregate(key, classes):
        per_class = {
            value: float(np.mean([
                row[key]["per_class_accuracy"][value] for row in folds
            ]))
            for value in classes
        }
        return {
            "overall_accuracy": float(np.mean([
                row[key]["overall_accuracy"] for row in folds
            ])),
            "min_class_accuracy": min(per_class.values()),
            "per_class_accuracy": per_class,
        }
    result = {
        "before_old": aggregate("before_old", old_classes),
        "base_before_old": aggregate("base_before_old", old_classes),
        "after_old": aggregate("after_old", old_classes),
        "base_after_old": aggregate("base_after_old", old_classes),
        "after_new": aggregate("after_new", new_classes),
        "base_after_new": aggregate("base_after_new", new_classes),
        "joint": aggregate("joint", joint_classes),
        "base_joint": aggregate("base_joint", joint_classes),
        "folds": folds,
        "old_score_bitwise_locked": all(
            row["old_score_bitwise_locked"] for row in folds
        ),
        "max_old_edge_count": max(len(row["old_pairs"]) for row in folds),
        "max_new_rivals_per_class": max(
            max((sum(value >= 0 for value in rivals) for rivals in row["new_rivals"]), default=0)
            for row in folds
        ),
    }
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
    result["candidate_vs_z0_per_class_non_degraded"] = {
        "before_old": {
            value: (
                result["before_old"]["per_class_accuracy"][value] + 1e-12
                >= result["base_before_old"]["per_class_accuracy"][value]
            )
            for value in old_classes
        },
        "after_old": {
            value: (
                result["after_old"]["per_class_accuracy"][value] + 1e-12
                >= result["base_after_old"]["per_class_accuracy"][value]
                and result["after_old"]["per_class_accuracy"][value] + 1e-12
                >= result["before_old"]["per_class_accuracy"][value]
            )
            for value in old_classes
        },
        "after_new": {
            value: (
                result["after_new"]["per_class_accuracy"][value] + 1e-12
                >= result["base_after_new"]["per_class_accuracy"][value]
            )
            for value in new_classes
        },
    }
    return result, tuple(trace)


def predict_all_registered(
    state: SprtdrState,
    artifact: RuntimeAuthorizedFeatureArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_state(state)
    rows = _rows(artifact)
    if len(rows) != 1:
        raise SprtdrError("exactly one physical query required")
    if (
        artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or artifact.feature_code_sha256 != state.feature_code_sha256
        or artifact.sealed_phase1_checkpoint_sha256
        != state.sealed_phase1_checkpoint_sha256
        or artifact.operator_id != state.operator_id
        or artifact.view_seed != state.view_seed
    ):
        raise SprtdrError("query runtime binding drift")
    scores = _score_numpy(rows, state)
    return np.asarray(state.classes)[np.argmax(scores, axis=1)], scores
