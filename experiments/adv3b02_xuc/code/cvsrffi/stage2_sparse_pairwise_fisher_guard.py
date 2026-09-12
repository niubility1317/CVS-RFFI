"""D14 support-only sparse pairwise Fisher registration guard.

The module consumes only runtime-authorized features extracted one physical
sample at a time from fixed received LEO_weak IQ.  Before registration it may
add at most three endpoint-disjoint old--old Fisher edges.  After registration
it freezes the complete Before old-score function bitwise and adds at most one
old rival edge for each new class; only the corresponding new score can move.
"""

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
SCHEMA = "cvs.phase2.sparse_pairwise_fisher_guard.v1"
MAX_STATE_BYTES = 256 * 1024
RIVAL_OLD_QUANTILE = 0.90
RIVAL_NEW_QUANTILE = 0.10


class SparsePairwiseFisherGuardError(ValueError):
    """Raised when the D14 support-only contract fails closed."""


@dataclass(frozen=True)
class SparsePairwiseFisherHyperparameters:
    candidate_id: str
    operator_id: str = "base"
    ridge: float = 0.05
    gamma_old: float = 0.05
    gamma_new: float = 0.05
    select_band_old: float = 0.20
    band_old: float = 0.20
    band_new: float = 0.20
    max_old_edges: int = 3
    force_zero: bool = False


@dataclass(frozen=True)
class SparsePairwiseFisherState:
    schema: str
    candidate_id: str
    classes: tuple[str, ...]
    prototypes: np.ndarray
    old_edge_pairs: np.ndarray
    old_edge_directions: np.ndarray
    old_edge_bias: np.ndarray
    new_rivals: np.ndarray
    new_edge_directions: np.ndarray
    new_edge_bias: np.ndarray
    hyperparameters: SparsePairwiseFisherHyperparameters
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
        float_names = (
            "prototypes",
            "old_edge_directions",
            "old_edge_bias",
            "new_edge_directions",
            "new_edge_bias",
        )
        int_names = ("old_edge_pairs", "new_rivals")
        for name in float_names:
            source = np.ascontiguousarray(getattr(self, name), dtype=np.float32)
            immutable = np.frombuffer(source.tobytes(), dtype=np.float32).reshape(
                source.shape
            )
            object.__setattr__(self, name, immutable)
        for name in int_names:
            source = np.ascontiguousarray(getattr(self, name), dtype=np.int64)
            immutable = np.frombuffer(source.tobytes(), dtype=np.int64).reshape(
                source.shape
            )
            object.__setattr__(self, name, immutable)
        computed = _state_content_sha256(self)
        if self.state_content_sha256 and self.state_content_sha256 != computed:
            raise SparsePairwiseFisherGuardError("state content SHA mismatch")
        object.__setattr__(self, "state_content_sha256", computed)
        _validate_state(self)


@dataclass(frozen=True)
class BeforeAfterSparsePairwiseFit:
    before_state: SparsePairwiseFisherState
    after_state: SparsePairwiseFisherState
    trace: tuple[dict[str, Any], ...]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sparse_pairwise_fisher_state(
    npz_path: Path,
    metadata_path: Path,
    *,
    expected_npz_sha256: str,
    expected_metadata_sha256: str,
) -> SparsePairwiseFisherState:
    """Rebuild and validate a sealed D14 state from externally pinned files."""

    if (
        len(expected_npz_sha256) != 64
        or len(expected_metadata_sha256) != 64
        or _sha256_path(npz_path) != expected_npz_sha256
        or _sha256_path(metadata_path) != expected_metadata_sha256
    ):
        raise SparsePairwiseFisherGuardError("sealed state external hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hp = metadata["hyperparameters"]
    with np.load(npz_path, allow_pickle=False) as loaded:
        required = {
            "prototypes",
            "old_edge_pairs",
            "old_edge_directions",
            "old_edge_bias",
            "new_rivals",
            "new_edge_directions",
            "new_edge_bias",
        }
        if set(loaded.files) != required:
            raise SparsePairwiseFisherGuardError("sealed state member drift")
        arrays = {name: np.array(loaded[name], copy=True) for name in required}
    return SparsePairwiseFisherState(
        schema=str(metadata["schema"]),
        candidate_id=str(metadata["candidate_id"]),
        classes=tuple(str(value) for value in metadata["classes"]),
        prototypes=arrays["prototypes"],
        old_edge_pairs=arrays["old_edge_pairs"],
        old_edge_directions=arrays["old_edge_directions"],
        old_edge_bias=arrays["old_edge_bias"],
        new_rivals=arrays["new_rivals"],
        new_edge_directions=arrays["new_edge_directions"],
        new_edge_bias=arrays["new_edge_bias"],
        hyperparameters=SparsePairwiseFisherHyperparameters(
            candidate_id=str(metadata["candidate_id"]),
            operator_id=str(hp["operator_id"]),
            ridge=float(hp["ridge"]),
            gamma_old=float(hp["gamma_old"]),
            gamma_new=float(hp["gamma_new"]),
            select_band_old=float(hp["select_band_old"]),
            band_old=float(hp["band_old"]),
            band_new=float(hp["band_new"]),
            max_old_edges=int(hp["max_old_edges"]),
            force_zero=bool(hp["force_zero"]),
        ),
        feature_dim=int(metadata["feature_dim"]),
        k_shot=int(metadata["k_shot"]),
        old_class_count=int(metadata["old_class_count"]),
        registration_generation=int(metadata["registration_generation"]),
        resource=dict(metadata["resource"]),
        support_feature_artifact_sha256=str(
            metadata["support_feature_artifact_sha256"]
        ),
        support_selection_sha256=str(metadata["support_selection_sha256"]),
        sealed_runtime_sha256=str(metadata["sealed_runtime_sha256"]),
        feature_code_sha256=str(metadata["feature_code_sha256"]),
        sealed_phase1_checkpoint_sha256=str(
            metadata["sealed_phase1_checkpoint_sha256"]
        ),
        operator_id=str(metadata["operator_id"]),
        view_seed=int(metadata["view_seed"]),
        state_content_sha256=str(metadata["state_content_sha256"]),
    )


def _validate_hyperparameters(
    value: SparsePairwiseFisherHyperparameters,
) -> None:
    finite_nonnegative = (
        value.ridge,
        value.gamma_old,
        value.gamma_new,
        value.select_band_old,
        value.band_old,
        value.band_new,
    )
    if (
        not value.candidate_id
        or value.operator_id != "base"
        or not all(np.isfinite(item) and item >= 0.0 for item in finite_nonnegative)
        or value.ridge <= 0.0
        or not 0 <= int(value.max_old_edges) <= 3
        or (
            value.force_zero
            and (
                value.gamma_old != 0.0
                or value.gamma_new != 0.0
                or int(value.max_old_edges) != 0
            )
        )
    ):
        raise SparsePairwiseFisherGuardError("hyperparameter drift")


def _artifact_rows(value: RuntimeAuthorizedFeatureArtifact) -> np.ndarray:
    if not isinstance(value, RuntimeAuthorizedFeatureArtifact):
        raise SparsePairwiseFisherGuardError(
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
    rows = _artifact_rows(artifact)
    label_rows = np.asarray(labels).astype(str)
    rank_rows = np.asarray(ranks, dtype=np.int64)
    if len(rows) != len(label_rows) or len(rows) != len(rank_rows):
        raise SparsePairwiseFisherGuardError("support alignment drift")
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
        raise SparsePairwiseFisherGuardError("strict physical K-shot support drift")
    return rows, label_rows, rank_rows, tuple(sorted(classes.tolist()))


def _validate_artifact_binding(
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
        raise SparsePairwiseFisherGuardError(
            "before/after runtime/checkpoint/operator binding drift"
        )


def _validate_old_lineage_exact_reuse(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: np.ndarray,
    before_ranks: np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: np.ndarray,
    after_ranks: np.ndarray,
    old_classes: Sequence[str],
) -> None:
    def keyed(
        artifact: RuntimeAuthorizedFeatureArtifact,
        labels: np.ndarray,
        ranks: np.ndarray,
        allowed: set[str],
    ) -> dict[tuple[str, int], tuple[str, str, str]]:
        return {
            (str(labels[index]), int(ranks[index])): (
                artifact.physical_sample_ids[index],
                artifact.parent_received_iq_sha256[index],
                artifact.per_row_feature_sha256[index],
            )
            for index in range(len(labels))
            if str(labels[index]) in allowed
        }

    allowed = set(old_classes)
    if keyed(before_artifact, before_labels, before_ranks, allowed) != keyed(
        after_artifact, after_labels, after_ranks, allowed
    ):
        raise SparsePairwiseFisherGuardError(
            "after old support lineage exact-reuse lock failed"
        )


def _normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), EPS)


def _class_statistics(
    rows: np.ndarray, labels: np.ndarray, classes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalize(rows)
    means = np.stack(
        [np.mean(normalized[labels == label], axis=0) for label in classes]
    ).astype(np.float32)
    prototypes = _normalize(means).astype(np.float32)
    variances = np.stack(
        [
            np.var(normalized[labels == label], axis=0, ddof=0)
            for label in classes
        ]
    ).astype(np.float32)
    return prototypes, variances


def _support_selection_sha256(
    artifact: RuntimeAuthorizedFeatureArtifact,
    labels: np.ndarray,
    ranks: np.ndarray,
    selection: np.ndarray | None = None,
) -> str:
    selected = (
        np.ones(len(labels), dtype=bool)
        if selection is None
        else np.asarray(selection, dtype=bool)
    )
    if len(selected) != len(labels) or len(labels) != len(artifact.features):
        raise SparsePairwiseFisherGuardError("support selection alignment drift")
    records = [
        {
            "label": str(labels[index]),
            "rank": int(ranks[index]),
            "physical_sample_id": artifact.physical_sample_ids[index],
            "parent_received_iq_sha256": artifact.parent_received_iq_sha256[index],
            "feature_sha256": artifact.per_row_feature_sha256[index],
        }
        for index in np.flatnonzero(selected)
    ]
    return hashlib.sha256(
        json.dumps(
            records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _fisher_edge(
    first: int,
    second: int,
    prototypes: np.ndarray,
    variances: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    pooled = variances[first] + variances[second] + float(ridge)
    w = (prototypes[first] - prototypes[second]) / pooled
    midpoint = (prototypes[first] + prototypes[second]) / 2.0
    denominator = float(np.sqrt(np.sum(np.square(w) * pooled)))
    if (
        not np.isfinite(w).all()
        or not np.isfinite(midpoint).all()
        or not np.isfinite(denominator)
        or denominator <= EPS
    ):
        return None
    direction = np.asarray(w / denominator, dtype=np.float32)
    bias = np.asarray(-np.dot(direction, midpoint), dtype=np.float32)
    if not np.isfinite(direction).all() or not np.isfinite(bias).all():
        return None
    return direction, bias


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
    candidates: list[tuple[float, tuple[tuple[str, str], ...], tuple[tuple[int, int], ...]]] = []

    def visit(
        remaining: tuple[int, ...],
        selected: tuple[tuple[int, int], ...],
        total: float,
    ) -> None:
        canonical = tuple(
            sorted(
                (
                    tuple(sorted((classes[first], classes[second])))
                    for first, second in selected
                )
            )
        )
        candidates.append((float(total), canonical, tuple(sorted(selected))))
        if not remaining or len(selected) >= max_edges:
            return
        first = remaining[0]
        visit(remaining[1:], selected, total)
        for position, second in enumerate(remaining[1:], start=1):
            pair = tuple(sorted((first, second)))
            if pair not in positive:
                continue
            next_remaining = remaining[1:position] + remaining[position + 1 :]
            visit(
                next_remaining,
                selected + (pair,),
                total + positive[pair],
            )

    visit(tuple(range(len(classes))), (), 0.0)
    best_weight = max(value[0] for value in candidates)
    tied = [value for value in candidates if abs(value[0] - best_weight) <= 1.0e-12]
    tied.sort(key=lambda value: (-len(value[2]), value[1]))
    return tied[0][2]


def _select_old_edges(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    variances: np.ndarray,
    hyperparameters: SparsePairwiseFisherHyperparameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    if hyperparameters.force_zero or hyperparameters.max_old_edges == 0:
        dim = int(rows.shape[1])
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            (),
        )
    _, counts = np.unique(labels, return_counts=True)
    if not len(counts) or int(np.min(counts)) < 2:
        dim = int(rows.shape[1])
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            (
                {
                    "kind": "before_old_old_edges_closed",
                    "reason": "internal_loo_requires_at_least_two_samples_per_class",
                },
            ),
        )
    normalized = _normalize(rows)
    lookup = {label: index for index, label in enumerate(classes)}
    evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for index, held in enumerate(normalized):
        keep = np.ones(len(rows), dtype=bool)
        keep[index] = False
        loo_prototypes, _ = _class_statistics(rows[keep], labels[keep], classes)
        score = held @ loo_prototypes.T
        top = np.argsort(score, kind="stable")[-2:][::-1]
        truth = lookup[str(labels[index])]
        if int(top[0]) != truth:
            pair = tuple(sorted((truth, int(top[0]))))
            row = evidence.setdefault(
                pair, {"weight": 0.0, "error_count": 0, "near_count": 0}
            )
            row["weight"] += 1.0
            row["error_count"] += 1
        elif hyperparameters.select_band_old > 0.0:
            runner_up = int(top[1])
            margin = float(score[truth] - score[runner_up])
            if margin <= hyperparameters.select_band_old:
                pair = tuple(sorted((truth, runner_up)))
                row = evidence.setdefault(
                    pair, {"weight": 0.0, "error_count": 0, "near_count": 0}
                )
                row["weight"] += max(
                    0.0, 1.0 - margin / hyperparameters.select_band_old
                )
                row["near_count"] += 1
    valid_edges: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    valid_weights: dict[tuple[int, int], float] = {}
    for pair, value in evidence.items():
        edge = _fisher_edge(
            pair[0], pair[1], prototypes, variances, hyperparameters.ridge
        )
        if edge is not None and float(value["weight"]) > 0.0:
            valid_edges[pair] = edge
            valid_weights[pair] = float(value["weight"])
    chosen = list(
        _maximum_weight_matching(
            valid_weights,
            classes,
            max_edges=int(hyperparameters.max_old_edges),
        )
    )
    diagnostics: list[dict[str, Any]] = [
        {
            "kind": "before_old_old_edge",
            "first_class": classes[pair[0]],
            "second_class": classes[pair[1]],
            "internal_loo_collision_weight": valid_weights[pair],
            "internal_loo_error_count": int(evidence[pair]["error_count"]),
            "internal_loo_near_count": int(evidence[pair]["near_count"]),
            "matching_policy": "deterministic_max_weight_endpoint_disjoint",
            "endpoint_disjoint": True,
        }
        for pair in chosen
    ]
    pairs = np.asarray(chosen, dtype=np.int64).reshape(-1, 2)
    values = [valid_edges[(a, b)] for a, b in chosen]
    dim = int(rows.shape[1])
    edge_w = (
        np.stack([value[0] for value in values])
        if values
        else np.empty((0, dim), dtype=np.float32)
    )
    biases = (
        np.stack([value[1] for value in values])
        if values
        else np.empty((0,), dtype=np.float32)
    )
    return pairs, edge_w, biases, tuple(diagnostics)


def _old_scores_from_train_statistics(
    normalized_rows: np.ndarray,
    prototypes: np.ndarray,
    variances: np.ndarray,
    old_edge_pairs: np.ndarray,
    hyperparameters: SparsePairwiseFisherHyperparameters,
) -> np.ndarray:
    base = normalized_rows @ prototypes.T
    result = np.array(base, dtype=np.float32, copy=True)
    if hyperparameters.gamma_old == 0.0 or not len(old_edge_pairs):
        return result
    top2 = np.argsort(base, axis=1, kind="stable")[:, -2:]
    for first, second in old_edge_pairs:
        first = int(first)
        second = int(second)
        edge = _fisher_edge(
            first, second, prototypes, variances, hyperparameters.ridge
        )
        if edge is None:
            continue
        direction, bias = edge
        pair_hit = np.all(
            np.sort(top2, axis=1) == np.asarray([first, second]), axis=1
        )
        active = pair_hit & (
            np.abs(base[:, first] - base[:, second])
            <= hyperparameters.band_old
        )
        if not np.any(active):
            continue
        correction = np.float32(hyperparameters.gamma_old) * np.clip(
            normalized_rows[active] @ direction + bias, -1.0, 1.0
        ).astype(np.float32)
        result[active, first] += correction
        result[active, second] -= correction
    return result


def _select_new_rivals(
    rows: np.ndarray,
    labels: np.ndarray,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    variances: np.ndarray,
    old_edge_pairs: np.ndarray,
    *,
    old_class_count: int,
    hyperparameters: SparsePairwiseFisherHyperparameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
    new_count = len(classes) - old_class_count
    rivals = np.full(new_count, -1, dtype=np.int64)
    edge_w = np.zeros((new_count, prototypes.shape[1]), dtype=np.float32)
    biases = np.zeros(new_count, dtype=np.float32)
    if hyperparameters.force_zero:
        return rivals, edge_w, biases, ()
    _, counts = np.unique(labels, return_counts=True)
    if not len(counts) or int(np.min(counts)) < 2:
        return (
            rivals,
            edge_w,
            biases,
            (
                {
                    "kind": "after_new_old_rival_edges_closed",
                    "reason": "internal_loo_requires_at_least_two_samples_per_class",
                },
            ),
        )
    normalized = _normalize(rows)
    old_classes = classes[:old_class_count]
    old_mask = np.isin(labels, old_classes)
    full_old_prototypes = prototypes[:old_class_count]
    full_old_variances = variances[:old_class_count]
    before_correct_old_margins: dict[tuple[int, int], list[float]] = {
        (new_index, rival): []
        for new_index in range(old_class_count, len(classes))
        for rival in range(old_class_count)
    }
    new_loo_margins: dict[tuple[int, int], list[float]] = {
        (new_index, rival): []
        for new_index in range(old_class_count, len(classes))
        for rival in range(old_class_count)
    }
    for index, held in enumerate(normalized):
        truth = classes.index(str(labels[index]))
        if truth < old_class_count:
            keep_old = old_mask.copy()
            keep_old[index] = False
            loo_old_prototypes, loo_old_variances = _class_statistics(
                rows[keep_old],
                labels[keep_old],
                old_classes,
            )
            before_scores = _old_scores_from_train_statistics(
                held[None, :],
                loo_old_prototypes,
                loo_old_variances,
                old_edge_pairs,
                hyperparameters,
            )[0]
            if int(np.argmax(before_scores)) != truth:
                continue
            for new_index in range(old_class_count, len(classes)):
                new_score = float(held @ prototypes[new_index])
                before_correct_old_margins[(new_index, truth)].append(
                    new_score - float(before_scores[truth])
                )
        else:
            keep = np.ones(len(rows), dtype=bool)
            keep[index] = False
            loo_prototypes, _ = _class_statistics(
                rows[keep], labels[keep], classes
            )
            new_score = float(held @ loo_prototypes[truth])
            before_scores = _old_scores_from_train_statistics(
                held[None, :],
                full_old_prototypes,
                full_old_variances,
                old_edge_pairs,
                hyperparameters,
            )[0]
            for rival in range(old_class_count):
                new_loo_margins[(truth, rival)].append(
                    new_score - float(before_scores[rival])
                )
    diagnostics = []
    for new_index in range(old_class_count, len(classes)):
        risks = np.full(old_class_count, -np.inf, dtype=np.float64)
        risk_records: list[dict[str, Any]] = []
        for rival in range(old_class_count):
            old_margins = before_correct_old_margins[(new_index, rival)]
            new_margins = new_loo_margins[(new_index, rival)]
            edge = _fisher_edge(
                new_index, rival, prototypes, variances, hyperparameters.ridge
            )
            valid = bool(
                old_margins
                and new_margins
                and np.isfinite(old_margins).all()
                and np.isfinite(new_margins).all()
                and edge is not None
            )
            risk = (
                float(
                    np.quantile(
                        np.asarray(old_margins),
                        RIVAL_OLD_QUANTILE,
                        method="linear",
                    )
                    - np.quantile(
                        np.asarray(new_margins),
                        RIVAL_NEW_QUANTILE,
                        method="linear",
                    )
                )
                if valid
                else -np.inf
            )
            risks[rival] = risk
            risk_records.append(
                {
                    "old_rival": classes[rival],
                    "valid": valid,
                    "risk": None if not np.isfinite(risk) else risk,
                    "before_correct_old_margin_count": len(old_margins),
                    "new_internal_loo_margin_count": len(new_margins),
                    "before_correct_policy": (
                        "train_only_d14_before_with_locked_old_edges"
                    ),
                }
            )
        if not np.any(np.isfinite(risks)):
            diagnostics.append(
                {
                    "kind": "after_new_old_rival_edge_closed",
                    "new_class": classes[new_index],
                    "reason": "no_valid_before_correct_old_and_new_loo_margin",
                    "risk_records": risk_records,
                }
            )
            continue
        rival = int(np.argmax(risks))
        edge = _fisher_edge(
            new_index, rival, prototypes, variances, hyperparameters.ridge
        )
        if edge is None:
            continue
        offset = new_index - old_class_count
        rivals[offset] = rival
        edge_w[offset], biases[offset] = edge
        diagnostics.append(
            {
                "kind": "after_new_old_rival_edge",
                "new_class": classes[new_index],
                "old_rival": classes[rival],
                "risk_formula": (
                    "Q90(new_minus_rival_on_before_correct_old)"
                    "-Q10(new_minus_rival_on_new)"
                ),
                "risk": float(risks[rival]),
                "quantile_method": "linear",
                "one_rival_only": True,
                "risk_records": risk_records,
            }
        )
    return rivals, edge_w, biases, tuple(diagnostics)


def _evidence(
    direction: np.ndarray, bias: float, normalized_rows: np.ndarray
) -> np.ndarray:
    return normalized_rows @ direction + np.float32(bias)


def _before_scores(
    rows: np.ndarray, state: SparsePairwiseFisherState
) -> np.ndarray:
    normalized = _normalize(rows)
    base = normalized @ state.prototypes[: state.old_class_count].T
    result = np.array(base, dtype=np.float32, copy=True)
    if state.hyperparameters.gamma_old == 0.0 or not len(state.old_edge_pairs):
        return result
    top2 = np.argsort(base, axis=1, kind="stable")[:, -2:]
    for edge_index, pair in enumerate(state.old_edge_pairs):
        first, second = int(pair[0]), int(pair[1])
        pair_hit = np.all(
            np.sort(top2, axis=1) == np.asarray([first, second]), axis=1
        )
        margin = np.abs(base[:, first] - base[:, second])
        active = pair_hit & (margin <= state.hyperparameters.band_old)
        if not np.any(active):
            continue
        evidence = _evidence(
            state.old_edge_directions[edge_index],
            float(state.old_edge_bias[edge_index]),
            normalized[active],
        )
        correction = np.float32(state.hyperparameters.gamma_old) * np.clip(
            evidence, -1.0, 1.0
        ).astype(np.float32)
        result[active, first] += correction
        result[active, second] -= correction
    return result


def _score_numpy(
    rows: np.ndarray, state: SparsePairwiseFisherState
) -> np.ndarray:
    _validate_state(state)
    normalized = _normalize(rows)
    before = _before_scores(rows, state)
    if state.registration_generation == 0:
        return before
    new_base = normalized @ state.prototypes[state.old_class_count :].T
    result = np.concatenate([before, new_base.astype(np.float32)], axis=1)
    frozen = np.array(result[:, : state.old_class_count], copy=True)
    if state.hyperparameters.gamma_new > 0.0:
        for class_index in range(state.old_class_count, len(state.classes)):
            offset = class_index - state.old_class_count
            rival = int(state.new_rivals[offset])
            if rival < 0:
                continue
            rival_is_before_argmax = np.argmax(before, axis=1) == rival
            margin = np.abs(
                new_base[:, class_index - state.old_class_count]
                - before[:, rival]
            )
            active = rival_is_before_argmax & (
                margin <= state.hyperparameters.band_new
            )
            if not np.any(active):
                continue
            evidence = _evidence(
                state.new_edge_directions[offset],
                float(state.new_edge_bias[offset]),
                normalized[active],
            )
            result[active, class_index] += np.float32(
                state.hyperparameters.gamma_new
            ) * np.clip(evidence, -1.0, 1.0).astype(np.float32)
    if not np.array_equal(result[:, : state.old_class_count], frozen):
        raise SparsePairwiseFisherGuardError("After old score bitwise freeze violated")
    return result


def _array_bytes(state: SparsePairwiseFisherState) -> int:
    return int(
        state.prototypes.nbytes
        + state.old_edge_pairs.nbytes
        + state.old_edge_directions.nbytes
        + state.old_edge_bias.nbytes
        + state.new_rivals.nbytes
        + state.new_edge_directions.nbytes
        + state.new_edge_bias.nbytes
    )


def _selection_tensor_sha256(state: SparsePairwiseFisherState) -> str:
    digest = hashlib.sha256()
    for value in (
        state.prototypes,
        state.old_edge_pairs,
        state.old_edge_directions,
        state.old_edge_bias,
        state.new_rivals,
        state.new_edge_directions,
        state.new_edge_bias,
    ):
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _state_content_sha256(state: SparsePairwiseFisherState) -> str:
    digest = hashlib.sha256()
    digest.update(state.schema.encode("utf-8"))
    digest.update(state.candidate_id.encode("utf-8"))
    digest.update(json.dumps(state.classes, separators=(",", ":")).encode("utf-8"))
    for value in (
        state.prototypes,
        state.old_edge_pairs,
        state.old_edge_directions,
        state.old_edge_bias,
        state.new_rivals,
        state.new_edge_directions,
        state.new_edge_bias,
    ):
        digest.update(str(value.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(
        json.dumps(
            {
                "feature_dim": state.feature_dim,
                "k_shot": state.k_shot,
                "old_class_count": state.old_class_count,
                "registration_generation": state.registration_generation,
                "support_feature_artifact_sha256": (
                    state.support_feature_artifact_sha256
                ),
                "support_selection_sha256": state.support_selection_sha256,
                "sealed_runtime_sha256": state.sealed_runtime_sha256,
                "feature_code_sha256": state.feature_code_sha256,
                "sealed_phase1_checkpoint_sha256": (
                    state.sealed_phase1_checkpoint_sha256
                ),
                "operator_id": state.operator_id,
                "view_seed": state.view_seed,
                "hyperparameters": {
                    "candidate_id": state.hyperparameters.candidate_id,
                    "operator_id": state.hyperparameters.operator_id,
                    "ridge": state.hyperparameters.ridge,
                    "gamma_old": state.hyperparameters.gamma_old,
                    "gamma_new": state.hyperparameters.gamma_new,
                    "select_band_old": state.hyperparameters.select_band_old,
                    "band_old": state.hyperparameters.band_old,
                    "band_new": state.hyperparameters.band_new,
                    "max_old_edges": state.hyperparameters.max_old_edges,
                    "force_zero": state.hyperparameters.force_zero,
                },
                "resource": dict(state.resource),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _validate_state(state: SparsePairwiseFisherState) -> None:
    hp = state.hyperparameters
    _validate_hyperparameters(hp)
    endpoints = state.old_edge_pairs.reshape(-1).tolist()
    total_bytes = _array_bytes(state)
    new_count = len(state.classes) - state.old_class_count
    float_arrays = (
        state.prototypes,
        state.old_edge_directions,
        state.old_edge_bias,
        state.new_edge_directions,
        state.new_edge_bias,
    )
    active_new = np.flatnonzero(state.new_rivals >= 0)
    disabled_new = np.flatnonzero(state.new_rivals < 0)
    pair_tuples = [tuple(map(int, value)) for value in state.old_edge_pairs]
    if (
        state.schema != SCHEMA
        or hp.operator_id != state.operator_id
        or state.operator_id != "base"
        or state.feature_dim < 1
        or not 1 <= state.old_class_count <= len(state.classes)
        or state.registration_generation not in (0, 1)
        or (
            state.registration_generation == 0
            and state.old_class_count != len(state.classes)
        )
        or (
            state.registration_generation == 1
            and state.old_class_count >= len(state.classes)
        )
        or state.k_shot < 1
        or state.prototypes.shape != (len(state.classes), state.feature_dim)
        or state.old_edge_pairs.ndim != 2
        or state.old_edge_pairs.shape[1:] != (2,)
        or len(state.old_edge_pairs) > 3
        or len(endpoints) != len(set(endpoints))
        or any(
            first < 0
            or second < 0
            or first >= state.old_class_count
            or second >= state.old_class_count
            or first == second
            or first > second
            for first, second in state.old_edge_pairs
        )
        or len(pair_tuples) != len(set(pair_tuples))
        or state.old_edge_directions.shape
        != (len(state.old_edge_pairs), state.feature_dim)
        or state.old_edge_bias.shape != (len(state.old_edge_pairs),)
        or state.new_rivals.shape != (new_count,)
        or state.new_edge_directions.shape != (new_count, state.feature_dim)
        or state.new_edge_bias.shape != (new_count,)
        or np.any(state.new_rivals >= state.old_class_count)
        or np.any(state.new_rivals < -1)
        or not all(np.isfinite(value).all() for value in float_arrays)
        or any(
            float(np.linalg.norm(value)) <= EPS
            for value in state.old_edge_directions
        )
        or any(
            float(np.linalg.norm(state.new_edge_directions[index])) <= EPS
            for index in active_new
        )
        or np.any(state.new_edge_directions[disabled_new] != 0.0)
        or np.any(state.new_edge_bias[disabled_new] != 0.0)
        or total_bytes > MAX_STATE_BYTES
        or int(state.resource.get("trainable_parameters", -1)) != 0
        or int(state.resource.get("adapt_epochs", -1)) != 0
        or int(state.resource.get("persistent_array_state_bytes", -1))
        != total_bytes
        or int(state.resource.get("old_edge_count", -1))
        != len(state.old_edge_pairs)
        or int(state.resource.get("new_edge_count", -1))
        != int(np.sum(state.new_rivals >= 0))
        or int(state.resource.get("new_edge_count", -1)) > new_count
        or bool(state.resource.get("dense_query_graph", True))
        or state.state_content_sha256 != _state_content_sha256(state)
        or any(
            len(value) != 64
            for value in (
                state.support_feature_artifact_sha256,
                state.support_selection_sha256,
                state.sealed_runtime_sha256,
                state.feature_code_sha256,
                state.sealed_phase1_checkpoint_sha256,
            )
        )
    ):
        raise SparsePairwiseFisherGuardError("state content/resource/binding drift")
    if hp.force_zero and (
        len(state.old_edge_pairs)
        or np.any(state.new_rivals >= 0)
        or hp.gamma_old != 0.0
        or hp.gamma_new != 0.0
    ):
        raise SparsePairwiseFisherGuardError("true zero fallback drift")


def _make_state(
    artifact: RuntimeAuthorizedFeatureArtifact,
    *,
    classes: tuple[str, ...],
    prototypes: np.ndarray,
    old_edge_pairs: np.ndarray,
    old_edge_directions: np.ndarray,
    old_edge_bias: np.ndarray,
    new_rivals: np.ndarray,
    new_edge_directions: np.ndarray,
    new_edge_bias: np.ndarray,
    hyperparameters: SparsePairwiseFisherHyperparameters,
    k_shot: int,
    old_class_count: int,
    registration_generation: int,
    support_selection_sha256: str,
) -> SparsePairwiseFisherState:
    provisional = {
        "trainable_parameters": 0,
        "adapt_epochs": 0,
        "closed_form_solve_count": 1,
        "old_edge_count": int(len(old_edge_pairs)),
        "new_edge_count": int(np.sum(new_rivals >= 0)),
        "prototype_cosine_mac_per_sample": int(len(classes) * prototypes.shape[1]),
        "fisher_edge_mac_per_sample_upper_bound": int(
            (len(old_edge_pairs) + np.sum(new_rivals >= 0))
            * (prototypes.shape[1] + 3)
        ),
        "backbone_forwards_per_physical_sample": 1,
        "fft_branches_per_physical_sample": 0,
        "dense_query_graph": False,
    }
    array_bytes = int(
        prototypes.nbytes
        + old_edge_pairs.nbytes
        + old_edge_directions.nbytes
        + old_edge_bias.nbytes
        + new_rivals.nbytes
        + new_edge_directions.nbytes
        + new_edge_bias.nbytes
    )
    provisional["persistent_array_state_bytes"] = array_bytes
    return SparsePairwiseFisherState(
        schema=SCHEMA,
        candidate_id=hyperparameters.candidate_id,
        classes=classes,
        prototypes=prototypes,
        old_edge_pairs=old_edge_pairs,
        old_edge_directions=old_edge_directions,
        old_edge_bias=old_edge_bias,
        new_rivals=new_rivals,
        new_edge_directions=new_edge_directions,
        new_edge_bias=new_edge_bias,
        hyperparameters=hyperparameters,
        feature_dim=int(prototypes.shape[1]),
        k_shot=int(k_shot),
        old_class_count=int(old_class_count),
        registration_generation=int(registration_generation),
        resource=provisional,
        support_feature_artifact_sha256=artifact.artifact_sha256,
        support_selection_sha256=support_selection_sha256,
        sealed_runtime_sha256=artifact.sealed_runtime_sha256,
        feature_code_sha256=artifact.feature_code_sha256,
        sealed_phase1_checkpoint_sha256=artifact.sealed_phase1_checkpoint_sha256,
        operator_id=artifact.operator_id,
        view_seed=artifact.view_seed,
    )


def _fit_selected(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    old_rows: np.ndarray,
    old_labels: np.ndarray,
    old_ranks: np.ndarray,
    old_classes: tuple[str, ...],
    old_selection: np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    joint_rows: np.ndarray,
    joint_labels: np.ndarray,
    joint_ranks: np.ndarray,
    joint_classes: tuple[str, ...],
    joint_selection: np.ndarray,
    *,
    hyperparameters: SparsePairwiseFisherHyperparameters,
) -> BeforeAfterSparsePairwiseFit:
    _validate_hyperparameters(hyperparameters)
    selected_old_rows = old_rows[old_selection]
    selected_old_labels = old_labels[old_selection]
    selected_joint_rows = joint_rows[joint_selection]
    selected_joint_labels = joint_labels[joint_selection]
    old_prototypes, old_variances = _class_statistics(
        selected_old_rows, selected_old_labels, old_classes
    )
    joint_prototypes, joint_variances = _class_statistics(
        selected_joint_rows, selected_joint_labels, joint_classes
    )
    if not np.array_equal(
        old_prototypes, joint_prototypes[: len(old_classes)]
    ) or not np.array_equal(
        old_variances, joint_variances[: len(old_classes)]
    ):
        raise SparsePairwiseFisherGuardError(
            "old prototype/statistics bitwise freeze violated"
        )
    old_pairs, old_w, old_mid, old_diag = _select_old_edges(
        selected_old_rows,
        selected_old_labels,
        old_classes,
        old_prototypes,
        old_variances,
        hyperparameters,
    )
    before_rivals = np.empty((0,), dtype=np.int64)
    before_state = _make_state(
        before_artifact,
        classes=old_classes,
        prototypes=old_prototypes,
        old_edge_pairs=old_pairs,
        old_edge_directions=old_w,
        old_edge_bias=old_mid,
        new_rivals=before_rivals,
        new_edge_directions=np.empty(
            (0, old_prototypes.shape[1]), dtype=np.float32
        ),
        new_edge_bias=np.empty((0,), dtype=np.float32),
        hyperparameters=hyperparameters,
        k_shot=int(np.sum(old_selection & (old_labels == old_classes[0]))),
        old_class_count=len(old_classes),
        registration_generation=0,
        support_selection_sha256=_support_selection_sha256(
            before_artifact, old_labels, old_ranks, old_selection
        ),
    )
    rivals, new_w, new_mid, new_diag = _select_new_rivals(
        selected_joint_rows,
        selected_joint_labels,
        joint_classes,
        joint_prototypes,
        joint_variances,
        old_pairs,
        old_class_count=len(old_classes),
        hyperparameters=hyperparameters,
    )
    after_state = _make_state(
        after_artifact,
        classes=joint_classes,
        prototypes=joint_prototypes,
        old_edge_pairs=np.array(old_pairs, copy=True),
        old_edge_directions=np.array(old_w, copy=True),
        old_edge_bias=np.array(old_mid, copy=True),
        new_rivals=rivals,
        new_edge_directions=new_w,
        new_edge_bias=new_mid,
        hyperparameters=hyperparameters,
        k_shot=int(np.sum(joint_selection & (joint_labels == joint_classes[0]))),
        old_class_count=len(old_classes),
        registration_generation=1,
        support_selection_sha256=_support_selection_sha256(
            after_artifact, joint_labels, joint_ranks, joint_selection
        ),
    )
    if (
        not np.array_equal(before_state.old_edge_pairs, after_state.old_edge_pairs)
        or not np.array_equal(
            before_state.old_edge_directions,
            after_state.old_edge_directions,
        )
        or not np.array_equal(
            before_state.old_edge_bias, after_state.old_edge_bias
        )
    ):
        raise SparsePairwiseFisherGuardError("Before pairwise state lock failed")
    return BeforeAfterSparsePairwiseFit(
        before_state=before_state,
        after_state=after_state,
        trace=(
            {
                "phase": "before_sparse_pairwise_closed_form",
                "candidate_id": hyperparameters.candidate_id,
                "operator_id": hyperparameters.operator_id,
                "old_edge_count": len(old_pairs),
                "old_edge_pairs": old_pairs.tolist(),
                "selection_diagnostics": [dict(value) for value in old_diag],
                "support_selection_sha256": before_state.support_selection_sha256,
                "trainable_parameters": 0,
                "adapt_epochs": 0,
            },
            {
                "phase": "after_sparse_pairwise_closed_form",
                "candidate_id": hyperparameters.candidate_id,
                "operator_id": hyperparameters.operator_id,
                "old_edge_count": len(old_pairs),
                "new_edge_count": int(np.sum(rivals >= 0)),
                "old_edge_pairs": old_pairs.tolist(),
                "new_rivals": rivals.tolist(),
                "selection_diagnostics": [
                    dict(value) for value in old_diag + new_diag
                ],
                "support_selection_sha256": after_state.support_selection_sha256,
                "trainable_parameters": 0,
                "adapt_epochs": 0,
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
    hyperparameters: SparsePairwiseFisherHyperparameters,
) -> BeforeAfterSparsePairwiseFit:
    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=k_shot
    )
    joint_rows, joint_labels, joint_ranks, found_classes = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=k_shot
    )
    if not set(old_classes) < set(found_classes):
        raise SparsePairwiseFisherGuardError("new-class registration set drift")
    _validate_artifact_binding(before_artifact, after_artifact)
    _validate_old_lineage_exact_reuse(
        before_artifact,
        old_labels,
        old_ranks,
        after_artifact,
        joint_labels,
        joint_ranks,
        old_classes,
    )
    joint_classes = old_classes + tuple(sorted(set(found_classes) - set(old_classes)))
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
        hyperparameters=hyperparameters,
    )


def _leave_two_out_masks(labels: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, ...]:
    if set(np.unique(ranks).tolist()) != set(range(10)):
        raise SparsePairwiseFisherGuardError("joint L2O requires strict K10 ranks")
    masks = []
    for first in range(0, 10, 2):
        held = np.isin(ranks, (first, first + 1))
        if any(np.sum(held & (labels == label)) != 2 for label in np.unique(labels)):
            raise SparsePairwiseFisherGuardError("leave-two-out physical fold drift")
        masks.append(held)
    return tuple(masks)


def _prediction_metrics(
    truth: np.ndarray, predictions: np.ndarray, classes: Sequence[str]
) -> dict[str, Any]:
    per_class = {
        label: float(np.mean(predictions[truth == label] == label))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean(predictions == truth)),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _aggregate_metrics(
    folds: Sequence[Mapping[str, Any]], classes: Sequence[str], key: str
) -> dict[str, Any]:
    per_class = {
        label: float(np.mean([row[key]["per_class_accuracy"][label] for row in folds]))
        for label in classes
    }
    return {
        "overall_accuracy": float(np.mean([row[key]["overall_accuracy"] for row in folds])),
        "min_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _harmonic(old: float, new: float) -> float:
    return 0.0 if old + new <= 0.0 else 2.0 * old * new / (old + new)


def evaluate_joint_leave_two_out(
    before_artifact: RuntimeAuthorizedFeatureArtifact,
    before_labels: Sequence[str] | np.ndarray,
    before_ranks: Sequence[int] | np.ndarray,
    after_artifact: RuntimeAuthorizedFeatureArtifact,
    after_labels: Sequence[str] | np.ndarray,
    after_ranks: Sequence[int] | np.ndarray,
    *,
    hyperparameters: SparsePairwiseFisherHyperparameters,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Evaluate strict K10 support-only joint old/new physical L2O."""

    old_rows, old_labels, old_ranks, old_classes = _validate_support(
        before_artifact, before_labels, before_ranks, k_shot=10
    )
    joint_rows, joint_labels, joint_ranks, found_classes = _validate_support(
        after_artifact, after_labels, after_ranks, k_shot=10
    )
    if not set(old_classes) < set(found_classes):
        raise SparsePairwiseFisherGuardError("new-class registration set drift")
    _validate_artifact_binding(before_artifact, after_artifact)
    _validate_old_lineage_exact_reuse(
        before_artifact,
        old_labels,
        old_ranks,
        after_artifact,
        joint_labels,
        joint_ranks,
        old_classes,
    )
    joint_classes = old_classes + tuple(sorted(set(found_classes) - set(old_classes)))
    new_classes = joint_classes[len(old_classes) :]
    old_masks = _leave_two_out_masks(old_labels, old_ranks)
    joint_masks = _leave_two_out_masks(joint_labels, joint_ranks)
    old_in_joint = np.isin(joint_labels, old_classes)
    zero_hp = SparsePairwiseFisherHyperparameters(
        candidate_id="d14_true_zero_base",
        operator_id="base",
        ridge=hyperparameters.ridge,
        gamma_old=0.0,
        gamma_new=0.0,
        band_old=0.0,
        band_new=0.0,
        max_old_edges=0,
        force_zero=True,
    )
    folds: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for fold, (held_old, held_joint) in enumerate(zip(old_masks, joint_masks)):
        train_old = ~held_old
        train_joint = ~held_joint
        fitted = _fit_selected(
            before_artifact,
            old_rows,
            old_labels,
            old_ranks,
            old_classes,
            train_old,
            after_artifact,
            joint_rows,
            joint_labels,
            joint_ranks,
            joint_classes,
            train_joint,
            hyperparameters=hyperparameters,
        )
        base = _fit_selected(
            before_artifact,
            old_rows,
            old_labels,
            old_ranks,
            old_classes,
            train_old,
            after_artifact,
            joint_rows,
            joint_labels,
            joint_ranks,
            joint_classes,
            train_joint,
            hyperparameters=zero_hp,
        )
        held_old_joint = held_joint & old_in_joint
        held_new_joint = held_joint & ~old_in_joint
        before_scores = _score_numpy(old_rows[held_old], fitted.before_state)
        base_before_scores = _score_numpy(old_rows[held_old], base.before_state)
        after_old_scores = _score_numpy(
            joint_rows[held_old_joint], fitted.after_state
        )
        after_new_scores = _score_numpy(
            joint_rows[held_new_joint], fitted.after_state
        )
        base_after_old_scores = _score_numpy(
            joint_rows[held_old_joint], base.after_state
        )
        base_after_new_scores = _score_numpy(
            joint_rows[held_new_joint], base.after_state
        )
        before_old_on_after = _score_numpy(
            joint_rows[held_old_joint], fitted.before_state
        )
        before_new_on_after = _score_numpy(
            joint_rows[held_new_joint], fitted.before_state
        )
        before_predictions = np.asarray(old_classes)[
            np.argmax(before_scores, axis=1)
        ]
        base_before_predictions = np.asarray(old_classes)[
            np.argmax(base_before_scores, axis=1)
        ]
        after_old_predictions = np.asarray(joint_classes)[
            np.argmax(after_old_scores, axis=1)
        ]
        after_new_predictions = np.asarray(joint_classes)[
            np.argmax(after_new_scores, axis=1)
        ]
        base_after_old_predictions = np.asarray(joint_classes)[
            np.argmax(base_after_old_scores, axis=1)
        ]
        base_after_new_predictions = np.asarray(joint_classes)[
            np.argmax(base_after_new_scores, axis=1)
        ]
        row = {
            "fold": fold,
            "candidate_id": hyperparameters.candidate_id,
            "before_old": _prediction_metrics(
                old_labels[held_old], before_predictions, old_classes
            ),
            "base_before_old": _prediction_metrics(
                old_labels[held_old], base_before_predictions, old_classes
            ),
            "after_old": _prediction_metrics(
                joint_labels[held_old_joint], after_old_predictions, old_classes
            ),
            "after_new": _prediction_metrics(
                joint_labels[held_new_joint], after_new_predictions, new_classes
            ),
            "base_after_old": _prediction_metrics(
                joint_labels[held_old_joint],
                base_after_old_predictions,
                old_classes,
            ),
            "base_after_new": _prediction_metrics(
                joint_labels[held_new_joint],
                base_after_new_predictions,
                new_classes,
            ),
            "joint_accuracy": float(
                np.mean(
                    np.concatenate([after_old_predictions, after_new_predictions])
                    == np.concatenate(
                        [joint_labels[held_old_joint], joint_labels[held_new_joint]]
                    )
                )
            ),
            "base_joint_accuracy": float(
                np.mean(
                    np.concatenate(
                        [base_after_old_predictions, base_after_new_predictions]
                    )
                    == np.concatenate(
                        [joint_labels[held_old_joint], joint_labels[held_new_joint]]
                    )
                )
            ),
            "old_score_columns_bitwise_equal_before_after": bool(
                np.array_equal(
                    after_old_scores[:, : len(old_classes)], before_old_on_after
                )
                and np.array_equal(
                    after_new_scores[:, : len(old_classes)], before_new_on_after
                )
            ),
            "old_edge_pairs": fitted.before_state.old_edge_pairs.tolist(),
            "new_rivals": fitted.after_state.new_rivals.tolist(),
            "before_support_selection_sha256": (
                fitted.before_state.support_selection_sha256
            ),
            "after_support_selection_sha256": (
                fitted.after_state.support_selection_sha256
            ),
            "before_selection_tensor_sha256": _selection_tensor_sha256(
                fitted.before_state
            ),
            "after_selection_tensor_sha256": _selection_tensor_sha256(
                fitted.after_state
            ),
            "old_train_rows_per_class": 8,
            "new_train_rows_per_class": 8,
            "old_held_rows_per_class": 2,
            "new_held_rows_per_class": 2,
        }
        row["h_old_new"] = _harmonic(
            row["after_old"]["overall_accuracy"],
            row["after_new"]["overall_accuracy"],
        )
        row["base_h_old_new"] = _harmonic(
            row["base_after_old"]["overall_accuracy"],
            row["base_after_new"]["overall_accuracy"],
        )
        row["old_forgetting"] = (
            row["before_old"]["overall_accuracy"]
            - row["after_old"]["overall_accuracy"]
        )
        folds.append(row)
        trace.extend({"fold": fold, **item} for item in fitted.trace)
        trace.append({"phase": "joint_l2o_fold_summary", **row})
    keys = (
        ("before_old", old_classes),
        ("base_before_old", old_classes),
        ("after_old", old_classes),
        ("after_new", new_classes),
        ("base_after_old", old_classes),
        ("base_after_new", new_classes),
    )
    metrics = {
        key: _aggregate_metrics(folds, classes, key) for key, classes in keys
    }
    joint_accuracy = float(np.mean([row["joint_accuracy"] for row in folds]))
    base_joint_accuracy = float(
        np.mean([row["base_joint_accuracy"] for row in folds])
    )
    h_value = _harmonic(
        metrics["after_old"]["overall_accuracy"],
        metrics["after_new"]["overall_accuracy"],
    )
    base_h = _harmonic(
        metrics["base_after_old"]["overall_accuracy"],
        metrics["base_after_new"]["overall_accuracy"],
    )
    result = {
        "selection_policy": (
            "joint_physical_leave_two_out_old_new_each_held2_all_registered"
        ),
        **metrics,
        "joint_accuracy": joint_accuracy,
        "base_joint_accuracy": base_joint_accuracy,
        "h_old_new": h_value,
        "base_h_old_new": base_h,
        "old_forgetting": (
            metrics["before_old"]["overall_accuracy"]
            - metrics["after_old"]["overall_accuracy"]
        ),
        "before_old_per_class_non_degraded_vs_base": all(
            metrics["before_old"]["per_class_accuracy"][label] + 1.0e-12
            >= metrics["base_before_old"]["per_class_accuracy"][label]
            for label in old_classes
        ),
        "after_old_per_class_non_degraded_vs_before": all(
            metrics["after_old"]["per_class_accuracy"][label] + 1.0e-12
            >= metrics["before_old"]["per_class_accuracy"][label]
            for label in old_classes
        ),
        "after_old_per_class_non_degraded_vs_base": all(
            metrics["after_old"]["per_class_accuracy"][label] + 1.0e-12
            >= metrics["base_after_old"]["per_class_accuracy"][label]
            for label in old_classes
        ),
        "after_new_per_class_non_degraded_vs_base": all(
            metrics["after_new"]["per_class_accuracy"][label] + 1.0e-12
            >= metrics["base_after_new"]["per_class_accuracy"][label]
            for label in new_classes
        ),
        "old_score_columns_bitwise_equal_before_after": all(
            row["old_score_columns_bitwise_equal_before_after"] for row in folds
        ),
        "max_old_edge_count": max(len(row["old_edge_pairs"]) for row in folds),
        "all_old_edges_endpoint_disjoint": all(
            len(sum(row["old_edge_pairs"], []))
            == len(set(sum(row["old_edge_pairs"], [])))
            for row in folds
        ),
        "max_new_rivals_per_class": 1,
        "folds": folds,
    }
    return result, tuple(trace)


def predict_all_registered(
    state: SparsePairwiseFisherState,
    query_artifact: RuntimeAuthorizedFeatureArtifact,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict exactly one physical query over all registered classes."""

    _validate_state(state)
    rows = _artifact_rows(query_artifact)
    if len(rows) != 1:
        raise SparsePairwiseFisherGuardError(
            "formal prediction requires exactly one physical query"
        )
    if (
        query_artifact.sealed_runtime_sha256 != state.sealed_runtime_sha256
        or query_artifact.feature_code_sha256 != state.feature_code_sha256
        or query_artifact.sealed_phase1_checkpoint_sha256
        != state.sealed_phase1_checkpoint_sha256
        or query_artifact.operator_id != state.operator_id
        or query_artifact.view_seed != state.view_seed
    ):
        raise SparsePairwiseFisherGuardError(
            "query runtime/code/checkpoint/operator binding mismatch"
        )
    scores = _score_numpy(rows, state)
    predictions = np.asarray(state.classes)[np.argmax(scores, axis=1)]
    return predictions, scores
