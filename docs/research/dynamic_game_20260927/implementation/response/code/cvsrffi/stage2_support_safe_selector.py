"""Pure registered-support safety selector for fixed received LEO_weak IQ views.

This module has no query, truth-sidecar, role-oracle, quota, dataset, clean,
source, scorer, or runner interface. It consumes support deletion predictions
whose physical-sample lineage is already sealed. A candidate is eligible only
when its aggregate and floor evidence remain within pre-registered
non-inferiority limits relative to a matched baseline.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


DEFAULT_MAX_ACCURACY_DROP = 0.02
DEFAULT_MAX_FLOOR_DROP = 0.02
DEFAULT_WILSON_Z = 1.6448536269514722
DEFAULT_BETA_ALPHA = 1.0
DEFAULT_BETA_BETA = 1.0
ALLOWED_STABILITY_PROTOCOLS = {
    "support_leave_two_physical_samples_out",
    "support_stratified_physical_subsampling",
}


class SupportSafeSelectorError(ValueError):
    """Raised when support-only evidence or protocol invariants drift."""


@dataclass(frozen=True)
class SupportDeletionEvidence:
    """Predictions for support samples held out from a support-only fit."""

    protocol: str
    physical_sample_ids: tuple[str, ...]
    parent_received_iq_sha256: tuple[str, ...]
    labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    margins: tuple[float, ...]


@dataclass(frozen=True)
class SupportSliceEvidence:
    """Matched evidence for one registration state and one LEO scenario."""

    registration_state: str
    scenario: str
    loo: SupportDeletionEvidence
    stability: SupportDeletionEvidence


@dataclass(frozen=True)
class SupportCandidateEvidence:
    """All support-only evidence for one pre-registered candidate."""

    name: str
    view_count: int
    slices: tuple[SupportSliceEvidence, ...]
    views_count_as_additional_physical_samples: bool = False
    additional_leo_channel_state_generation: bool = False
    query_rows_used_for_selection: int = 0
    query_labels_used_for_selection: bool = False


def wilson_lower_bound(
    correct: int,
    total: int,
    *,
    z: float = DEFAULT_WILSON_Z,
) -> float:
    """One-sided Wilson lower confidence bound for a binomial proportion."""

    if total < 1 or correct < 0 or correct > total:
        raise SupportSafeSelectorError("invalid binomial count")
    if not math.isfinite(float(z)) or float(z) <= 0.0:
        raise SupportSafeSelectorError("Wilson z must be positive")
    n = float(total)
    p = float(correct) / n
    z2 = float(z) ** 2
    center = p + z2 / (2.0 * n)
    radius = float(z) * math.sqrt(
        p * (1.0 - p) / n + z2 / (4.0 * n * n)
    )
    return max(0.0, (center - radius) / (1.0 + z2 / n))


def beta_smoothed_accuracy(
    correct: int,
    total: int,
    *,
    alpha: float = DEFAULT_BETA_ALPHA,
    beta: float = DEFAULT_BETA_BETA,
) -> float:
    """Posterior mean under a fixed Beta prior."""

    if total < 1 or correct < 0 or correct > total:
        raise SupportSafeSelectorError("invalid binomial count")
    if (
        not math.isfinite(float(alpha))
        or not math.isfinite(float(beta))
        or float(alpha) <= 0.0
        or float(beta) <= 0.0
    ):
        raise SupportSafeSelectorError("Beta prior parameters must be positive")
    return (float(correct) + float(alpha)) / (
        float(total) + float(alpha) + float(beta)
    )


def stratified_deletion_folds(
    labels: Sequence[str],
    *,
    width_per_class: int,
) -> tuple[tuple[int, ...], ...]:
    """Create deterministic class-stratified physical-sample deletion folds."""

    values = np.asarray(tuple(str(value) for value in labels))
    if values.ndim != 1 or len(values) < 2 or any(not value for value in values):
        raise SupportSafeSelectorError("support labels must be non-empty")
    width = int(width_per_class)
    if width not in {1, 2}:
        raise SupportSafeSelectorError("deletion width must be one or two")
    classes = tuple(sorted(set(values.tolist())))
    by_class = [np.flatnonzero(values == label).tolist() for label in classes]
    if min(len(indices) for indices in by_class) <= width:
        raise SupportSafeSelectorError(
            "each class must retain at least one physical sample per fold"
        )
    folds: list[tuple[int, ...]] = []
    for offset in range(0, max(len(indices) for indices in by_class), width):
        held: list[int] = []
        for indices in by_class:
            held.extend(indices[offset : offset + width])
        if held:
            folds.append(tuple(sorted(held)))
    flattened = [index for fold in folds for index in fold]
    if sorted(flattened) != list(range(len(values))):
        raise SupportSafeSelectorError(
            "stratified deletion folds must cover each support sample once"
        )
    return tuple(folds)


def collect_deletion_evidence(
    *,
    labels: Sequence[str],
    physical_sample_ids: Sequence[str],
    parent_received_iq_sha256: Sequence[str],
    predictor: Callable[[np.ndarray, np.ndarray], tuple[Sequence[str], Sequence[float]]],
    width_per_class: int,
) -> SupportDeletionEvidence:
    """Evaluate a candidate on support-only deletion folds.

    ``predictor`` receives train indices and held-out indices. It must fit only
    on the train indices and return one prediction and margin per held sample.
    """

    label_values = np.asarray(tuple(str(value) for value in labels))
    sample_ids = tuple(str(value) for value in physical_sample_ids)
    hashes = tuple(str(value).lower() for value in parent_received_iq_sha256)
    if (
        len(label_values) != len(sample_ids)
        or len(label_values) != len(hashes)
        or len(set(sample_ids)) != len(sample_ids)
        or len(set(hashes)) != len(hashes)
    ):
        raise SupportSafeSelectorError("support lineage alignment or uniqueness drift")
    folds = stratified_deletion_folds(
        label_values,
        width_per_class=int(width_per_class),
    )
    predicted_by_index: list[str | None] = [None] * len(label_values)
    margin_by_index: list[float | None] = [None] * len(label_values)
    all_indices = np.arange(len(label_values), dtype=np.int64)
    for held_tuple in folds:
        held = np.asarray(held_tuple, dtype=np.int64)
        keep = np.ones(len(label_values), dtype=bool)
        keep[held] = False
        train = all_indices[keep]
        predicted, margins = predictor(train, held)
        predicted_values = tuple(str(value) for value in predicted)
        margin_values = tuple(float(value) for value in margins)
        if (
            len(predicted_values) != len(held)
            or len(margin_values) != len(held)
            or any(not value for value in predicted_values)
            or not np.isfinite(np.asarray(margin_values)).all()
        ):
            raise SupportSafeSelectorError("support deletion predictor output drift")
        for index, prediction, margin in zip(
            held.tolist(), predicted_values, margin_values
        ):
            if predicted_by_index[index] is not None:
                raise SupportSafeSelectorError(
                    "support sample evaluated more than once"
                )
            predicted_by_index[index] = prediction
            margin_by_index[index] = margin
    if any(value is None for value in predicted_by_index + margin_by_index):
        raise SupportSafeSelectorError("support deletion evidence is incomplete")
    protocol = (
        "support_leave_two_physical_samples_out"
        if int(width_per_class) == 2
        else "support_leave_one_physical_sample_out"
    )
    return SupportDeletionEvidence(
        protocol=protocol,
        physical_sample_ids=sample_ids,
        parent_received_iq_sha256=hashes,
        labels=tuple(label_values.tolist()),
        predicted_labels=tuple(str(value) for value in predicted_by_index),
        margins=tuple(float(value) for value in margin_by_index),
    )


def _validate_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_deletion_evidence(
    evidence: SupportDeletionEvidence,
    *,
    expected_protocols: set[str],
) -> None:
    count = len(evidence.labels)
    if (
        evidence.protocol not in expected_protocols
        or count < 2
        or len(evidence.physical_sample_ids) != count
        or len(evidence.parent_received_iq_sha256) != count
        or len(evidence.predicted_labels) != count
        or len(evidence.margins) != count
        or len(set(evidence.physical_sample_ids)) != count
        or len(set(evidence.parent_received_iq_sha256)) != count
        or any(not value for value in evidence.labels)
        or any(not value for value in evidence.predicted_labels)
        or any(
            not _validate_sha256(value)
            for value in evidence.parent_received_iq_sha256
        )
        or not np.isfinite(np.asarray(evidence.margins, dtype=np.float64)).all()
    ):
        raise SupportSafeSelectorError("invalid support deletion evidence")


def _slice_key(value: SupportSliceEvidence) -> tuple[str, str]:
    return value.registration_state, value.scenario


def _candidate_summary(
    candidate: SupportCandidateEvidence,
    *,
    wilson_z: float,
    beta_alpha: float,
    beta_beta: float,
) -> dict[str, Any]:
    loo_correct: list[bool] = []
    stability_correct: list[bool] = []
    loo_floor_wilson: list[float] = []
    loo_floor_beta: list[float] = []
    stability_floor_wilson: list[float] = []
    stability_floor_beta: list[float] = []
    slice_rows: list[dict[str, Any]] = []
    for support_slice in candidate.slices:
        _validate_deletion_evidence(
            support_slice.loo,
            expected_protocols={"support_leave_one_physical_sample_out"},
        )
        _validate_deletion_evidence(
            support_slice.stability,
            expected_protocols=ALLOWED_STABILITY_PROTOCOLS,
        )
        if (
            support_slice.loo.physical_sample_ids
            != support_slice.stability.physical_sample_ids
            or support_slice.loo.parent_received_iq_sha256
            != support_slice.stability.parent_received_iq_sha256
            or support_slice.loo.labels != support_slice.stability.labels
        ):
            raise SupportSafeSelectorError("LOO/stability support lineage drift")
        labels = np.asarray(support_slice.loo.labels)
        loo_prediction = np.asarray(support_slice.loo.predicted_labels)
        stability_prediction = np.asarray(
            support_slice.stability.predicted_labels
        )
        loo_match = loo_prediction == labels
        stability_match = stability_prediction == labels
        loo_correct.extend(loo_match.tolist())
        stability_correct.extend(stability_match.tolist())
        per_class: dict[str, Any] = {}
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            total = int(np.sum(mask))
            loo_hits = int(np.sum(loo_match[mask]))
            stability_hits = int(np.sum(stability_match[mask]))
            loo_wilson = wilson_lower_bound(loo_hits, total, z=wilson_z)
            loo_beta = beta_smoothed_accuracy(
                loo_hits,
                total,
                alpha=beta_alpha,
                beta=beta_beta,
            )
            stability_wilson = wilson_lower_bound(
                stability_hits, total, z=wilson_z
            )
            stability_beta = beta_smoothed_accuracy(
                stability_hits,
                total,
                alpha=beta_alpha,
                beta=beta_beta,
            )
            loo_floor_wilson.append(loo_wilson)
            loo_floor_beta.append(loo_beta)
            stability_floor_wilson.append(stability_wilson)
            stability_floor_beta.append(stability_beta)
            per_class[label] = {
                "physical_support_count": total,
                "loo_correct": loo_hits,
                "loo_wilson_lower": loo_wilson,
                "loo_beta_smoothed": loo_beta,
                "stability_correct": stability_hits,
                "stability_wilson_lower": stability_wilson,
                "stability_beta_smoothed": stability_beta,
            }
        slice_rows.append(
            {
                "registration_state": support_slice.registration_state,
                "scenario": support_slice.scenario,
                "loo_protocol": support_slice.loo.protocol,
                "stability_protocol": support_slice.stability.protocol,
                "loo_accuracy": float(np.mean(loo_match)),
                "stability_accuracy": float(np.mean(stability_match)),
                "per_class": per_class,
            }
        )
    return {
        "name": candidate.name,
        "view_count": int(candidate.view_count),
        "loo_overall_accuracy": float(np.mean(loo_correct)),
        "stability_overall_accuracy": float(np.mean(stability_correct)),
        "worst_loo_class_wilson_lower": min(loo_floor_wilson),
        "mean_loo_class_beta_smoothed": float(np.mean(loo_floor_beta)),
        "worst_stability_class_wilson_lower": min(stability_floor_wilson),
        "mean_stability_class_beta_smoothed": float(
            np.mean(stability_floor_beta)
        ),
        "slices": slice_rows,
    }


def _validate_matched_candidates(
    candidates: Sequence[SupportCandidateEvidence],
) -> None:
    if not candidates:
        raise SupportSafeSelectorError("candidate evidence is empty")
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise SupportSafeSelectorError("candidate names must be unique")
    reference = candidates[0]
    reference_slices = {_slice_key(row): row for row in reference.slices}
    if len(reference_slices) != len(reference.slices) or not reference_slices:
        raise SupportSafeSelectorError("support slice keys must be unique")
    for candidate in candidates:
        if (
            candidate.view_count < 1
            or candidate.views_count_as_additional_physical_samples
            or candidate.additional_leo_channel_state_generation
            or candidate.query_rows_used_for_selection != 0
            or candidate.query_labels_used_for_selection
        ):
            raise SupportSafeSelectorError(
                "candidate violates fixed-received-IQ support-only boundary"
            )
        candidate_slices = {_slice_key(row): row for row in candidate.slices}
        if set(candidate_slices) != set(reference_slices):
            raise SupportSafeSelectorError("candidate support slice set drift")
        for key, reference_slice in reference_slices.items():
            candidate_slice = candidate_slices[key]
            for field in ("loo", "stability"):
                left = getattr(reference_slice, field)
                right = getattr(candidate_slice, field)
                if (
                    left.physical_sample_ids != right.physical_sample_ids
                    or left.parent_received_iq_sha256
                    != right.parent_received_iq_sha256
                    or left.labels != right.labels
                ):
                    raise SupportSafeSelectorError(
                        "candidates are not matched on physical support"
                    )
    for registration_state in sorted(
        {key[0] for key in reference_slices}
    ):
        scenario_ids: dict[str, set[str]] = {}
        scenario_hashes: dict[str, set[str]] = {}
        for (state, scenario), support_slice in reference_slices.items():
            if state != registration_state:
                continue
            scenario_ids[scenario] = set(
                support_slice.loo.physical_sample_ids
            )
            scenario_hashes[scenario] = set(
                support_slice.loo.parent_received_iq_sha256
            )
        scenarios = sorted(scenario_ids)
        for first_index, first in enumerate(scenarios):
            for second in scenarios[first_index + 1 :]:
                if (
                    scenario_ids[first] & scenario_ids[second]
                    or scenario_hashes[first] & scenario_hashes[second]
                ):
                    raise SupportSafeSelectorError(
                        "cross-scenario physical support reuse"
                    )


def select_support_safe_candidate(
    candidates: Sequence[SupportCandidateEvidence],
    *,
    baseline_name: str,
    max_overall_drop: float = DEFAULT_MAX_ACCURACY_DROP,
    max_floor_drop: float = DEFAULT_MAX_FLOOR_DROP,
    wilson_z: float = DEFAULT_WILSON_Z,
    beta_alpha: float = DEFAULT_BETA_ALPHA,
    beta_beta: float = DEFAULT_BETA_BETA,
) -> dict[str, Any]:
    """Select one candidate using only matched registered-support evidence."""

    _validate_matched_candidates(candidates)
    if not 0.0 <= float(max_overall_drop) <= 0.25:
        raise SupportSafeSelectorError("max overall drop is out of range")
    if not 0.0 <= float(max_floor_drop) <= 0.25:
        raise SupportSafeSelectorError("max floor drop is out of range")
    summaries = [
        _candidate_summary(
            candidate,
            wilson_z=float(wilson_z),
            beta_alpha=float(beta_alpha),
            beta_beta=float(beta_beta),
        )
        for candidate in candidates
    ]
    by_name = {row["name"]: row for row in summaries}
    if baseline_name not in by_name:
        raise SupportSafeSelectorError("baseline candidate is missing")
    baseline = by_name[baseline_name]
    for row in summaries:
        row["loo_overall_noninferiority_pass"] = bool(
            row["loo_overall_accuracy"] + float(max_overall_drop)
            >= baseline["loo_overall_accuracy"]
        )
        row["stability_overall_noninferiority_pass"] = bool(
            row["stability_overall_accuracy"] + float(max_overall_drop)
            >= baseline["stability_overall_accuracy"]
        )
        row["loo_floor_noninferiority_pass"] = bool(
            row["worst_loo_class_wilson_lower"] + float(max_floor_drop)
            >= baseline["worst_loo_class_wilson_lower"]
        )
        row["stability_floor_noninferiority_pass"] = bool(
            row["worst_stability_class_wilson_lower"] + float(max_floor_drop)
            >= baseline["worst_stability_class_wilson_lower"]
        )
        row["eligible"] = all(
            row[field]
            for field in (
                "loo_overall_noninferiority_pass",
                "stability_overall_noninferiority_pass",
                "loo_floor_noninferiority_pass",
                "stability_floor_noninferiority_pass",
            )
        )
        row["selection_key"] = [
            row["worst_stability_class_wilson_lower"],
            row["worst_loo_class_wilson_lower"],
            row["mean_stability_class_beta_smoothed"],
            row["mean_loo_class_beta_smoothed"],
            row["stability_overall_accuracy"],
            row["loo_overall_accuracy"],
            -int(row["view_count"]),
        ]
    eligible = [row for row in summaries if row["eligible"]]
    if not eligible:
        raise SupportSafeSelectorError("no support-safe candidate remains")
    ranking = sorted(
        eligible,
        key=lambda row: tuple(float(value) for value in row["selection_key"]),
        reverse=True,
    )
    rejected = sorted(
        [row for row in summaries if not row["eligible"]],
        key=lambda row: row["name"],
    )
    return {
        "schema": "cvs.phase2.support_safe_selector.v1",
        "selection_data": "registered_support_only",
        "query_rows_used_for_selection": 0,
        "query_labels_used_for_selection": False,
        "baseline_name": baseline_name,
        "max_overall_drop": float(max_overall_drop),
        "max_floor_drop": float(max_floor_drop),
        "wilson_z": float(wilson_z),
        "beta_prior": {
            "alpha": float(beta_alpha),
            "beta": float(beta_beta),
        },
        "selected_candidate": ranking[0]["name"],
        "ranking": ranking,
        "rejected": rejected,
        "complexity_tie_break": "lower_view_count",
        "views_count_as_additional_physical_samples": False,
        "additional_leo_channel_state_generation": False,
    }


def public_interface_is_support_only() -> bool:
    """Machine-check public selector signatures for forbidden query/oracle inputs."""

    forbidden = {
        "query",
        "truth",
        "role",
        "quota",
        "batch_assignment",
        "source",
        "clean",
    }
    public = (
        collect_deletion_evidence,
        select_support_safe_candidate,
        stratified_deletion_folds,
    )
    return all(
        not any(
            token in parameter.lower()
            for token in forbidden
            for parameter in inspect.signature(function).parameters
        )
        for function in public
    )


__all__ = [
    "SupportCandidateEvidence",
    "SupportDeletionEvidence",
    "SupportSafeSelectorError",
    "SupportSliceEvidence",
    "beta_smoothed_accuracy",
    "collect_deletion_evidence",
    "public_interface_is_support_only",
    "select_support_safe_candidate",
    "stratified_deletion_folds",
    "wilson_lower_bound",
]
