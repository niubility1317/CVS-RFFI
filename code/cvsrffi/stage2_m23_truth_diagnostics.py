"""Truth-last paired prediction diagnostics for the M2.3 F0-F5 matrix."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_metric_scorer import (
    load_verified_scoring_sidecar,
    load_verified_sealed_prediction,
    score_prediction_arrays,
)


M23_PAIRED_DIAGNOSTIC_SCHEMA = "cvs.erbt_idr.m23.paired_flip_diagnostic.v1"
M23_FOUR_STATE_SCORE_SCHEMA = "cvs.erbt_idr.m23.four_state_score.v1"


class M23TruthDiagnosticError(ValueError):
    """Raised when paired truth-side rows are not the same physical query set."""


def _vectors(*values: Any) -> list[np.ndarray]:
    result = [np.asarray(value).astype(str) for value in values]
    if not result or any(value.ndim != 1 for value in result):
        raise M23TruthDiagnosticError("paired diagnostic columns must be vectors")
    if len({len(value) for value in result}) != 1 or len(result[0]) <= 0:
        raise M23TruthDiagnosticError("paired diagnostic row counts differ")
    return result


def _mcnemar_exact(help_count: int, harm_count: int) -> float:
    discordant = int(help_count + harm_count)
    if discordant == 0:
        return 1.0
    tail = min(int(help_count), int(harm_count))
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2.0**discordant)
    return float(min(1.0, 2.0 * probability))


def _counts(
    reference: np.ndarray,
    candidate: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    ref_correct = reference[mask] == truth[mask]
    candidate_correct = candidate[mask] == truth[mask]
    help_mask = (~ref_correct) & candidate_correct
    harm_mask = ref_correct & (~candidate_correct)
    count = int(np.sum(mask))
    return {
        "query_count": count,
        "reference_accuracy": float(np.mean(ref_correct)) if count else None,
        "candidate_accuracy": float(np.mean(candidate_correct)) if count else None,
        "accuracy_delta": (
            float(np.mean(candidate_correct) - np.mean(ref_correct)) if count else None
        ),
        "N_help": int(np.sum(help_mask)),
        "N_harm": int(np.sum(harm_mask)),
    }


def _cluster_bootstrap(
    reference_correct: np.ndarray,
    candidate_correct: np.ndarray,
    clusters: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(clusters)
    if repeats <= 0 or len(unique) <= 0:
        raise M23TruthDiagnosticError("bootstrap controls are invalid")
    rng = np.random.default_rng(int(seed))
    observations = np.empty(int(repeats), dtype=np.float64)
    per_cluster = {
        item: candidate_correct[clusters == item].astype(np.float64)
        - reference_correct[clusters == item].astype(np.float64)
        for item in unique
    }
    for repeat in range(int(repeats)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([per_cluster[item] for item in sampled])
        observations[repeat] = float(np.mean(values))
    return {
        "cluster_key": "true_class",
        "cluster_count": int(len(unique)),
        "repeats": int(repeats),
        "seed": int(seed),
        "accuracy_delta_mean": float(np.mean(observations)),
        "accuracy_delta_ci95": [
            float(np.quantile(observations, 0.025)),
            float(np.quantile(observations, 0.975)),
        ],
    }


def paired_flip_summary(
    reference_prediction: Any,
    candidate_prediction: Any,
    true_class: Any,
    *,
    scenarios: Any,
    true_roles: Any,
    true_classes: Any,
    receiver: str,
    k_shot: int,
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 2301,
) -> dict[str, Any]:
    """Summarise exactly paired F1-versus-candidate prediction flips."""

    reference, candidate, truth, scenario, role, class_rows = _vectors(
        reference_prediction,
        candidate_prediction,
        true_class,
        scenarios,
        true_roles,
        true_classes,
    )
    if not str(receiver).strip() or int(k_shot) <= 0:
        raise M23TruthDiagnosticError("receiver/K-shot binding is incomplete")
    class_role: dict[str, str] = {}
    for handle, current_role in zip(class_rows, role):
        prior = class_role.setdefault(str(handle), str(current_role))
        if prior != str(current_role):
            raise M23TruthDiagnosticError("class handle maps to multiple roles")
    if any(value not in class_role for value in np.concatenate([reference, candidate])):
        raise M23TruthDiagnosticError("prediction references an unregistered class")

    ref_correct = reference == truth
    candidate_correct = candidate == truth
    help_mask = (~ref_correct) & candidate_correct
    harm_mask = ref_correct & (~candidate_correct)
    transitions = {
        "correct_to_correct": int(np.sum(ref_correct & candidate_correct)),
        "wrong_to_correct": int(np.sum(help_mask)),
        "correct_to_wrong": int(np.sum(harm_mask)),
        "wrong_to_same_wrong": int(
            np.sum((~ref_correct) & (~candidate_correct) & (reference == candidate))
        ),
        "wrong_to_different_wrong": int(
            np.sum((~ref_correct) & (~candidate_correct) & (reference != candidate))
        ),
    }
    all_mask = np.ones(len(truth), dtype=bool)
    overall = _counts(reference, candidate, truth, all_mask)

    def stratified(values: np.ndarray) -> dict[str, Any]:
        return {
            str(item): _counts(reference, candidate, truth, values == item)
            for item in sorted(set(values.tolist()))
        }

    reference_roles = np.asarray([class_role[value] for value in reference])
    candidate_roles = np.asarray([class_role[value] for value in candidate])
    direction = {
        "reference_old_to_new": int(
            np.sum((role == "target_old") & (reference_roles == "target_new"))
        ),
        "candidate_old_to_new": int(
            np.sum((role == "target_old") & (candidate_roles == "target_new"))
        ),
        "reference_new_to_old": int(
            np.sum((role == "target_new") & (reference_roles == "target_old"))
        ),
        "candidate_new_to_old": int(
            np.sum((role == "target_new") & (candidate_roles == "target_old"))
        ),
        "reference_old_to_wrong_old": int(
            np.sum(
                (role == "target_old")
                & (reference_roles == "target_old")
                & (reference != truth)
            )
        ),
        "candidate_old_to_wrong_old": int(
            np.sum(
                (role == "target_old")
                & (candidate_roles == "target_old")
                & (candidate != truth)
            )
        ),
        "reference_new_to_wrong_new": int(
            np.sum(
                (role == "target_new")
                & (reference_roles == "target_new")
                & (reference != truth)
            )
        ),
        "candidate_new_to_wrong_new": int(
            np.sum(
                (role == "target_new")
                & (candidate_roles == "target_new")
                & (candidate != truth)
            )
        ),
    }
    return {
        "schema": M23_PAIRED_DIAGNOSTIC_SCHEMA,
        "receiver": str(receiver),
        "k_shot": int(k_shot),
        "query_count": len(truth),
        "reference_accuracy": overall["reference_accuracy"],
        "candidate_accuracy": overall["candidate_accuracy"],
        "accuracy_delta": overall["accuracy_delta"],
        "N_help": overall["N_help"],
        "N_harm": overall["N_harm"],
        "error_transition_counts": transitions,
        "role_error_direction_counts": direction,
        "mcnemar_exact_pvalue": _mcnemar_exact(overall["N_help"], overall["N_harm"]),
        "cluster_bootstrap": _cluster_bootstrap(
            ref_correct,
            candidate_correct,
            class_rows,
            repeats=int(bootstrap_repeats),
            seed=int(bootstrap_seed),
        ),
        "by_scenario": stratified(scenario),
        "by_role": stratified(role),
        "by_true_class": stratified(class_rows),
    }


def _harmonic(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or left + right <= 0.0:
        return None
    return float(2.0 * left * right / (left + right))


def four_state_summary_from_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert scorer rows into explicit DA/registration states and effects."""

    if not predictions:
        raise M23TruthDiagnosticError("four-state scoring requires prediction rows")
    columns = {
        "DA0_REG0": "identity_before",
        "DA1_REG0": "candidate_before",
        "DA0_REG1": "identity_after",
        "DA1_REG1": "candidate_after",
    }
    scenarios = sorted({str(row["scenario"]) for row in predictions})
    scenario_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows = [row for row in predictions if str(row["scenario"]) == scenario]
        states: dict[str, dict[str, Any]] = {}
        for state, column in columns.items():
            old_rows = [row for row in rows if row["evaluation_role"] == "target_old"]
            new_rows = [row for row in rows if row["evaluation_role"] == "target_new"]
            if not old_rows or (state.endswith("REG1") and not new_rows):
                raise M23TruthDiagnosticError("four-state role coverage is incomplete")
            old_accuracy = float(
                np.mean(
                    [row[column] == row["true_class_handle"] for row in old_rows]
                )
            )
            if state.endswith("REG0"):
                new_accuracy = None
            else:
                new_accuracy = float(
                    np.mean(
                        [row[column] == row["true_class_handle"] for row in new_rows]
                    )
                )
            states[state] = {
                "old_accuracy": old_accuracy,
                "new_accuracy": new_accuracy,
                "H_old_new": _harmonic(old_accuracy, new_accuracy),
                "new_class_metric_status": (
                    "N/A_UNREGISTERED" if state.endswith("REG0") else "DEFINED"
                ),
            }

        def delta(left: str, right: str, metric: str) -> float | None:
            left_value = states[left][metric]
            right_value = states[right][metric]
            if left_value is None or right_value is None:
                return None
            return float(left_value - right_value)

        metrics = ("old_accuracy", "new_accuracy", "H_old_new")
        effects = {
            "DA1_REG0_minus_DA0_REG0": {
                metric: delta("DA1_REG0", "DA0_REG0", metric) for metric in metrics
            },
            "DA1_REG1_minus_DA0_REG1": {
                metric: delta("DA1_REG1", "DA0_REG1", metric) for metric in metrics
            },
            "DA0_REG1_minus_DA0_REG0": {
                metric: delta("DA0_REG1", "DA0_REG0", metric) for metric in metrics
            },
            "DA1_REG1_minus_DA1_REG0": {
                metric: delta("DA1_REG1", "DA1_REG0", metric) for metric in metrics
            },
        }
        interaction: dict[str, float | None] = {}
        for metric in metrics:
            after = effects["DA1_REG1_minus_DA0_REG1"][metric]
            before = effects["DA1_REG0_minus_DA0_REG0"][metric]
            interaction[metric] = (
                float(after - before) if after is not None and before is not None else None
            )
        effects["difference_in_differences"] = interaction
        scenario_rows.append(
            {
                "scenario": scenario,
                "query_count": len(rows),
                "states": states,
                "effects": effects,
            }
        )
    return {
        "schema": M23_FOUR_STATE_SCORE_SCHEMA,
        "status": "PASS",
        "state_names": list(columns),
        "scenario_rows": scenario_rows,
    }


def score_m23_four_state_artifact(
    *,
    prediction_path: str,
    prediction_artifact_sha256: str,
    prediction_seal_sha256: str,
    scoring_manifest_path: str,
    scoring_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify one prediction before opening truth and score all four states."""

    binding, arrays, prediction_audit = load_verified_sealed_prediction(
        prediction_path,
        expected_prediction_artifact_sha256=prediction_artifact_sha256,
        expected_prediction_seal_sha256=prediction_seal_sha256,
    )
    truth, scoring_manifest, scoring_audit = load_verified_scoring_sidecar(
        scoring_manifest_path,
        expected_scoring_manifest_sha256=scoring_manifest_sha256,
    )
    if (
        scoring_manifest["predictor_package_root_sha256"]
        != binding["predictor_package_root_sha256"]
        or scoring_manifest["predictor_package_seal_sha256"]
        != binding["predictor_package_seal_sha256"]
    ):
        raise M23TruthDiagnosticError("four-state artifact/scoring binding drift")
    _rows, predictions = score_prediction_arrays(
        binding=binding,
        arrays=arrays,
        truth=truth,
    )
    summary = four_state_summary_from_predictions(predictions)
    return {
        **summary,
        "row_id": binding["row_id"],
        "receiver": binding["receiver"],
        "k_shot": int(binding["k_shot"]),
        "prediction_artifact_sha256": prediction_audit[
            "prediction_artifact_sha256"
        ],
        "scoring_manifest_sha256": scoring_audit["scoring_manifest_sha256"],
        "truth_sidecar_sha256": scoring_audit["truth_sidecar_sha256"],
        "truth_opened_after_prediction_commit": True,
        "scorer_output_must_not_feed_predictor": True,
    }


def score_m23_paired_artifacts(
    *,
    reference_prediction_path: str,
    reference_artifact_sha256: str,
    reference_seal_sha256: str,
    candidate_prediction_path: str,
    candidate_artifact_sha256: str,
    candidate_seal_sha256: str,
    scoring_manifest_path: str,
    scoring_manifest_sha256: str,
    bootstrap_repeats: int = 2000,
    bootstrap_seed: int = 2301,
) -> dict[str, Any]:
    """Verify both prediction artifacts before the first truth-side open."""

    reference_binding, reference_arrays, reference_audit = load_verified_sealed_prediction(
        reference_prediction_path,
        expected_prediction_artifact_sha256=reference_artifact_sha256,
        expected_prediction_seal_sha256=reference_seal_sha256,
    )
    candidate_binding, candidate_arrays, candidate_audit = load_verified_sealed_prediction(
        candidate_prediction_path,
        expected_prediction_artifact_sha256=candidate_artifact_sha256,
        expected_prediction_seal_sha256=candidate_seal_sha256,
    )
    identity_fields = (
        "stage",
        "receiver",
        "k_shot",
        "predictor_package_root_sha256",
        "predictor_package_seal_sha256",
    )
    if any(reference_binding[field] != candidate_binding[field] for field in identity_fields):
        raise M23TruthDiagnosticError("paired artifacts do not share a physical row")
    for field in ("query_tokens", "scenarios"):
        if not np.array_equal(reference_arrays[field], candidate_arrays[field]):
            raise M23TruthDiagnosticError("paired artifacts do not share query identity")

    truth, scoring_manifest, scoring_audit = load_verified_scoring_sidecar(
        scoring_manifest_path,
        expected_scoring_manifest_sha256=scoring_manifest_sha256,
    )
    if (
        scoring_manifest["predictor_package_root_sha256"]
        != reference_binding["predictor_package_root_sha256"]
        or scoring_manifest["predictor_package_seal_sha256"]
        != reference_binding["predictor_package_seal_sha256"]
    ):
        raise M23TruthDiagnosticError("paired artifacts/scoring sidecar binding drift")
    _reference_rows, reference_predictions = score_prediction_arrays(
        binding=reference_binding,
        arrays=reference_arrays,
        truth=truth,
    )
    _candidate_rows, candidate_predictions = score_prediction_arrays(
        binding=candidate_binding,
        arrays=candidate_arrays,
        truth=truth,
    )
    reference_map = {
        (row["scenario"], row["query_token"]): row for row in reference_predictions
    }
    candidate_map = {
        (row["scenario"], row["query_token"]): row for row in candidate_predictions
    }
    if set(reference_map) != set(candidate_map):
        raise M23TruthDiagnosticError("truth-scored paired query sets differ")
    keys = sorted(reference_map)
    reference = np.asarray([reference_map[key]["candidate_after"] for key in keys])
    candidate = np.asarray([candidate_map[key]["candidate_after"] for key in keys])
    true_class = np.asarray([reference_map[key]["true_class_handle"] for key in keys])
    roles = np.asarray([reference_map[key]["evaluation_role"] for key in keys])
    scenarios = np.asarray([key[0] for key in keys])
    if any(value is None for value in true_class.tolist()):
        raise M23TruthDiagnosticError("paired diagnostic contains an unscored query")
    summary = paired_flip_summary(
        reference,
        candidate,
        true_class,
        scenarios=scenarios,
        true_roles=roles,
        true_classes=true_class,
        receiver=str(reference_binding["receiver"]),
        k_shot=int(reference_binding["k_shot"]),
        bootstrap_repeats=int(bootstrap_repeats),
        bootstrap_seed=int(bootstrap_seed),
    )
    return {
        **summary,
        "reference_row_id": reference_binding["row_id"],
        "candidate_row_id": candidate_binding["row_id"],
        "reference_prediction_artifact_sha256": reference_audit[
            "prediction_artifact_sha256"
        ],
        "candidate_prediction_artifact_sha256": candidate_audit[
            "prediction_artifact_sha256"
        ],
        "scoring_manifest_sha256": scoring_audit["scoring_manifest_sha256"],
        "truth_sidecar_sha256": scoring_audit["truth_sidecar_sha256"],
        "truth_opened_after_both_prediction_commits": True,
        "scorer_output_must_not_feed_predictor": True,
    }


__all__ = [
    "M23_FOUR_STATE_SCORE_SCHEMA",
    "M23_PAIRED_DIAGNOSTIC_SCHEMA",
    "M23TruthDiagnosticError",
    "four_state_summary_from_predictions",
    "paired_flip_summary",
    "score_m23_four_state_artifact",
    "score_m23_paired_artifacts",
]
