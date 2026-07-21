"""Paired D92 baseline and licensed role-only Oracle query evaluation.

The D92 fit and INT8 score path run exactly once.  The ordinary D92 prediction
is committed before the role capsule is opened.  The licensed branch then masks
only the opposite-role columns of the captured score matrix; it never refits or
re-forwards a query.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_d42_unified_shrinkage_lda import (
    score_d42_unified_shrinkage_lda,
)
from cvsrffi.stage2_d92_licensed_role_oracle import (
    LICENSE_STATUS,
    decode_d92_licensed_role_oracle,
)
from cvsrffi.stage2_d92_query_evaluation import (
    CANDIDATE_D92,
    run_d92_query_evaluation,
)
from cvsrffi.stage2_diag_cosine_exploration import (
    _canonical_json_bytes,
    _sha256_file,
    _write_json_new,
    _write_npz_new,
)


CANDIDATE_D92_ROLE_ORACLE = "d92_role_oracle_licensed_upper_bound"
SCHEMA = "cvs.phase2.d92.licensed_role_oracle.paired_query_evaluation.v1"


class D92RoleOracleQueryEvaluationError(ValueError):
    """Raised when paired D92/Role-Oracle evidence loses closure."""


def _model_state_sha256(state: Any) -> str:
    metadata = {
        "schema": str(state.schema),
        "classes": [str(value) for value in state.classes],
        "old_class_count": int(state.old_class_count),
        "covariance_policy": str(state.covariance_policy),
    }
    digest = hashlib.sha256(_canonical_json_bytes(metadata))
    for name in (
        "log_diag_fp32",
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
        "coef_fp32",
        "intercept_fp32",
    ):
        value = np.ascontiguousarray(getattr(state, name))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _read_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != (
            "query_tokens",
            "scenarios",
            "predicted_class_handles",
        ):
            raise D92RoleOracleQueryEvaluationError(
                "baseline prediction artifact schema drift"
            )
        result = {name: archive[name].astype(str) for name in archive.files}
    if len({len(value) for value in result.values()}) != 1:
        raise D92RoleOracleQueryEvaluationError(
            "baseline prediction artifact row alignment drift"
        )
    return result


def _readonly_snapshot(path: Path, *, name: str) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise D92RoleOracleQueryEvaluationError(f"{name} cannot be opened safely") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o222:
            raise D92RoleOracleQueryEvaluationError(
                f"{name} must be a read-only regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    return raw, hashlib.sha256(raw).hexdigest()


def _read_role_capsule(path: Path, *, expected_sha256: str) -> Mapping[str, Any]:
    raw, actual_sha256 = _readonly_snapshot(path, name="role capsule")
    if actual_sha256 != expected_sha256:
        raise D92RoleOracleQueryEvaluationError("role capsule SHA256 drift")
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise D92RoleOracleQueryEvaluationError("role capsule is unreadable") from exc
    if not isinstance(payload, dict):
        raise D92RoleOracleQueryEvaluationError("role capsule must be a mapping")
    return payload


def _verify_baseline_commit(
    root: Path, *, state: str, expected_state: Mapping[str, Any]
) -> None:
    receipt_path = root / state / "execution_receipt.json"
    prediction_path = root / state / "prediction_artifact.npz"
    commit_path = root / state / "COMMIT.json"
    receipt_raw, receipt_sha = _readonly_snapshot(
        receipt_path, name=f"{state} baseline receipt"
    )
    prediction_raw, prediction_sha = _readonly_snapshot(
        prediction_path, name=f"{state} baseline prediction"
    )
    commit_raw, commit_sha = _readonly_snapshot(
        commit_path, name=f"{state} baseline COMMIT"
    )
    try:
        receipt = json.loads(receipt_raw.decode("utf-8-sig"))
        commit = json.loads(commit_raw.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise D92RoleOracleQueryEvaluationError(
            f"{state} baseline COMMIT closure is unreadable"
        ) from exc
    members = commit.get("members", [])
    expected_members = {
        "execution_receipt.json": (receipt_sha, len(receipt_raw)),
        "prediction_artifact.npz": (prediction_sha, len(prediction_raw)),
    }
    actual_members = {
        item.get("relative_path"): (item.get("sha256"), int(item.get("size_bytes", -1)))
        for item in members
        if isinstance(item, dict)
    }
    if (
        receipt.get("schema")
        != "cvs.phase2.diag_cosine_exploration_receipt.v1"
        or commit.get("schema")
        != "cvs.phase2.diag_cosine_exploration_commit.v1"
        or commit.get("execution_receipt_sha256") != receipt_sha
        or commit.get("prediction_artifact_sha256") != prediction_sha
        or any(actual_members.get(name) != value for name, value in expected_members.items())
        or expected_state.get("prediction_artifact_sha256") != prediction_sha
        or expected_state.get("execution_receipt_sha256") != receipt_sha
        or expected_state.get("commit_sha256") != commit_sha
    ):
        raise D92RoleOracleQueryEvaluationError(
            f"{state} baseline COMMIT closure drift"
        )


def _publish_shared_scores(
    destination: Path,
    *,
    state: str,
    classes: tuple[str, ...],
    model_state_sha256_by_row: np.ndarray,
    baseline: Mapping[str, np.ndarray],
    scores: np.ndarray,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    score_sha256 = _write_npz_new(
        destination / "shared_score_matrix.npz",
        query_tokens=np.asarray(baseline["query_tokens"]).astype(str),
        scenarios=np.asarray(baseline["scenarios"]).astype(str),
        registered_class_handles=np.asarray(classes).astype(str),
        model_state_sha256=np.asarray(model_state_sha256_by_row).astype(str),
        scores=np.asarray(scores, dtype=np.float32),
    )
    state_hashes = sorted(set(np.asarray(model_state_sha256_by_row).astype(str).tolist()))
    receipt = {
        "schema": "cvs.phase2.d92.licensed_role_oracle.shared_score.v1",
        "state": state,
        "baseline_candidate": CANDIDATE_D92,
        "licensed_candidate": CANDIDATE_D92_ROLE_ORACLE,
        "scenario_score_pass_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "score_compute_count_per_query": 1,
        "query_reforward_count_for_oracle": 0,
        "oracle_refit_count": 0,
        "registered_class_count": len(classes),
        "query_count": int(len(scores)),
        "shared_score_matrix_sha256": score_sha256,
        "model_state_set_sha256": hashlib.sha256(
            _canonical_json_bytes(state_hashes)
        ).hexdigest(),
    }
    receipt_sha256 = _write_json_new(destination / "receipt.json", receipt)
    members = [
        {
            "relative_path": value.name,
            "sha256": _sha256_file(value),
            "size_bytes": value.stat().st_size,
        }
        for value in sorted(destination.iterdir(), key=lambda item: item.name)
    ]
    commit = {
        "schema": "cvs.phase2.d92.licensed_role_oracle.shared_score_commit.v1",
        "members": members,
        "artifact_root_sha256": hashlib.sha256(
            _canonical_json_bytes(members)
        ).hexdigest(),
        "receipt_sha256": receipt_sha256,
        "shared_score_matrix_sha256": score_sha256,
        "model_state_set_sha256": receipt["model_state_set_sha256"],
    }
    commit_sha256 = _write_json_new(destination / "COMMIT.json", commit)
    return {
        **receipt,
        "receipt_sha256": receipt_sha256,
        "commit_sha256": commit_sha256,
        "output_root": str(destination),
    }


def _publish_oracle_prediction(
    destination: Path,
    *,
    state: str,
    baseline: Mapping[str, np.ndarray],
    predictions: np.ndarray,
    baseline_state: Mapping[str, Any],
    shared_score: Mapping[str, Any],
    role_capsule_sha256: str,
    role_capsule_used: bool,
    decision_audit: Mapping[str, Any],
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    prediction_sha256 = _write_npz_new(
        destination / "prediction_artifact.npz",
        query_tokens=np.asarray(baseline["query_tokens"]).astype(str),
        scenarios=np.asarray(baseline["scenarios"]).astype(str),
        predicted_class_handles=np.asarray(predictions).astype(str),
    )
    receipt = {
        "schema": "cvs.phase2.d92.licensed_role_oracle.execution_receipt.v1",
        "status": LICENSE_STATUS,
        "claim_scope": "licensed_role_oracle_upper_bound_only",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "formal_protocol_valid": False,
        "promotion_eligible": False,
        "candidate": CANDIDATE_D92_ROLE_ORACLE,
        "baseline_candidate": CANDIDATE_D92,
        "registration_state": state,
        "licensed_protocol_deviation": "query_old_new_role_oracle_only",
        "query_decision_policy": (
            "per_sample_role_partition_all_registered_classes_within_role"
        ),
        "query_role_oracle_access": bool(role_capsule_used),
        "query_tx_id_access": False,
        "query_truth_label_access": False,
        "query_true_batch_class_count_access": False,
        "query_class_quota_access": False,
        "query_batch_global_assignment": False,
        "query_query_graph_used": False,
        "oracle_refit_count": 0,
        "oracle_query_reforward_count": 0,
        "baseline_prediction_sha256": baseline_state[
            "prediction_artifact_sha256"
        ],
        "baseline_commit_sha256": baseline_state["commit_sha256"],
        "shared_score_matrix_sha256": shared_score[
            "shared_score_matrix_sha256"
        ],
        "shared_score_commit_sha256": shared_score["commit_sha256"],
        "role_capsule_sha256": role_capsule_sha256,
        "decision_audit": dict(decision_audit),
        "prediction_artifact_sha256": prediction_sha256,
    }
    receipt_sha256 = _write_json_new(destination / "execution_receipt.json", receipt)
    members = [
        {
            "relative_path": value.name,
            "sha256": _sha256_file(value),
            "size_bytes": value.stat().st_size,
        }
        for value in sorted(destination.iterdir(), key=lambda item: item.name)
    ]
    commit = {
        "schema": "cvs.phase2.d92.licensed_role_oracle.commit.v1",
        "members": members,
        "artifact_root_sha256": hashlib.sha256(
            _canonical_json_bytes(members)
        ).hexdigest(),
        "execution_receipt_sha256": receipt_sha256,
        "prediction_artifact_sha256": prediction_sha256,
        "shared_score_matrix_sha256": shared_score[
            "shared_score_matrix_sha256"
        ],
    }
    commit_sha256 = _write_json_new(destination / "COMMIT.json", commit)
    return {
        "registration_state": state,
        "prediction_artifact_sha256": prediction_sha256,
        "execution_receipt_sha256": receipt_sha256,
        "commit_sha256": commit_sha256,
        "output_root": str(destination),
        "shared_score_matrix_sha256": shared_score[
            "shared_score_matrix_sha256"
        ],
    }


def run_d92_role_oracle_query_evaluation(
    *,
    role_capsule_path: str | Path | None = None,
    expected_role_capsule_sha256: str | None = None,
    role_capsule_factory: (
        Callable[[], tuple[str | Path, Mapping[str, Any]]] | None
    ) = None,
    output_root: str | Path,
    **d92_kwargs: Any,
) -> dict[str, Any]:
    """Publish fresh paired D92 and role-only Oracle predictions."""

    from cvsrffi import stage2_d81_query_evaluation as d81_eval

    output = Path(output_root)
    if output.exists() and (
        not output.is_dir() or output.is_symlink() or any(output.iterdir())
    ):
        raise D92RoleOracleQueryEvaluationError(
            f"paired evaluation output is not an empty directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    if (role_capsule_path is None) == (role_capsule_factory is None):
        raise D92RoleOracleQueryEvaluationError(
            "provide exactly one role capsule path or delayed factory"
        )
    baseline_root = output / "baseline"
    baseline_root.mkdir()

    captured: list[dict[str, Any]] = []
    original_predict = d81_eval.predict_d42_unified_shrinkage_lda

    def capture_predict(state: Any, features: np.ndarray) -> np.ndarray:
        scores = score_d42_unified_shrinkage_lda(state, features)
        classes = tuple(str(value) for value in np.asarray(state.classes).tolist())
        predictions = np.asarray(classes)[np.argmax(scores, axis=1)]
        captured.append(
            {
                "classes": classes,
                "scores": np.asarray(scores, dtype=np.float32).copy(),
                "model_state_sha256": _model_state_sha256(state),
                "predictions": np.asarray(predictions).astype(str),
            }
        )
        return predictions

    try:
        d81_eval.predict_d42_unified_shrinkage_lda = capture_predict
        baseline_result = run_d92_query_evaluation(
            output_root=baseline_root, **d92_kwargs
        )
    finally:
        d81_eval.predict_d42_unified_shrinkage_lda = original_predict

    for state in ("before", "after"):
        _verify_baseline_commit(
            baseline_root,
            state=state,
            expected_state=baseline_result["states"][state],
        )

    if len(captured) != 2 * len(FORMAL_LEO_WEAK_SCENARIOS):
        raise D92RoleOracleQueryEvaluationError(
            "D92 paired score capture count drift"
        )
    grouped = {"before": captured[0::2], "after": captured[1::2]}
    baselines = {
        state: _read_prediction(
            baseline_root / state / "prediction_artifact.npz"
        )
        for state in ("before", "after")
    }
    score_rows: dict[str, np.ndarray] = {}
    classes_by_state: dict[str, tuple[str, ...]] = {}
    shared_scores: dict[str, dict[str, Any]] = {}
    for state in ("before", "after"):
        records = grouped[state]
        classes = records[0]["classes"]
        if any(record["classes"] != classes for record in records):
            raise D92RoleOracleQueryEvaluationError(
                f"{state} registered class order drift across scenarios"
            )
        matrix = np.concatenate([record["scores"] for record in records], axis=0)
        state_hash_by_row = np.concatenate(
            [
                np.asarray([record["model_state_sha256"]] * len(record["scores"]))
                for record in records
            ]
        )
        captured_predictions = np.concatenate(
            [record["predictions"] for record in records]
        ).astype(str)
        baseline_predictions = baselines[state]["predicted_class_handles"]
        if (
            matrix.shape[0] != len(baseline_predictions)
            or not np.array_equal(captured_predictions, baseline_predictions)
        ):
            raise D92RoleOracleQueryEvaluationError(
                f"{state} captured scores do not reproduce committed baseline"
            )
        score_rows[state] = matrix
        classes_by_state[state] = classes
        shared_scores[state] = _publish_shared_scores(
            output / "shared_scores" / state,
            state=state,
            classes=classes,
            model_state_sha256_by_row=state_hash_by_row,
            baseline=baselines[state],
            scores=matrix,
        )

    # The baseline and shared-score COMMIT files now exist.  Only after that
    # boundary may the licensed decoder open the role-only capsule.
    projection_receipt: Mapping[str, Any] | None = None
    if role_capsule_factory is not None:
        supplied_path, projection_receipt = role_capsule_factory()
        role_path = Path(supplied_path)
        if (
            set(projection_receipt)
            != {"path", "source_truth_sha256", "capsule_sha256", "readonly"}
            or projection_receipt.get("path") != str(role_path)
            or projection_receipt.get("readonly") is not True
            or not isinstance(projection_receipt.get("source_truth_sha256"), str)
            or len(str(projection_receipt.get("source_truth_sha256"))) != 64
        ):
            raise D92RoleOracleQueryEvaluationError(
                "role capsule projection receipt drift"
            )
        expected_role_capsule_sha256 = str(projection_receipt["capsule_sha256"])
    else:
        role_path = Path(role_capsule_path)
        expected_role_capsule_sha256 = str(expected_role_capsule_sha256 or "")
    if len(str(expected_role_capsule_sha256)) != 64:
        raise D92RoleOracleQueryEvaluationError("expected role capsule SHA256 missing")
    role_capsule = _read_role_capsule(
        role_path, expected_sha256=str(expected_role_capsule_sha256)
    )
    role_capsule_sha256 = str(expected_role_capsule_sha256)
    old_classes = classes_by_state["before"]
    all_classes = classes_by_state["after"]
    if all_classes[: len(old_classes)] != old_classes:
        raise D92RoleOracleQueryEvaluationError(
            "D92 old registry is not the final registry prefix"
        )
    decoded = decode_d92_licensed_role_oracle(
        score_rows["after"],
        all_classes,
        old_classes,
        baselines["after"]["query_tokens"].tolist(),
        role_capsule,
    )
    if tuple(decoded["baseline_predictions"]) != tuple(
        baselines["after"]["predicted_class_handles"].tolist()
    ):
        raise D92RoleOracleQueryEvaluationError(
            "licensed decoder baseline does not match committed D92 baseline"
        )

    oracle_root = output / "oracle"
    oracle_root.mkdir()
    before_audit = {
        **dict(decoded["audit"]),
        "query_role_oracle_access": False,
        "role_capsule_used": False,
        "before_prediction_bit_exact_to_baseline": True,
    }
    oracle_states = {
        "before": _publish_oracle_prediction(
            oracle_root / "before",
            state="before",
            baseline=baselines["before"],
            predictions=baselines["before"]["predicted_class_handles"],
            baseline_state=baseline_result["states"]["before"],
            shared_score=shared_scores["before"],
            role_capsule_sha256=role_capsule_sha256,
            role_capsule_used=False,
            decision_audit=before_audit,
        ),
        "after": _publish_oracle_prediction(
            oracle_root / "after",
            state="after",
            baseline=baselines["after"],
            predictions=np.asarray(decoded["role_oracle_predictions"]).astype(str),
            baseline_state=baseline_result["states"]["after"],
            shared_score=shared_scores["after"],
            role_capsule_sha256=role_capsule_sha256,
            role_capsule_used=True,
            decision_audit=decoded["audit"],
        ),
    }
    paired = {
        "schema": SCHEMA,
        "status": LICENSE_STATUS,
        "claim_scope": "licensed_role_oracle_upper_bound_only",
        "formal_protocol_valid": False,
        "promotion_eligible": False,
        "candidate": CANDIDATE_D92_ROLE_ORACLE,
        "baseline_candidate": CANDIDATE_D92,
        "receiver": baseline_result["receiver"],
        "seed": baseline_result["seed"],
        "k_shot": baseline_result["k_shot"],
        "new_class_count": baseline_result["new_class_count"],
        "licensed_protocol_deviation": "query_old_new_role_oracle_only",
        "role_capsule_sha256": role_capsule_sha256,
        "role_capsule_projection": (
            dict(projection_receipt) if projection_receipt is not None else None
        ),
        "baseline_states": baseline_result["states"],
        "oracle_states": oracle_states,
        "shared_scores": shared_scores,
        "resource": {
            **dict(baseline_result["resource"]),
            "oracle_additional_fit_count": 0,
            "oracle_additional_backbone_forward_count": 0,
            "oracle_additional_score_matrix_compute_count": 0,
            "oracle_persistent_model_state_bytes": 0,
        },
    }
    paired_sha256 = _write_json_new(output / "paired_receipt.json", paired)
    return {**paired, "paired_receipt_sha256": paired_sha256, "output_root": str(output)}


__all__ = [
    "CANDIDATE_D92_ROLE_ORACLE",
    "D92RoleOracleQueryEvaluationError",
    "LICENSE_STATUS",
    "SCHEMA",
    "run_d92_role_oracle_query_evaluation",
]
