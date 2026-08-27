"""Independent scorer for sealed CVS Phase2 Stage2-B/C predictions.

The scorer accepts only the production single-container ``.cvspred`` artifact
and delegates every prediction-container integrity, immutability, member, and
detached-seal check to :func:`stage2_prediction_artifact.verify_prediction_artifact`.
It opens scorer-side truth only after that verification succeeds.  This module
has no dataset, training, Torch, predictor-runtime, or legacy-runner imports.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np

from cvsrffi.stage2_prediction_artifact import (
    NPZ_FIELD_ALLOWLIST,
    PHASE2_FULL_CONTRACT,
    PredictionArtifactError,
    verify_prediction_artifact,
)


FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
PREDICTION_NPZ_MEMBERS = tuple(NPZ_FIELD_ALLOWLIST)
SCORING_MANIFEST_SCHEMA = "cvs.phase2.scoring_sidecar_manifest.v2"
TRUTH_SIDECAR_SCHEMA = "cvs.phase2.query_truth_sidecar.v3"
FORMAL_ROWS_SCHEMA = "cvs.phase2.formal_metric_rows.v1"
FORMAL_PREDICTIONS_SCHEMA = "cvs.phase2.formal_scored_predictions.v1"
SCORING_RECEIPT_SCHEMA = "cvs.phase2.scoring_receipt.v1"

SCORING_MANIFEST_KEYS = {
    "schema",
    "predictor_package_root_sha256",
    "predictor_package_seal_sha256",
    "truth_sidecar_json",
    "truth_sidecar_sha256",
    "scorer_output_must_not_feed_predictor",
}
TRUTH_TOP_LEVEL_KEYS = {"schema", "stage", "receiver", "seed", "rows"}
TRUTH_ROW_REQUIRED_KEYS = {
    "scenario",
    "query_token",
    "true_class_index",
    "true_class_handle",
    "transmitter_label",
    "evaluation_role",
    "receiver_label",
}
TRUTH_ROW_OPTIONAL_KEYS = {"day_label", "signal_label", "physical_sample_id"}

SHA256_RE = re.compile(r"[0-9a-f]{64}")
QUERY_TOKEN_RE = re.compile(r"qid_[0-9a-f]{32,64}")
CLASS_HANDLE_RE = re.compile(r"cls_[0-9a-f]{32,64}")


class Stage2ScoringError(ValueError):
    """Raised when prediction verification, truth validation, or joining fails."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def _exact_object(value: Any, keys: set[str], *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise Stage2ScoringError(f"{context} exact schema drift")
    return value


def _relative_leaf(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise Stage2ScoringError(f"{context} must be a POSIX-relative leaf path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or len(parsed.parts) != 1 or parsed.name in {".", ".."}:
        raise Stage2ScoringError(f"unsafe {context}")
    return parsed.as_posix()


def _regular_file(path: Path, *, context: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Stage2ScoringError(f"missing {context}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise Stage2ScoringError(f"{context} must be a regular non-symlink file")
    return path


@contextmanager
def _open_regular_same_fd(path: Path, *, context: str) -> Iterator[BinaryIO]:
    """Open scorer metadata without following a final symlink and retain one fd."""

    _regular_file(path, context=context)
    before = path.stat(follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage2ScoringError(f"cannot open {context}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise Stage2ScoringError(f"opened {context} is not a regular file")
        if (before.st_dev, before.st_ino, before.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise Stage2ScoringError(f"{context} identity changed before open")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(descriptor)


def _hash_handle(handle: BinaryIO) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    handle.seek(0)
    return digest.hexdigest(), size


def _load_json_handle(handle: BinaryIO, *, context: str) -> dict[str, Any]:
    handle.seek(0)
    try:
        payload = json.loads(handle.read().decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage2ScoringError(f"invalid JSON for {context}") from exc
    finally:
        handle.seek(0)
    if not isinstance(payload, dict):
        raise Stage2ScoringError(f"{context} JSON root must be an object")
    return payload


def _validate_hash_value(actual: str, expected: str, *, context: str) -> str:
    if not _is_sha256(expected):
        raise Stage2ScoringError(f"invalid expected SHA256 for {context}")
    if actual != expected.lower():
        raise Stage2ScoringError(f"{context} detached hash mismatch")
    return actual


def _normalize_stage(value: str) -> str:
    mapping = {
        "Stage2-A": "stage2a",
        "Stage2-B": "stage2b",
        "Stage2-C": "stage2c",
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise Stage2ScoringError(f"unsupported prediction stage: {value!r}") from exc


def load_verified_sealed_prediction(
    prediction_artifact_path: str | Path,
    *,
    expected_prediction_artifact_sha256: str,
    expected_prediction_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Verify the production ``.cvspred`` container before truth is opened."""

    path = Path(prediction_artifact_path)
    if path.suffix != ".cvspred":
        raise Stage2ScoringError("prediction artifact must use the .cvspred suffix")
    try:
        verified = verify_prediction_artifact(
            path,
            expected_artifact_sha256=expected_prediction_artifact_sha256,
            expected_seal_sha256=expected_prediction_seal_sha256,
        )
    except PredictionArtifactError as exc:
        raise Stage2ScoringError(
            f"prediction artifact verification failed: {exc}"
        ) from exc
    manifest = verified["manifest"]
    arrays = verified["arrays"]
    for name in (
        "candidate_after",
        "candidate_before",
        "identity_after",
        "identity_before",
        "direct",
    ):
        if any(
            CLASS_HANDLE_RE.fullmatch(value) is None
            for value in np.asarray(arrays[name]).astype(str).tolist()
        ):
            raise Stage2ScoringError(
                f"prediction artifact contains a non-opaque class handle: {name}"
            )
    scenario_values = np.asarray(arrays["scenarios"]).astype(str).tolist()
    scenario_runs: list[str] = []
    for value in scenario_values:
        if not scenario_runs or scenario_runs[-1] != value:
            scenario_runs.append(value)
    if tuple(scenario_runs) != FORMAL_LEO_WEAK_SCENARIOS:
        raise Stage2ScoringError(
            "prediction scenario sequence must exactly equal the three formal LEO weak scenarios"
        )
    query_tokens = np.asarray(arrays["query_tokens"]).astype(str)
    scenario_array = np.asarray(scenario_values)
    token_sets = [
        set(query_tokens[scenario_array == scenario].tolist())
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    ]
    if any(not values for values in token_sets) or any(
        token_sets[left] & token_sets[right]
        for left in range(len(token_sets))
        for right in range(left + 1, len(token_sets))
    ):
        raise Stage2ScoringError(
            "the three formal scenarios must contain pairwise-disjoint query_token sets"
        )
    scenarios = list(FORMAL_LEO_WEAK_SCENARIOS)
    binding = {
        "stage": _normalize_stage(manifest["stage"]),
        "row_id": manifest["row_id"],
        "receiver": manifest["receiver"],
        "k_shot": manifest["k_shot"],
        "candidate_lock_sha256": manifest["candidate_lock_sha256"],
        "predictor_package_root_sha256": manifest["package_root_sha256"],
        "predictor_package_seal_sha256": manifest["package_seal_sha256"],
        "scenarios": scenarios,
        "resource_receipt": manifest["resource_receipt"],
        "adapter_resource_verification": verified[
            "adapter_resource_verification"
        ],
    }
    adapter_resource = binding["adapter_resource_verification"]
    expected_adapter_resource = {
        "status": "NOT_PROVABLE_FROM_PREDICTION_ARTIFACT",
        "reason_code": "ADAPTER_MATRIX_NOT_EMBEDDED",
        "adapter_matrix_embedded": False,
        "trainable_parameter_count_verified": False,
        "persistent_state_bytes_verified": False,
        "formal_adapter_resource_claim_allowed": False,
    }
    if adapter_resource != expected_adapter_resource:
        raise Stage2ScoringError(
            "prediction artifact cannot prove adapter resources and must fail closed"
        )
    audit = {
        "prediction_artifact": verified["path"],
        "prediction_artifact_sha256": verified["artifact_sha256"],
        "prediction_payload_sha256": verified["payload_sha256"],
        "prediction_manifest_sha256": verified["manifest_sha256"],
        "prediction_seal_sha256": verified["seal_sha256"],
        "prediction_resource_receipt_sha256": verified[
            "resource_receipt_sha256"
        ],
        "prediction_immutable_state": verified["immutable_state"],
    }
    return binding, arrays, audit


def load_verified_scoring_sidecar(
    scoring_manifest_path: str | Path,
    *,
    expected_scoring_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load scorer-only truth under a detached scoring-manifest trust root."""

    manifest_path = Path(scoring_manifest_path)
    with _open_regular_same_fd(manifest_path, context="scoring manifest") as handle:
        actual_manifest_sha256, _manifest_size = _hash_handle(handle)
        manifest_sha256 = _validate_hash_value(
            actual_manifest_sha256,
            expected_scoring_manifest_sha256,
            context="scoring manifest",
        )
        manifest = _exact_object(
            _load_json_handle(handle, context="scoring manifest"),
            SCORING_MANIFEST_KEYS,
            context="scoring manifest",
        )
    if manifest["schema"] != SCORING_MANIFEST_SCHEMA:
        raise Stage2ScoringError("scoring manifest schema drift")
    if manifest["scorer_output_must_not_feed_predictor"] is not True:
        raise Stage2ScoringError("scorer feedback guard missing")
    for field in (
        "predictor_package_root_sha256",
        "predictor_package_seal_sha256",
        "truth_sidecar_sha256",
    ):
        if not _is_sha256(manifest[field]):
            raise Stage2ScoringError(f"invalid scoring manifest SHA256: {field}")
    truth_name = _relative_leaf(
        manifest["truth_sidecar_json"], context="truth sidecar path"
    )
    truth_path = manifest_path.parent / truth_name
    with _open_regular_same_fd(truth_path, context="truth sidecar") as handle:
        actual_truth_sha256, _truth_size = _hash_handle(handle)
        truth_sha256 = _validate_hash_value(
            actual_truth_sha256,
            manifest["truth_sidecar_sha256"],
            context="truth sidecar",
        )
        truth = _exact_object(
            _load_json_handle(handle, context="truth sidecar"),
            TRUTH_TOP_LEVEL_KEYS,
            context="truth sidecar",
        )
    if truth["schema"] != TRUTH_SIDECAR_SCHEMA:
        raise Stage2ScoringError("truth sidecar schema drift")
    _validate_truth_rows(truth)
    return truth, manifest, {
        "scoring_manifest": str(manifest_path),
        "scoring_manifest_sha256": manifest_sha256,
        "truth_sidecar": str(truth_path),
        "truth_sidecar_sha256": truth_sha256,
    }


def _validate_truth_rows(
    truth: Mapping[str, Any],
    *,
    require_scenario: bool = True,
) -> None:
    if truth["stage"] not in {"stage2a", "stage2b", "stage2c"}:
        raise Stage2ScoringError(
            "truth sidecar stage must be stage2a, stage2b, or stage2c"
        )
    if not isinstance(truth["receiver"], str) or not truth["receiver"]:
        raise Stage2ScoringError("truth sidecar receiver must be nonempty")
    if not isinstance(truth["seed"], int) or isinstance(truth["seed"], bool):
        raise Stage2ScoringError("truth sidecar seed must be an integer")
    rows = truth["rows"]
    if not isinstance(rows, list) or not rows:
        raise Stage2ScoringError("truth sidecar rows must be nonempty")
    seen_tokens: set[str] = set()
    seen_scenarios: set[str] = set()
    tx_to_role: dict[str, str] = {}
    tx_to_class: dict[str, int] = {}
    tx_to_handle: dict[str, str] = {}
    class_to_tx: dict[int, str] = {}
    handle_to_tx: dict[str, str] = {}
    class_to_handle: dict[int, str] = {}
    handle_to_class: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise Stage2ScoringError("truth sidecar row must be an object")
        keys = set(row)
        required = (
            TRUTH_ROW_REQUIRED_KEYS
            if require_scenario
            else TRUTH_ROW_REQUIRED_KEYS - {"scenario"}
        )
        allowed = required | TRUTH_ROW_OPTIONAL_KEYS
        if not required.issubset(keys) or not keys.issubset(
            allowed
        ):
            raise Stage2ScoringError("truth sidecar row exact schema drift")
        token = row["query_token"]
        if not isinstance(token, str) or QUERY_TOKEN_RE.fullmatch(token) is None:
            raise Stage2ScoringError("truth query token is not opaque")
        if token in seen_tokens:
            raise Stage2ScoringError("duplicate truth query_token")
        seen_tokens.add(token)
        if require_scenario:
            scenario = row["scenario"]
            if scenario not in FORMAL_LEO_WEAK_SCENARIOS:
                raise Stage2ScoringError("truth scenario contamination")
            seen_scenarios.add(scenario)
        role = row["evaluation_role"]
        if role not in {"target_old", "target_new"}:
            raise Stage2ScoringError("truth evaluation_role contamination")
        if row["receiver_label"] != truth["receiver"]:
            raise Stage2ScoringError("truth receiver contamination")
        tx = row["transmitter_label"]
        if not isinstance(tx, str) or not tx:
            raise Stage2ScoringError("truth transmitter_label must be nonempty")
        if tx in tx_to_role and tx_to_role[tx] != role:
            raise Stage2ScoringError("transmitter role contamination")
        tx_to_role[tx] = role
        true_class = row["true_class_index"]
        true_handle = row["true_class_handle"]
        if truth["stage"] in {"stage2a", "stage2b"} and role == "target_new":
            if true_class is not None or true_handle is not None:
                stage_name = (
                    "Stage2-A"
                    if truth["stage"] == "stage2a"
                    else "Stage2-B"
                )
                raise Stage2ScoringError(
                    f"{stage_name} target-new reference cannot have a "
                    "registered true class"
                )
            continue
        if not isinstance(true_class, int) or isinstance(true_class, bool) or true_class < 0:
            raise Stage2ScoringError("scored truth class index must be a nonnegative integer")
        if not isinstance(true_handle, str) or CLASS_HANDLE_RE.fullmatch(true_handle) is None:
            raise Stage2ScoringError("scored truth class handle must be opaque")
        day_label = row.get("day_label")
        if not isinstance(day_label, str) or not day_label.strip():
            raise Stage2ScoringError("scored truth day_label must be nonempty")
        if true_handle in handle_to_class and handle_to_class[true_handle] != true_class:
            raise Stage2ScoringError(
                "true class handle maps to multiple class indices"
            )
        if true_class in class_to_handle and class_to_handle[true_class] != true_handle:
            raise Stage2ScoringError(
                "true class index maps to multiple opaque class handles"
            )
        if tx in tx_to_class and tx_to_class[tx] != true_class:
            raise Stage2ScoringError("transmitter maps to multiple true class indices")
        if tx in tx_to_handle and tx_to_handle[tx] != true_handle:
            raise Stage2ScoringError("transmitter maps to multiple true class handles")
        if true_class in class_to_tx and class_to_tx[true_class] != tx:
            raise Stage2ScoringError("true class index maps to multiple transmitters")
        if true_handle in handle_to_tx and handle_to_tx[true_handle] != tx:
            raise Stage2ScoringError("true class handle maps to multiple transmitters")
        tx_to_class[tx] = true_class
        tx_to_handle[tx] = true_handle
        class_to_tx[true_class] = tx
        handle_to_tx[true_handle] = tx
        class_to_handle[true_class] = true_handle
        handle_to_class[true_handle] = true_class
    roles = {row["evaluation_role"] for row in rows}
    if require_scenario and seen_scenarios != set(
        FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise Stage2ScoringError(
            "truth sidecar does not cover all formal scenarios"
        )
    if "target_old" not in roles:
        raise Stage2ScoringError("truth sidecar has no target-old query")
    if truth["stage"] == "stage2c" and "target_new" not in roles:
        raise Stage2ScoringError("Stage2-C truth sidecar has no target-new query")


def _accuracy(predictions: np.ndarray, truth: np.ndarray) -> float:
    if len(truth) == 0:
        raise Stage2ScoringError("cannot compute accuracy on an empty group")
    return float(np.mean(predictions == truth))


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right == 0.0 else float(2.0 * left * right / (left + right))


def _observed_p95(values: np.ndarray) -> int:
    try:
        return int(np.percentile(values, 95, method="higher"))
    except TypeError:
        return int(np.percentile(values, 95, interpolation="higher"))


def score_prediction_arrays(
    *,
    binding: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    truth: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact-join truth to five fixed streams and compute per-scenario rows."""

    _validate_truth_rows(truth)
    if binding["stage"] != truth["stage"]:
        raise Stage2ScoringError("prediction/truth stage mismatch")
    if binding["receiver"] != truth["receiver"]:
        raise Stage2ScoringError("prediction/truth receiver mismatch")
    truth_by_key = {
        (row["scenario"], row["query_token"]): row
        for row in truth["rows"]
    }
    prediction_tokens = np.asarray(arrays["query_tokens"]).astype(str)
    prediction_scenarios = np.asarray(arrays["scenarios"]).astype(str)
    prediction_streams = {
        name: np.asarray(arrays[name]).astype(str)
        for name in (
            "candidate_after",
            "candidate_before",
            "identity_after",
            "identity_before",
            "direct",
        )
    }
    handle_to_class_index: dict[str, int] = {}
    class_index_to_handle: dict[int, str] = {}
    for truth_row in truth["rows"]:
        true_class = truth_row["true_class_index"]
        true_handle = truth_row["true_class_handle"]
        if true_class is None:
            continue
        if (
            true_handle in handle_to_class_index
            and handle_to_class_index[true_handle] != true_class
        ) or (
            true_class in class_index_to_handle
            and class_index_to_handle[true_class] != true_handle
        ):
            raise Stage2ScoringError(
                "inconsistent opaque class handle to class index mapping"
            )
        handle_to_class_index[true_handle] = true_class
        class_index_to_handle[true_class] = true_handle
    shared_view_counts = np.asarray(arrays["shared_view_counts"], dtype=np.int64)
    formal_rows: list[dict[str, Any]] = []
    formal_predictions: list[dict[str, Any]] = []

    for scenario in binding["scenarios"]:
        indices = np.flatnonzero(prediction_scenarios == scenario)
        scenario_tokens = prediction_tokens[indices].tolist()
        truth_tokens = {
            token
            for truth_scenario, token in truth_by_key
            if truth_scenario == scenario
        }
        if set(scenario_tokens) != truth_tokens:
            missing = sorted(truth_tokens - set(scenario_tokens))
            extra = sorted(set(scenario_tokens) - truth_tokens)
            raise Stage2ScoringError(
                f"prediction/truth token mismatch for {scenario}: "
                f"missing={missing}:extra={extra}"
            )
        ordered_truth = [
            truth_by_key[(scenario, token)]
            for token in scenario_tokens
        ]
        old_mask = np.asarray(
            [row["evaluation_role"] == "target_old" for row in ordered_truth],
            dtype=bool,
        )
        new_mask = np.asarray(
            [row["evaluation_role"] == "target_new" for row in ordered_truth],
            dtype=bool,
        )
        scored_truth = np.asarray(
            [
                "" if row["true_class_handle"] is None else row["true_class_handle"]
                for row in ordered_truth
            ]
        ).astype(str)
        old_truth = scored_truth[old_mask]
        streams = {name: values[indices] for name, values in prediction_streams.items()}
        scenario_view_counts = shared_view_counts[indices]
        for name in streams:
            if any(
                CLASS_HANDLE_RE.fullmatch(value) is None
                for value in streams[name][old_mask].tolist()
            ):
                raise Stage2ScoringError(
                    f"missing opaque {name} prediction on target-old query"
                )
        if binding["stage"] == "stage2c":
            for name in ("candidate_after", "identity_after"):
                if any(
                    CLASS_HANDLE_RE.fullmatch(value) is None
                    for value in streams[name][new_mask].tolist()
                ):
                    raise Stage2ScoringError(
                        f"missing opaque {name} prediction on Stage2-C target-new query"
                    )

        candidate_old_after = _accuracy(streams["candidate_after"][old_mask], old_truth)
        candidate_old_before = _accuracy(streams["candidate_before"][old_mask], old_truth)
        identity_old_after = _accuracy(streams["identity_after"][old_mask], old_truth)
        identity_old_before = _accuracy(streams["identity_before"][old_mask], old_truth)
        direct_old = _accuracy(streams["direct"][old_mask], old_truth)
        candidate_class_acc: dict[str, float] = {}
        candidate_class_count: dict[str, int] = {}
        old_transmitters = sorted(
            {
                row["transmitter_label"]
                for row in ordered_truth
                if row["evaluation_role"] == "target_old"
            }
        )
        for tx in old_transmitters:
            mask = np.asarray(
                [
                    row["evaluation_role"] == "target_old"
                    and row["transmitter_label"] == tx
                    for row in ordered_truth
                ],
                dtype=bool,
            )
            candidate_class_acc[tx] = _accuracy(
                streams["candidate_after"][mask], scored_truth[mask]
            )
            candidate_class_count[tx] = int(np.sum(mask))
        min_old_class_acc = min(candidate_class_acc.values())

        if binding["stage"] == "stage2c":
            seen_new_acc: float | None = _accuracy(
                streams["candidate_after"][new_mask], scored_truth[new_mask]
            )
            h_old_new: float | None = _harmonic(candidate_old_after, seen_new_acc)
        else:
            seen_new_acc = None
            h_old_new = None

        candidate_forgetting = candidate_old_before - candidate_old_after
        identity_forgetting = identity_old_before - identity_old_after
        is_k1 = int(binding["k_shot"]) == 1
        formal_rows.append(
            {
                "row_id": binding["row_id"],
                "stage": binding["stage"],
                "receiver_label": binding["receiver"],
                "scenario": scenario,
                "k_shot": int(binding["k_shot"]),
                "candidate_lock_sha256": binding["candidate_lock_sha256"],
                "predictor_package_root_sha256": binding[
                    "predictor_package_root_sha256"
                ],
                "query_count": len(indices),
                "target_old_query_count": int(np.sum(old_mask)),
                "target_new_query_count": int(np.sum(new_mask)),
                "old_acc": candidate_old_after,
                "old_acc_before_increment": candidate_old_before,
                "old_acc_after_increment": candidate_old_after,
                "min_old_class_acc": min_old_class_acc,
                "seen_new_acc": seen_new_acc,
                "H_old_new": h_old_new,
                "candidate_average_forgetting": candidate_forgetting,
                "candidate_old_adaptation_gain": -candidate_forgetting,
                "identity_old_acc_before_increment": identity_old_before,
                "identity_old_acc_after_increment": identity_old_after,
                "identity_average_forgetting": identity_forgetting,
                "identity_old_adaptation_gain": -identity_forgetting,
                "direct_adv3b02_old_acc": direct_old,
                "delta_vs_direct_ADV3B02_K1": (
                    candidate_old_after - direct_old if is_k1 else None
                ),
                "delta_vs_identity_K1": (
                    candidate_old_after - identity_old_after if is_k1 else None
                ),
                "identity_delta_vs_direct_ADV3B02_K1": (
                    identity_old_after - direct_old if is_k1 else None
                ),
                "candidate_old_class_acc": candidate_class_acc,
                "candidate_old_class_count": candidate_class_count,
                "shared_view_count_mean": float(np.mean(scenario_view_counts)),
                "shared_view_count_p95": _observed_p95(scenario_view_counts),
                "view1_count": int(np.sum(scenario_view_counts == 1)),
                "view3_count": int(np.sum(scenario_view_counts == 3)),
                "view5_count": int(np.sum(scenario_view_counts == 5)),
                **PHASE2_FULL_CONTRACT,
            }
        )

        for position, truth_row in enumerate(ordered_truth):
            true_class = truth_row["true_class_index"]
            scored = true_class is not None
            predicted_class: int | None = None
            if scored:
                candidate_after_handle = str(streams["candidate_after"][position])
                try:
                    predicted_class = handle_to_class_index[candidate_after_handle]
                except KeyError as exc:
                    raise Stage2ScoringError(
                        "candidate_after class handle is absent from joined truth mapping"
                    ) from exc
            result: dict[str, Any] = {
                "row_id": binding["row_id"],
                "stage": binding["stage"],
                "receiver_label": truth_row["receiver_label"],
                "scenario": scenario,
                "query_token": scenario_tokens[position],
                "evaluation_role": truth_row["evaluation_role"],
                "transmitter_label": truth_row["transmitter_label"],
                "true_class_index": true_class,
                "predicted_class_index": predicted_class,
                "true_class_handle": truth_row["true_class_handle"],
                "day_label": truth_row.get("day_label"),
                "candidate_lock_sha256": binding["candidate_lock_sha256"],
                "shared_view_count": int(scenario_view_counts[position]),
            }
            for name in streams:
                predicted = str(streams[name][position])
                result[name] = predicted
                result[f"{name}_correct"] = (
                    int(predicted == truth_row["true_class_handle"]) if scored else None
                )
            formal_predictions.append(result)

    return formal_rows, formal_predictions


def score_sealed_prediction(
    prediction_artifact_path: str | Path,
    scoring_manifest_path: str | Path,
    *,
    expected_prediction_artifact_sha256: str,
    expected_prediction_seal_sha256: str,
    expected_scoring_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify one production artifact, then score each contained scenario."""

    binding, arrays, prediction_audit = load_verified_sealed_prediction(
        prediction_artifact_path,
        expected_prediction_artifact_sha256=expected_prediction_artifact_sha256,
        expected_prediction_seal_sha256=expected_prediction_seal_sha256,
    )
    # Truth is intentionally opened only after the .cvspred container verifies.
    truth, scoring_manifest, scoring_audit = load_verified_scoring_sidecar(
        scoring_manifest_path,
        expected_scoring_manifest_sha256=expected_scoring_manifest_sha256,
    )
    if (
        binding["predictor_package_root_sha256"]
        != scoring_manifest["predictor_package_root_sha256"]
    ):
        raise Stage2ScoringError("prediction/scoring package root hash mismatch")
    if (
        binding["predictor_package_seal_sha256"]
        != scoring_manifest["predictor_package_seal_sha256"]
    ):
        raise Stage2ScoringError("prediction/scoring package seal hash mismatch")
    rows, predictions = score_prediction_arrays(
        binding=binding, arrays=arrays, truth=truth
    )
    rows_payload = {"schema": FORMAL_ROWS_SCHEMA, "rows": rows}
    predictions_payload = {
        "schema": FORMAL_PREDICTIONS_SCHEMA,
        "predictions": predictions,
    }
    receipt = {
        "schema": SCORING_RECEIPT_SCHEMA,
        "status": "PASS",
        "row_id": binding["row_id"],
        "stage": binding["stage"],
        "receiver": binding["receiver"],
        "k_shot": int(binding["k_shot"]),
        "candidate_lock_sha256": binding["candidate_lock_sha256"],
        "predictor_package_root_sha256": binding[
            "predictor_package_root_sha256"
        ],
        "predictor_package_seal_sha256": binding[
            "predictor_package_seal_sha256"
        ],
        "prediction_artifact_sha256": prediction_audit[
            "prediction_artifact_sha256"
        ],
        "prediction_payload_sha256": prediction_audit["prediction_payload_sha256"],
        "prediction_manifest_sha256": prediction_audit[
            "prediction_manifest_sha256"
        ],
        "prediction_seal_sha256": prediction_audit["prediction_seal_sha256"],
        "prediction_resource_receipt_sha256": prediction_audit[
            "prediction_resource_receipt_sha256"
        ],
        "prediction_immutable_state": prediction_audit[
            "prediction_immutable_state"
        ],
        "prediction_resource_receipt": binding["resource_receipt"],
        "adapter_resource_verification": binding[
            "adapter_resource_verification"
        ],
        "formal_adapter_resource_claim_allowed": False,
        "scoring_manifest_sha256": scoring_audit["scoring_manifest_sha256"],
        "truth_sidecar_sha256": scoring_audit["truth_sidecar_sha256"],
        "scenario_count": len(binding["scenarios"]),
        "formal_row_count": len(rows),
        "formal_prediction_count": len(predictions),
        "join_policy": "exact_scenario_query_token",
        "truth_join_after_prediction_only": True,
        "scorer_output_must_not_feed_predictor": True,
        **PHASE2_FULL_CONTRACT,
        "formal_rows_sha256": sha256_bytes(canonical_json_bytes(rows_payload) + b"\n"),
        "formal_predictions_sha256": sha256_bytes(
            canonical_json_bytes(predictions_payload) + b"\n"
        ),
    }
    if not all(
        math.isfinite(value)
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    ):
        raise Stage2ScoringError("non-finite formal metric")
    return rows_payload, predictions_payload, receipt


def _open_exclusive_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    return os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")


def write_scoring_outputs_exclusive(
    *,
    formal_rows_path: str | Path,
    formal_predictions_path: str | Path,
    scoring_receipt_path: str | Path,
    formal_rows: Mapping[str, Any],
    formal_predictions: Mapping[str, Any],
    scoring_receipt: Mapping[str, Any],
) -> None:
    """Write all scorer outputs with O_EXCL; existing paths are never replaced."""

    outputs = [
        (Path(formal_rows_path), formal_rows),
        (Path(formal_predictions_path), formal_predictions),
        (Path(scoring_receipt_path), scoring_receipt),
    ]
    resolved = [path.resolve(strict=False) for path, _payload in outputs]
    if len(set(resolved)) != len(resolved):
        raise Stage2ScoringError("scorer output paths must be distinct")
    if any(path.exists() for path, _payload in outputs):
        raise FileExistsError("scorer output already exists; overwrite is forbidden")
    handles: list[tuple[Path, Any]] = []
    try:
        for path, _payload in outputs:
            handles.append((path, _open_exclusive_text(path)))
    except BaseException:
        for path, handle in handles:
            handle.close()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    try:
        for (_path, handle), (_output_path, payload) in zip(handles, outputs):
            handle.write(canonical_json_bytes(payload).decode("utf-8"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        for _path, handle in handles:
            handle.close()
