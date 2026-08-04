"""Independent truth-side scorer for the frozen NEXT-R2 proxy24 output.

The prediction runner never opens query labels.  This module is the separate
truth-side closure: it first verifies the complete 24-key/96-state prediction
capsule, manifest, state files and completion receipt, and only then opens the
pinned ``ls_label_join`` archive.  It intentionally emits the four explicit
DA/registration states and never emits a formal Target claim.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage2_next_r2_matrix as matrix


PREDICTION_CAPSULE_SCHEMA = "cvs.stage2.next_r2.prediction_capsule.v1"
STATE_RECEIPT_SCHEMA = "cvs.stage2.next_r2.proxy24.state_receipt.v1"
SEALED_MANIFEST_SCHEMA = "cvs.stage2.next_r2.proxy24.sealed_manifest.v1"
COMPLETION_SCHEMA = "cvs.stage2.next_r2.proxy24.completion.v1"
SCORE_SCHEMA = "cvs.stage2.next_r2.proxy24.score.v1"
SCORING_COMPLETION_SCHEMA = "cvs.stage2.next_r2.proxy24.scoring_completion.v1"

_STATE_NPZ_MEMBERS = (
    "query_physical_ids",
    "registered_classes",
    "scores",
    "predictions",
)
_TRUTH_MEMBERS = (
    "z_dom",
    "pre_relu",
    "receiver_ids",
    "day_ids",
    "tx_labels",
    "physical_ids",
)
_FORBIDDEN_KEYS = frozenset(
    {
        "truth",
        "truth_label",
        "query_truth",
        "query_labels",
        "query_role",
        "query_roles",
        "class_quota",
        "batch_class_count",
        "true_batch_class_counts",
    }
)


class NextR2ScoreError(ValueError):
    """Prediction closure, truth join, or metric contract drifted."""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path, *, name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise NextR2ScoreError(f"{name} must be a regular non-symlink file")
    return _sha(path.read_bytes())


def _require_sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise NextR2ScoreError(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR2ScoreError(f"{name} must be a lowercase SHA256") from error
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return matrix.canonical_bytes(value)


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return matrix.canonical_sha256(value)


def _id_root(values: Sequence[str]) -> str:
    return _sha(_canonical({"ids": tuple(values)}))


def _prediction_root(values: Sequence[str]) -> str:
    return _sha(_canonical({"predictions": tuple(values)}))


def _array_sha(value: np.ndarray) -> str:
    return _sha(np.ascontiguousarray(value).tobytes(order="C"))


def _json_load(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NextR2ScoreError(f"{name} must be a regular non-symlink JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR2ScoreError(f"{name} must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise NextR2ScoreError(f"{name} must be a JSON object")
    return value


def _reject_forbidden(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_KEYS:
                raise NextR2ScoreError(f"{name} contains forbidden field {key}")
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _regular_child(root: Path, relative: object, *, name: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise NextR2ScoreError(f"{name} must be a relative path")
    raw_path = root / relative
    if raw_path.is_symlink():
        raise NextR2ScoreError(f"{name} must not be a symlink")
    path = raw_path.resolve(strict=False)
    root_resolved = root.resolve(strict=True)
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise NextR2ScoreError(f"{name} escapes run root") from error
    if path.is_symlink() or not path.is_file():
        raise NextR2ScoreError(f"{name} must be a regular run-root file")
    return path


def _npz_bytes(value: bytes, *, members: Sequence[str], name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(value), allow_pickle=False) as archive:
            if tuple(archive.files) != tuple(members):
                raise NextR2ScoreError(f"{name} member/order drift")
            result = {member: np.asarray(archive[member]) for member in members}
    except (OSError, ValueError) as error:
        raise NextR2ScoreError(f"{name} is not a no-pickle NPZ") from error
    if any(item.dtype.hasobject for item in result.values()):
        raise NextR2ScoreError(f"{name} contains an object array")
    return result


def _strings(value: object, *, name: str, rows: int | None = None) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or (rows is not None and array.shape[0] != rows) or array.dtype.kind not in "US":
        raise NextR2ScoreError(f"{name} must be a non-object string vector")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result):
        raise NextR2ScoreError(f"{name} contains an empty value")
    return result


def _read_capsule(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    expected = _require_sha(expected_sha256, name="prediction_capsule_sha256")
    raw = path.read_bytes() if path.is_file() and not path.is_symlink() else None
    if raw is None or _sha(raw) != expected:
        raise NextR2ScoreError("prediction capsule SHA256 drift")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR2ScoreError("prediction capsule must be UTF-8 canonical JSON") from error
    if not isinstance(decoded, Mapping) or _canonical(decoded) != raw:
        raise NextR2ScoreError("prediction capsule canonical bytes drift")
    payload = dict(decoded)
    observed = payload.pop("capsule_content_sha256", None)
    if observed != _canonical_sha(payload):
        raise NextR2ScoreError("prediction capsule content SHA drift")
    if (
        payload.get("schema") != PREDICTION_CAPSULE_SCHEMA
        or not isinstance(payload.get("capsule_id"), str)
        or not payload.get("capsule_id")
        or not isinstance(payload.get("split_id"), str)
        or not payload.get("split_id")
        or payload.get("truth_opened_for_capsule_build") is not True
        or payload.get("query_labels_persisted") is not False
    ):
        raise NextR2ScoreError("prediction capsule legality header drift")
    _reject_forbidden(payload, name="prediction_capsule")
    for field in (
        "selected_iq_archive_sha256",
        "selected_iq_receipt_sha256",
        "label_join_archive_sha256",
        "physical_id_root_sha256",
        "matrix_sha256",
    ):
        _require_sha(payload.get(field), name=f"capsule.{field}")
    plan = matrix.validate_next_r2_proxy24_plan(payload.get("plan", {}))
    if payload["matrix_sha256"] != plan["matrix_sha256"]:
        raise NextR2ScoreError("prediction capsule matrix binding drift")
    raw_keys = payload.get("keys", ())
    if not isinstance(raw_keys, (tuple, list)):
        raise NextR2ScoreError("prediction capsule keys must be a sequence")
    keys = tuple(raw_keys)
    if len(keys) != matrix.OUTER_KEY_COUNT:
        raise NextR2ScoreError("prediction capsule outer-key count drift")
    for key, planned in zip(keys, plan["keys"], strict=True):
        if not isinstance(key, Mapping):
            raise NextR2ScoreError("prediction capsule key is not an object")
        if (
            key.get("outer_key_id") != planned["outer_key_id"]
            or key.get("held_receiver") != planned["held_receiver"]
            or key.get("held_class") != planned["held_class"]
            or key.get("active_k") != planned["active_k"]
            or not isinstance(key.get("registrations"), Mapping)
            or set(key.get("registrations", {})) != {"REG0", "REG1"}
        ):
            raise NextR2ScoreError("prediction capsule key/plan binding drift")
        for registration, classes in (("REG0", planned["retained_classes"]), ("REG1", planned["all_registered_classes"])):
            item = key["registrations"][registration]
            if not isinstance(item, Mapping):
                raise NextR2ScoreError("prediction capsule registration must be an object")
            if tuple(item.get("registered_classes", ())) != tuple(classes):
                raise NextR2ScoreError("prediction capsule registered class drift")
            support_labels = tuple(item.get("support_labels", ()))
            support_ids = tuple(item.get("support_physical_ids", ()))
            query_ids = tuple(item.get("query_physical_ids", ()))
            support_indices = tuple(item.get("support_indices", ()))
            query_indices = tuple(item.get("query_indices", ()))
            expected_support = len(classes) * int(planned["active_k"])
            expected_query = len(classes) * matrix.QUERY_PER_CLASS
            if (
                len(support_ids) != expected_support
                or len(query_ids) != expected_query
                or len(support_indices) != expected_support
                or len(query_indices) != expected_query
                or len(support_labels) != expected_support
                or len(set(support_ids)) != len(support_ids)
                or len(set(query_ids)) != len(query_ids)
                or set(support_ids) & set(query_ids)
                or set(support_labels) != set(classes)
                or any(support_labels.count(cls) != int(planned["active_k"]) for cls in classes)
                or any(type(index) is not int or index < 0 or index >= 588 for index in support_indices + query_indices)
            ):
                raise NextR2ScoreError("prediction capsule support/query closure drift")
    by_pair: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    for key in keys:
        by_pair.setdefault((str(key["held_receiver"]), str(key["held_class"])), {})[int(key["active_k"])] = key
    if len(by_pair) != matrix.SELECTED_RECEIVER_COUNT * matrix.CLASS_COUNT or any(set(item) != set(matrix.K_VALUES) for item in by_pair.values()):
        raise NextR2ScoreError("prediction capsule K-pair coverage drift")
    for pair in by_pair.values():
        for registration in ("REG0", "REG1"):
            k1 = pair[1]["registrations"][registration]
            k5 = pair[5]["registrations"][registration]
            if set(k1["support_indices"]) - set(k5["support_indices"]) or tuple(k1["query_indices"]) != tuple(k5["query_indices"]):
                raise NextR2ScoreError("prediction capsule K1/K5 nesting drift")
        if set(pair[1]["registrations"]["REG0"]["support_indices"]) - set(pair[1]["registrations"]["REG1"]["support_indices"]) or set(pair[1]["registrations"]["REG0"]["query_indices"]) - set(pair[1]["registrations"]["REG1"]["query_indices"]):
            raise NextR2ScoreError("prediction capsule REG0 subset drift")
    ready = dict(payload)
    ready["capsule_content_sha256"] = observed
    return MappingProxyType(ready)


def _verify_header(
    run_root: Path,
    *,
    capsule: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    completion_path = run_root / "completion.json"
    plan_path = run_root / "plan.json"
    manifest_path = run_root / "manifest.json"
    prereg_path = run_root / "preregistration.json"
    completion = _json_load(completion_path, name="completion.json")
    plan = _json_load(plan_path, name="plan.json")
    manifest = _json_load(manifest_path, name="manifest.json")
    prereg = _json_load(prereg_path, name="preregistration.json")
    for field, path in (("plan_sha256", plan_path), ("manifest_sha256", manifest_path), ("preregistration_sha256", prereg_path)):
        expected = _require_sha(completion.get(field), name=f"completion.{field}")
        if _sha_file(path, name=field) != expected:
            raise NextR2ScoreError(f"{field} drift")
    frozen_plan = matrix.validate_next_r2_proxy24_plan(plan)
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("status") != "ARTIFACTS_COMPLETE_NOT_SCORED"
        or completion.get("outer_keys_completed") != matrix.OUTER_KEY_COUNT
        or completion.get("states_completed") != matrix.STATE_PREDICTION_COUNT
        or completion.get("all_states_sealed") is not True
        or completion.get("label_join_opened") is not False
        or completion.get("query_labels_present") is not False
        or completion.get("truth_opened") is not False
        or completion.get("scoring_performed") is not False
    ):
        raise NextR2ScoreError("prediction completion is not ARTIFACTS_COMPLETE_NOT_SCORED")
    if (
        prereg.get("candidate_id") != matrix.CANDIDATE_ID
        or prereg.get("protocol_schema") != matrix.PROTOCOL_SCHEMA
        or prereg.get("matrix_sha256") != frozen_plan["matrix_sha256"]
        or prereg.get("capsule_id") != capsule["capsule_id"]
        or prereg.get("split_id") != capsule["split_id"]
        or prereg.get("capsule_file_sha256") != capsule.get("_file_sha256", prereg.get("capsule_file_sha256"))
        or prereg.get("query_labels_present") is not False
        or prereg.get("truth_input") is not None
        or prereg.get("truth_scoring_in_process") is not False
    ):
        raise NextR2ScoreError("preregistration binding or truth-isolation drift")
    if (
        manifest.get("schema") != SEALED_MANIFEST_SCHEMA
        or manifest.get("candidate_id") != matrix.CANDIDATE_ID
        or manifest.get("matrix_sha256") != frozen_plan["matrix_sha256"]
        or manifest.get("outer_key_count") != matrix.OUTER_KEY_COUNT
        or manifest.get("state_prediction_count") != matrix.STATE_PREDICTION_COUNT
        or manifest.get("all_states_sealed") is not True
        or manifest.get("sealed_before_scoring") is not True
        or manifest.get("truth_opened") is not False
    ):
        raise NextR2ScoreError("sealed manifest header drift")
    manifest_without_sha = dict(manifest)
    observed_manifest_sha = manifest_without_sha.pop("sealed_manifest_sha256", None)
    if observed_manifest_sha != _canonical_sha(manifest_without_sha):
        raise NextR2ScoreError("sealed manifest SHA drift")
    return frozen_plan, manifest, completion, prereg


def _state_record(
    root: Path,
    item: Mapping[str, Any],
    outer: matrix.NextR2OuterKey,
    state_id: str,
    *,
    capsule_id: str,
    split_id: str,
) -> Mapping[str, Any]:
    if item.get("outer_key_id") != outer.outer_key_id or item.get("state_id") != state_id:
        raise NextR2ScoreError("state artifact identity drift")
    json_path = _regular_child(root, item.get("json_path"), name="state JSON")
    npz_path = _regular_child(root, item.get("npz_path"), name="state NPZ")
    if _sha_file(json_path, name="state JSON") != _require_sha(item.get("json_sha256"), name="manifest.json_sha256"):
        raise NextR2ScoreError("state JSON hash drift")
    if _sha_file(npz_path, name="state NPZ") != _require_sha(item.get("npz_sha256"), name="manifest.npz_sha256"):
        raise NextR2ScoreError("state NPZ hash drift")
    payload = _json_load(json_path, name="state receipt")
    _reject_forbidden(payload, name="state receipt")
    if (
        payload.get("schema") != STATE_RECEIPT_SCHEMA
        or payload.get("outer_key_id") != outer.outer_key_id
        or payload.get("state_id") != state_id
        or payload.get("truth_present") is not False
        or payload.get("score_present") is not False
    ):
        raise NextR2ScoreError("state receipt legality header drift")
    receipt = payload.get("receipt")
    seal = payload.get("seal")
    if not isinstance(receipt, Mapping) or not isinstance(seal, Mapping):
        raise NextR2ScoreError("state receipt/seal missing")
    if (
        receipt.get("schema") != STATE_RECEIPT_SCHEMA
        or receipt.get("state_id") != state_id
        or receipt.get("candidate_id") != matrix.CANDIDATE_ID
        or receipt.get("protocol_schema") != matrix.PROTOCOL_SCHEMA
        or receipt.get("capsule_id") != capsule_id
        or receipt.get("split_id") != split_id
        or receipt.get("outer_key_id") != outer.outer_key_id
        or receipt.get("active_k") != outer.active_k
        or not isinstance(receipt.get("registered_classes"), (tuple, list))
        or tuple(receipt.get("registered_classes", ())) != matrix.registered_classes_for_state(outer, state_id)
    ):
        raise NextR2ScoreError("state receipt binding drift")
    if any(receipt.get(field) != expected for field, expected in (("query_truth_input_count", 0), ("query_rows_used_for_fit", 0), ("query_state_updates", 0), ("query_selection_count", 0))):
        raise NextR2ScoreError("state query isolation counters drift")
    receipt_without_sha = dict(receipt)
    observed_receipt_sha = receipt_without_sha.pop("state_receipt_sha256", None)
    if observed_receipt_sha != _canonical_sha(receipt_without_sha):
        raise NextR2ScoreError("state receipt SHA drift")
    seal_without_sha = dict(seal)
    observed_seal_sha = seal_without_sha.pop("state_seal_sha256", None)
    if observed_seal_sha != _canonical_sha(seal_without_sha):
        raise NextR2ScoreError("state seal SHA drift")
    if observed_seal_sha != _require_sha(item.get("state_seal_sha256"), name="manifest.state_seal_sha256"):
        raise NextR2ScoreError("manifest/state seal binding drift")
    try:
        npz = _npz_bytes(npz_path.read_bytes(), members=_STATE_NPZ_MEMBERS, name="state NPZ")
    except OSError as error:
        raise NextR2ScoreError("state NPZ read failed") from error
    query_ids = _strings(npz["query_physical_ids"], name="state.query_physical_ids")
    classes = _strings(npz["registered_classes"], name="state.registered_classes")
    predictions = _strings(npz["predictions"], name="state.predictions", rows=len(query_ids))
    scores = np.asarray(npz["scores"])
    if scores.dtype != np.dtype("<f4") or scores.ndim != 2 or scores.shape != (len(query_ids), len(classes)) or not scores.flags.c_contiguous or not np.isfinite(scores).all():
        raise NextR2ScoreError("state score matrix shape/dtype/value drift")
    expected_classes = matrix.registered_classes_for_state(outer, state_id)
    if classes != expected_classes or any(value not in classes for value in predictions):
        raise NextR2ScoreError("state registered class/prediction drift")
    seal_classes = seal.get("registered_classes")
    seal_query_root = seal.get("query_physical_id_root")
    if (
        not isinstance(seal_classes, (tuple, list))
        or tuple(seal_classes) != classes
        or not isinstance(seal_query_root, str)
        or not seal_query_root
    ):
        raise NextR2ScoreError("state seal class/query binding drift")
    if seal.get("query_physical_id_root") != _id_root(query_ids):
        raise NextR2ScoreError("state query physical root drift")
    if (
        receipt.get("query_physical_id_root") != _id_root(query_ids)
        or receipt.get("scores_sha256") != _array_sha(scores)
        or receipt.get("predictions_sha256") != _prediction_root(predictions)
        or seal.get("scores_sha256") != _array_sha(scores)
        or seal.get("predictions_sha256") != _prediction_root(predictions)
    ):
        raise NextR2ScoreError("state prediction array hash drift")
    if payload.get("npz_path") != item.get("npz_path") or payload.get("npz_sha256") != item.get("npz_sha256"):
        raise NextR2ScoreError("state receipt/manifest NPZ binding drift")
    for path_field, sha_field, required in (
        ("bssdg_wire_path", "bssdg_wire_sha256", True),
        ("cvfr_wire_path", "cvfr_wire_sha256", state_id in matrix.DA1_STATES),
    ):
        wire_path_value = payload.get(path_field)
        wire_sha_value = payload.get(sha_field)
        if path_field == "cvfr_wire_path" and state_id not in matrix.DA1_STATES:
            if wire_path_value is not None or wire_sha_value is not None:
                raise NextR2ScoreError("DA0 state must not carry a CVFR wire")
            continue
        if wire_path_value is None and wire_sha_value is None and not required:
            continue
        if not isinstance(wire_path_value, str) or not isinstance(wire_sha_value, str):
            raise NextR2ScoreError(f"{path_field}/{sha_field} missing")
        wire_path = _regular_child(root, wire_path_value, name=path_field)
        if _sha_file(wire_path, name=path_field) != _require_sha(wire_sha_value, name=sha_field):
            raise NextR2ScoreError(f"{path_field} hash drift")
    return MappingProxyType({
        "outer_key": outer,
        "state_id": state_id,
        "query_ids": query_ids,
        "classes": classes,
        "predictions": predictions,
        "scores": scores,
        "receipt": receipt,
        "seal": seal,
        "json_path": str(json_path),
        "npz_path": str(npz_path),
        "json_sha256": item["json_sha256"],
        "npz_sha256": item["npz_sha256"],
    })


def _validate_prediction_closure(
    run_root: Path,
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    capsule_id: str,
    split_id: str,
) -> tuple[Mapping[str, Any], ...]:
    raw_artifacts = manifest.get("states", ())
    if not isinstance(raw_artifacts, (tuple, list)):
        raise NextR2ScoreError("manifest states must be a sequence")
    artifacts = tuple(raw_artifacts)
    if len(artifacts) != matrix.STATE_PREDICTION_COUNT:
        raise NextR2ScoreError("prediction state artifact count is not 96")
    expected = tuple((str(key["outer_key_id"]), state_id) for key in plan["keys"] for state_id in matrix.STATE_IDS)
    observed: list[tuple[str, str]] = []
    records: list[Mapping[str, Any]] = []
    seen_file_hashes: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise NextR2ScoreError("manifest state entry must be an object")
        pair = (str(item.get("outer_key_id", "")), str(item.get("state_id", "")))
        observed.append(pair)
        for field in ("json_sha256", "npz_sha256"):
            digest = _require_sha(item.get(field), name=f"manifest.{field}")
            if digest in seen_file_hashes:
                raise NextR2ScoreError("manifest artifact file hash repeated")
            seen_file_hashes.add(digest)
        outer = matrix.outer_key_from_mapping(next(key for key in plan["keys"] if key["outer_key_id"] == pair[0])) if pair[0] in {key["outer_key_id"] for key in plan["keys"]} else None
        if outer is None or pair[1] not in matrix.STATE_IDS:
            raise NextR2ScoreError("manifest state identity drift")
        records.append(_state_record(run_root, item, outer, pair[1], capsule_id=capsule_id, split_id=split_id))
    if tuple(observed) != expected or len({(r["outer_key"].outer_key_id, r["state_id"]) for r in records}) != matrix.STATE_PREDICTION_COUNT:
        raise NextR2ScoreError("prediction state matrix incomplete or reordered")
    return tuple(records)


def _harmonic(a: float, n: float) -> float:
    return 0.0 if a + n <= 0.0 else 2.0 * a * n / (a + n)


def _metric_row(record: Mapping[str, Any], truth: Mapping[str, str]) -> Mapping[str, Any]:
    outer: matrix.NextR2OuterKey = record["outer_key"]
    state_id = str(record["state_id"])
    query_ids = tuple(record["query_ids"])
    predictions = tuple(record["predictions"])
    classes = tuple(record["classes"])
    if any(qid not in truth for qid in query_ids):
        raise NextR2ScoreError("truth archive lacks a prediction physical_id")
    pairs = [(prediction, truth[qid], qid) for qid, prediction in zip(query_ids, predictions, strict=True)]
    if any(label not in classes for _, label, _ in pairs):
        raise NextR2ScoreError("truth label is outside the state registered classes")
    retained_classes = tuple(outer.retained_classes)
    retained = [(prediction, label) for prediction, label, _ in pairs if label in retained_classes]
    if not retained or any(not any(label == cls for _, label in retained) for cls in retained_classes):
        raise NextR2ScoreError("state retained truth coverage is incomplete")
    per_class: dict[str, dict[str, Any]] = {}
    for cls in classes:
        class_pairs = [(prediction, label) for prediction, label, _ in pairs if label == cls]
        correct = sum(int(prediction == label) for prediction, label in class_pairs)
        per_class[cls] = {"correct": correct, "count": len(class_pairs), "accuracy": (correct / len(class_pairs) if class_pairs else None)}
    retained_correct = sum(int(prediction == label) for prediction, label in retained)
    a_retained = retained_correct / len(retained)
    floor = min(float(per_class[cls]["accuracy"]) for cls in retained_classes)
    total_correct = sum(int(prediction == label) for prediction, label, _ in pairs)
    result: dict[str, Any] = {
        "outer_key_id": outer.outer_key_id,
        "held_receiver": outer.held_receiver,
        "held_class": outer.held_class,
        "active_k": outer.active_k,
        "state_id": state_id,
        "registered_classes": classes,
        "A_retained": a_retained,
        "F_retained": floor,
        "retained_correct_count": retained_correct,
        "retained_query_count": len(retained),
        "total_correct_count": total_correct,
        "total_query_count": len(pairs),
        "retained_per_class": {cls: per_class[cls] for cls in retained_classes},
        "per_class": per_class,
    }
    if state_id in matrix.REG1_STATES:
        new_pairs = [(prediction, label) for prediction, label, _ in pairs if label == outer.held_class]
        if not new_pairs:
            raise NextR2ScoreError("REG1 held-class truth coverage is incomplete")
        new_correct = sum(int(prediction == label) for prediction, label in new_pairs)
        n_seen_new = new_correct / len(new_pairs)
        result.update({
            "N_seen_new": n_seen_new,
            "H_retained_new": _harmonic(a_retained, n_seen_new),
            "new_correct_count": new_correct,
            "new_query_count": len(new_pairs),
            "registration_metric_status": "DEFINED_AFTER_REGISTRATION",
        })
    else:
        result.update({
            "N_seen_new": None,
            "H_retained_new": None,
            "new_correct_count": None,
            "new_query_count": None,
            "registration_metric_status": "NA_BEFORE_REGISTRATION",
        })
    return MappingProxyType(result)


def _aggregate(rows: Sequence[Mapping[str, Any]], *, active_k: int, state_id: str) -> Mapping[str, Any]:
    selected = [row for row in rows if row["active_k"] == active_k and row["state_id"] == state_id]
    if len(selected) != matrix.SELECTED_RECEIVER_COUNT * matrix.CLASS_COUNT:
        raise NextR2ScoreError("aggregate requires all 12 outer keys")
    retained_correct = sum(int(row["retained_correct_count"]) for row in selected)
    retained_query = sum(int(row["retained_query_count"]) for row in selected)
    total_correct = sum(int(row["total_correct_count"]) for row in selected)
    total_query = sum(int(row["total_query_count"]) for row in selected)
    # Build the six-class pooled retained floor from every held fold.  Each
    # class is absent only in its own held fold, and is covered by the others.
    class_ids = tuple(sorted({cls for row in selected for cls in row["per_class"]}))
    pooled = {cls: {"correct": 0, "count": 0} for cls in class_ids}
    for row in selected:
        for cls, value in row["retained_per_class"].items():
            pooled[cls]["correct"] += int(value["correct"])
            pooled[cls]["count"] += int(value["count"])
    if any(value["count"] < 1 for value in pooled.values()):
        raise NextR2ScoreError("pooled retained floor lacks six-class coverage")
    pooled_per_class = {
        cls: {**value, "accuracy": value["correct"] / value["count"]}
        for cls, value in sorted(pooled.items())
    }
    result: dict[str, Any] = {
        "active_k": active_k,
        "state_id": state_id,
        "outer_key_count": len(selected),
        "A_retained": retained_correct / retained_query,
        "F_retained": min(float(value["accuracy"]) for value in pooled_per_class.values()),
        "retained_correct_count": retained_correct,
        "retained_query_count": retained_query,
        "total_correct_count": total_correct,
        "total_query_count": total_query,
        "retained_per_class": pooled_per_class,
    }
    if state_id in matrix.REG1_STATES:
        new_correct = sum(int(row["new_correct_count"]) for row in selected)
        new_query = sum(int(row["new_query_count"]) for row in selected)
        n_seen_new = new_correct / new_query
        result.update({
            "N_seen_new": n_seen_new,
            "H_retained_new": _harmonic(result["A_retained"], n_seen_new),
            "new_correct_count": new_correct,
            "new_query_count": new_query,
            "registration_metric_status": "DEFINED_AFTER_REGISTRATION",
        })
    else:
        result.update({
            "N_seen_new": None,
            "H_retained_new": None,
            "new_correct_count": None,
            "new_query_count": None,
            "registration_metric_status": "NA_BEFORE_REGISTRATION",
        })
    return MappingProxyType(result)


def _difference(aggregates: Mapping[str, Mapping[str, Any]], left: str, right: str, *, fields: Sequence[str]) -> Mapping[str, Any]:
    result: dict[str, Any] = {f"delta_{field}": aggregates[left][field] - aggregates[right][field] for field in fields}
    result["left_state"] = left
    result["right_state"] = right
    result["delta_total_correct_count"] = aggregates[left]["total_correct_count"] - aggregates[right]["total_correct_count"]
    result["delta_total_query_count"] = aggregates[left]["total_query_count"] - aggregates[right]["total_query_count"]
    return MappingProxyType(result)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NextR2ScoreError("non-finite score output")
        return value
    return value


def score_next_r2_proxy24(
    *,
    run_root: str | Path,
    prediction_capsule: str | Path,
    prediction_capsule_sha256: str,
    ls_label_join_archive: str | Path,
    ls_label_join_archive_sha256: str,
) -> Mapping[str, Any]:
    """Validate a complete run, then open truth and score all 96 states."""

    root = Path(run_root).resolve(strict=True)
    if not root.is_dir():
        raise NextR2ScoreError("run_root must be a directory")
    capsule_path = Path(prediction_capsule)
    capsule_sha = _require_sha(prediction_capsule_sha256, name="prediction_capsule_sha256")
    capsule = _read_capsule(capsule_path, capsule_sha)
    # The file SHA is pinned separately from the content SHA, so preregistration
    # can bind the exact file without ever opening truth.
    capsule = MappingProxyType({**capsule, "_file_sha256": capsule_sha})
    plan, manifest, completion, prereg = _verify_header(root, capsule=capsule)
    if (
        capsule["matrix_sha256"] != plan["matrix_sha256"]
        or capsule["split_id"] != completion.get("split_id")
        or capsule["capsule_id"] != completion.get("capsule_id")
        or prereg.get("capsule_file_sha256") != capsule_sha
        or prereg.get("capsule_content_sha256") != capsule["capsule_content_sha256"]
        or capsule["label_join_archive_sha256"] != _require_sha(ls_label_join_archive_sha256, name="ls_label_join_archive_sha256")
    ):
        raise NextR2ScoreError("capsule/run completion binding drift")
    records = _validate_prediction_closure(
        root,
        manifest,
        plan,
        capsule_id=str(capsule["capsule_id"]),
        split_id=str(capsule["split_id"]),
    )
    # Only after the full 96-state closure has passed is the truth archive read.
    truth_path = Path(ls_label_join_archive)
    truth_sha = _require_sha(ls_label_join_archive_sha256, name="ls_label_join_archive_sha256")
    if truth_path.is_symlink() or not truth_path.is_file() or _sha_file(truth_path, name="ls_label_join_archive") != truth_sha:
        raise NextR2ScoreError("ls_label_join_archive SHA256 drift")
    truth_npz = _npz_bytes(truth_path.read_bytes(), members=_TRUTH_MEMBERS, name="ls_label_join_archive")
    physical_ids = _strings(truth_npz["physical_ids"], name="truth.physical_ids")
    labels = _strings(truth_npz["tx_labels"], name="truth.tx_labels", rows=len(physical_ids))
    if len(set(physical_ids)) != len(physical_ids):
        raise NextR2ScoreError("truth physical_id catalog is not unique")
    truth = dict(zip(physical_ids, labels, strict=True))
    class_registry = tuple(plan["class_registry"])
    if any(value not in class_registry for value in truth.values()):
        raise NextR2ScoreError("truth label is outside frozen class registry")
    state_scores = [_metric_row(record, truth) for record in records]
    aggregates: dict[str, dict[str, Mapping[str, Any]]] = {}
    differences: dict[str, dict[str, Mapping[str, Any]]] = {}
    for active_k in matrix.K_VALUES:
        agg = {state_id: _aggregate(state_scores, active_k=active_k, state_id=state_id) for state_id in matrix.STATE_IDS}
        aggregates[str(active_k)] = agg
        differences[str(active_k)] = {
            "DA_PRE": _difference(agg, "DA1_REG0", "DA0_REG0", fields=("A_retained", "F_retained")),
            "DA_POST": _difference(agg, "DA1_REG1", "DA0_REG1", fields=("A_retained", "N_seen_new", "H_retained_new", "F_retained")),
            "REGISTRATION_NO_DA": _difference(agg, "DA0_REG1", "DA0_REG0", fields=("A_retained", "F_retained")),
            "REGISTRATION_WITH_DA": _difference(agg, "DA1_REG1", "DA1_REG0", fields=("A_retained", "F_retained")),
        }
        differences[str(active_k)]["DID_RETAINED"] = MappingProxyType({
            "delta_A_retained": differences[str(active_k)]["DA_POST"]["delta_A_retained"] - differences[str(active_k)]["DA_PRE"]["delta_A_retained"],
            "delta_F_retained": differences[str(active_k)]["DA_POST"]["delta_F_retained"] - differences[str(active_k)]["DA_PRE"]["delta_F_retained"],
        })
    identity_statuses = [
        str(record["receipt"].get("cvfr_status", ""))
        for record in records
        if record["state_id"] in matrix.DA1_STATES
    ]
    pair_map = {(record["outer_key"].outer_key_id, record["state_id"]): record for record in records}
    prediction_identity = all(
        pair_map[(outer_key["outer_key_id"], "DA1_REG0")]["predictions"] == pair_map[(outer_key["outer_key_id"], "DA0_REG0")]["predictions"]
        and pair_map[(outer_key["outer_key_id"], "DA1_REG1")]["predictions"] == pair_map[(outer_key["outer_key_id"], "DA0_REG1")]["predictions"]
        for outer_key in plan["keys"]
    )
    all_da_identity = bool(identity_statuses) and all(status.startswith("DA_IDENTITY") for status in identity_statuses)
    k5 = differences["5"]["DA_POST"]
    primary = {
        "delta_H_retained_new": k5["delta_H_retained_new"],
        "delta_total_correct_count": k5["delta_total_correct_count"],
        "delta_A_retained": k5["delta_A_retained"],
        "delta_N_seen_new": k5["delta_N_seen_new"],
        "delta_F_retained": k5["delta_F_retained"],
    }
    no_function = all_da_identity or prediction_identity
    primary["no_function"] = no_function
    primary["pass"] = (not no_function and primary["delta_H_retained_new"] > 0.0 and primary["delta_total_correct_count"] > 0 and primary["delta_A_retained"] >= 0.0 and primary["delta_N_seen_new"] >= 0.0 and primary["delta_F_retained"] >= 0.0)
    decision = "NO_FUNCTION" if no_function else ("PASS_TO_SOURCE_HELD_REVIEW" if primary["pass"] else "REJECT_CANDIDATE")
    output = {
        "schema": SCORE_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "run_id": completion.get("run_id"),
        "capsule_id": capsule["capsule_id"],
        "split_id": capsule["split_id"],
        "matrix_sha256": plan["matrix_sha256"],
        "prediction_capsule_sha256": capsule_sha,
        "prediction_capsule_content_sha256": capsule["capsule_content_sha256"],
        "ls_label_join_archive_sha256": truth_sha,
        "outer_key_count": matrix.OUTER_KEY_COUNT,
        "state_prediction_count": matrix.STATE_PREDICTION_COUNT,
        "truth_opened_after_complete_predictions": True,
        "partial_scoring_used": False,
        "formal_target_claim": False,
        "formal_new_registration_claim": False,
        "state_scores": state_scores,
        "aggregates_by_k_and_state": aggregates,
        "four_state_differences_by_k": differences,
        "k5_primary": primary,
        "decision": decision,
        "prediction_identity_all_pairs": prediction_identity,
        "da_identity_all_states": all_da_identity,
        "truth_query_count_opened": len(truth),
        "truth_label_join_only": True,
    }
    return MappingProxyType(_plain(output))


# Friendly aliases used by small local tools/tests.
score_next_r2 = score_next_r2_proxy24
score_run = score_next_r2_proxy24


__all__ = [
    "COMPLETION_SCHEMA",
    "NextR2ScoreError",
    "SCORE_SCHEMA",
    "SCORING_COMPLETION_SCHEMA",
    "score_next_r2",
    "score_next_r2_proxy24",
    "score_run",
]
