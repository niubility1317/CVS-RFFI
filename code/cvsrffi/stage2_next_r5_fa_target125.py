"""Immutable prediction closure and truth-side scoring for NEXT-R5 Target125.

The prediction plane accepts only opaque query IDs, registered class handles
and labels predicted by the frozen FA-RDCE3/qKNN core.  It writes 1,350 unique
prediction artifacts and represents the 150 K1 DA1 surfaces as exact aliases.
Truth labels are accepted only by :func:`score_target125_truth_catalog` after
the prediction manifest has passed its complete 1,500-surface closure check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import stage2_next_r5_fa_target125_matrix as matrix


PREDICTION_ARTIFACT_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.prediction_artifact.v1"
PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.prediction_manifest.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.truth_catalog.v1"
SCORE_SCHEMA = "cvs.phase2.next_r5.fa_rdce3_qknn.target125.score.v1"


class NextR5FATarget125Error(ValueError):
    """Raised when Target125 prediction closure or truth-side scoring drifts."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise NextR5FATarget125Error("canonical Target125 payload is invalid") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NextR5FATarget125Error(f"{name} must be a lowercase SHA256")
    return value


def _tokens(value: Any, *, name: str, expected: int | None = None, unique: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise NextR5FATarget125Error(f"{name} must be a sequence")
    result = tuple(value)
    if (
        (expected is not None and len(result) != expected)
        or not result
        or any(type(item) is not str or not item for item in result)
        or (unique and len(set(result)) != len(result))
    ):
        raise NextR5FATarget125Error(f"{name} is malformed")
    return result


def _plain(value: Any, *, name: str) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item, name=f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NextR5FATarget125Error(f"{name} contains a non-finite float")
        return value
    raise NextR5FATarget125Error(f"{name} has a non-JSON value")


_FORBIDDEN_PREDICTOR_KEYS = frozenset(
    {
        "truth",
        "truth_label",
        "truth_labels",
        "query_truth",
        "query_label",
        "query_labels",
        "query_role",
        "query_roles",
        "role",
        "roles",
        "class_quota",
        "true_batch_class_count",
        "global_reassignment",
        "hungarian",
        "optimal_transport",
        "clean_iq",
        "raw_iq",
        "source_feature",
        "source_features",
        "source_cache",
    }
)


def _reject_forbidden(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_PREDICTOR_KEYS:
                raise NextR5FATarget125Error(f"{name} exposes forbidden predictor field {key}")
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NextR5FATarget125Error("unsafe immutable output parent")
    raw = _canonical_bytes(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path, *, expected_sha256: str, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise NextR5FATarget125Error(f"{name} must be a regular file")
    if _sha256_file(path) != _sha(expected_sha256, f"{name} SHA256"):
        raise NextR5FATarget125Error(f"{name} SHA mismatch")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR5FATarget125Error(f"{name} must be UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise NextR5FATarget125Error(f"{name} must contain an object")
    return result


def _surface_metadata(surface: matrix.Target125StateSurface) -> dict[str, Any]:
    return {
        "surface_id": surface.surface_id,
        "scene_row_id": surface.scene_row_id,
        "outer_id": surface.outer_id,
        "receiver": surface.receiver,
        "seed": surface.seed,
        "k_shot": surface.k_shot,
        "new_count": surface.new_count,
        "source_pool_k": surface.source_pool_k,
        "scene": surface.scene,
        "state": surface.state,
        "state_name_zh": matrix.STATE_NAMES_ZH[surface.state],
        "registration_phase": surface.registration_phase,
        "da_phase": surface.da_phase,
        "metric_availability": dict(matrix.METRIC_AVAILABILITY[surface.state]),
        "unique_prediction": surface.unique_prediction,
        "alias_of_surface_id": surface.alias_of_surface_id,
    }


_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "protocol_schema",
        "surface",
        "registered_classes",
        "ordered_query_physical_ids",
        "ordered_query_physical_ids_sha256",
        "predicted_labels",
        "predicted_labels_sha256",
        "state_receipt",
        "access_ledger",
        "truth_open",
        "immutable",
        "artifact_receipt_sha256",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "surface",
        "registered_classes",
        "ordered_query_physical_ids",
        "ordered_query_physical_ids_sha256",
        "prediction_artifact",
        "prediction_artifact_sha256",
        "state_receipt",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "protocol_schema",
        "matrix_receipt_sha256",
        "manifest_sealed",
        "truth_open",
        "outer_job_count",
        "scene_row_count",
        "logical_state_surface_count",
        "unique_prediction_count",
        "alias_count",
        "state_names_zh",
        "metric_availability",
        "access_ledger",
        "surfaces",
        "manifest_receipt_sha256",
    }
)


def _validate_surface(surface: Any) -> matrix.Target125StateSurface:
    if not isinstance(surface, matrix.Target125StateSurface):
        raise NextR5FATarget125Error("prediction requires an exact frozen Target125 surface")
    # Individual dataclass construction validates IDs; the full plan validation
    # below proves that no out-of-order partial topology slips through.
    return surface


def seal_unique_prediction(
    *,
    output_dir: str | Path,
    surface: matrix.Target125StateSurface,
    registered_classes: Sequence[str],
    query_physical_ids: Sequence[str],
    predicted_labels: Sequence[str],
    state_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Seal one non-alias prediction artifact; no truth sidecar is readable."""

    surface = _validate_surface(surface)
    if not surface.unique_prediction:
        raise NextR5FATarget125Error("K1 DA1 aliases must not create a duplicate prediction artifact")
    classes = _tokens(registered_classes, name="registered_classes")
    query_ids = _tokens(query_physical_ids, name="query_physical_ids")
    labels = _tokens(predicted_labels, name="predicted_labels", expected=len(query_ids), unique=False)
    if any(label not in classes for label in labels):
        raise NextR5FATarget125Error("prediction is outside its registered class registry")
    receipt = _plain(state_receipt, name="state_receipt")
    _reject_forbidden(receipt, name="state_receipt")
    payload: dict[str, Any] = {
        "schema": PREDICTION_ARTIFACT_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "surface": _surface_metadata(surface),
        "registered_classes": list(classes),
        "ordered_query_physical_ids": list(query_ids),
        "ordered_query_physical_ids_sha256": canonical_sha256(list(query_ids)),
        "predicted_labels": list(labels),
        "predicted_labels_sha256": canonical_sha256(list(labels)),
        "state_receipt": receipt,
        "access_ledger": dict(matrix.ACCESS_LEDGER),
        "truth_open": False,
        "immutable": True,
    }
    payload["artifact_receipt_sha256"] = canonical_sha256(payload)
    root = Path(output_dir)
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise NextR5FATarget125Error("prediction output root must be a normal directory")
    root.mkdir(parents=True, exist_ok=True)
    prediction_root = root / "predictions"
    prediction_root.mkdir(exist_ok=True)
    path = prediction_root / f"{surface.surface_id}.json"
    file_sha = _write_json_new(path, payload)
    return MappingProxyType(
        {
            "surface": _surface_metadata(surface),
            "registered_classes": list(classes),
            "ordered_query_physical_ids": list(query_ids),
            "ordered_query_physical_ids_sha256": payload["ordered_query_physical_ids_sha256"],
            "prediction_artifact": f"predictions/{surface.surface_id}.json",
            "prediction_artifact_sha256": file_sha,
            "state_receipt": receipt,
        }
    )


def seal_k1_alias(
    *,
    surface: matrix.Target125StateSurface,
    source_record: Mapping[str, Any],
    state_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Represent K1 DA1 as a logical alias without writing another prediction."""

    surface = _validate_surface(surface)
    if surface.unique_prediction or surface.alias_of_surface_id is None:
        raise NextR5FATarget125Error("only a frozen K1 DA1 surface can be an alias")
    source = _validate_record(source_record, expected_surface_id=surface.alias_of_surface_id)
    expected = matrix.surface_by_id(surface.alias_of_surface_id)
    if (
        source["surface"]["registration_phase"] != surface.registration_phase
        or expected.registration_phase != surface.registration_phase
    ):
        raise NextR5FATarget125Error("K1 alias registration phase drift")
    receipt = _plain(state_receipt, name="K1 alias state_receipt")
    _reject_forbidden(receipt, name="K1 alias state_receipt")
    if receipt.get("exact_prediction_alias") is not True or receipt.get("alias_of_surface_id") != surface.alias_of_surface_id:
        raise NextR5FATarget125Error("K1 alias receipt must prove the exact source surface")
    return MappingProxyType(
        {
            "surface": _surface_metadata(surface),
            "registered_classes": list(source["registered_classes"]),
            "ordered_query_physical_ids": list(source["ordered_query_physical_ids"]),
            "ordered_query_physical_ids_sha256": source["ordered_query_physical_ids_sha256"],
            "prediction_artifact": source["prediction_artifact"],
            "prediction_artifact_sha256": source["prediction_artifact_sha256"],
            "state_receipt": receipt,
        }
    )


def _exact_mapping(value: Any, expected: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise NextR5FATarget125Error(f"{name} field closure drift")
    return value


def _validate_record(value: Mapping[str, Any], *, expected_surface_id: str | None = None) -> dict[str, Any]:
    record = _exact_mapping(value, _RECORD_FIELDS, "prediction record")
    surface_raw = record["surface"]
    if not isinstance(surface_raw, Mapping):
        raise NextR5FATarget125Error("prediction record surface must be a mapping")
    surface_id = surface_raw.get("surface_id")
    surface = matrix.surface_by_id(str(surface_id))
    if expected_surface_id is not None and surface.surface_id != expected_surface_id:
        raise NextR5FATarget125Error("prediction record surface ID drift")
    if dict(surface_raw) != _surface_metadata(surface):
        raise NextR5FATarget125Error("prediction record frozen surface metadata drift")
    classes = _tokens(record["registered_classes"], name="record.registered_classes")
    query_ids = _tokens(record["ordered_query_physical_ids"], name="record.query_ids")
    if record["ordered_query_physical_ids_sha256"] != canonical_sha256(list(query_ids)):
        raise NextR5FATarget125Error("prediction record query root receipt drift")
    artifact = record["prediction_artifact"]
    if type(artifact) is not str or not artifact.startswith("predictions/") or Path(artifact).is_absolute():
        raise NextR5FATarget125Error("prediction artifact must be a portable predictions-relative path")
    _sha(record["prediction_artifact_sha256"], "prediction artifact SHA256")
    receipt = _plain(record["state_receipt"], name="record.state_receipt")
    _reject_forbidden(receipt, name="record.state_receipt")
    return {
        "surface": _surface_metadata(surface),
        "registered_classes": list(classes),
        "ordered_query_physical_ids": list(query_ids),
        "ordered_query_physical_ids_sha256": record["ordered_query_physical_ids_sha256"],
        "prediction_artifact": artifact,
        "prediction_artifact_sha256": record["prediction_artifact_sha256"],
        "state_receipt": receipt,
    }


def build_prediction_manifest(
    *, output_dir: str | Path, records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Seal the full 1,500 logical surface closure after all shards return."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise NextR5FATarget125Error("records must be an ordered sequence")
    plan = matrix.freeze_next_r5_fa_target125_matrix()
    if len(records) != matrix.LOGICAL_STATE_SURFACE_COUNT:
        raise NextR5FATarget125Error("Target125 requires exactly 1,500 logical surface records")
    parsed = [
        _validate_record(record, expected_surface_id=surface.surface_id)
        for record, surface in zip(records, plan.surfaces, strict=True)
    ]
    matrix.audit_logical_surface_coverage(
        [item["surface"]["surface_id"] for item in parsed]
    )
    unique = [item for item in parsed if item["surface"]["unique_prediction"]]
    matrix.audit_unique_prediction_coverage(
        [item["surface"]["surface_id"] for item in unique]
    )
    aliases = [item for item in parsed if not item["surface"]["unique_prediction"]]
    if len(aliases) != matrix.ALIAS_COUNT:
        raise NextR5FATarget125Error("K1 alias count drift")
    by_surface = {item["surface"]["surface_id"]: item for item in parsed}
    for alias in aliases:
        target = alias["surface"]["alias_of_surface_id"]
        source = by_surface.get(target)
        if source is None or not source["surface"]["unique_prediction"]:
            raise NextR5FATarget125Error("K1 alias source coverage drift")
        if (
            alias["registered_classes"] != source["registered_classes"]
            or alias["ordered_query_physical_ids"] != source["ordered_query_physical_ids"]
            or alias["prediction_artifact"] != source["prediction_artifact"]
            or alias["prediction_artifact_sha256"] != source["prediction_artifact_sha256"]
        ):
            raise NextR5FATarget125Error("K1 alias must reuse exact class/query/artifact bytes")
    payload: dict[str, Any] = {
        "schema": PREDICTION_MANIFEST_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "matrix_receipt_sha256": plan.matrix_receipt_sha256,
        "manifest_sealed": True,
        "truth_open": False,
        "outer_job_count": matrix.OUTER_JOB_COUNT,
        "scene_row_count": matrix.SCENE_ROW_COUNT,
        "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
        "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT,
        "alias_count": matrix.ALIAS_COUNT,
        "state_names_zh": dict(matrix.STATE_NAMES_ZH),
        "metric_availability": {key: dict(value) for key, value in matrix.METRIC_AVAILABILITY.items()},
        "access_ledger": dict(matrix.ACCESS_LEDGER),
        "surfaces": parsed,
    }
    payload["manifest_receipt_sha256"] = canonical_sha256(payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "prediction_manifest.json"
    file_sha = _write_json_new(path, payload)
    return MappingProxyType(
        {
            "prediction_manifest": str(path),
            "prediction_manifest_file_sha256": file_sha,
            "prediction_manifest_sha256": payload["manifest_receipt_sha256"],
            "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
            "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT,
            "alias_count": matrix.ALIAS_COUNT,
        }
    )


def _resolve_artifact(manifest_path: Path, relative_path: str) -> Path:
    candidate = manifest_path.parent / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(manifest_path.parent.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise NextR5FATarget125Error("prediction artifact escapes the manifest root") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise NextR5FATarget125Error("prediction artifact must be a regular non-symlink file")
    return resolved


def _validate_artifact(
    *,
    manifest_path: Path,
    record: Mapping[str, Any],
    expected_surface: matrix.Target125StateSurface,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source_id = expected_surface.alias_of_surface_id or expected_surface.surface_id
    source_surface = matrix.surface_by_id(source_id)
    artifact_path = _resolve_artifact(manifest_path, str(record["prediction_artifact"]))
    artifact = _read_json(
        artifact_path,
        expected_sha256=str(record["prediction_artifact_sha256"]),
        name=f"prediction artifact {expected_surface.surface_id}",
    )
    _exact_mapping(artifact, _ARTIFACT_FIELDS, "prediction artifact")
    _reject_forbidden(artifact, name="prediction artifact")
    if (
        artifact["schema"] != PREDICTION_ARTIFACT_SCHEMA
        or artifact["candidate_id"] != matrix.CANDIDATE_ID
        or artifact["protocol_schema"] != matrix.PROTOCOL_SCHEMA
        or artifact["truth_open"] is not False
        or artifact["immutable"] is not True
        or artifact["access_ledger"] != matrix.ACCESS_LEDGER
        or artifact["surface"] != _surface_metadata(source_surface)
    ):
        raise NextR5FATarget125Error("prediction artifact identity/seal drift")
    payload = dict(artifact)
    receipt = payload.pop("artifact_receipt_sha256")
    if _sha(receipt, "artifact receipt") != canonical_sha256(payload):
        raise NextR5FATarget125Error("prediction artifact receipt drift")
    classes = _tokens(artifact["registered_classes"], name="artifact.registered_classes")
    query_ids = _tokens(artifact["ordered_query_physical_ids"], name="artifact.query_ids")
    labels = _tokens(artifact["predicted_labels"], name="artifact.predictions", expected=len(query_ids), unique=False)
    if any(label not in classes for label in labels):
        raise NextR5FATarget125Error("prediction artifact label registry drift")
    if (
        list(classes) != record["registered_classes"]
        or list(query_ids) != record["ordered_query_physical_ids"]
        or artifact["ordered_query_physical_ids_sha256"] != record["ordered_query_physical_ids_sha256"]
        or artifact["predicted_labels_sha256"] != canonical_sha256(list(labels))
    ):
        raise NextR5FATarget125Error("prediction record/artifact binding drift")
    return query_ids, labels


def _validate_da_query_id_parity(
    plan: matrix.Target125MatrixPlan,
    decoded: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]],
) -> None:
    """Require DA0/DA1 to use the exact same ordered query rows per REG state."""

    for outer in plan.outer_rows:
        for scene in matrix.SCENES:
            scene_row_id = matrix.make_scene_row_id(outer.outer_id, scene)
            for da0, da1 in (("DA0_REG0", "DA1_REG0"), ("DA0_REG1", "DA1_REG1")):
                da0_ids = decoded[matrix.make_surface_id(scene_row_id, da0)][0]
                da1_ids = decoded[matrix.make_surface_id(scene_row_id, da1)][0]
                if da0_ids != da1_ids:
                    raise NextR5FATarget125Error(
                        "DA0/DA1 ordered query physical-ID parity drift"
                    )


def validate_prediction_manifest(
    *, prediction_manifest_path: str | Path, expected_prediction_manifest_file_sha256: str
) -> Mapping[str, Any]:
    """Validate complete truth-closed 1,500/1,350/150 prediction closure."""

    path = Path(prediction_manifest_path)
    document = _read_json(
        path,
        expected_sha256=expected_prediction_manifest_file_sha256,
        name="Target125 prediction manifest",
    )
    _exact_mapping(document, _MANIFEST_FIELDS, "prediction manifest")
    _reject_forbidden(document, name="prediction manifest")
    plan = matrix.freeze_next_r5_fa_target125_matrix()
    if (
        document["schema"] != PREDICTION_MANIFEST_SCHEMA
        or document["candidate_id"] != matrix.CANDIDATE_ID
        or document["protocol_schema"] != matrix.PROTOCOL_SCHEMA
        or document["matrix_receipt_sha256"] != plan.matrix_receipt_sha256
        or document["manifest_sealed"] is not True
        or document["truth_open"] is not False
        or document["outer_job_count"] != matrix.OUTER_JOB_COUNT
        or document["scene_row_count"] != matrix.SCENE_ROW_COUNT
        or document["logical_state_surface_count"] != matrix.LOGICAL_STATE_SURFACE_COUNT
        or document["unique_prediction_count"] != matrix.UNIQUE_PREDICTION_COUNT
        or document["alias_count"] != matrix.ALIAS_COUNT
        or document["state_names_zh"] != matrix.STATE_NAMES_ZH
        or document["metric_availability"] != matrix.METRIC_AVAILABILITY
        or document["access_ledger"] != matrix.ACCESS_LEDGER
    ):
        raise NextR5FATarget125Error("prediction manifest header/count drift")
    payload = dict(document)
    receipt = payload.pop("manifest_receipt_sha256")
    if _sha(receipt, "prediction manifest receipt") != canonical_sha256(payload):
        raise NextR5FATarget125Error("prediction manifest receipt drift")
    raw_records = document["surfaces"]
    if not isinstance(raw_records, list) or len(raw_records) != matrix.LOGICAL_STATE_SURFACE_COUNT:
        raise NextR5FATarget125Error("prediction manifest surface cardinality drift")
    records = [
        _validate_record(record, expected_surface_id=surface.surface_id)
        for record, surface in zip(raw_records, plan.surfaces, strict=True)
    ]
    by_surface = {item["surface"]["surface_id"]: item for item in records}
    decoded: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for record, surface in zip(records, plan.surfaces, strict=True):
        if not surface.unique_prediction:
            source = by_surface.get(surface.alias_of_surface_id or "")
            if source is None or (
                record["registered_classes"] != source["registered_classes"]
                or record["ordered_query_physical_ids"] != source["ordered_query_physical_ids"]
                or record["prediction_artifact"] != source["prediction_artifact"]
                or record["prediction_artifact_sha256"] != source["prediction_artifact_sha256"]
            ):
                raise NextR5FATarget125Error("K1 alias record is not exact")
        decoded[surface.surface_id] = _validate_artifact(
            manifest_path=path, record=record, expected_surface=surface
        )
    matrix.audit_logical_surface_coverage(decoded)
    matrix.audit_unique_prediction_coverage(
        surface.surface_id for surface in plan.surfaces if surface.unique_prediction
    )
    _validate_da_query_id_parity(plan, decoded)
    return MappingProxyType(
        {
            "manifest": document,
            "manifest_path": path.resolve(strict=True),
            "manifest_file_sha256": expected_prediction_manifest_file_sha256,
            "manifest_receipt_sha256": receipt,
            "records": MappingProxyType(by_surface),
            "decoded": MappingProxyType(decoded),
        }
    )


def _rates(
    *, classes: Sequence[str], predictions: Sequence[str], labels: Sequence[str]
) -> tuple[float, float, dict[str, dict[str, int]]]:
    counts = {label: {"correct_count": 0, "query_count": 0} for label in classes}
    for predicted, truth in zip(predictions, labels, strict=True):
        if truth not in counts:
            raise NextR5FATarget125Error("truth label escapes this registered-class state")
        counts[truth]["query_count"] += 1
        counts[truth]["correct_count"] += int(predicted == truth)
    if any(item["query_count"] == 0 for item in counts.values()):
        raise NextR5FATarget125Error("truth catalog lacks class coverage")
    accuracies = [100.0 * item["correct_count"] / item["query_count"] for item in counts.values()]
    return sum(accuracies) / len(accuracies), min(accuracies), counts


def _harmonic(left: float, right: float) -> float:
    return 0.0 if left + right == 0.0 else 2.0 * left * right / (left + right)


def score_target125_truth_catalog(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    truth_catalog: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    """Score a complete independent truth catalog after prediction closure.

    ``truth_catalog`` maps every logical ``surface_id`` to labels in the
    corresponding sealed query-ID order.  It is a scorer-only input: the
    prediction artifact validation completes before the catalog is touched.
    """

    prediction = validate_prediction_manifest(
        prediction_manifest_path=prediction_manifest_path,
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )
    if not isinstance(truth_catalog, Mapping):
        raise NextR5FATarget125Error("truth catalog must be a scorer-side mapping")
    plan = matrix.freeze_next_r5_fa_target125_matrix()
    expected_ids = tuple(surface.surface_id for surface in plan.surfaces)
    if tuple(truth_catalog) != expected_ids:
        raise NextR5FATarget125Error("truth catalog must close the ordered 1,500-surface matrix")
    state_rows: list[dict[str, Any]] = []
    by_scene: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for surface in plan.surfaces:
        record = prediction["records"][surface.surface_id]
        query_ids, labels_pred = prediction["decoded"][surface.surface_id]
        labels_truth = _tokens(
            truth_catalog[surface.surface_id],
            name=f"truth[{surface.surface_id}]",
            expected=len(query_ids),
            unique=False,
        )
        classes = tuple(record["registered_classes"])
        old_classes = classes[: matrix.OLD_CLASS_COUNT]
        old_truth = [(p, y) for p, y in zip(labels_pred, labels_truth, strict=True) if y in old_classes]
        if not old_truth:
            raise NextR5FATarget125Error("truth catalog lacks old-class rows")
        old_ba, old_floor, old_counts = _rates(
            classes=old_classes,
            predictions=[item[0] for item in old_truth],
            labels=[item[1] for item in old_truth],
        )
        row: dict[str, Any] = {
            **_surface_metadata(surface),
            "old_balanced_accuracy": old_ba,
            "old_floor": old_floor,
            "old_correct_count": sum(int(p == y) for p, y in old_truth),
            "old_query_count": len(old_truth),
            "old_by_class": old_counts,
            "total_correct_count": sum(int(p == y) for p, y in zip(labels_pred, labels_truth, strict=True)),
            "total_query_count": len(labels_truth),
        }
        if surface.registration_phase == "REG0":
            if any(label not in old_classes for label in labels_truth):
                raise NextR5FATarget125Error("REG0 contains a non-registered new-class truth label")
            row.update({"seen_new_acc": "N/A", "H_old_new": "N/A", "all_floor": old_floor})
        else:
            new_classes = classes[matrix.OLD_CLASS_COUNT :]
            new_truth = [(p, y) for p, y in zip(labels_pred, labels_truth, strict=True) if y in new_classes]
            if len(old_truth) + len(new_truth) != len(labels_truth):
                raise NextR5FATarget125Error("REG1 truth falls outside its old/new registry")
            new_ba, _new_floor, new_counts = _rates(
                classes=new_classes,
                predictions=[item[0] for item in new_truth],
                labels=[item[1] for item in new_truth],
            )
            all_ba, all_floor, all_counts = _rates(
                classes=classes, predictions=labels_pred, labels=labels_truth
            )
            row.update(
                {
                    "seen_new_acc": new_ba,
                    "H_old_new": _harmonic(old_ba, new_ba),
                    "all_floor": all_floor,
                    "new_correct_count": sum(int(p == y) for p, y in new_truth),
                    "new_query_count": len(new_truth),
                    "new_by_class": new_counts,
                    "all_balanced_accuracy": all_ba,
                    "all_by_class": all_counts,
                }
            )
        row["state_metric_receipt_sha256"] = canonical_sha256(row)
        state_rows.append(row)
        by_scene.setdefault((surface.outer_id, surface.scene), {})[surface.state] = row
    if len(state_rows) != matrix.LOGICAL_STATE_SURFACE_COUNT:
        raise NextR5FATarget125Error("state metric output coverage drift")
    contrasts: list[dict[str, Any]] = []
    for outer in plan.outer_rows:
        for scene in matrix.SCENES:
            states = by_scene.get((outer.outer_id, scene), {})
            if tuple(states) != matrix.STATES:
                raise NextR5FATarget125Error("four-state score contrast coverage drift")
            before = states["DA0_REG0"]
            after_da = states["DA1_REG0"]
            reg = states["DA0_REG1"]
            joint = states["DA1_REG1"]
            contrast = {
                "outer_id": outer.outer_id,
                "receiver": outer.receiver,
                "seed": outer.seed,
                "k_shot": outer.k_shot,
                "new_count": outer.new_count,
                "scene": scene,
                "DA_effect_REG0_old_ba": after_da["old_balanced_accuracy"] - before["old_balanced_accuracy"],
                "DA_effect_REG1_old_ba": joint["old_balanced_accuracy"] - reg["old_balanced_accuracy"],
                "registration_effect_DA0_old_ba": reg["old_balanced_accuracy"] - before["old_balanced_accuracy"],
                "registration_effect_DA1_old_ba": joint["old_balanced_accuracy"] - after_da["old_balanced_accuracy"],
                "joint_interaction_old_ba": (joint["old_balanced_accuracy"] - reg["old_balanced_accuracy"]) - (after_da["old_balanced_accuracy"] - before["old_balanced_accuracy"]),
                "DA_effect_REG1_seen_new_acc": joint["seen_new_acc"] - reg["seen_new_acc"],
                "DA_effect_REG1_H_old_new": joint["H_old_new"] - reg["H_old_new"],
                "DA_effect_REG1_total_correct_count": joint["total_correct_count"] - reg["total_correct_count"],
                "DA_effect_REG0_old_floor": after_da["old_floor"] - before["old_floor"],
                "DA_effect_REG1_old_floor": joint["old_floor"] - reg["old_floor"],
                "registration_effect_DA0_old_floor": reg["old_floor"] - before["old_floor"],
                "registration_effect_DA1_old_floor": joint["old_floor"] - after_da["old_floor"],
                "joint_interaction_old_floor": (joint["old_floor"] - reg["old_floor"]) - (after_da["old_floor"] - before["old_floor"]),
                "DA_effect_REG0_all_floor": after_da["all_floor"] - before["all_floor"],
                "DA_effect_REG1_all_floor": joint["all_floor"] - reg["all_floor"],
                "registration_effect_DA0_all_floor": reg["all_floor"] - before["all_floor"],
                "registration_effect_DA1_all_floor": joint["all_floor"] - after_da["all_floor"],
                "joint_interaction_all_floor": (joint["all_floor"] - reg["all_floor"]) - (after_da["all_floor"] - before["all_floor"]),
                "DA_effect_REG0_total_correct_count": after_da["total_correct_count"] - before["total_correct_count"],
                "registration_effect_DA0_total_correct_count": reg["total_correct_count"] - before["total_correct_count"],
                "registration_effect_DA1_total_correct_count": joint["total_correct_count"] - after_da["total_correct_count"],
                "joint_interaction_total_correct_count": (joint["total_correct_count"] - reg["total_correct_count"]) - (after_da["total_correct_count"] - before["total_correct_count"]),
            }
            contrast["contrast_receipt_sha256"] = canonical_sha256(contrast)
            contrasts.append(contrast)
    return MappingProxyType(
        {
            "schema": SCORE_SCHEMA,
            "candidate_id": matrix.CANDIDATE_ID,
            "protocol_schema": matrix.PROTOCOL_SCHEMA,
            "prediction_manifest_sha256": prediction["manifest_receipt_sha256"],
            "prediction_closure_verified": True,
            "truth_open": True,
            "outer_job_count": matrix.OUTER_JOB_COUNT,
            "scene_row_count": matrix.SCENE_ROW_COUNT,
            "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
            "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT,
            "alias_count": matrix.ALIAS_COUNT,
            "state_names_zh": dict(matrix.STATE_NAMES_ZH),
            "state_rows": state_rows,
            "four_state_contrasts": contrasts,
        }
    )


def _preflight_d92_apply_bundle(
    reference: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load the D92 bundle verifier only on the post-seal truth side."""

    from .somph_diagnostic_bundle_loader import preflight_somph_predictor_bundle

    return preflight_somph_predictor_bundle(
        reference["package_root"],
        detached_seal_path=reference["detached_seal_path"],
        expected_seal_sha256=reference["expected_seal_sha256"],
    )


def _sealed_d92_apply_registry(
    *,
    source_row: Mapping[str, Any],
    outer: matrix.Target125OuterRow,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Recover row-local D92 handles from the sealed source-context packages.

    The FA asset carries a Phase1-global old-class order, whereas the D92
    packages deliberately expose per-row opaque ``cls_*`` handles.  Truth
    opening therefore must use the latter.  This function preflights only the
    sealed apply-package manifests; it never reads prediction labels, query
    truth, or query IQ.
    """

    if (
        source_row.get("receiver") != outer.receiver
        or source_row.get("seed") != outer.seed
        or source_row.get("k_shot") != outer.k_shot
        or source_row.get("new_count") != outer.new_count
        or source_row.get("active_k") != outer.k_shot
        or source_row.get("source_pool_k") != outer.source_pool_k
    ):
        raise NextR5FATarget125Error("truth-side source-context outer binding drift")
    packages = source_row.get("packages")
    expected_package_keys = {
        "before_enrollment",
        "before_apply",
        "after_enrollment",
        "after_apply",
    }
    if not isinstance(packages, Mapping) or set(packages) != expected_package_keys:
        raise NextR5FATarget125Error("truth-side sealed D92 package map drift")

    def load_registry(
        package_name: str,
        *,
        expected_stage: str,
        expected_registration_state: str,
        expected_count: int,
    ) -> tuple[str, ...]:
        reference = packages[package_name]
        expected_reference_keys = {
            "package_root",
            "detached_seal_path",
            "expected_seal_sha256",
        }
        if not isinstance(reference, Mapping) or set(reference) != expected_reference_keys:
            raise NextR5FATarget125Error(
                "truth-side sealed D92 apply-package reference drift"
            )
        try:
            manifest, _seal, _audit = _preflight_d92_apply_bundle(reference)
        except Exception as error:
            raise NextR5FATarget125Error(
                "truth-side sealed D92 apply-package preflight failed"
            ) from error
        if not isinstance(manifest, Mapping) or (
            manifest.get("profile") != "apply_only"
            or manifest.get("stage") != expected_stage
            or manifest.get("registration_state") != expected_registration_state
            or manifest.get("receiver") != outer.receiver
            or manifest.get("seed") != outer.seed
            or manifest.get("k_shot") != outer.source_pool_k
            or manifest.get("registered_class_count") != expected_count
        ):
            raise NextR5FATarget125Error(
                "truth-side sealed D92 apply-package identity/registry drift"
            )
        rows = manifest.get("registered_classes")
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise NextR5FATarget125Error(
                "truth-side sealed D92 apply-package class registry drift"
            )
        handles: list[str] = []
        for class_index, item in enumerate(rows):
            if (
                not isinstance(item, Mapping)
                or set(item) != {"class_index", "class_handle"}
                or item.get("class_index") != class_index
                or type(item.get("class_handle")) is not str
                or not item["class_handle"]
            ):
                raise NextR5FATarget125Error(
                    "truth-side sealed D92 apply-package class-index bridge drift"
                )
            handles.append(item["class_handle"])
        return _tokens(
            handles,
            name="truth-side sealed D92 apply-package class handles",
            expected=expected_count,
        )

    old_classes = load_registry(
        "before_apply",
        expected_stage="stage2b",
        expected_registration_state="before",
        expected_count=matrix.OLD_CLASS_COUNT,
    )
    all_classes = load_registry(
        "after_apply",
        expected_stage="stage2c",
        expected_registration_state="after",
        expected_count=matrix.OLD_CLASS_COUNT + outer.new_count,
    )
    if all_classes[: matrix.OLD_CLASS_COUNT] != old_classes:
        raise NextR5FATarget125Error(
            "truth-side D92 after registry does not preserve the old-class prefix"
        )
    return old_classes, all_classes[matrix.OLD_CLASS_COUNT :]


def _validate_prediction_registry_binding(
    *,
    prediction: Mapping[str, Any],
    outer: matrix.Target125OuterRow,
    old_classes: tuple[str, ...],
    new_classes: tuple[str, ...],
) -> None:
    """Bind the truth-side D92 registry to r10's truth-free class metadata.

    This reads only the sealed ``registered_classes`` metadata, never the
    prediction labels or query truth.  It prevents a row-local class-handle
    mismatch from being silently scored after the prediction manifest is fixed.
    """

    records = prediction.get("records")
    if not isinstance(records, Mapping):
        raise NextR5FATarget125Error("sealed prediction registry records are missing")
    expected_by_state = {
        "DA0_REG0": old_classes,
        "DA1_REG0": old_classes,
        "DA0_REG1": (*old_classes, *new_classes),
        "DA1_REG1": (*old_classes, *new_classes),
    }
    for scene in matrix.SCENES:
        scene_row_id = matrix.make_scene_row_id(outer.outer_id, scene)
        for state, expected in expected_by_state.items():
            surface_id = matrix.make_surface_id(scene_row_id, state)
            record = records.get(surface_id)
            if not isinstance(record, Mapping):
                raise NextR5FATarget125Error(
                    "sealed prediction registry surface coverage drift"
                )
            actual = _tokens(
                record.get("registered_classes"),
                name=f"sealed prediction registry[{surface_id}]",
                expected=len(expected),
            )
            if actual != expected:
                raise NextR5FATarget125Error(
                    "sealed prediction/D92 registry binding drift"
                )


def build_target125_truth_catalog(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
    output_path: str | Path,
) -> Mapping[str, Any]:
    """Open D92 truth only after the complete Target125 prediction seal."""

    prediction = validate_prediction_manifest(
        prediction_manifest_path=prediction_manifest_path,
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )
    try:
        from . import stage2_d108_truth_scorer as d108_truth
        from . import stage2_next_r5_fa_target125_runtime as runtime

        _target_plan, target_context, source_plan, source_context = runtime._load_prepared_next_r5_inputs(  # type: ignore[attr-defined]
            plan_manifest_path=plan_manifest_path,
            expected_plan_file_sha256=expected_plan_file_sha256,
            context_manifest_path=context_manifest_path,
            expected_context_file_sha256=expected_context_file_sha256,
        )
    except Exception as error:
        raise NextR5FATarget125Error("Target125 scorer-side prepared input reload failed") from error
    frozen = matrix.freeze_next_r5_fa_target125_matrix()
    surfaces: list[dict[str, Any]] = []
    for index, outer in enumerate(frozen.outer_rows):
        target_row = target_context["rows"][index]
        source_index = target_row.get("source_row_index") if isinstance(target_row, Mapping) else None
        if type(source_index) is not int or source_index not in range(matrix.OUTER_JOB_COUNT):
            raise NextR5FATarget125Error("truth-side Target125 source-row binding drift")
        source_row = source_context["rows"][source_index]
        if not isinstance(source_row, Mapping):
            raise NextR5FATarget125Error("truth-side D108 source row is malformed")
        old_classes, new_classes = _sealed_d92_apply_registry(
            source_row=source_row,
            outer=outer,
        )
        _validate_prediction_registry_binding(
            prediction=prediction,
            outer=outer,
            old_classes=old_classes,
            new_classes=new_classes,
        )
        d108_outer = {
            "receiver": outer.receiver,
            "seed": outer.seed,
            "k_shot": outer.k_shot,
            "new_count": outer.new_count,
            "old_classes": list(old_classes),
            "new_classes": list(new_classes),
        }
        try:
            sidecar = d108_truth._load_d92_truth_sidecar_for_outer(  # type: ignore[attr-defined]
                plan=source_plan, row=source_row, outer=d108_outer
            )
        except Exception as error:
            raise NextR5FATarget125Error("D92 truth-sidecar binding failed") from error
        seen_after_query_ids: set[str] = set()
        for scene in matrix.SCENES:
            scene_row_id = matrix.make_scene_row_id(outer.outer_id, scene)
            reg0_surface_id = matrix.make_surface_id(scene_row_id, "DA0_REG0")
            reg1_surface_id = matrix.make_surface_id(scene_row_id, "DA0_REG1")
            reg0_query_ids = prediction["decoded"][reg0_surface_id][0]
            reg1_query_ids = prediction["decoded"][reg1_surface_id][0]
            if seen_after_query_ids.intersection(reg1_query_ids) or any(
                query_id not in sidecar for query_id in reg1_query_ids
            ):
                raise NextR5FATarget125Error("D92 truth-side query scene/coverage drift")
            reg1_truth_rows = [sidecar[query_id] for query_id in reg1_query_ids]
            expected_reg0_query_ids = tuple(
                query_id
                for query_id, truth_row in zip(reg1_query_ids, reg1_truth_rows, strict=True)
                if truth_row.get("role") == "target_old"
            )
            if tuple(reg0_query_ids) != expected_reg0_query_ids:
                raise NextR5FATarget125Error(
                    "REG0 query IDs must equal the ordered target_old subset of REG1"
                )
            seen_after_query_ids.update(reg1_query_ids)
            for state in matrix.STATES:
                surface_id = matrix.make_surface_id(
                    matrix.make_scene_row_id(outer.outer_id, scene), state
                )
                query_ids, _labels_pred = prediction["decoded"][surface_id]
                try:
                    labels = [sidecar[query_id]["label"] for query_id in query_ids]
                except (KeyError, TypeError) as error:
                    raise NextR5FATarget125Error("truth-side query physical-ID binding drift") from error
                surfaces.append(
                    {
                        "surface_id": surface_id,
                        "ordered_query_physical_ids": list(query_ids),
                        "labels": labels,
                    }
                )
        if seen_after_query_ids != set(sidecar):
            raise NextR5FATarget125Error("D92 truth-side query coverage drift")
    if len(surfaces) != matrix.LOGICAL_STATE_SURFACE_COUNT:
        raise NextR5FATarget125Error("Target125 truth catalog surface coverage drift")
    catalog: dict[str, Any] = {
        "schema": TRUTH_CATALOG_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "truth_open": True,
        "prediction_manifest_sha256": prediction["manifest_receipt_sha256"],
        "outer_job_count": matrix.OUTER_JOB_COUNT,
        "scene_row_count": matrix.SCENE_ROW_COUNT,
        "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
        "surfaces": surfaces,
    }
    catalog["truth_catalog_sha256"] = canonical_sha256(catalog)
    destination = Path(output_path)
    return MappingProxyType(
        {
            "truth_catalog": str(destination),
            "truth_catalog_file_sha256": _write_json_new(destination, catalog),
            "truth_catalog_sha256": catalog["truth_catalog_sha256"],
            "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
        }
    )


def _load_target125_truth_catalog(
    *,
    truth_catalog_path: str | Path,
    expected_truth_catalog_file_sha256: str,
    prediction_manifest_receipt_sha256: str,
    prediction_query_ids: Mapping[str, Sequence[str]],
) -> Mapping[str, Sequence[str]]:
    document = _read_json(
        Path(truth_catalog_path),
        expected_sha256=expected_truth_catalog_file_sha256,
        name="Target125 independent truth catalog",
    )
    required = {
        "schema", "candidate_id", "protocol_schema", "truth_open", "prediction_manifest_sha256",
        "outer_job_count", "scene_row_count", "logical_state_surface_count", "surfaces", "truth_catalog_sha256",
    }
    if set(document) != required or (
        document["schema"] != TRUTH_CATALOG_SCHEMA
        or document["candidate_id"] != matrix.CANDIDATE_ID
        or document["protocol_schema"] != matrix.PROTOCOL_SCHEMA
        or document["truth_open"] is not True
        or document["prediction_manifest_sha256"] != prediction_manifest_receipt_sha256
        or document["outer_job_count"] != matrix.OUTER_JOB_COUNT
        or document["scene_row_count"] != matrix.SCENE_ROW_COUNT
        or document["logical_state_surface_count"] != matrix.LOGICAL_STATE_SURFACE_COUNT
    ):
        raise NextR5FATarget125Error("Target125 truth catalog identity/count drift")
    payload = dict(document)
    receipt = payload.pop("truth_catalog_sha256")
    if _sha(receipt, "truth catalog receipt") != canonical_sha256(payload):
        raise NextR5FATarget125Error("Target125 truth catalog receipt drift")
    raw_surfaces = document["surfaces"]
    if not isinstance(raw_surfaces, list) or len(raw_surfaces) != matrix.LOGICAL_STATE_SURFACE_COUNT:
        raise NextR5FATarget125Error("Target125 truth catalog surfaces drift")
    expected_ids = tuple(surface.surface_id for surface in matrix.freeze_next_r5_fa_target125_matrix().surfaces)
    truths: dict[str, Sequence[str]] = {}
    for raw, expected_id in zip(raw_surfaces, expected_ids, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {"surface_id", "ordered_query_physical_ids", "labels"}:
            raise NextR5FATarget125Error("Target125 truth surface field closure drift")
        query_ids = _tokens(raw["ordered_query_physical_ids"], name=f"truth[{expected_id}].query_ids")
        labels = _tokens(raw["labels"], name=f"truth[{expected_id}].labels", expected=len(query_ids), unique=False)
        if raw["surface_id"] != expected_id or expected_id in truths:
            raise NextR5FATarget125Error("Target125 truth surface/order drift")
        sealed_query_ids = prediction_query_ids.get(expected_id)
        if sealed_query_ids is None or tuple(query_ids) != tuple(sealed_query_ids):
            raise NextR5FATarget125Error(
                "Target125 truth query physical-ID order does not match the sealed prediction"
            )
        truths[expected_id] = labels
    return MappingProxyType(truths)


def score_target125_from_files(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    truth_catalog_path: str | Path,
    expected_truth_catalog_file_sha256: str,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Score a closed prediction manifest using an already opened catalog."""

    prediction = validate_prediction_manifest(
        prediction_manifest_path=prediction_manifest_path,
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )
    truth = _load_target125_truth_catalog(
        truth_catalog_path=truth_catalog_path,
        expected_truth_catalog_file_sha256=expected_truth_catalog_file_sha256,
        prediction_manifest_receipt_sha256=prediction["manifest_receipt_sha256"],
        prediction_query_ids={
            surface_id: decoded[0]
            for surface_id, decoded in prediction["decoded"].items()
        },
    )
    result = dict(
        score_target125_truth_catalog(
            prediction_manifest_path=prediction_manifest_path,
            expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
            truth_catalog=truth,
        )
    )
    result["truth_catalog_file_sha256"] = _sha(expected_truth_catalog_file_sha256, "truth catalog SHA256")
    result["truth_catalog_sha256"] = canonical_sha256(
        _read_json(Path(truth_catalog_path), expected_sha256=expected_truth_catalog_file_sha256, name="Target125 independent truth catalog")
    )
    result["score_receipt_sha256"] = canonical_sha256(result)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable score output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise NextR5FATarget125Error("unsafe score output parent")
    destination.mkdir()
    score_path = destination / "score.json"
    return MappingProxyType(
        {
            "score": str(score_path),
            "score_file_sha256": _write_json_new(score_path, result),
            "score_receipt_sha256": result["score_receipt_sha256"],
            "logical_state_surface_count": matrix.LOGICAL_STATE_SURFACE_COUNT,
        }
    )


__all__ = [
    "NextR5FATarget125Error",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA",
    "SCORE_SCHEMA",
    "TRUTH_CATALOG_SCHEMA",
    "build_target125_truth_catalog",
    "build_prediction_manifest",
    "canonical_sha256",
    "score_target125_from_files",
    "score_target125_truth_catalog",
    "seal_k1_alias",
    "seal_unique_prediction",
    "validate_prediction_manifest",
]
