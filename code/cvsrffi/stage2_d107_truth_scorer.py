"""Independent, fail-closed truth-side scorer for the D107 SCMKRR Target125.

The predictor must finish and seal every one of the 3,000 immutable surfaces
before this module opens an independent truth catalog.  This file deliberately
does not import the D106 K router (or any predictor): D107 evaluates all four
causal arms as submitted and never selects, routes, retries, or early-stops on
performance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
from typing import Any


PROTOCOL_SCHEMA = "p2_min_v1"
PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.d107.scmkrr.target125.prediction_manifest.v1"
PREDICTION_ARTIFACT_SCHEMA = "cvs.phase2.d107.scmkrr.target125.prediction_artifact.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d107.scmkrr.target125.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.phase2.d107.scmkrr.target125.truth_open_event.v1"
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d107.scmkrr.target125.score_manifest.v1"

ARMS = ("M0", "M_DA", "M_HEAD", "M_JOINT")
PHASES = ("before", "after")
OUTER_JOB_COUNT = 125
SCENE_ROW_COUNT = 375
ARM_PAIR_COUNT = 1_500
SURFACE_COUNT = 3_000
TRUTH_SURFACE_COUNT = 750
SCENE_ARM_METRIC_ROW_COUNT = 1_500
OUTER_ARM_AGGREGATE_ROW_COUNT = 500

_ACCESS_LEDGER_FIELDS = (
    "clean_source_runtime_access",
    "query_fit_access",
    "query_update_access",
    "query_truth_access",
    "query_role_access",
    "query_selection_access",
)
_PREDICTION_MANIFEST_FIELDS = {
    "schema",
    "candidate_id",
    "protocol_schema",
    "manifest_sealed",
    "truth_open",
    "outer_job_count",
    "scene_row_count",
    "arm_pair_count",
    "surface_count",
    "scenes",
    "arms",
    "phases",
    "outer_rows",
    "access_ledger",
    "surfaces",
    "manifest_sha256",
}
_OUTER_ROW_FIELDS = {
    "outer_id",
    "receiver",
    "seed",
    "k_shot",
    "new_count",
    "old_classes",
    "new_classes",
}
_SURFACE_FIELDS = {
    "surface_id",
    "outer_id",
    "receiver",
    "seed",
    "k_shot",
    "new_count",
    "scene",
    "arm",
    "phase",
    "registered_classes",
    "prediction_artifact",
    "prediction_artifact_sha256",
    "ordered_query_physical_ids",
    "ordered_query_physical_ids_sha256",
    "predicted_labels",
    "predicted_labels_sha256",
    "access_ledger",
    "truth_open",
    "immutable",
}
_ARTIFACT_FIELDS = (
    _SURFACE_FIELDS
    - {"prediction_artifact", "prediction_artifact_sha256"}
    | {"schema", "artifact_receipt_sha256"}
)
_TRUTH_CATALOG_FIELDS = {
    "schema",
    "truth_open",
    "prediction_manifest_sha256",
    "outer_job_count",
    "scene_row_count",
    "truth_surface_count",
    "scenes",
    "phases",
    "surfaces",
    "truth_catalog_sha256",
}
_TRUTH_SURFACE_FIELDS = {
    "outer_id",
    "receiver",
    "seed",
    "k_shot",
    "new_count",
    "scene",
    "phase",
    "ordered_query_physical_ids",
    "labels",
}
_ALLOWED_PREDICTION_TRUTH_ROLE_FIELDS = {
    "truth_open",
    "query_truth_access",
    "query_role_access",
}
_FORMAL_COMPARATORS = frozenset({"D62", "D92", "SVRN"})
_D91_DEVELOPMENT_STATUS = "D91_DEVELOPMENT_ONLY_15_ROWS_NON_PROMOTABLE"


class D107TruthScorerError(ValueError):
    """Raised when a D107 prediction or truth-side artifact fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D107TruthScorerError("canonical JSON payload is invalid") from error


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA256 used by D107 immutable receipts."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise D107TruthScorerError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise D107TruthScorerError(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise D107TruthScorerError(f"{name} must be non-empty trimmed text")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise D107TruthScorerError(f"{name} must be an integer >= {minimum}")
    return value


def _require_exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise D107TruthScorerError(f"{name} field closure drift")
    return value


def _require_string_list(value: Any, name: str, *, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise D107TruthScorerError(f"{name} must be a non-empty list")
    values = tuple(_require_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if unique and len(values) != len(set(values)):
        raise D107TruthScorerError(f"{name} contains duplicate values")
    return values


def _access_ledger(value: Any, name: str) -> dict[str, bool]:
    ledger = _require_exact_keys(value, set(_ACCESS_LEDGER_FIELDS), name)
    if any(ledger[field] is not False for field in _ACCESS_LEDGER_FIELDS):
        raise D107TruthScorerError(f"{name} must deny every query/source access")
    return {field: False for field in _ACCESS_LEDGER_FIELDS}


def _reject_forbidden_prediction_truth_role_fields(value: Any, name: str) -> None:
    """Reject leaked truth/role material before accepting prediction payloads."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _require_text(raw_key, f"{name} key")
            lowered = key.lower()
            if (
                ("truth" in lowered or "role" in lowered)
                and key not in _ALLOWED_PREDICTION_TRUTH_ROLE_FIELDS
            ):
                raise D107TruthScorerError(
                    f"{name} contains forbidden truth/role field: {key}"
                )
            _reject_forbidden_prediction_truth_role_fields(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_prediction_truth_role_fields(item, f"{name}[{index}]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_regular(
    path: str | Path,
    *,
    name: str,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], int]:
    candidate = Path(path)
    expected = _require_sha256(expected_file_sha256, f"expected {name} file SHA256")
    if candidate.is_symlink() or not candidate.is_file():
        raise D107TruthScorerError(f"{name} must be a regular file")
    mode = candidate.stat().st_mode
    if not stat.S_ISREG(mode):
        raise D107TruthScorerError(f"{name} must be a regular file")
    observed = _sha256_file(candidate)
    if observed != expected:
        raise D107TruthScorerError(f"{name} SHA mismatch")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D107TruthScorerError(f"{name} is not valid UTF-8 JSON") from error
    if type(payload) is not dict:
        raise D107TruthScorerError(f"{name} must be a JSON object")
    return payload, candidate.stat().st_size


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise D107TruthScorerError("immutable output parent is unsafe")
    payload = _canonical_bytes(value)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise D107TruthScorerError(f"unable to write immutable output: {path}") from error
    return _sha256_file(path)


def _receipt(value: Mapping[str, Any], field: str, name: str) -> str:
    receipt = _require_sha256(value.get(field), f"{name}.{field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if canonical_sha256(unsigned) != receipt:
        raise D107TruthScorerError(f"{name} canonical receipt mismatch")
    return receipt


def _expected_outer_id(row: Mapping[str, Any]) -> str:
    return (
        f"d107-rx-{row['receiver']}__seed-{row['seed']}"
        f"__k-{row['k_shot']}__new-{row['new_count']}"
    )


def _expected_surface_id(
    outer_id: str, scene: str, arm: str, phase: str
) -> str:
    return f"{outer_id}__scene-{scene}__arm-{arm}__phase-{phase}"


def _validate_outer_rows(value: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list) or len(value) != OUTER_JOB_COUNT:
        raise D107TruthScorerError("outer_rows must contain exactly 125 rows")
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    identity_keys: set[tuple[str, int, int, int]] = set()
    for index, raw in enumerate(value):
        row = _require_exact_keys(raw, _OUTER_ROW_FIELDS, f"outer_rows[{index}]")
        receiver = _require_text(row["receiver"], f"outer_rows[{index}].receiver")
        seed = _require_int(row["seed"], f"outer_rows[{index}].seed")
        k_shot = _require_int(row["k_shot"], f"outer_rows[{index}].k_shot", minimum=1)
        new_count = _require_int(
            row["new_count"], f"outer_rows[{index}].new_count", minimum=1
        )
        outer_id = _require_text(row["outer_id"], f"outer_rows[{index}].outer_id")
        expected_outer_id = _expected_outer_id(
            {
                "receiver": receiver,
                "seed": seed,
                "k_shot": k_shot,
                "new_count": new_count,
            }
        )
        if outer_id != expected_outer_id:
            raise D107TruthScorerError("outer row ID binding drift")
        old_classes = _require_string_list(row["old_classes"], f"outer_rows[{index}].old_classes")
        new_classes = _require_string_list(row["new_classes"], f"outer_rows[{index}].new_classes")
        if set(old_classes) & set(new_classes):
            raise D107TruthScorerError("outer row old/new class overlap")
        if len(new_classes) != new_count:
            raise D107TruthScorerError("outer row new_count/class coverage drift")
        identity = (receiver, seed, k_shot, new_count)
        if outer_id in by_id or identity in identity_keys:
            raise D107TruthScorerError("duplicate outer row identity")
        normalized = {
            "outer_id": outer_id,
            "receiver": receiver,
            "seed": seed,
            "k_shot": k_shot,
            "new_count": new_count,
            "old_classes": old_classes,
            "new_classes": new_classes,
        }
        rows.append(normalized)
        by_id[outer_id] = normalized
        identity_keys.add(identity)
    return rows, by_id


def _validate_surface_scalar_fields(
    surface: Mapping[str, Any], outer: Mapping[str, Any], *, index: int, scenes: tuple[str, ...]
) -> tuple[str, str, str, tuple[str, ...]]:
    outer_id = _require_text(surface["outer_id"], f"surfaces[{index}].outer_id")
    if outer_id != outer["outer_id"]:
        raise D107TruthScorerError("surface outer row binding drift")
    for field in ("receiver", "seed", "k_shot", "new_count"):
        if surface[field] != outer[field]:
            raise D107TruthScorerError("surface outer metadata binding drift")
    scene = _require_text(surface["scene"], f"surfaces[{index}].scene")
    arm = _require_text(surface["arm"], f"surfaces[{index}].arm")
    phase = _require_text(surface["phase"], f"surfaces[{index}].phase")
    if scene not in scenes or arm not in ARMS or phase not in PHASES:
        raise D107TruthScorerError("surface scene/arm/phase binding drift")
    surface_id = _require_text(surface["surface_id"], f"surfaces[{index}].surface_id")
    if surface_id != _expected_surface_id(outer_id, scene, arm, phase):
        raise D107TruthScorerError("surface ID binding drift")
    registered = _require_string_list(
        surface["registered_classes"], f"surfaces[{index}].registered_classes"
    )
    expected_registered = (
        outer["old_classes"]
        if phase == "before"
        else (*outer["old_classes"], *outer["new_classes"])
    )
    if registered != expected_registered:
        raise D107TruthScorerError("surface registered-class closure drift")
    return scene, arm, phase, registered


def _validate_list_receipts(
    surface: Mapping[str, Any], *, index: int, registered: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    query_ids = _require_string_list(
        surface["ordered_query_physical_ids"],
        f"surfaces[{index}].ordered_query_physical_ids",
    )
    labels = _require_string_list(
        surface["predicted_labels"], f"surfaces[{index}].predicted_labels", unique=False
    )
    if len(query_ids) != len(labels):
        raise D107TruthScorerError("surface prediction/query length drift")
    if canonical_sha256(list(query_ids)) != _require_sha256(
        surface["ordered_query_physical_ids_sha256"],
        f"surfaces[{index}].ordered_query_physical_ids_sha256",
    ):
        raise D107TruthScorerError("surface ordered query-ID SHA mismatch")
    if canonical_sha256(list(labels)) != _require_sha256(
        surface["predicted_labels_sha256"],
        f"surfaces[{index}].predicted_labels_sha256",
    ):
        raise D107TruthScorerError("surface predicted-label SHA mismatch")
    if any(label not in registered for label in labels):
        raise D107TruthScorerError("surface prediction falls outside registered classes")
    return query_ids, labels


def _resolve_prediction_artifact(manifest_path: Path, raw_path: Any) -> Path:
    value = _require_text(raw_path, "prediction_artifact")
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts or "\\" in value:
        raise D107TruthScorerError("prediction artifact path must be portable and relative")
    root = manifest_path.parent.resolve(strict=True)
    candidate = root.joinpath(*portable.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise D107TruthScorerError("prediction artifact escapes manifest root") from error
    if candidate.is_symlink() or resolved.is_symlink():
        raise D107TruthScorerError("prediction artifact must not be a symlink")
    return resolved


def _validate_artifact(
    *,
    artifact: Mapping[str, Any],
    surface: Mapping[str, Any],
    index: int,
) -> None:
    _reject_forbidden_prediction_truth_role_fields(artifact, f"artifact[{index}]")
    document = _require_exact_keys(artifact, _ARTIFACT_FIELDS, f"artifact[{index}]")
    if document["schema"] != PREDICTION_ARTIFACT_SCHEMA:
        raise D107TruthScorerError("prediction artifact schema drift")
    _receipt(document, "artifact_receipt_sha256", f"artifact[{index}]")
    for field in _SURFACE_FIELDS - {"prediction_artifact", "prediction_artifact_sha256"}:
        if document[field] != surface[field]:
            raise D107TruthScorerError("prediction artifact/surface binding drift")


def _validate_prediction_manifest(
    prediction_manifest_path: Path,
    *,
    expected_prediction_manifest_file_sha256: str,
) -> dict[str, Any]:
    manifest, manifest_size = _read_json_regular(
        prediction_manifest_path,
        name="D107 prediction manifest",
        expected_file_sha256=expected_prediction_manifest_file_sha256,
    )
    _reject_forbidden_prediction_truth_role_fields(manifest, "prediction manifest")
    document = _require_exact_keys(manifest, _PREDICTION_MANIFEST_FIELDS, "prediction manifest")
    if document["schema"] != PREDICTION_MANIFEST_SCHEMA:
        raise D107TruthScorerError("prediction manifest schema drift")
    if not _require_text(document["candidate_id"], "candidate_id").startswith("D107-SCMKRR"):
        raise D107TruthScorerError("prediction manifest candidate binding drift")
    if document["protocol_schema"] != PROTOCOL_SCHEMA:
        raise D107TruthScorerError("prediction manifest protocol schema drift")
    if document["manifest_sealed"] is not True or document["truth_open"] is not False:
        raise D107TruthScorerError("prediction manifest must be sealed before truth opening")
    counts = {
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT,
        "surface_count": SURFACE_COUNT,
    }
    if any(document[field] != expected for field, expected in counts.items()):
        raise D107TruthScorerError("prediction manifest 125/375/1500/3000 coverage drift")
    scenes = _require_string_list(document["scenes"], "prediction manifest.scenes")
    if len(scenes) != 3:
        raise D107TruthScorerError("prediction manifest must contain exactly three scenes")
    if document["arms"] != list(ARMS) or document["phases"] != list(PHASES):
        raise D107TruthScorerError("prediction manifest arm/phase order drift")
    _access_ledger(document["access_ledger"], "prediction manifest.access_ledger")
    manifest_receipt = _receipt(document, "manifest_sha256", "prediction manifest")
    outer_rows, outer_by_id = _validate_outer_rows(document["outer_rows"])
    raw_surfaces = document["surfaces"]
    if not isinstance(raw_surfaces, list) or len(raw_surfaces) != SURFACE_COUNT:
        raise D107TruthScorerError("prediction manifest must contain exactly 3000 surfaces")

    surfaces: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    artifact_paths: set[Path] = set()
    artifact_bytes = 0
    support_slots = 0
    total_prediction_queries = 0
    for index, raw_surface in enumerate(raw_surfaces):
        _reject_forbidden_prediction_truth_role_fields(raw_surface, f"surfaces[{index}]")
        surface = _require_exact_keys(raw_surface, _SURFACE_FIELDS, f"surfaces[{index}]")
        outer_id = _require_text(surface["outer_id"], f"surfaces[{index}].outer_id")
        outer = outer_by_id.get(outer_id)
        if outer is None:
            raise D107TruthScorerError("surface references an unknown outer row")
        scene, arm, phase, registered = _validate_surface_scalar_fields(
            surface, outer, index=index, scenes=scenes
        )
        query_ids, labels = _validate_list_receipts(surface, index=index, registered=registered)
        if surface["truth_open"] is not False or surface["immutable"] is not True:
            raise D107TruthScorerError("surface must be immutable and truth-closed")
        _access_ledger(surface["access_ledger"], f"surfaces[{index}].access_ledger")
        expected_path = f"predictions/{surface['surface_id']}.json"
        if surface["prediction_artifact"] != expected_path:
            raise D107TruthScorerError("prediction artifact path binding drift")
        artifact_path = _resolve_prediction_artifact(prediction_manifest_path, surface["prediction_artifact"])
        if artifact_path in artifact_paths:
            raise D107TruthScorerError("prediction artifact is reused across surfaces")
        artifact, artifact_size = _read_json_regular(
            artifact_path,
            name=f"prediction artifact {surface['surface_id']}",
            expected_file_sha256=_require_sha256(
                surface["prediction_artifact_sha256"],
                f"surfaces[{index}].prediction_artifact_sha256",
            ),
        )
        _validate_artifact(artifact=artifact, surface=surface, index=index)
        key = (outer_id, scene, arm, phase)
        if key in surfaces:
            raise D107TruthScorerError("duplicate prediction surface")
        surfaces[key] = {
            "surface": dict(surface),
            "query_ids": query_ids,
            "predictions": labels,
            "registered_classes": registered,
            "artifact_path": artifact_path,
            "artifact_size": artifact_size,
        }
        artifact_paths.add(artifact_path)
        artifact_bytes += artifact_size
        support_slots += outer["k_shot"] * len(registered)
        total_prediction_queries += len(query_ids)

    expected_keys = {
        (outer["outer_id"], scene, arm, phase)
        for outer in outer_rows
        for scene in scenes
        for arm in ARMS
        for phase in PHASES
    }
    if set(surfaces) != expected_keys:
        raise D107TruthScorerError("prediction surface coverage drift")
    _validate_matched_prediction_surfaces(surfaces, outer_rows=outer_rows, scenes=scenes)
    return {
        "manifest": document,
        "manifest_path": prediction_manifest_path.resolve(strict=True),
        "manifest_file_sha256": expected_prediction_manifest_file_sha256,
        "manifest_size": manifest_size,
        "manifest_receipt_sha256": manifest_receipt,
        "outer_rows": outer_rows,
        "outer_by_id": outer_by_id,
        "scenes": scenes,
        "surfaces": surfaces,
        "artifact_bytes": artifact_bytes,
        "support_slots": support_slots,
        "prediction_query_count": total_prediction_queries,
    }


def _validate_matched_prediction_surfaces(
    surfaces: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    outer_rows: Sequence[Mapping[str, Any]],
    scenes: Sequence[str],
) -> None:
    """Validate all-four-arm ID parity and before/after causal pair coverage."""

    for outer in outer_rows:
        for scene in scenes:
            for phase in PHASES:
                rows = [surfaces[(outer["outer_id"], scene, arm, phase)] for arm in ARMS]
                baseline = rows[0]["query_ids"]
                if any(row["query_ids"] != baseline for row in rows[1:]):
                    raise D107TruthScorerError("four-arm ordered query-ID parity drift")
            for arm in ARMS:
                before = surfaces[(outer["outer_id"], scene, arm, "before")]
                after = surfaces[(outer["outer_id"], scene, arm, "after")]
                if before["query_ids"] != after["query_ids"]:
                    raise D107TruthScorerError("before/after ordered query-ID parity drift")
                if before["registered_classes"] != tuple(outer["old_classes"]):
                    raise D107TruthScorerError("before old-class registry drift")
                if after["registered_classes"] != (
                    *outer["old_classes"],
                    *outer["new_classes"],
                ):
                    raise D107TruthScorerError("after old/new registry drift")


def validate_d107_prediction_manifest(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
) -> dict[str, Any]:
    """Validate only the sealed, truth-free D107 prediction closure.

    This function never receives, stats, reads, or opens a truth catalog.
    """

    return _validate_prediction_manifest(
        Path(prediction_manifest_path),
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )


def _validate_truth_catalog(
    truth_catalog_path: str | Path,
    *,
    expected_truth_catalog_file_sha256: str,
    prediction: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]], str, int]:
    """Open and validate truth only after a durable truth-open event exists."""

    catalog, catalog_size = _read_json_regular(
        truth_catalog_path,
        name="D107 independent truth catalog",
        expected_file_sha256=expected_truth_catalog_file_sha256,
    )
    document = _require_exact_keys(catalog, _TRUTH_CATALOG_FIELDS, "truth catalog")
    if document["schema"] != TRUTH_CATALOG_SCHEMA or document["truth_open"] is not True:
        raise D107TruthScorerError("truth catalog opening/schema drift")
    if document["prediction_manifest_sha256"] != prediction["manifest_receipt_sha256"]:
        raise D107TruthScorerError("truth catalog prediction manifest binding drift")
    expected_counts = {
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "truth_surface_count": TRUTH_SURFACE_COUNT,
    }
    if any(document[field] != expected for field, expected in expected_counts.items()):
        raise D107TruthScorerError("truth catalog 125/375/750 coverage drift")
    if document["scenes"] != list(prediction["scenes"]) or document["phases"] != list(PHASES):
        raise D107TruthScorerError("truth catalog scene/phase binding drift")
    truth_receipt = _receipt(document, "truth_catalog_sha256", "truth catalog")
    raw_surfaces = document["surfaces"]
    if not isinstance(raw_surfaces, list) or len(raw_surfaces) != TRUTH_SURFACE_COUNT:
        raise D107TruthScorerError("truth catalog must contain exactly 750 surfaces")
    truths: dict[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    outer_by_id = prediction["outer_by_id"]
    scenes = prediction["scenes"]
    for index, raw in enumerate(raw_surfaces):
        truth = _require_exact_keys(raw, _TRUTH_SURFACE_FIELDS, f"truth surfaces[{index}]")
        outer_id = _require_text(truth["outer_id"], f"truth surfaces[{index}].outer_id")
        outer = outer_by_id.get(outer_id)
        if outer is None:
            raise D107TruthScorerError("truth surface references an unknown outer row")
        for field in ("receiver", "seed", "k_shot", "new_count"):
            if truth[field] != outer[field]:
                raise D107TruthScorerError("truth surface outer metadata binding drift")
        scene = _require_text(truth["scene"], f"truth surfaces[{index}].scene")
        phase = _require_text(truth["phase"], f"truth surfaces[{index}].phase")
        if scene not in scenes or phase not in PHASES:
            raise D107TruthScorerError("truth surface scene/phase drift")
        query_ids = _require_string_list(
            truth["ordered_query_physical_ids"],
            f"truth surfaces[{index}].ordered_query_physical_ids",
        )
        labels = _require_string_list(truth["labels"], f"truth surfaces[{index}].labels", unique=False)
        if len(query_ids) != len(labels):
            raise D107TruthScorerError("truth query/label length drift")
        registry = set((*outer["old_classes"], *outer["new_classes"]))
        if any(label not in registry for label in labels):
            raise D107TruthScorerError("truth label falls outside the phase registry")
        expected_classes = (
            set(outer["old_classes"])
            if phase == "before"
            else registry
        )
        if not expected_classes.issubset(set(labels)):
            raise D107TruthScorerError("truth class coverage is incomplete")
        key = (outer_id, scene, phase)
        if key in truths:
            raise D107TruthScorerError("duplicate truth surface")
        prediction_ids = prediction["surfaces"][(outer_id, scene, ARMS[0], phase)]["query_ids"]
        if query_ids != prediction_ids:
            raise D107TruthScorerError("truth/prediction ordered query-ID mismatch")
        truths[key] = (query_ids, labels)
    expected_keys = {
        (outer["outer_id"], scene, phase)
        for outer in prediction["outer_rows"]
        for scene in scenes
        for phase in PHASES
    }
    if set(truths) != expected_keys:
        raise D107TruthScorerError("truth surface coverage drift")
    _validate_before_after_old_query_matching(truths, prediction)
    return truths, truth_receipt, catalog_size


def _validate_before_after_old_query_matching(
    truths: Mapping[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]],
    prediction: Mapping[str, Any],
) -> None:
    for outer in prediction["outer_rows"]:
        old = set(outer["old_classes"])
        for scene in prediction["scenes"]:
            before_ids, before_labels = truths[(outer["outer_id"], scene, "before")]
            after_ids, after_labels = truths[(outer["outer_id"], scene, "after")]
            before_old_ids = tuple(
                query_id
                for query_id, label in zip(before_ids, before_labels, strict=True)
                if label in old
            )
            after_old_ids = tuple(
                query_id
                for query_id, label in zip(after_ids, after_labels, strict=True)
                if label in old
            )
            if after_old_ids != before_old_ids:
                raise D107TruthScorerError("before/after old-query physical-ID mismatch")


def _rate(correct: int, total: int, name: str) -> float:
    if type(correct) is not int or type(total) is not int or total <= 0:
        raise D107TruthScorerError(f"{name} has an empty/invalid denominator")
    return 100.0 * correct / total


def _harmonic(old_accuracy: float, new_accuracy: float) -> float:
    denominator = old_accuracy + new_accuracy
    return 0.0 if denominator == 0.0 else 2.0 * old_accuracy * new_accuracy / denominator


def _class_counts(
    predictions: Sequence[str], labels: Sequence[str], classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    values = {label: {"correct_count": 0, "query_count": 0} for label in classes}
    for prediction, label in zip(predictions, labels, strict=True):
        if label not in values:
            raise D107TruthScorerError("old-class count received a non-old label")
        values[label]["query_count"] += 1
        values[label]["correct_count"] += int(prediction == label)
    if any(item["query_count"] == 0 for item in values.values()):
        raise D107TruthScorerError("old-class floor coverage drift")
    return values


def _floor(class_counts: Mapping[str, Mapping[str, int]], name: str) -> float:
    return min(
        _rate(values["correct_count"], values["query_count"], f"{name}.{label}")
        for label, values in class_counts.items()
    )


def _build_metric_row(
    *,
    outer: Mapping[str, Any],
    scene: str,
    arm: str,
    before_predictions: Sequence[str],
    before_truth: Sequence[str],
    after_predictions: Sequence[str],
    after_truth: Sequence[str],
) -> dict[str, Any]:
    old = set(outer["old_classes"])
    new = set(outer["new_classes"])
    before_old_pairs = [
        (prediction, label)
        for prediction, label in zip(before_predictions, before_truth, strict=True)
        if label in old
    ]
    before_old_correct = sum(prediction == label for prediction, label in before_old_pairs)
    after_old_pairs = [
        (prediction, label)
        for prediction, label in zip(after_predictions, after_truth, strict=True)
        if label in old
    ]
    after_new_pairs = [
        (prediction, label)
        for prediction, label in zip(after_predictions, after_truth, strict=True)
        if label in new
    ]
    if len(after_old_pairs) + len(after_new_pairs) != len(after_truth):
        raise D107TruthScorerError("after metric truth role partition drift")
    before_counts = _class_counts(
        [prediction for prediction, _ in before_old_pairs],
        [label for _, label in before_old_pairs],
        outer["old_classes"],
    )
    after_counts = _class_counts(
        [prediction for prediction, _ in after_old_pairs],
        [label for _, label in after_old_pairs],
        outer["old_classes"],
    )
    after_old_correct = sum(prediction == label for prediction, label in after_old_pairs)
    seen_new_correct = sum(prediction == label for prediction, label in after_new_pairs)
    before_old_query_count = len(before_old_pairs)
    after_old_query_count = len(after_old_pairs)
    seen_new_query_count = len(after_new_pairs)
    before_old = _rate(before_old_correct, before_old_query_count, "before_old")
    after_old = _rate(after_old_correct, after_old_query_count, "after_old")
    seen_new = _rate(seen_new_correct, seen_new_query_count, "seen_new")
    post_total_correct = after_old_correct + seen_new_correct
    post_total_query = after_old_query_count + seen_new_query_count
    total_correct = before_old_correct + post_total_correct
    total_query = before_old_query_count + post_total_query
    values = {
        "outer_id": outer["outer_id"],
        "receiver": outer["receiver"],
        "seed": outer["seed"],
        "k_shot": outer["k_shot"],
        "new_count": outer["new_count"],
        "scene": scene,
        "arm": arm,
        "before_old": before_old,
        "after_old": after_old,
        "before_old_floor": _floor(before_counts, "before_old_floor"),
        "after_old_floor": _floor(after_counts, "after_old_floor"),
        "seen_new": seen_new,
        "H_old_new": _harmonic(after_old, seen_new),
        "forgetting": before_old - after_old,
        "before_old_correct_count": before_old_correct,
        "before_old_query_count": before_old_query_count,
        "after_old_correct_count": after_old_correct,
        "after_old_query_count": after_old_query_count,
        "old_correct_count": after_old_correct,
        "old_query_count": after_old_query_count,
        "new_correct_count": seen_new_correct,
        "new_query_count": seen_new_query_count,
        "post_registration_total_correct_count": post_total_correct,
        "post_registration_total_query_count": post_total_query,
        "total_correct_count": total_correct,
        "total_query_count": total_query,
        "before_old_by_class": before_counts,
        "after_old_by_class": after_counts,
    }
    if any(
        not math.isfinite(value)
        for key, value in values.items()
        if key
        in {
            "before_old",
            "after_old",
            "before_old_floor",
            "after_old_floor",
            "seen_new",
            "H_old_new",
            "forgetting",
        }
    ):
        raise D107TruthScorerError("non-finite metric row")
    values["metric_row_receipt_sha256"] = canonical_sha256(values)
    return values


def _merge_class_counts(
    rows: Sequence[Mapping[str, Any]], field: str, classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    totals = {label: {"correct_count": 0, "query_count": 0} for label in classes}
    for row in rows:
        counts = row[field]
        if set(counts) != set(classes):
            raise D107TruthScorerError("aggregate old-class coverage drift")
        for label in classes:
            totals[label]["correct_count"] += counts[label]["correct_count"]
            totals[label]["query_count"] += counts[label]["query_count"]
    return totals


def _aggregate_outer_arm(
    *, outer: Mapping[str, Any], arm: str, scene_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(scene_rows) != 3 or any(row["arm"] != arm for row in scene_rows):
        raise D107TruthScorerError("outer-arm aggregate source coverage drift")
    count_fields = (
        "before_old_correct_count",
        "before_old_query_count",
        "after_old_correct_count",
        "after_old_query_count",
        "new_correct_count",
        "new_query_count",
        "post_registration_total_correct_count",
        "post_registration_total_query_count",
        "total_correct_count",
        "total_query_count",
    )
    totals = {field: sum(int(row[field]) for row in scene_rows) for field in count_fields}
    before_counts = _merge_class_counts(scene_rows, "before_old_by_class", outer["old_classes"])
    after_counts = _merge_class_counts(scene_rows, "after_old_by_class", outer["old_classes"])
    before_old = _rate(
        totals["before_old_correct_count"], totals["before_old_query_count"], "aggregate.before_old"
    )
    after_old = _rate(
        totals["after_old_correct_count"], totals["after_old_query_count"], "aggregate.after_old"
    )
    seen_new = _rate(totals["new_correct_count"], totals["new_query_count"], "aggregate.seen_new")
    result = {
        "outer_id": outer["outer_id"],
        "receiver": outer["receiver"],
        "seed": outer["seed"],
        "k_shot": outer["k_shot"],
        "new_count": outer["new_count"],
        "arm": arm,
        "scenes": [row["scene"] for row in scene_rows],
        "aggregation": "micro_average_across_three_scenes",
        "before_old": before_old,
        "after_old": after_old,
        "before_old_floor": _floor(before_counts, "aggregate.before_old_floor"),
        "after_old_floor": _floor(after_counts, "aggregate.after_old_floor"),
        "seen_new": seen_new,
        "H_old_new": _harmonic(after_old, seen_new),
        "forgetting": before_old - after_old,
        **totals,
        "old_correct_count": totals["after_old_correct_count"],
        "old_query_count": totals["after_old_query_count"],
        "before_old_by_class": before_counts,
        "after_old_by_class": after_counts,
        "source_scene_metric_row_receipt_sha256s": [
            row["metric_row_receipt_sha256"] for row in scene_rows
        ],
    }
    result["outer_arm_receipt_sha256"] = canonical_sha256(result)
    return result


def _score_rows(
    prediction: Mapping[str, Any],
    truths: Mapping[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_same_rows: list[dict[str, Any]] = []
    per_outer_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for outer in prediction["outer_rows"]:
        for scene in prediction["scenes"]:
            arm_rows: list[dict[str, Any]] = []
            before_truth = truths[(outer["outer_id"], scene, "before")][1]
            after_truth = truths[(outer["outer_id"], scene, "after")][1]
            for arm in ARMS:
                before = prediction["surfaces"][(outer["outer_id"], scene, arm, "before")]
                after = prediction["surfaces"][(outer["outer_id"], scene, arm, "after")]
                row = _build_metric_row(
                    outer=outer,
                    scene=scene,
                    arm=arm,
                    before_predictions=before["predictions"],
                    before_truth=before_truth,
                    after_predictions=after["predictions"],
                    after_truth=after_truth,
                )
                arm_rows.append(row)
                per_outer_arm[(outer["outer_id"], arm)].append(row)
            scene_same_row: dict[str, Any] = {
                "outer_id": outer["outer_id"],
                "receiver": outer["receiver"],
                "seed": outer["seed"],
                "k_shot": outer["k_shot"],
                "new_count": outer["new_count"],
                "scene": scene,
                "arms": arm_rows,
            }
            scene_same_row["scene_same_row_receipt_sha256"] = canonical_sha256(scene_same_row)
            scene_same_rows.append(scene_same_row)
    outer_arm_rows = [
        _aggregate_outer_arm(
            outer=outer,
            arm=arm,
            scene_rows=per_outer_arm[(outer["outer_id"], arm)],
        )
        for outer in prediction["outer_rows"]
        for arm in ARMS
    ]
    if (
        len(scene_same_rows) != SCENE_ROW_COUNT
        or sum(len(row["arms"]) for row in scene_same_rows) != SCENE_ARM_METRIC_ROW_COUNT
        or len(outer_arm_rows) != OUTER_ARM_AGGREGATE_ROW_COUNT
    ):
        raise D107TruthScorerError("metric output row coverage drift")
    return scene_same_rows, outer_arm_rows


def _score_summary(
    prediction: Mapping[str, Any], *, truth_catalog_size: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    coverage = {
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT,
        "prediction_surface_count": SURFACE_COUNT,
        "truth_surface_count": TRUTH_SURFACE_COUNT,
        "scene_same_row_count": SCENE_ROW_COUNT,
        "scene_arm_metric_row_count": SCENE_ARM_METRIC_ROW_COUNT,
        "outer_arm_aggregate_row_count": OUTER_ARM_AGGREGATE_ROW_COUNT,
        "prediction_closure_verified": True,
        "before_after_old_query_matching_verified": True,
        "four_arm_causal_coverage_verified": True,
    }
    resources = {
        "prediction_manifest_bytes": prediction["manifest_size"],
        "prediction_artifact_bytes": prediction["artifact_bytes"],
        "truth_catalog_bytes": truth_catalog_size,
        "unique_prediction_artifact_count": SURFACE_COUNT,
        "prediction_query_count": prediction["prediction_query_count"],
        "registered_support_slot_count": prediction["support_slots"],
        "resource_source": "sealed_artifact_inventory_only",
    }
    verdict = {
        "primary_candidate_arm": "M_JOINT",
        "causal_arms": list(ARMS),
        "causal_table_preserved": True,
        "coverage_verdict": "COMPLETE_125_TRUTH_OPEN_AND_SCORED",
        "target_thresholds_declared": False,
        "target_verdict": "NO_TARGET_THRESHOLD_DECLARED",
        "performance_early_stop_or_selection_performed": False,
        "D91_status": _D91_DEVELOPMENT_STATUS,
    }
    return coverage, resources, verdict


def score_d107_target125(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    truth_catalog_path: str | Path,
    expected_truth_catalog_file_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Seal-check all predictions, then truth-open and score the complete matrix.

    No performance result affects dispatch, candidate selection, thresholds, or
    retries.  The return value is an artifact handoff, not a promotion verdict.
    """

    prediction = validate_d107_prediction_manifest(
        prediction_manifest_path=prediction_manifest_path,
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )
    expected_truth_sha = _require_sha256(
        expected_truth_catalog_file_sha256, "expected truth catalog file SHA256"
    )
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable score output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D107TruthScorerError("unsafe score output parent")
    destination.mkdir()

    event: dict[str, Any] = {
        "schema": TRUTH_OPEN_EVENT_SCHEMA,
        "prediction_manifest_sha256": prediction["manifest_receipt_sha256"],
        "prediction_manifest_file_sha256": prediction["manifest_file_sha256"],
        "expected_truth_catalog_file_sha256": expected_truth_sha,
        "prediction_closure_verified": True,
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT,
        "surface_count": SURFACE_COUNT,
    }
    event["truth_open_event_receipt_sha256"] = canonical_sha256(event)
    event_path = destination / "truth_open_event.json"
    event_file_sha = _write_json_new(event_path, event)

    truths, truth_receipt, truth_catalog_size = _validate_truth_catalog(
        truth_catalog_path,
        expected_truth_catalog_file_sha256=expected_truth_sha,
        prediction=prediction,
    )
    scene_same_rows, outer_arm_rows = _score_rows(prediction, truths)
    coverage, resources, verdict = _score_summary(
        prediction, truth_catalog_size=truth_catalog_size
    )
    score: dict[str, Any] = {
        "schema": SCORE_MANIFEST_SCHEMA,
        "candidate_id": prediction["manifest"]["candidate_id"],
        "protocol_schema": PROTOCOL_SCHEMA,
        "prediction_manifest_sha256": prediction["manifest_receipt_sha256"],
        "prediction_manifest_file_sha256": prediction["manifest_file_sha256"],
        "truth_catalog_sha256": truth_receipt,
        "truth_catalog_file_sha256": expected_truth_sha,
        "truth_open_event_receipt_sha256": event["truth_open_event_receipt_sha256"],
        "scene_same_row_count": len(scene_same_rows),
        "scene_arm_metric_row_count": sum(len(row["arms"]) for row in scene_same_rows),
        "outer_arm_aggregate_row_count": len(outer_arm_rows),
        "scene_same_rows": scene_same_rows,
        "outer_arm_aggregate_rows": outer_arm_rows,
        "coverage_summary": coverage,
        "resource_summary": resources,
        "target_verdict_summary": verdict,
    }
    score["score_manifest_sha256"] = canonical_sha256(score)
    score_path = destination / "score_manifest.json"
    score_file_sha = _write_json_new(score_path, score)
    return {
        "truth_open_event": str(event_path),
        "truth_open_event_file_sha256": event_file_sha,
        "score_manifest": str(score_path),
        "score_manifest_file_sha256": score_file_sha,
        "score_manifest_sha256": score["score_manifest_sha256"],
        "scene_same_row_count": len(scene_same_rows),
        "scene_arm_metric_row_count": sum(len(row["arms"]) for row in scene_same_rows),
        "outer_arm_aggregate_row_count": len(outer_arm_rows),
    }


def _comparator_identity(row: Mapping[str, Any], name: str) -> tuple[str, int, int, int, str]:
    required = ("receiver", "seed", "k_shot", "new_count", "scene")
    if any(field not in row for field in required):
        raise D107TruthScorerError(f"{name} comparator row identity is incomplete")
    return (
        _require_text(row["receiver"], f"{name}.receiver"),
        _require_int(row["seed"], f"{name}.seed"),
        _require_int(row["k_shot"], f"{name}.k_shot", minimum=1),
        _require_int(row["new_count"], f"{name}.new_count", minimum=1),
        _require_text(row["scene"], f"{name}.scene"),
    )


def pair_d107_same_row(
    *,
    scene_same_rows: Sequence[Mapping[str, Any]],
    comparator_id: str,
    comparator_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pair comparator evidence only on an identical outer/scene identity.

    D62, D92 and SVRN retain an evidence-only same-row label.  D91 is fenced
    to precisely fifteen development rows and is never returned as formal
    confirmation evidence.
    """

    comparator = _require_text(comparator_id, "comparator_id")
    if comparator not in _FORMAL_COMPARATORS and comparator != "D91":
        raise D107TruthScorerError("unsupported comparator ID")
    if not isinstance(comparator_rows, Sequence) or isinstance(comparator_rows, (str, bytes)):
        raise D107TruthScorerError("comparator_rows must be a sequence")
    if comparator == "D91" and len(comparator_rows) != 15:
        raise D107TruthScorerError("D91 must be labelled as exactly 15 development rows")
    comparator_by_identity: dict[tuple[str, int, int, int, str], Mapping[str, Any]] = {}
    for index, row in enumerate(comparator_rows):
        if not isinstance(row, Mapping):
            raise D107TruthScorerError("comparator row must be a mapping")
        identity = _comparator_identity(row, f"comparator_rows[{index}]")
        if identity in comparator_by_identity:
            raise D107TruthScorerError("duplicate comparator same-row identity")
        comparator_by_identity[identity] = row
    pairs: list[dict[str, Any]] = []
    for d107_row in scene_same_rows:
        if not isinstance(d107_row, Mapping):
            raise D107TruthScorerError("D107 scene row must be a mapping")
        identity = _comparator_identity(d107_row, "D107 scene row")
        comparator_row = comparator_by_identity.get(identity)
        if comparator_row is None:
            continue
        pair: dict[str, Any] = {
            "comparator_id": comparator,
            "pairing_status": (
                _D91_DEVELOPMENT_STATUS
                if comparator == "D91"
                else "SAME_ROW_EVIDENCE_ONLY_NOT_PROMOTION"
            ),
            "receiver": identity[0],
            "seed": identity[1],
            "k_shot": identity[2],
            "new_count": identity[3],
            "scene": identity[4],
            "d107_scene_same_row": dict(d107_row),
            "comparator_row": dict(comparator_row),
        }
        pair["same_row_pair_receipt_sha256"] = canonical_sha256(pair)
        pairs.append(pair)
    result: dict[str, Any] = {
        "comparator_id": comparator,
        "pairing_status": (
            _D91_DEVELOPMENT_STATUS
            if comparator == "D91"
            else "SAME_ROW_EVIDENCE_ONLY_NOT_PROMOTION"
        ),
        "comparator_input_row_count": len(comparator_rows),
        "matched_row_count": len(pairs),
        "pairs": pairs,
    }
    result["pairing_receipt_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "ARMS",
    "ARM_PAIR_COUNT",
    "D107TruthScorerError",
    "OUTER_ARM_AGGREGATE_ROW_COUNT",
    "OUTER_JOB_COUNT",
    "PHASES",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PREDICTION_MANIFEST_SCHEMA",
    "PROTOCOL_SCHEMA",
    "SCENE_ARM_METRIC_ROW_COUNT",
    "SCENE_ROW_COUNT",
    "SCORE_MANIFEST_SCHEMA",
    "SURFACE_COUNT",
    "TRUTH_CATALOG_SCHEMA",
    "TRUTH_OPEN_EVENT_SCHEMA",
    "TRUTH_SURFACE_COUNT",
    "canonical_sha256",
    "pair_d107_same_row",
    "score_d107_target125",
    "validate_d107_prediction_manifest",
]
