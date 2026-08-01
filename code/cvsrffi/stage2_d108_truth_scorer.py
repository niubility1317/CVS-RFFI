"""Independent, fail-closed truth-side scorer for D108 Target125.

Prediction artifacts must form one immutable 3,000-surface closure before
this module opens any independently held D92 truth sidecar.  The scorer never
routes, retries, tunes, or selects a D108 arm from performance evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from .stage2_d108_matrix_protocol import (
    ARMS,
    ARM_PAIR_COUNT,
    CANDIDATE_ID,
    OUTER_JOB_COUNT,
    PHASES,
    PROTOCOL_SCHEMA,
    SCENE_ROW_COUNT,
    SURFACE_COUNT,
    canonical_bytes,
    canonical_sha256,
)


PREDICTION_MANIFEST_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target125.prediction_manifest.v1"
PREDICTION_ARTIFACT_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target125.prediction_artifact.v1"
TRUTH_CATALOG_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target125.truth_catalog.v1"
TRUTH_OPEN_EVENT_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target125.truth_open_event.v1"
SCORE_MANIFEST_SCHEMA = "cvs.phase2.d108.cbrrc_smme.target125.score_manifest.v1"

D92_TRUTH_SIDECAR_SCHEMA = "cvs.phase2.query_truth_sidecar.v2"
D92_OFFLINE_BUILD_SCHEMA = "cvs.phase2.somph_offline_row_pair_build.v2"

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
    _SURFACE_FIELDS - {"prediction_artifact", "prediction_artifact_sha256"}
) | {"schema", "artifact_receipt_sha256"}
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
_D92_TRUTH_SIDECAR_FIELDS = {"schema", "stage", "receiver", "seed", "rows"}
_D92_TRUTH_ROW_FIELDS = {
    "query_token",
    "true_class_index",
    "true_class_handle",
    "transmitter_label",
    "evaluation_role",
    "receiver_label",
    "day_label",
    "signal_label",
    "physical_sample_id",
}
_D92_QUERY_TOKEN_RE = re.compile(r"qid_[0-9a-f]{32,64}")
_ALLOWED_PREDICTION_TRUTH_ROLE_FIELDS = {
    "truth_open",
    "query_truth_access",
    "query_role_access",
}
_FORMAL_COMPARATORS = frozenset({"D62", "D92", "SVRN"})
_D91_DEVELOPMENT_STATUS = "D91_DEVELOPMENT_ONLY_15_ROWS_NON_PROMOTABLE"


class D108TruthScorerError(ValueError):
    """Raised when a D108 prediction or truth artifact fails closed."""


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise D108TruthScorerError(f"{name} must be a lowercase SHA256")
    if any(character not in "0123456789abcdef" for character in value):
        raise D108TruthScorerError(f"{name} must be a lowercase SHA256")
    return value


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise D108TruthScorerError(f"{name} must be non-empty trimmed text")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise D108TruthScorerError(f"{name} must be an integer >= {minimum}")
    return value


def _exact_fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise D108TruthScorerError(f"{name} field closure drift")
    return value


def _string_list(value: Any, name: str, *, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise D108TruthScorerError(f"{name} must be a non-empty list")
    result = tuple(_require_text(item, f"{name}[{index}]") for index, item in enumerate(value))
    if unique and len(result) != len(set(result)):
        raise D108TruthScorerError(f"{name} contains duplicate values")
    return result


def _access_ledger(value: Any, name: str) -> dict[str, bool]:
    ledger = _exact_fields(value, set(_ACCESS_LEDGER_FIELDS), name)
    if any(ledger[field] is not False for field in _ACCESS_LEDGER_FIELDS):
        raise D108TruthScorerError(f"{name} must deny every query/source access")
    return {field: False for field in _ACCESS_LEDGER_FIELDS}


def _reject_prediction_truth_role_fields(value: Any, name: str) -> None:
    """Reject truth or semantic-role material in every prediction payload."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _require_text(raw_key, f"{name} key")
            lowered = key.lower()
            if (
                ("truth" in lowered or "role" in lowered)
                and key not in _ALLOWED_PREDICTION_TRUTH_ROLE_FIELDS
            ):
                raise D108TruthScorerError(
                    f"{name} contains forbidden truth/role field: {key}"
                )
            _reject_prediction_truth_role_fields(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_prediction_truth_role_fields(item, f"{name}[{index}]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_regular(
    path: str | Path, *, name: str, expected_file_sha256: str
) -> tuple[dict[str, Any], int]:
    candidate = Path(path)
    expected = _require_sha256(expected_file_sha256, f"expected {name} file SHA256")
    if candidate.is_symlink() or not candidate.is_file():
        raise D108TruthScorerError(f"{name} must be a regular file")
    if not stat.S_ISREG(candidate.stat().st_mode):
        raise D108TruthScorerError(f"{name} must be a regular file")
    if _sha256_file(candidate) != expected:
        raise D108TruthScorerError(f"{name} SHA mismatch")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D108TruthScorerError(f"{name} is not valid UTF-8 JSON") from error
    if type(payload) is not dict:
        raise D108TruthScorerError(f"{name} must be a JSON object")
    return payload, candidate.stat().st_size


def _read_json_unpinned_regular(
    path: str | Path, *, name: str
) -> tuple[dict[str, Any], str, Path]:
    """Read an independent D92 receipt which pins its sidecar instead."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise D108TruthScorerError(f"{name} must be a regular file")
    if not stat.S_ISREG(candidate.stat().st_mode):
        raise D108TruthScorerError(f"{name} must be a regular file")
    try:
        raw = candidate.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D108TruthScorerError(f"{name} is not valid UTF-8 JSON") from error
    if type(payload) is not dict:
        raise D108TruthScorerError(f"{name} must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest(), candidate.resolve(strict=True)


def _regular_directory(path: str | Path, *, name: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_dir():
        raise D108TruthScorerError(f"{name} must be a regular non-symlink directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise D108TruthScorerError(f"{name} cannot be resolved") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise D108TruthScorerError(f"{name} must be a regular non-symlink directory")
    return resolved


def _safe_job_id(value: Any) -> str:
    job_id = _require_text(value, "source_d92_job_id")
    if job_id in {".", ".."} or any(character in job_id for character in ("/", "\\", ":")):
        raise D108TruthScorerError("source_d92_job_id is not a safe path component")
    return job_id


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise D108TruthScorerError("immutable output parent is unsafe")
    payload = canonical_bytes(value)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise D108TruthScorerError(f"unable to write immutable output: {path}") from error
    return _sha256_file(path)


def _receipt(value: Mapping[str, Any], field: str, name: str) -> str:
    receipt = _require_sha256(value.get(field), f"{name}.{field}")
    unsigned = dict(value)
    unsigned.pop(field)
    if canonical_sha256(unsigned) != receipt:
        raise D108TruthScorerError(f"{name} canonical receipt mismatch")
    return receipt


def _expected_outer_id(row: Mapping[str, Any]) -> str:
    return (
        f"d108-rx-{row['receiver']}__seed-{row['seed']}"
        f"__k-{row['k_shot']}__new-{row['new_count']}"
    )


def _expected_surface_id(outer_id: str, scene: str, arm: str, phase: str) -> str:
    return f"{outer_id}__scene-{scene}__arm-{arm}__phase-{phase}"


def _validate_outer_rows(value: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list) or len(value) != OUTER_JOB_COUNT:
        raise D108TruthScorerError("outer_rows must contain exactly 125 rows")
    rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, int, int, int]] = set()
    for index, raw in enumerate(value):
        row = _exact_fields(raw, _OUTER_ROW_FIELDS, f"outer_rows[{index}]")
        receiver = _require_text(row["receiver"], f"outer_rows[{index}].receiver")
        seed = _require_int(row["seed"], f"outer_rows[{index}].seed")
        k_shot = _require_int(row["k_shot"], f"outer_rows[{index}].k_shot", minimum=1)
        new_count = _require_int(row["new_count"], f"outer_rows[{index}].new_count", minimum=1)
        outer_id = _require_text(row["outer_id"], f"outer_rows[{index}].outer_id")
        if outer_id != _expected_outer_id(
            {"receiver": receiver, "seed": seed, "k_shot": k_shot, "new_count": new_count}
        ):
            raise D108TruthScorerError("outer row ID binding drift")
        old = _string_list(row["old_classes"], f"outer_rows[{index}].old_classes")
        new = _string_list(row["new_classes"], f"outer_rows[{index}].new_classes")
        if set(old) & set(new) or len(new) != new_count:
            raise D108TruthScorerError("outer row class coverage drift")
        identity = (receiver, seed, k_shot, new_count)
        if outer_id in by_id or identity in identities:
            raise D108TruthScorerError("duplicate outer row identity")
        normalized = {
            "outer_id": outer_id,
            "receiver": receiver,
            "seed": seed,
            "k_shot": k_shot,
            "new_count": new_count,
            "old_classes": old,
            "new_classes": new,
        }
        rows.append(normalized)
        by_id[outer_id] = normalized
        identities.add(identity)
    return rows, by_id


def _resolve_prediction_artifact(manifest_path: Path, raw_path: Any) -> Path:
    value = _require_text(raw_path, "prediction_artifact")
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts or "\\" in value:
        raise D108TruthScorerError("prediction artifact path must be portable and relative")
    root = manifest_path.parent.resolve(strict=True)
    candidate = root.joinpath(*portable.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise D108TruthScorerError("prediction artifact escapes manifest root") from error
    if candidate.is_symlink() or resolved.is_symlink():
        raise D108TruthScorerError("prediction artifact must not be a symlink")
    return resolved


def _validate_surface(
    *,
    surface: Mapping[str, Any],
    outer: Mapping[str, Any],
    index: int,
    scenes: Sequence[str],
    manifest_path: Path,
) -> dict[str, Any]:
    _reject_prediction_truth_role_fields(surface, f"surfaces[{index}]")
    document = _exact_fields(surface, _SURFACE_FIELDS, f"surfaces[{index}]")
    outer_id = _require_text(document["outer_id"], f"surfaces[{index}].outer_id")
    if outer_id != outer["outer_id"] or any(
        document[field] != outer[field]
        for field in ("receiver", "seed", "k_shot", "new_count")
    ):
        raise D108TruthScorerError("surface outer metadata binding drift")
    scene = _require_text(document["scene"], f"surfaces[{index}].scene")
    arm = _require_text(document["arm"], f"surfaces[{index}].arm")
    phase = _require_text(document["phase"], f"surfaces[{index}].phase")
    if scene not in scenes or arm not in ARMS or phase not in PHASES:
        raise D108TruthScorerError("surface scene/arm/phase binding drift")
    if document["surface_id"] != _expected_surface_id(outer_id, scene, arm, phase):
        raise D108TruthScorerError("surface ID binding drift")
    registered = _string_list(document["registered_classes"], f"surfaces[{index}].registered_classes")
    expected_registered = outer["old_classes"] if phase == "before" else (
        *outer["old_classes"], *outer["new_classes"]
    )
    if registered != expected_registered:
        raise D108TruthScorerError("surface registered-class closure drift")
    query_ids = _string_list(
        document["ordered_query_physical_ids"],
        f"surfaces[{index}].ordered_query_physical_ids",
    )
    labels = _string_list(document["predicted_labels"], f"surfaces[{index}].predicted_labels", unique=False)
    if len(query_ids) != len(labels):
        raise D108TruthScorerError("surface prediction/query length drift")
    if canonical_sha256(list(query_ids)) != _require_sha256(
        document["ordered_query_physical_ids_sha256"],
        f"surfaces[{index}].ordered_query_physical_ids_sha256",
    ) or canonical_sha256(list(labels)) != _require_sha256(
        document["predicted_labels_sha256"],
        f"surfaces[{index}].predicted_labels_sha256",
    ):
        raise D108TruthScorerError("surface list SHA mismatch")
    if any(label not in registered for label in labels):
        raise D108TruthScorerError("surface prediction falls outside registered classes")
    if document["truth_open"] is not False or document["immutable"] is not True:
        raise D108TruthScorerError("surface must be immutable and truth-closed")
    _access_ledger(document["access_ledger"], f"surfaces[{index}].access_ledger")
    expected_path = f"predictions/{document['surface_id']}.json"
    if document["prediction_artifact"] != expected_path:
        raise D108TruthScorerError("prediction artifact path binding drift")
    artifact_path = _resolve_prediction_artifact(manifest_path, document["prediction_artifact"])
    artifact, artifact_size = _read_json_regular(
        artifact_path,
        name=f"prediction artifact {document['surface_id']}",
        expected_file_sha256=_require_sha256(
            document["prediction_artifact_sha256"],
            f"surfaces[{index}].prediction_artifact_sha256",
        ),
    )
    _reject_prediction_truth_role_fields(artifact, f"artifact[{index}]")
    artifact = _exact_fields(artifact, _ARTIFACT_FIELDS, f"artifact[{index}]")
    if artifact["schema"] != PREDICTION_ARTIFACT_SCHEMA:
        raise D108TruthScorerError("prediction artifact schema drift")
    _receipt(artifact, "artifact_receipt_sha256", f"artifact[{index}]")
    for field in _SURFACE_FIELDS - {"prediction_artifact", "prediction_artifact_sha256"}:
        if artifact[field] != document[field]:
            raise D108TruthScorerError("prediction artifact/surface binding drift")
    return {
        "surface": dict(document),
        "outer_id": outer_id,
        "scene": scene,
        "arm": arm,
        "phase": phase,
        "query_ids": query_ids,
        "predictions": labels,
        "registered_classes": registered,
        "artifact_path": artifact_path,
        "artifact_size": artifact_size,
    }


def _validate_matched_prediction_surfaces(
    surfaces: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    outer_rows: Sequence[Mapping[str, Any]],
    scenes: Sequence[str],
) -> None:
    for outer in outer_rows:
        for scene in scenes:
            for phase in PHASES:
                rows = [surfaces[(outer["outer_id"], scene, arm, phase)] for arm in ARMS]
                if any(row["query_ids"] != rows[0]["query_ids"] for row in rows[1:]):
                    raise D108TruthScorerError("four-arm ordered query-ID parity drift")
            for arm in ARMS:
                before = surfaces[(outer["outer_id"], scene, arm, "before")]
                after = surfaces[(outer["outer_id"], scene, arm, "after")]
                if before["registered_classes"] != tuple(outer["old_classes"]) or after[
                    "registered_classes"
                ] != (*outer["old_classes"], *outer["new_classes"]):
                    raise D108TruthScorerError("before/after registered-class closure drift")


def _validate_prediction_manifest(
    prediction_manifest_path: Path, *, expected_prediction_manifest_file_sha256: str
) -> dict[str, Any]:
    manifest, manifest_size = _read_json_regular(
        prediction_manifest_path,
        name="D108 prediction manifest",
        expected_file_sha256=expected_prediction_manifest_file_sha256,
    )
    _reject_prediction_truth_role_fields(manifest, "prediction manifest")
    document = _exact_fields(manifest, _PREDICTION_MANIFEST_FIELDS, "prediction manifest")
    if (
        document["schema"] != PREDICTION_MANIFEST_SCHEMA
        or document["candidate_id"] != CANDIDATE_ID
        or document["protocol_schema"] != PROTOCOL_SCHEMA
        or document["manifest_sealed"] is not True
        or document["truth_open"] is not False
    ):
        raise D108TruthScorerError("prediction manifest identity/seal drift")
    if any(
        document[field] != expected
        for field, expected in {
            "outer_job_count": OUTER_JOB_COUNT,
            "scene_row_count": SCENE_ROW_COUNT,
            "arm_pair_count": ARM_PAIR_COUNT,
            "surface_count": SURFACE_COUNT,
        }.items()
    ):
        raise D108TruthScorerError("prediction manifest 125/375/1500/3000 coverage drift")
    scenes = _string_list(document["scenes"], "prediction manifest.scenes")
    if len(scenes) != 3 or document["arms"] != list(ARMS) or document["phases"] != list(PHASES):
        raise D108TruthScorerError("prediction manifest scene/arm/phase order drift")
    _access_ledger(document["access_ledger"], "prediction manifest.access_ledger")
    manifest_receipt = _receipt(document, "manifest_sha256", "prediction manifest")
    outer_rows, outer_by_id = _validate_outer_rows(document["outer_rows"])
    raw_surfaces = document["surfaces"]
    if not isinstance(raw_surfaces, list) or len(raw_surfaces) != SURFACE_COUNT:
        raise D108TruthScorerError("prediction manifest must contain exactly 3000 surfaces")
    surfaces: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    artifact_paths: set[Path] = set()
    artifact_bytes = support_slots = query_count = 0
    for index, raw_surface in enumerate(raw_surfaces):
        if not isinstance(raw_surface, Mapping):
            raise D108TruthScorerError("prediction surface must be an object")
        outer_id = _require_text(raw_surface.get("outer_id"), f"surfaces[{index}].outer_id")
        outer = outer_by_id.get(outer_id)
        if outer is None:
            raise D108TruthScorerError("surface references an unknown outer row")
        item = _validate_surface(
            surface=raw_surface,
            outer=outer,
            index=index,
            scenes=scenes,
            manifest_path=prediction_manifest_path,
        )
        key = (item["outer_id"], item["scene"], item["arm"], item["phase"])
        if key in surfaces or item["artifact_path"] in artifact_paths:
            raise D108TruthScorerError("duplicate prediction surface or reused artifact")
        surfaces[key] = item
        artifact_paths.add(item["artifact_path"])
        artifact_bytes += item["artifact_size"]
        support_slots += outer["k_shot"] * len(item["registered_classes"])
        query_count += len(item["query_ids"])
    expected_keys = {
        (outer["outer_id"], scene, arm, phase)
        for outer in outer_rows
        for scene in scenes
        for arm in ARMS
        for phase in PHASES
    }
    if set(surfaces) != expected_keys:
        raise D108TruthScorerError("prediction surface coverage drift")
    _validate_matched_prediction_surfaces(surfaces, outer_rows=outer_rows, scenes=scenes)
    return {
        "manifest": dict(document),
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
        "prediction_query_count": query_count,
    }


def validate_d108_prediction_manifest(
    *, prediction_manifest_path: str | Path, expected_prediction_manifest_file_sha256: str
) -> dict[str, Any]:
    """Validate only the sealed, truth-free D108 prediction closure."""

    return _validate_prediction_manifest(
        Path(prediction_manifest_path),
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )


def _load_prepared_d108_truth_inputs(
    *,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from .stage2_d108_target125_runner import _prepared_inputs

        return _prepared_inputs(
            plan_manifest_path=Path(plan_manifest_path),
            expected_plan_file_sha256=expected_plan_file_sha256,
            context_manifest_path=Path(context_manifest_path),
            expected_context_file_sha256=expected_context_file_sha256,
        )
    except D108TruthScorerError:
        raise
    except Exception as error:
        raise D108TruthScorerError("D108 prepared plan/context validation failed") from error


def _validate_d92_truth_sidecar(
    sidecar: Mapping[str, Any], *, outer: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    document = _exact_fields(sidecar, _D92_TRUTH_SIDECAR_FIELDS, "D92 truth sidecar")
    if (
        document["schema"] != D92_TRUTH_SIDECAR_SCHEMA
        or document["stage"] != "stage2c"
        or document["receiver"] != outer["receiver"]
        or document["seed"] != outer["seed"]
    ):
        raise D108TruthScorerError("D92 truth sidecar identity drift")
    rows = document["rows"]
    if not isinstance(rows, list) or not rows:
        raise D108TruthScorerError("D92 truth sidecar rows must be non-empty")
    old = tuple(outer["old_classes"])
    new = tuple(outer["new_classes"])
    registry = (*old, *new)
    class_index = {label: index for index, label in enumerate(registry)}
    by_token: dict[str, dict[str, str]] = {}
    physical_ids: set[str] = set()
    role_classes = {"target_old": set(), "target_new": set()}
    for index, raw in enumerate(rows):
        row = _exact_fields(raw, _D92_TRUTH_ROW_FIELDS, f"D92 truth rows[{index}]")
        token = _require_text(row["query_token"], f"D92 truth rows[{index}].query_token")
        if _D92_QUERY_TOKEN_RE.fullmatch(token) is None or token in by_token:
            raise D108TruthScorerError("D92 truth sidecar query-token closure drift")
        label = _require_text(row["true_class_handle"], f"D92 truth rows[{index}].true_class_handle")
        role = _require_text(row["evaluation_role"], f"D92 truth rows[{index}].evaluation_role")
        expected_role = set(old if role == "target_old" else new if role == "target_new" else ())
        if not expected_role or label not in expected_role:
            raise D108TruthScorerError("D92 truth sidecar registry/role binding drift")
        if _require_int(row["true_class_index"], f"D92 truth rows[{index}].true_class_index") != class_index[label]:
            raise D108TruthScorerError("D92 truth sidecar class-index binding drift")
        if row["receiver_label"] != document["receiver"]:
            raise D108TruthScorerError("D92 truth sidecar receiver-label drift")
        _require_text(row["transmitter_label"], f"D92 truth rows[{index}].transmitter_label")
        _require_text(row["day_label"], f"D92 truth rows[{index}].day_label")
        _require_text(row["signal_label"], f"D92 truth rows[{index}].signal_label")
        physical_id = _require_text(row["physical_sample_id"], f"D92 truth rows[{index}].physical_sample_id")
        if physical_id in physical_ids:
            raise D108TruthScorerError("D92 truth sidecar physical-ID reuse drift")
        physical_ids.add(physical_id)
        role_classes[role].add(label)
        by_token[token] = {"label": label, "role": role, "physical_sample_id": physical_id}
    if role_classes["target_old"] != set(old) or role_classes["target_new"] != set(new):
        raise D108TruthScorerError("D92 truth sidecar class coverage drift")
    return by_token


def _load_d92_truth_sidecar_for_outer(
    *, plan: Mapping[str, Any], row: Mapping[str, Any], outer: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    identity = plan.get("identity")
    if not isinstance(identity, Mapping):
        raise D108TruthScorerError("D108 plan identity is missing")
    d92_root = _regular_directory(
        _require_text(identity.get("d92_output_root"), "D92 output root"), name="D92 output root"
    )
    jobs_root = _regular_directory(d92_root / "jobs", name="D92 jobs root")
    job_root = _regular_directory(jobs_root / _safe_job_id(row.get("source_d92_job_id")), name="D92 source job root")
    try:
        job_root.relative_to(jobs_root)
    except ValueError as error:
        raise D108TruthScorerError("D92 source job escapes the jobs root") from error
    packages = row.get("packages")
    if not isinstance(packages, Mapping) or not isinstance(packages.get("after_apply"), Mapping):
        raise D108TruthScorerError("D108 row after-apply package binding is missing")
    apply_root = _regular_directory(
        _require_text(packages["after_apply"].get("package_root"), "D92 after-apply package root"),
        name="D92 after-apply package root",
    )
    expected_apply = _regular_directory(
        job_root / "offline" / "predictor" / "after" / "apply_only_staging",
        name="expected D92 after-apply package root",
    )
    if apply_root != expected_apply:
        raise D108TruthScorerError("D108 after-apply/source-job path binding drift")
    receipt, _receipt_sha, _receipt_path = _read_json_unpinned_regular(
        job_root / "offline" / "offline_build_receipt.json", name="D92 offline-build receipt"
    )
    if (
        receipt.get("schema") != D92_OFFLINE_BUILD_SCHEMA
        or receipt.get("receiver") != outer["receiver"]
        or receipt.get("seed") != outer["seed"]
        or receipt.get("k_shot") != row.get("source_pool_k")
        or receipt.get("new_class_count") != outer["new_count"]
    ):
        raise D108TruthScorerError("D92 offline-build receipt outer-row binding drift")
    states = receipt.get("states")
    if not isinstance(states, Mapping) or not isinstance(states.get("after"), Mapping):
        raise D108TruthScorerError("D92 offline-build receipt after-state is missing")
    if _regular_directory(
        _require_text(states["after"].get("apply_staging_root"), "D92 receipt after apply root"),
        name="D92 receipt after apply root",
    ) != apply_root:
        raise D108TruthScorerError("D92 offline-build receipt/apply package binding drift")
    truth_path = job_root / "offline" / "scorer" / "truth_sidecar.json"
    receipt_truth_path = Path(_require_text(receipt.get("truth_sidecar"), "D92 receipt truth-sidecar path"))
    try:
        if receipt_truth_path.resolve(strict=True) != truth_path.resolve(strict=True):
            raise D108TruthScorerError("D92 receipt truth-sidecar path binding drift")
    except OSError as error:
        raise D108TruthScorerError("D92 receipt truth-sidecar path cannot be resolved") from error
    sidecar, _size = _read_json_regular(
        truth_path,
        name="D92 truth sidecar",
        expected_file_sha256=_require_sha256(receipt.get("truth_sidecar_sha256"), "D92 receipt truth-sidecar SHA256"),
    )
    return _validate_d92_truth_sidecar(sidecar, outer=outer)


def _prepared_rows_for_prediction(
    *, prediction: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    rows = context.get("rows")
    if not isinstance(rows, list) or len(rows) != OUTER_JOB_COUNT:
        raise D108TruthScorerError("D108 prepared context row coverage drift")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise D108TruthScorerError("D108 prepared context row must be an object")
        outer_id = _require_text(row.get("outer_id"), f"D108 context rows[{index}].outer_id")
        outer = prediction["outer_by_id"].get(outer_id)
        if outer is None or any(
            row.get(field) != outer[field]
            for field in ("receiver", "seed", "k_shot", "new_count")
        ) or outer_id in result:
            raise D108TruthScorerError("D108 prepared context/prediction row binding drift")
        result[outer_id] = row
    if set(result) != set(prediction["outer_by_id"]):
        raise D108TruthScorerError("D108 prepared context/prediction coverage drift")
    return result


def _build_outer_truth_surfaces(
    *, prediction: Mapping[str, Any], outer: Mapping[str, Any], sidecar_by_token: Mapping[str, Mapping[str, str]]
) -> list[dict[str, Any]]:
    """Join D92 sidecars by sealed query tokens, never by a scene heuristic."""

    old = tuple(outer["old_classes"])
    registry = (*old, *outer["new_classes"])
    seen_after_tokens: set[str] = set()
    result: list[dict[str, Any]] = []
    for scene in prediction["scenes"]:
        before_ids = prediction["surfaces"][(outer["outer_id"], scene, ARMS[0], "before")]["query_ids"]
        after_ids = prediction["surfaces"][(outer["outer_id"], scene, ARMS[0], "after")]["query_ids"]
        if seen_after_tokens.intersection(after_ids) or any(token not in sidecar_by_token for token in after_ids):
            raise D108TruthScorerError("D92 query-token scene/coverage drift")
        seen_after_tokens.update(after_ids)
        after_rows = [sidecar_by_token[token] for token in after_ids]
        expected_before_ids = tuple(
            token for token, item in zip(after_ids, after_rows, strict=True) if item["role"] == "target_old"
        )
        if before_ids != expected_before_ids:
            raise D108TruthScorerError("before prediction IDs must equal ordered D92 old-token subset")
        before_labels = [item["label"] for item in after_rows if item["role"] == "target_old"]
        after_labels = [item["label"] for item in after_rows]
        if set(before_labels) != set(old) or set(after_labels) != set(registry):
            raise D108TruthScorerError("D92 truth scene registry coverage drift")
        common = {
            "outer_id": outer["outer_id"],
            "receiver": outer["receiver"],
            "seed": outer["seed"],
            "k_shot": outer["k_shot"],
            "new_count": outer["new_count"],
            "scene": scene,
        }
        result.extend(
            (
                {**common, "phase": "before", "ordered_query_physical_ids": list(before_ids), "labels": before_labels},
                {**common, "phase": "after", "ordered_query_physical_ids": list(after_ids), "labels": after_labels},
            )
        )
    if seen_after_tokens != set(sidecar_by_token):
        raise D108TruthScorerError("D92 sidecar/query-token coverage drift")
    return result


def build_d108_target125_truth_catalog(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    plan_manifest_path: str | Path,
    expected_plan_file_sha256: str,
    context_manifest_path: str | Path,
    expected_context_file_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Build the immutable 750-surface catalog after prediction closure."""

    prediction = validate_d108_prediction_manifest(
        prediction_manifest_path=prediction_manifest_path,
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )
    plan, context = _load_prepared_d108_truth_inputs(
        plan_manifest_path=plan_manifest_path,
        expected_plan_file_sha256=expected_plan_file_sha256,
        context_manifest_path=context_manifest_path,
        expected_context_file_sha256=expected_context_file_sha256,
    )
    prepared_rows = _prepared_rows_for_prediction(prediction=prediction, context=context)
    truth_surfaces: list[dict[str, Any]] = []
    for outer in prediction["outer_rows"]:
        sidecar = _load_d92_truth_sidecar_for_outer(
            plan=plan, row=prepared_rows[outer["outer_id"]], outer=outer
        )
        truth_surfaces.extend(
            _build_outer_truth_surfaces(prediction=prediction, outer=outer, sidecar_by_token=sidecar)
        )
    if len(truth_surfaces) != TRUTH_SURFACE_COUNT:
        raise D108TruthScorerError("D108 truth catalog surface coverage drift")
    catalog: dict[str, Any] = {
        "schema": TRUTH_CATALOG_SCHEMA,
        "truth_open": True,
        "prediction_manifest_sha256": prediction["manifest_receipt_sha256"],
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "truth_surface_count": TRUTH_SURFACE_COUNT,
        "scenes": list(prediction["scenes"]),
        "phases": list(PHASES),
        "surfaces": truth_surfaces,
    }
    catalog["truth_catalog_sha256"] = canonical_sha256(catalog)
    destination = Path(output_path)
    return {
        "truth_catalog": str(destination),
        "truth_catalog_file_sha256": _write_json_new(destination, catalog),
        "truth_catalog_sha256": catalog["truth_catalog_sha256"],
        "outer_job_count": OUTER_JOB_COUNT,
        "scene_row_count": SCENE_ROW_COUNT,
        "truth_surface_count": TRUTH_SURFACE_COUNT,
    }


def _validate_truth_catalog(
    truth_catalog_path: str | Path,
    *,
    expected_truth_catalog_file_sha256: str,
    prediction: Mapping[str, Any],
) -> tuple[dict[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]], str, int]:
    catalog, catalog_size = _read_json_regular(
        truth_catalog_path,
        name="D108 independent truth catalog",
        expected_file_sha256=expected_truth_catalog_file_sha256,
    )
    document = _exact_fields(catalog, _TRUTH_CATALOG_FIELDS, "truth catalog")
    if (
        document["schema"] != TRUTH_CATALOG_SCHEMA
        or document["truth_open"] is not True
        or document["prediction_manifest_sha256"] != prediction["manifest_receipt_sha256"]
        or document["scenes"] != list(prediction["scenes"])
        or document["phases"] != list(PHASES)
    ):
        raise D108TruthScorerError("truth catalog identity/binding drift")
    if any(
        document[field] != expected
        for field, expected in {
            "outer_job_count": OUTER_JOB_COUNT,
            "scene_row_count": SCENE_ROW_COUNT,
            "truth_surface_count": TRUTH_SURFACE_COUNT,
        }.items()
    ):
        raise D108TruthScorerError("truth catalog 125/375/750 coverage drift")
    truth_receipt = _receipt(document, "truth_catalog_sha256", "truth catalog")
    raw_surfaces = document["surfaces"]
    if not isinstance(raw_surfaces, list) or len(raw_surfaces) != TRUTH_SURFACE_COUNT:
        raise D108TruthScorerError("truth catalog must contain exactly 750 surfaces")
    truths: dict[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for index, raw in enumerate(raw_surfaces):
        truth = _exact_fields(raw, _TRUTH_SURFACE_FIELDS, f"truth surfaces[{index}]")
        outer_id = _require_text(truth["outer_id"], f"truth surfaces[{index}].outer_id")
        outer = prediction["outer_by_id"].get(outer_id)
        if outer is None or any(
            truth[field] != outer[field] for field in ("receiver", "seed", "k_shot", "new_count")
        ):
            raise D108TruthScorerError("truth surface outer metadata binding drift")
        scene = _require_text(truth["scene"], f"truth surfaces[{index}].scene")
        phase = _require_text(truth["phase"], f"truth surfaces[{index}].phase")
        if scene not in prediction["scenes"] or phase not in PHASES:
            raise D108TruthScorerError("truth surface scene/phase drift")
        query_ids = _string_list(truth["ordered_query_physical_ids"], f"truth surfaces[{index}].ordered_query_physical_ids")
        labels = _string_list(truth["labels"], f"truth surfaces[{index}].labels", unique=False)
        if len(query_ids) != len(labels):
            raise D108TruthScorerError("truth query/label length drift")
        old = set(outer["old_classes"])
        registry = old | set(outer["new_classes"])
        expected = old if phase == "before" else registry
        if any(label not in expected for label in labels) or not expected.issubset(set(labels)):
            raise D108TruthScorerError("truth label registry coverage drift")
        key = (outer_id, scene, phase)
        if key in truths or query_ids != prediction["surfaces"][(outer_id, scene, ARMS[0], phase)]["query_ids"]:
            raise D108TruthScorerError("truth/prediction ordered query-ID mismatch")
        truths[key] = (query_ids, labels)
    expected_keys = {
        (outer["outer_id"], scene, phase)
        for outer in prediction["outer_rows"]
        for scene in prediction["scenes"]
        for phase in PHASES
    }
    if set(truths) != expected_keys:
        raise D108TruthScorerError("truth surface coverage drift")
    for outer in prediction["outer_rows"]:
        old = set(outer["old_classes"])
        for scene in prediction["scenes"]:
            before_ids, before_labels = truths[(outer["outer_id"], scene, "before")]
            after_ids, after_labels = truths[(outer["outer_id"], scene, "after")]
            if tuple(query for query, label in zip(after_ids, after_labels, strict=True) if label in old) != tuple(
                query for query, label in zip(before_ids, before_labels, strict=True) if label in old
            ):
                raise D108TruthScorerError("before/after old-query physical-ID mismatch")
    return truths, truth_receipt, catalog_size


def _rate(correct: int, total: int, name: str) -> float:
    if type(correct) is not int or type(total) is not int or total <= 0:
        raise D108TruthScorerError(f"{name} has an empty/invalid denominator")
    return 100.0 * correct / total


def _harmonic(old_accuracy: float, new_accuracy: float) -> float:
    return 0.0 if old_accuracy + new_accuracy == 0.0 else 2.0 * old_accuracy * new_accuracy / (old_accuracy + new_accuracy)


def _class_counts(
    predictions: Sequence[str], labels: Sequence[str], classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    values = {label: {"correct_count": 0, "query_count": 0} for label in classes}
    for prediction, label in zip(predictions, labels, strict=True):
        if label not in values:
            raise D108TruthScorerError("old-class count received a non-old label")
        values[label]["query_count"] += 1
        values[label]["correct_count"] += int(prediction == label)
    if any(item["query_count"] == 0 for item in values.values()):
        raise D108TruthScorerError("old-class floor coverage drift")
    return values


def _floor(counts: Mapping[str, Mapping[str, int]], name: str) -> float:
    return min(_rate(item["correct_count"], item["query_count"], f"{name}.{label}") for label, item in counts.items())


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
    before_pairs = [(p, y) for p, y in zip(before_predictions, before_truth, strict=True) if y in old]
    after_old = [(p, y) for p, y in zip(after_predictions, after_truth, strict=True) if y in old]
    after_new = [(p, y) for p, y in zip(after_predictions, after_truth, strict=True) if y in new]
    if len(after_old) + len(after_new) != len(after_truth):
        raise D108TruthScorerError("after metric truth role partition drift")
    before_counts = _class_counts([p for p, _ in before_pairs], [y for _, y in before_pairs], outer["old_classes"])
    after_counts = _class_counts([p for p, _ in after_old], [y for _, y in after_old], outer["old_classes"])
    before_correct = sum(p == y for p, y in before_pairs)
    after_old_correct = sum(p == y for p, y in after_old)
    new_correct = sum(p == y for p, y in after_new)
    before_queries, old_queries, new_queries = len(before_pairs), len(after_old), len(after_new)
    before_old = _rate(before_correct, before_queries, "before_old")
    after_old_acc = _rate(after_old_correct, old_queries, "after_old")
    seen_new = _rate(new_correct, new_queries, "seen_new")
    post_correct, post_queries = after_old_correct + new_correct, old_queries + new_queries
    values: dict[str, Any] = {
        "outer_id": outer["outer_id"], "receiver": outer["receiver"], "seed": outer["seed"],
        "k_shot": outer["k_shot"], "new_count": outer["new_count"], "scene": scene, "arm": arm,
        "before_old": before_old, "after_old": after_old_acc,
        "before_old_floor": _floor(before_counts, "before_old_floor"),
        "after_old_floor": _floor(after_counts, "after_old_floor"),
        "seen_new": seen_new, "H_old_new": _harmonic(after_old_acc, seen_new),
        "forgetting": before_old - after_old_acc,
        "before_old_correct_count": before_correct, "before_old_query_count": before_queries,
        "after_old_correct_count": after_old_correct, "after_old_query_count": old_queries,
        "old_correct_count": after_old_correct, "old_query_count": old_queries,
        "new_correct_count": new_correct, "new_query_count": new_queries,
        "post_registration_total_correct_count": post_correct,
        "post_registration_total_query_count": post_queries,
        "total_correct_count": before_correct + post_correct,
        "total_query_count": before_queries + post_queries,
        "before_old_by_class": before_counts, "after_old_by_class": after_counts,
    }
    if any(not math.isfinite(values[field]) for field in ("before_old", "after_old", "before_old_floor", "after_old_floor", "seen_new", "H_old_new", "forgetting")):
        raise D108TruthScorerError("non-finite metric row")
    values["metric_row_receipt_sha256"] = canonical_sha256(values)
    return values


def _merge_class_counts(
    rows: Sequence[Mapping[str, Any]], field: str, classes: Sequence[str]
) -> dict[str, dict[str, int]]:
    totals = {label: {"correct_count": 0, "query_count": 0} for label in classes}
    for row in rows:
        counts = row[field]
        if set(counts) != set(classes):
            raise D108TruthScorerError("aggregate old-class coverage drift")
        for label in classes:
            totals[label]["correct_count"] += counts[label]["correct_count"]
            totals[label]["query_count"] += counts[label]["query_count"]
    return totals


def _aggregate_outer_arm(
    *, outer: Mapping[str, Any], arm: str, scene_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(scene_rows) != 3 or any(row["arm"] != arm for row in scene_rows):
        raise D108TruthScorerError("outer-arm aggregate source coverage drift")
    count_fields = (
        "before_old_correct_count", "before_old_query_count", "after_old_correct_count",
        "after_old_query_count", "new_correct_count", "new_query_count",
        "post_registration_total_correct_count", "post_registration_total_query_count",
        "total_correct_count", "total_query_count",
    )
    totals = {field: sum(int(row[field]) for row in scene_rows) for field in count_fields}
    before_counts = _merge_class_counts(scene_rows, "before_old_by_class", outer["old_classes"])
    after_counts = _merge_class_counts(scene_rows, "after_old_by_class", outer["old_classes"])
    before_old = _rate(totals["before_old_correct_count"], totals["before_old_query_count"], "aggregate.before_old")
    after_old = _rate(totals["after_old_correct_count"], totals["after_old_query_count"], "aggregate.after_old")
    seen_new = _rate(totals["new_correct_count"], totals["new_query_count"], "aggregate.seen_new")
    result: dict[str, Any] = {
        "outer_id": outer["outer_id"], "receiver": outer["receiver"], "seed": outer["seed"],
        "k_shot": outer["k_shot"], "new_count": outer["new_count"], "arm": arm,
        "scenes": [row["scene"] for row in scene_rows], "aggregation": "micro_average_across_three_scenes",
        "before_old": before_old, "after_old": after_old,
        "before_old_floor": _floor(before_counts, "aggregate.before_old_floor"),
        "after_old_floor": _floor(after_counts, "aggregate.after_old_floor"),
        "seen_new": seen_new, "H_old_new": _harmonic(after_old, seen_new),
        "forgetting": before_old - after_old, **totals,
        "old_correct_count": totals["after_old_correct_count"], "old_query_count": totals["after_old_query_count"],
        "before_old_by_class": before_counts, "after_old_by_class": after_counts,
        "source_scene_metric_row_receipt_sha256s": [row["metric_row_receipt_sha256"] for row in scene_rows],
    }
    result["outer_arm_receipt_sha256"] = canonical_sha256(result)
    return result


def _score_rows(
    prediction: Mapping[str, Any], truths: Mapping[tuple[str, str, str], tuple[tuple[str, ...], tuple[str, ...]]]
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
                    outer=outer, scene=scene, arm=arm,
                    before_predictions=before["predictions"], before_truth=before_truth,
                    after_predictions=after["predictions"], after_truth=after_truth,
                )
                arm_rows.append(row)
                per_outer_arm[(outer["outer_id"], arm)].append(row)
            same: dict[str, Any] = {
                "outer_id": outer["outer_id"], "receiver": outer["receiver"], "seed": outer["seed"],
                "k_shot": outer["k_shot"], "new_count": outer["new_count"], "scene": scene, "arms": arm_rows,
            }
            same["scene_same_row_receipt_sha256"] = canonical_sha256(same)
            scene_same_rows.append(same)
    outer_arm_rows = [
        _aggregate_outer_arm(outer=outer, arm=arm, scene_rows=per_outer_arm[(outer["outer_id"], arm)])
        for outer in prediction["outer_rows"] for arm in ARMS
    ]
    if (
        len(scene_same_rows) != SCENE_ROW_COUNT
        or sum(len(row["arms"]) for row in scene_same_rows) != SCENE_ARM_METRIC_ROW_COUNT
        or len(outer_arm_rows) != OUTER_ARM_AGGREGATE_ROW_COUNT
    ):
        raise D108TruthScorerError("metric output row coverage drift")
    return scene_same_rows, outer_arm_rows


def _score_summary(
    prediction: Mapping[str, Any], *, truth_catalog_size: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    coverage = {
        "outer_job_count": OUTER_JOB_COUNT, "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT, "prediction_surface_count": SURFACE_COUNT,
        "truth_surface_count": TRUTH_SURFACE_COUNT, "scene_same_row_count": SCENE_ROW_COUNT,
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
        "primary_candidate_arm": "M_JOINT", "causal_arms": list(ARMS),
        "causal_table_preserved": True, "coverage_verdict": "COMPLETE_125_TRUTH_OPEN_AND_SCORED",
        "target_thresholds_declared": False, "target_verdict": "NO_TARGET_THRESHOLD_DECLARED",
        "performance_early_stop_or_selection_performed": False,
        "D91_status": _D91_DEVELOPMENT_STATUS,
    }
    return coverage, resources, verdict


def score_d108_target125(
    *,
    prediction_manifest_path: str | Path,
    expected_prediction_manifest_file_sha256: str,
    truth_catalog_path: str | Path,
    expected_truth_catalog_file_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Seal-check all predictions, then truth-open and score the full matrix."""

    prediction = validate_d108_prediction_manifest(
        prediction_manifest_path=prediction_manifest_path,
        expected_prediction_manifest_file_sha256=expected_prediction_manifest_file_sha256,
    )
    expected_truth_sha = _require_sha256(expected_truth_catalog_file_sha256, "expected truth catalog file SHA256")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable score output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D108TruthScorerError("unsafe score output parent")
    destination.mkdir()
    event: dict[str, Any] = {
        "schema": TRUTH_OPEN_EVENT_SCHEMA,
        "prediction_manifest_sha256": prediction["manifest_receipt_sha256"],
        "prediction_manifest_file_sha256": prediction["manifest_file_sha256"],
        "expected_truth_catalog_file_sha256": expected_truth_sha,
        "prediction_closure_verified": True,
        "outer_job_count": OUTER_JOB_COUNT, "scene_row_count": SCENE_ROW_COUNT,
        "arm_pair_count": ARM_PAIR_COUNT, "surface_count": SURFACE_COUNT,
    }
    event["truth_open_event_receipt_sha256"] = canonical_sha256(event)
    event_path = destination / "truth_open_event.json"
    event_file_sha = _write_json_new(event_path, event)
    truths, truth_receipt, truth_size = _validate_truth_catalog(
        truth_catalog_path,
        expected_truth_catalog_file_sha256=expected_truth_sha,
        prediction=prediction,
    )
    scene_same_rows, outer_arm_rows = _score_rows(prediction, truths)
    coverage, resources, verdict = _score_summary(prediction, truth_catalog_size=truth_size)
    score: dict[str, Any] = {
        "schema": SCORE_MANIFEST_SCHEMA, "candidate_id": prediction["manifest"]["candidate_id"],
        "protocol_schema": PROTOCOL_SCHEMA,
        "prediction_manifest_sha256": prediction["manifest_receipt_sha256"],
        "prediction_manifest_file_sha256": prediction["manifest_file_sha256"],
        "truth_catalog_sha256": truth_receipt, "truth_catalog_file_sha256": expected_truth_sha,
        "truth_open_event_receipt_sha256": event["truth_open_event_receipt_sha256"],
        "scene_same_row_count": len(scene_same_rows),
        "scene_arm_metric_row_count": sum(len(row["arms"]) for row in scene_same_rows),
        "outer_arm_aggregate_row_count": len(outer_arm_rows),
        "scene_same_rows": scene_same_rows, "outer_arm_aggregate_rows": outer_arm_rows,
        "coverage_summary": coverage, "resource_summary": resources, "target_verdict_summary": verdict,
    }
    score["score_manifest_sha256"] = canonical_sha256(score)
    score_path = destination / "score_manifest.json"
    score_file_sha = _write_json_new(score_path, score)
    return {
        "truth_open_event": str(event_path), "truth_open_event_file_sha256": event_file_sha,
        "score_manifest": str(score_path), "score_manifest_file_sha256": score_file_sha,
        "score_manifest_sha256": score["score_manifest_sha256"],
        "scene_same_row_count": len(scene_same_rows),
        "scene_arm_metric_row_count": sum(len(row["arms"]) for row in scene_same_rows),
        "outer_arm_aggregate_row_count": len(outer_arm_rows),
    }


def _comparator_identity(row: Mapping[str, Any], name: str) -> tuple[str, int, int, int, str]:
    required = ("receiver", "seed", "k_shot", "new_count", "scene")
    if any(field not in row for field in required):
        raise D108TruthScorerError(f"{name} comparator row identity is incomplete")
    return (
        _require_text(row["receiver"], f"{name}.receiver"),
        _require_int(row["seed"], f"{name}.seed"),
        _require_int(row["k_shot"], f"{name}.k_shot", minimum=1),
        _require_int(row["new_count"], f"{name}.new_count", minimum=1),
        _require_text(row["scene"], f"{name}.scene"),
    )


def pair_d108_same_row(
    *, scene_same_rows: Sequence[Mapping[str, Any]], comparator_id: str, comparator_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Pair comparator evidence only on identical outer/scene identities."""

    comparator = _require_text(comparator_id, "comparator_id")
    if comparator not in _FORMAL_COMPARATORS and comparator != "D91":
        raise D108TruthScorerError("unsupported comparator ID")
    if not isinstance(comparator_rows, Sequence) or isinstance(comparator_rows, (str, bytes)):
        raise D108TruthScorerError("comparator_rows must be a sequence")
    if comparator == "D91" and len(comparator_rows) != 15:
        raise D108TruthScorerError("D91 must be labelled as exactly 15 development rows")
    comparator_by_identity: dict[tuple[str, int, int, int, str], Mapping[str, Any]] = {}
    for index, row in enumerate(comparator_rows):
        if not isinstance(row, Mapping):
            raise D108TruthScorerError("comparator row must be a mapping")
        identity = _comparator_identity(row, f"comparator_rows[{index}]")
        if identity in comparator_by_identity:
            raise D108TruthScorerError("duplicate comparator same-row identity")
        comparator_by_identity[identity] = row
    pairs: list[dict[str, Any]] = []
    status = _D91_DEVELOPMENT_STATUS if comparator == "D91" else "SAME_ROW_EVIDENCE_ONLY_NOT_PROMOTION"
    for d108_row in scene_same_rows:
        if not isinstance(d108_row, Mapping):
            raise D108TruthScorerError("D108 scene row must be a mapping")
        identity = _comparator_identity(d108_row, "D108 scene row")
        comparator_row = comparator_by_identity.get(identity)
        if comparator_row is None:
            continue
        pair: dict[str, Any] = {
            "comparator_id": comparator, "pairing_status": status,
            "receiver": identity[0], "seed": identity[1], "k_shot": identity[2],
            "new_count": identity[3], "scene": identity[4],
            "d108_scene_same_row": dict(d108_row), "comparator_row": dict(comparator_row),
        }
        pair["same_row_pair_receipt_sha256"] = canonical_sha256(pair)
        pairs.append(pair)
    result: dict[str, Any] = {
        "comparator_id": comparator, "pairing_status": status,
        "comparator_input_row_count": len(comparator_rows), "matched_row_count": len(pairs), "pairs": pairs,
    }
    result["pairing_receipt_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "ARMS", "ARM_PAIR_COUNT", "D108TruthScorerError", "OUTER_ARM_AGGREGATE_ROW_COUNT",
    "OUTER_JOB_COUNT", "PHASES", "PREDICTION_ARTIFACT_SCHEMA", "PREDICTION_MANIFEST_SCHEMA",
    "PROTOCOL_SCHEMA", "SCENE_ARM_METRIC_ROW_COUNT", "SCENE_ROW_COUNT", "SCORE_MANIFEST_SCHEMA",
    "SURFACE_COUNT", "TRUTH_CATALOG_SCHEMA", "TRUTH_OPEN_EVENT_SCHEMA", "TRUTH_SURFACE_COUNT",
    "build_d108_target125_truth_catalog", "canonical_sha256", "pair_d108_same_row",
    "score_d108_target125", "validate_d108_prediction_manifest",
]
