#!/usr/bin/env python3
"""Mechanical three-stage launcher for NEXT-R4 FA-RDCE3 x CER-PLR160.

The entry point deliberately owns no method logic.  ``prepare`` seals only
the validated-once capsule/split metadata, row physical-ID bindings and the
per-outer-cell aggregate asset manifest.  ``predict`` reads that package,
forwards only received IQ through the checkpoint bridge, and calls the frozen
single-row runtime.  ``score`` validates the complete truth-free prediction
closure before opening the opaque truth sidecar.

The input cardinality is taken from the capsule metadata and received-IQ
archive.  There is no R3 588/14 gate, no TSL/F/L asset, and no tuning or
promotion logic in this script.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
for _path in (str(SCRIPT_ROOT), str(CODE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cvsrffi import stage2_next_r4_artifact as artifact  # noqa: E402
from cvsrffi import stage2_next_r4_fa_rdce3 as fa  # noqa: E402
from cvsrffi import stage2_next_r4_matrix as matrix  # noqa: E402
from cvsrffi import stage2_next_r4_runtime as runtime  # noqa: E402
from cvsrffi import stage2_next_r4_score as scorer  # noqa: E402
from cvsrffi import stage2_zid_student_t_qknn as qknn  # noqa: E402


RUNNER_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.runner.v1"
PREPARE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.prepare.v1"
PREDICTOR_PACKAGE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.predictor_package.v1"
ASSET_MANIFEST_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.asset_manifest.v1"
COMPLETION_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.completion.v1"
MANIFEST_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.manifest.v1"
RESOURCE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.resource.v1"
SMOKE_SCHEMA = "cvs.stage2.next_r4.fa_rdce3_cer_plr160.proxy24.real_checkpoint_smoke.v1"
MISSING_PREFIX = "MISSING_REAL_INPUT_ARTIFACTS"
GROUPED_QUERY_FIELDS = frozenset(
    {"query_ids_by_class", "query_observation_ids_by_class", "query_count_by_class"}
)
PREPARE_AUTHORITY_FIELDS = (
    "package_sha256",
    "truth_sha256",
    "received_iq_sha256",
    "checkpoint_sha256",
    "asset_manifest_sha256",
    "matrix_sha256",
)


class NextR4Proxy24Error(ValueError):
    """The frozen mechanical runner contract did not close."""


class MissingRealInputArtifacts(NextR4Proxy24Error):
    """A required real capsule, checkpoint or aggregate asset is unavailable."""


_FORBIDDEN = frozenset(
    {
        "truth",
        "truth_label",
        "query_truth",
        "query_label",
        "query_labels",
        "query_role",
        "query_roles",
        "class_quota",
        "batch_class_count",
        "true_batch_class_count",
        "true_batch_class_counts",
        "global_reassignment",
        "hungarian",
        "optimal_transport",
        "clean_iq",
        "source_samples",
        "source_features",
        "logits",
        "features",
    }
)


def _normal_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _reject_forbidden(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normal_key(key) in _FORBIDDEN:
                raise NextR4Proxy24Error(f"{name} contains forbidden field {key}")
            _reject_forbidden(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden(item, name=f"{name}[{index}]")


def _reject_grouped_query(value: Any, *, name: str) -> None:
    """Reject class-indexed query metadata outside the prepare builder."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normal_key(key) in GROUPED_QUERY_FIELDS:
                raise NextR4Proxy24Error(f"{name} exposes class-grouped query metadata: {key}")
            _reject_grouped_query(item, name=f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_grouped_query(item, name=f"{name}[{index}]")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _plain(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _require_sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise NextR4Proxy24Error(f"{name} must be a lowercase SHA256")
    try:
        int(value, 16)
    except ValueError as error:
        raise NextR4Proxy24Error(f"{name} must be a lowercase SHA256") from error
    return value


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise NextR4Proxy24Error(f"output overwrite refused: {path}")
    if not path.parent.is_dir():
        raise NextR4Proxy24Error(f"output parent is missing: {path.parent}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_plain(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_bytes_new(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise NextR4Proxy24Error(f"output overwrite refused: {path}")
    if not path.parent.is_dir():
        raise NextR4Proxy24Error(f"output parent is missing: {path.parent}")
    with path.open("xb") as handle:
        handle.write(value)


def _new_root(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not path.is_absolute() or path != resolved or path.exists() or not path.parent.is_dir():
        raise NextR4Proxy24Error("run/output root must be a new absolute child of an existing directory")
    path.mkdir()
    return path


def _require_file(path: Path | None, expected_sha256: str | None, name: str) -> bytes:
    if path is None or expected_sha256 is None:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} path/SHA256 is required")
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} is not a regular file")
    expected = _require_sha(expected_sha256, f"{name} SHA256")
    payload = path.read_bytes()
    if _sha(payload) != expected:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} SHA256 mismatch")
    return payload


def _read_json(path: Path, expected_sha256: str, name: str) -> Mapping[str, Any]:
    payload = _require_file(path, expected_sha256, name)
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} must be UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} must be a JSON object")
    _reject_forbidden(value, name=name)
    return value


def _string_array(value: Any, *, name: str, count: int | None = None, unique: bool = False) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "US" or (count is not None and len(array) != count):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} must be a string vector")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result) or (unique and len(set(result)) != len(result)):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} contains blank/duplicate IDs")
    return result


@dataclass(frozen=True, slots=True)
class ReceivedCapsule:
    received_iq: np.ndarray
    receiver_ids: tuple[str, ...]
    physical_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    scenario_names: tuple[str, ...]
    received_iq_sha256: str

    def __post_init__(self) -> None:
        values = np.asarray(self.received_iq)
        if (
            values.dtype != np.dtype("<f4")
            or values.ndim != 3
            or values.shape[0] < 1
            or values.shape[1] != 2
            or not np.isfinite(values).all()
        ):
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: received_iq must be finite little-endian float32 [N,2,T]")
        if len(self.physical_ids) != len(values) or len(self.observation_ids) != len(values):
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: received-IQ ID lengths drift")
        if len(set(self.physical_ids)) != len(self.physical_ids) or len(set(self.observation_ids)) != len(self.observation_ids):
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: received-IQ IDs must be unique")
        frozen = np.ascontiguousarray(values, dtype=np.float32).copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "received_iq", frozen)

    @property
    def index_by_physical(self) -> Mapping[str, int]:
        return {item: index for index, item in enumerate(self.physical_ids)}


def _load_received_capsule(path: Path, expected_sha256: str) -> ReceivedCapsule:
    payload = _require_file(path, expected_sha256, "received-IQ capsule")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            names = tuple(archive.files)
            required = {"received_iq", "receiver_ids", "physical_ids", "observation_ids"}
            allowed = required | {"scenario_names", "day_ids"}
            if not required.issubset(names) or any(item not in allowed for item in names):
                raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: received-IQ capsule member drift")
            arrays = {item: np.asarray(archive[item]) for item in names}
    except MissingRealInputArtifacts:
        raise
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: received-IQ capsule is not a no-pickle NPZ") from error
    iq = np.asarray(arrays["received_iq"])
    count = int(iq.shape[0]) if iq.ndim >= 1 else None
    scenario = _string_array(
        arrays.get("scenario_names", np.asarray(["leo_unknown"] * int(count or 0), dtype="<U32")),
        name="received.scenario_names",
        count=count,
    )
    # D106 capsules may carry day provenance for the builder.  Validate its
    # shape/content, then deliberately discard it at the predictor boundary.
    # It must never enter ReceivedCapsule, the package, or runtime state.
    if "day_ids" in arrays:
        _string_array(arrays["day_ids"], name="received.day_ids", count=count)
    return ReceivedCapsule(
        received_iq=iq,
        receiver_ids=_string_array(arrays["receiver_ids"], name="received.receiver_ids", count=count),
        physical_ids=_string_array(arrays["physical_ids"], name="received.physical_ids", count=count, unique=True),
        observation_ids=_string_array(arrays["observation_ids"], name="received.observation_ids", count=count, unique=True),
        scenario_names=scenario,
        received_iq_sha256=_sha(payload),
    )


def _ordered_root(values: Sequence[str]) -> str:
    return _sha("\n".join(str(item) for item in values).encode("utf-8"))


def _query_maps(row: Mapping[str, Any], classes: tuple[str, ...]) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, tuple[str, ...]]]:
    query = row.get("query_ids_by_class")
    observation = row.get("query_observation_ids_by_class")
    if not isinstance(query, Mapping) or not isinstance(observation, Mapping) or set(query) != set(classes) or set(observation) != set(classes):
        raise NextR4Proxy24Error("capsule row query class maps must close over the full class registry")
    q = {cls: tuple(query[cls]) for cls in classes}
    o = {cls: tuple(observation[cls]) for cls in classes}
    for cls in classes:
        if not q[cls] or len(q[cls]) != len(set(q[cls])) or any(type(item) is not str or not item for item in q[cls]):
            raise NextR4Proxy24Error("capsule query physical IDs are invalid")
        if not o[cls] or len(o[cls]) != len(set(o[cls])) or any(type(item) is not str or not item for item in o[cls]):
            raise NextR4Proxy24Error("capsule query observation IDs are invalid")
        if len(q[cls]) != len(o[cls]):
            raise NextR4Proxy24Error("query physical/observation count drift")
    return q, o


def _class_maps(row: Mapping[str, Any], field: str, classes: tuple[str, ...], expected: int) -> Mapping[str, tuple[str, ...]]:
    value = row.get(field)
    if not isinstance(value, Mapping) or set(value) != set(classes):
        raise NextR4Proxy24Error(f"capsule row {field} must close over all classes")
    result = {cls: tuple(value[cls]) for cls in classes}
    if any(len(result[cls]) != expected or any(type(item) is not str or not item for item in result[cls]) or len(set(result[cls])) != len(result[cls]) for cls in classes):
        raise NextR4Proxy24Error(f"capsule row {field} has invalid physical IDs")
    return result


def _load_asset_manifest(path: Path, expected_sha256: str, *, checkpoint_sha256: str, expected_keys: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
    document = _read_json(path, expected_sha256, "FA asset manifest")
    if document.get("schema") != ASSET_MANIFEST_SCHEMA:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset manifest schema drift")
    entries = document.get("entries")
    if not isinstance(entries, Mapping) or set(entries) != set(expected_keys):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset manifest outer-key coverage drift")
    out: dict[str, Mapping[str, Any]] = {}
    for key in expected_keys:
        entry = entries[key]
        if not isinstance(entry, Mapping) or set(entry) != {"asset_path", "asset_sha256", "checkpoint_sha256", "phase1_fit_physical_root_sha256"}:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset manifest entry fields drift for {key}")
        if entry["checkpoint_sha256"] != checkpoint_sha256:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset/checkpoint binding drift for {key}")
        _require_sha(entry["asset_sha256"], f"asset {key} SHA256")
        _require_sha(entry["phase1_fit_physical_root_sha256"], f"asset {key} Phase1 root")
        asset_path = Path(str(entry["asset_path"]))
        _require_file(asset_path, entry["asset_sha256"], f"FA asset {key}")
        try:
            checked = fa.deserialize_fa_rdce3_phase1_asset(asset_path.read_bytes())
        except Exception as error:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset {key} wire is invalid") from error
        if checked.checkpoint_sha256 != checkpoint_sha256:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset {key} checkpoint binding drift")
        out[key] = {
            "asset_path": str(asset_path.resolve()),
            "asset_sha256": entry["asset_sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "phase1_fit_physical_root_sha256": entry["phase1_fit_physical_root_sha256"],
        }
    return out


def _load_capsule_metadata(
    path: Path,
    expected_sha256: str,
    *,
    received: ReceivedCapsule,
    checkpoint_sha256: str,
    asset_manifest: Mapping[str, Mapping[str, Any]],
    asset_manifest_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, str]]:
    metadata = _read_json(path, expected_sha256, "capsule/split metadata")
    required = {"schema", "protocol_schema", "phase2_data_status", "capsule_id", "split_id", "validator_receipt_sha256", "class_registry", "held_receivers", "rows", "seed", "qknn_lock_by_k"}
    if not required.issubset(metadata):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule/split metadata fields are incomplete")
    if metadata["protocol_schema"] != matrix.PROTOCOL_SCHEMA or metadata["phase2_data_status"] != "VALIDATED_ONCE":
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule metadata is not VALIDATED_ONCE/p2_min_v1")
    capsule_id = _require_sha(metadata["capsule_id"], "capsule_id")
    split_id = _require_sha(metadata["split_id"], "split_id")
    validator = _require_sha(metadata["validator_receipt_sha256"], "validator_receipt_sha256")
    classes = tuple(sorted(tuple(metadata["class_registry"])))
    try:
        plan = matrix.build_next_r4_proxy24_plan(classes, held_receivers=tuple(metadata["held_receivers"]))
        matrix.validate_next_r4_proxy24_plan(plan)
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule class/receiver registry does not match NEXT-R4") from error
    raw_rows = metadata["rows"]
    if not isinstance(raw_rows, (list, tuple)) or len(raw_rows) != matrix.ROW_COUNT:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule metadata must bind the complete 24-row matrix")
    by_id = {str(item.get("row_id")): item for item in raw_rows if isinstance(item, Mapping)}
    if len(by_id) != matrix.ROW_COUNT:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule row identities are incomplete")
    received_ids = set(received.physical_ids) | set(received.observation_ids)
    row_out: list[Mapping[str, Any]] = []
    # Build the paired physical binding while the prepare side is still
    # allowed to inspect Phase1 member IDs.  Only its sealed count/root are
    # carried forward; predictor never receives the raw Phase1 IDs.
    raw_by_row_id = {str(item["row_id"]): item for item in plan["rows"]}
    binding_by_outer: dict[tuple[str, str], Mapping[str, Any]] = {}
    truth_by_query_id: dict[str, str] = {}
    for planned in plan["rows"]:
        raw = by_id.get(str(planned["row_id"]))
        if not isinstance(raw, Mapping) or any(raw.get(field) != planned[field] for field in ("row_id", "held_receiver", "held_class", "active_k")):
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule row identity drift")
        classes_all = tuple(planned["all_registered_classes"])
        q, qo = _query_maps(raw, classes_all)
        support1 = _class_maps(raw, "k1_support_ids_by_class", classes_all, 1)
        support5 = _class_maps(raw, "k5_support_ids_by_class", classes_all, 5)
        phase1_ids = tuple(raw.get("phase1_fit_ids", ()))
        if not phase1_ids or any(type(item) is not str or not item for item in phase1_ids) or len(set(phase1_ids)) != len(phase1_ids):
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: phase1_fit_ids are incomplete")
        if set(phase1_ids) & received_ids:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: Phase1/support/query IDs overlap received-IQ IDs")
        if any(item not in set(received.physical_ids) for values in (*support1.values(), *support5.values(), *q.values()) for item in values):
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule references an unknown received physical ID")
        if any(item not in set(received.observation_ids) for values in qo.values() for item in values):
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule references an unknown observation ID")
        pair_key = f"{planned['held_receiver']}|{planned['held_class']}"
        if pair_key not in asset_manifest:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset manifest lacks {pair_key}")
        if _ordered_root(phase1_ids) != asset_manifest[pair_key]["phase1_fit_physical_root_sha256"]:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: FA asset/Phase1 physical root drift for {pair_key}")
        outer = (str(planned["held_receiver"]), str(planned["held_class"]))
        if outer not in binding_by_outer:
            k1_planned = raw_by_row_id[
                next(
                    str(item["row_id"])
                    for item in plan["rows"]
                    if item["held_receiver"] == planned["held_receiver"]
                    and item["held_class"] == planned["held_class"]
                    and int(item["active_k"]) == 1
                )
            ]
            k5_planned = raw_by_row_id[
                next(
                    str(item["row_id"])
                    for item in plan["rows"]
                    if item["held_receiver"] == planned["held_receiver"]
                    and item["held_class"] == planned["held_class"]
                    and int(item["active_k"]) == 5
                )
            ]
            try:
                binding_by_outer[outer] = matrix.bind_next_r4_physical_ids(
                    row_k1=matrix.outer_key_from_mapping(k1_planned),
                    row_k5=matrix.outer_key_from_mapping(k5_planned),
                    phase1_fit_ids=phase1_ids,
                    k1_support_ids_by_class=support1,
                    k5_support_ids_by_class=support5,
                    query_ids_by_class=q,
                    query_observation_ids_by_class=qo,
                    query_ids_by_view={view: q for view in matrix.QUERY_VIEW_IDS},
                    query_observation_ids_by_view={view: qo for view in matrix.QUERY_VIEW_IDS},
                )
            except Exception as error:
                raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: row physical binding is invalid for {pair_key}") from error
        binding = binding_by_outer[outer]
        if binding["phase1_fit_physical_root_sha256"] != asset_manifest[pair_key]["phase1_fit_physical_root_sha256"]:
            raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: row binding/FA Phase1 root drift for {pair_key}")
        row_out.append({
            "row_id": planned["row_id"],
            "held_receiver": planned["held_receiver"],
            "held_class": planned["held_class"],
            "active_k": planned["active_k"],
            "physical_binding_receipt": _plain(binding),
        })
        for class_id in classes_all:
            for query_id in q[class_id]:
                truth_by_query_id[query_id] = class_id
    return {
        "schema": PREDICTOR_PACKAGE_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "protocol_schema": matrix.PROTOCOL_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": capsule_id,
        "split_id": split_id,
        "validator_receipt_sha256": validator,
        "checkpoint_sha256": checkpoint_sha256,
        "class_registry": list(classes),
        "held_receivers": list(matrix.HELD_RECEIVERS),
        "matrix_sha256": plan["matrix_sha256"],
        "seed": int(metadata["seed"]),
        "qknn_lock_by_k": _plain(metadata["qknn_lock_by_k"]),
        "received_iq_sha256": received.received_iq_sha256,
        # Bind the package to the exact manifest file supplied by the
        # operator.  The canonical in-memory mapping is intentionally not
        # substituted here because predict receives the manifest file hash.
        "asset_manifest_sha256": _require_sha(asset_manifest_sha256, "FA asset manifest SHA256"),
        "rows": row_out,
        "truth_free": True,
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }, truth_by_query_id


class _FeatureCache:
    def __init__(self, bridge: Any, capsule: ReceivedCapsule, checkpoint_sha256: str) -> None:
        if getattr(bridge, "checkpoint_sha256", checkpoint_sha256) != checkpoint_sha256:
            raise NextR4Proxy24Error("checkpoint bridge SHA256 drift")
        self.bridge = bridge
        self.capsule = capsule
        self.index = capsule.index_by_physical
        self.cache: dict[str, np.ndarray] = {}

    def take(self, physical_ids: Sequence[str]) -> np.ndarray:
        ids = tuple(physical_ids)
        if not ids or len(set(ids)) != len(ids) or any(item not in self.index for item in ids):
            raise NextR4Proxy24Error("bridge feature request physical-ID drift")
        missing = tuple(item for item in ids if item not in self.cache)
        if missing:
            indices = tuple(self.index[item] for item in missing)
            try:
                result = self.bridge.forward_indices(indices)
                features = result[1] if isinstance(result, tuple) and len(result) == 2 else result
            except Exception as error:
                raise NextR4Proxy24Error("received-IQ checkpoint bridge forward failed") from error
            values = np.asarray(features)
            if values.dtype != np.float32 or values.ndim != 2 or values.shape != (len(missing), runtime.Z_DIM) or not np.isfinite(values).all():
                raise NextR4Proxy24Error("checkpoint bridge feature contract drift")
            for physical_id, row in zip(missing, values, strict=True):
                frozen = np.ascontiguousarray(row, dtype=np.float32).copy()
                frozen.setflags(write=False)
                self.cache[physical_id] = frozen
        return np.ascontiguousarray(np.stack([self.cache[item] for item in ids]), dtype=np.float32)


def _checkpoint_smoke(cache: _FeatureCache) -> Mapping[str, Any]:
    ids = tuple(cache.capsule.physical_ids[: min(2, len(cache.capsule.physical_ids))])
    first = cache.take(ids).tobytes(order="C")
    cache.cache.clear()
    second = cache.take(ids).tobytes(order="C")
    if first != second:
        raise NextR4Proxy24Error("real checkpoint no-truth smoke was not repeatable")
    return {"schema": SMOKE_SCHEMA, "sample_count": len(ids), "canonical_repeat_exact": True, "truth_loaded": False}


def _load_checkpoint_bridge(args: argparse.Namespace, capsule: ReceivedCapsule, checkpoint_sha256: str) -> Any:
    """Build a received-IQ-only model bridge; tests may replace this function."""

    try:
        from cvsrffi.stage2_d105_phase1_bundle import (
            build_d105_exact_model_from_checkpoint,
            load_d105_exact_sha_bound_checkpoint,
        )
        import torch
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: checkpoint bridge dependencies unavailable") from error
    try:
        checkpoint, _ = load_d105_exact_sha_bound_checkpoint(args.checkpoint, checkpoint_sha256)
        model, _ = build_d105_exact_model_from_checkpoint(checkpoint, input_len=int(capsule.received_iq.shape[2]), device=args.device)
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: real checkpoint bridge load failed") from error

    class ReceivedIQBridge:
        def __init__(self) -> None:
            self.model = model.eval()
            self.checkpoint_sha256 = checkpoint_sha256
            self._linear = self.model.id_backbone.cls_head.joint_proj[0]
            self.device = torch.device(args.device)

        def forward_indices(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
            values = np.ascontiguousarray(capsule.received_iq[np.asarray(tuple(indices), dtype=np.int64)], dtype=np.float32)
            tensor = torch.frombuffer(memoryview(values), dtype=torch.float32, count=int(values.size)).reshape(tuple(values.shape)).clone().to(self.device)
            captured: list[Any] = []
            hook = self._linear.register_forward_hook(lambda _m, _i, output: captured.append(output))
            try:
                with torch.no_grad():
                    output = self.model.id_backbone(tensor, y=None, return_aux=True, domain_labels=None)
            finally:
                hook.remove()
            if not isinstance(output, Mapping) or "logits" not in output or len(captured) != 1:
                raise NextR4Proxy24Error("checkpoint bridge output contract drift")
            pre = captured[0]
            if pre.ndim != 2 or int(pre.shape[1]) != runtime.Z_DIM:
                raise NextR4Proxy24Error("checkpoint bridge joint_proj dimension drift")
            logits = np.asarray(output["logits"].detach().cpu().tolist(), dtype=np.float32)
            signed = np.asarray(pre.detach().cpu().tolist(), dtype=np.float32)
            # Runtime expects canonical nonnegative, unit R0 rows.  The model's
            # post-ReLU auxiliary output is the only permitted bridge view.
            r0 = np.asarray(np.maximum(signed, 0.0), dtype=np.float32)
            norms = np.linalg.norm(r0.astype(np.float64), axis=1, keepdims=True)
            if np.any(norms <= 0.0):
                raise NextR4Proxy24Error("checkpoint bridge produced a zero R0 row")
            r0 = np.ascontiguousarray(r0 / norms, dtype=np.float32)
            return logits, r0

    return ReceivedIQBridge()


def _lock_for_k(package: Mapping[str, Any], active_k: int) -> qknn.Phase1ZIDStudentTLock:
    values = package.get("qknn_lock_by_k")
    if not isinstance(values, Mapping) or str(active_k) not in values or not isinstance(values[str(active_k)], Mapping):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: qKNN lock for K{active_k} is missing")
    payload = dict(values[str(active_k)])
    payload["active_k"] = active_k
    try:
        return qknn.Phase1ZIDStudentTLock(**payload)
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: qKNN lock for K{active_k} is invalid") from error


def _fa_binding(*, row: Mapping[str, Any], package: Mapping[str, Any], support_ids: Sequence[str]) -> fa.FARDCE3RuntimeBinding:
    retained = tuple(item for item in package["class_registry"] if item != row["held_class"])
    support_root = _ordered_root(support_ids)
    authority = _sha(_canonical({"row_id": row["row_id"], "support_physical_ids": list(support_ids), "capsule_id": package["capsule_id"], "split_id": package["split_id"]}))
    return fa.FARDCE3RuntimeBinding(
        checkpoint_sha256=package["checkpoint_sha256"],
        capsule_id=package["capsule_id"],
        split_id=package["split_id"],
        row_id=row["row_id"],
        seed=int(package["seed"]),
        active_k=int(row["active_k"]),
        old_classes=retained,
        support_physical_root_sha256=support_root,
        support_authority_sha256=authority,
    )


def _registration_inputs(*, row: Mapping[str, Any], package: Mapping[str, Any], binding: Mapping[str, Any], cache: _FeatureCache) -> tuple[runtime.NextR4RegistrationInput, runtime.NextR4RegistrationInput]:
    classes = tuple(package["class_registry"])
    retained = tuple(item for item in classes if item != row["held_class"])
    _reject_grouped_query(binding, name="physical binding")
    qids = tuple(binding["query_physical_ids"])
    observations = tuple(binding["query_observation_ids"])
    query = cache.take(qids)
    support_k = binding["k1_support_ids_by_class"] if int(row["active_k"]) == 1 else binding["k5_support_ids_by_class"]
    all_support = {cls: cache.take(tuple(support_k[cls])) for cls in classes}
    reg0 = runtime.NextR4RegistrationInput(
        registration_id="REG0",
        registered_classes=retained,
        support_r0_by_class={cls: all_support[cls] for cls in retained},
        support_physical_ids_by_class={cls: tuple(support_k[cls]) for cls in retained},
        query_r0_zid160=query,
        query_physical_ids=qids,
        query_observation_ids=observations,
    )
    reg1 = runtime.NextR4RegistrationInput(
        registration_id="REG1",
        registered_classes=classes,
        support_r0_by_class=all_support,
        support_physical_ids_by_class={cls: tuple(support_k[cls]) for cls in classes},
        query_r0_zid160=query.copy(),
        query_physical_ids=qids,
        query_observation_ids=observations,
    )
    return reg0, reg1


def _validate_runtime_result(
    *, result: Mapping[str, Any], planned: Mapping[str, Any], binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Check one runtime result before it enters the immutable artifact."""

    if not isinstance(result, Mapping):
        raise NextR4Proxy24Error("runtime result must be a mapping")
    _reject_grouped_query(result, name="runtime result")
    for field in ("row_id", "held_receiver", "held_class", "active_k"):
        if result.get(field) != planned[field]:
            raise NextR4Proxy24Error(f"runtime result {field} drift")
    result_binding = result.get("binding_receipt")
    try:
        checked_binding = matrix.validate_next_r4_binding(result_binding)
    except Exception as error:
        raise NextR4Proxy24Error("runtime result binding receipt is invalid") from error
    if dict(checked_binding) != dict(binding) or checked_binding.get("binding_sha256") != binding.get("binding_sha256"):
        raise NextR4Proxy24Error("runtime result binding SHA/receipt drift")
    registrations = result.get("registrations")
    if not isinstance(registrations, Mapping) or set(registrations) != set(matrix.REGISTRATION_IDS):
        raise NextR4Proxy24Error("runtime result REG0/REG1 closure drift")
    query_ids = tuple(binding["query_physical_ids"])
    observation_ids = tuple(binding["query_observation_ids"])
    expected_states = {
        "REG0": tuple(matrix.REG0_STATES),
        "REG1": tuple(matrix.REG1_STATES),
    }
    for registration_id, states_expected in expected_states.items():
        registration = registrations[registration_id]
        states = registration.get("states") if isinstance(registration, Mapping) else None
        if not isinstance(states, Mapping) or set(states) != set(states_expected):
            raise NextR4Proxy24Error(f"runtime result {registration_id} state closure drift")
        for state_id in states_expected:
            state = states[state_id]
            if not isinstance(state, Mapping):
                raise NextR4Proxy24Error(f"runtime result {state_id} must be a mapping")
            if tuple(state.get("query_physical_ids", ())) != query_ids or tuple(state.get("query_observation_ids", ())) != observation_ids:
                raise NextR4Proxy24Error(f"runtime result {state_id} query binding drift")
    return result


def _load_prepare_receipt(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    receipt = _read_json(path, expected_sha256, "prepare receipt")
    if receipt.get("schema") != PREPARE_SCHEMA or receipt.get("candidate_id") != matrix.CANDIDATE_ID:
        raise NextR4Proxy24Error("prepare receipt schema/candidate drift")
    if receipt.get("truth_in_predictor_package") is not False or receipt.get("phase2_data_status") != "VALIDATED_ONCE":
        raise NextR4Proxy24Error("prepare receipt truth/protocol flags drift")
    observed = receipt.get("prepare_receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("prepare_receipt_sha256", None)
    if _require_sha(observed, "prepare receipt payload SHA256") != _sha(_canonical(unsigned)):
        raise NextR4Proxy24Error("prepare receipt payload hash drift")
    for field in PREPARE_AUTHORITY_FIELDS:
        _require_sha(receipt.get(field), f"prepare receipt {field}")
    _reject_grouped_query(receipt, name="prepare receipt")
    return receipt


def run_prepare(args: argparse.Namespace) -> Mapping[str, Any]:
    checkpoint_sha = _require_sha(args.checkpoint_sha256, "checkpoint SHA256")
    received = _load_received_capsule(args.received_iq, args.received_iq_sha256)
    # Metadata is parsed before any output root is created, so missing real
    # inputs leave no misleading partial run.
    metadata = _read_json(args.capsule_metadata, args.capsule_metadata_sha256, "capsule/split metadata")
    classes = tuple(sorted(tuple(metadata.get("class_registry", ()))))
    try:
        plan = matrix.build_next_r4_proxy24_plan(classes, held_receivers=tuple(metadata.get("held_receivers", ())))
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: capsule metadata registry is invalid") from error
    expected_keys = tuple(f"{receiver}|{held}" for receiver in matrix.HELD_RECEIVERS for held in classes)
    asset_manifest = _load_asset_manifest(args.fa_asset_manifest, args.fa_asset_manifest_sha256, checkpoint_sha256=checkpoint_sha, expected_keys=expected_keys)
    package, truth = _load_capsule_metadata(
        args.capsule_metadata,
        args.capsule_metadata_sha256,
        received=received,
        checkpoint_sha256=checkpoint_sha,
        asset_manifest=asset_manifest,
        asset_manifest_sha256=args.fa_asset_manifest_sha256,
    )
    root = _new_root(args.output_dir)
    _write_json_new(root / "predictor_package.json", package)
    _write_json_new(root / "truth.json", truth)
    _write_json_new(root / "asset_manifest.json", {"schema": ASSET_MANIFEST_SCHEMA, "entries": asset_manifest})
    receipt = {
        "schema": PREPARE_SCHEMA,
        "candidate_id": matrix.CANDIDATE_ID,
        "matrix_sha256": package["matrix_sha256"],
        "package_sha256": _sha_file(root / "predictor_package.json"),
        "truth_sha256": _sha_file(root / "truth.json"),
        "received_iq_sha256": received.received_iq_sha256,
        "checkpoint_sha256": checkpoint_sha,
        "asset_manifest_sha256": args.fa_asset_manifest_sha256,
        "truth_in_predictor_package": False,
        "phase2_data_status": "VALIDATED_ONCE",
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
    }
    receipt["prepare_receipt_sha256"] = _sha(_canonical(receipt))
    _write_json_new(root / "prepare_receipt.json", receipt)
    return {"output_dir": str(root), "package": str(root / "predictor_package.json"), "truth": str(root / "truth.json"), "receipt": str(root / "prepare_receipt.json"), "truth_in_predictor_package": False}


def _load_package(path: Path, expected_sha256: str, *, received: ReceivedCapsule, checkpoint_sha256: str, asset_manifest_sha256: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    package = _read_json(path, expected_sha256, "predictor package")
    _reject_grouped_query(package, name="predictor package")
    if package.get("schema") != PREDICTOR_PACKAGE_SCHEMA or package.get("candidate_id") != matrix.CANDIDATE_ID or package.get("protocol_schema") != matrix.PROTOCOL_SCHEMA or package.get("truth_free") is not True or package.get("truth_loaded") is not False:
        raise NextR4Proxy24Error("predictor package schema/provenance drift")
    if package.get("received_iq_sha256") != received.received_iq_sha256 or package.get("checkpoint_sha256") != checkpoint_sha256 or package.get("asset_manifest_sha256") != asset_manifest_sha256:
        raise NextR4Proxy24Error("predictor package input binding drift")
    plan = matrix.validate_next_r4_proxy24_plan(matrix.build_next_r4_proxy24_plan(tuple(package["class_registry"]), held_receivers=tuple(package["held_receivers"])))
    if package.get("matrix_sha256") != plan["matrix_sha256"] or len(package.get("rows", ())) != matrix.ROW_COUNT:
        raise NextR4Proxy24Error("predictor package matrix closure drift")
    package_rows = package["rows"]
    package_by_id = {str(item.get("row_id")): item for item in package_rows if isinstance(item, Mapping)}
    if len(package_by_id) != matrix.ROW_COUNT:
        raise NextR4Proxy24Error("predictor package row identity closure drift")
    for planned in plan["rows"]:
        item = package_by_id.get(str(planned["row_id"]))
        if not isinstance(item, Mapping) or any(item.get(field) != planned[field] for field in ("row_id", "held_receiver", "held_class", "active_k")):
            raise NextR4Proxy24Error("predictor package row identity drift")
    for item in package_rows:
        if not isinstance(item, Mapping) or "phase1_fit_ids" in item or "physical_binding_receipt" not in item:
            raise NextR4Proxy24Error("predictor package contains raw Phase1 IDs or lacks sealed binding")
        try:
            binding = matrix.validate_next_r4_binding(item["physical_binding_receipt"])
        except Exception as error:
            raise NextR4Proxy24Error("predictor package physical binding drift") from error
        if tuple(binding.get("registered_classes", ())) != tuple(package["class_registry"]):
            raise NextR4Proxy24Error("predictor package binding class registry drift")
    _reject_forbidden(package, name="predictor package")
    return package, plan


def run_predict(args: argparse.Namespace) -> Mapping[str, Any]:
    received = _load_received_capsule(args.received_iq, args.received_iq_sha256)
    checkpoint_sha = _require_sha(args.checkpoint_sha256, "checkpoint SHA256")
    prepare_receipt_sha = _require_sha(args.prepare_receipt_sha256, "prepare receipt SHA256")
    package_sha = _require_sha(args.package_sha256, "predictor package SHA256")
    asset_manifest_sha = _require_sha(args.fa_asset_manifest_sha256, "FA asset manifest SHA256")
    prepare_receipt = _load_prepare_receipt(args.prepare_receipt, prepare_receipt_sha)
    if (
        prepare_receipt["received_iq_sha256"] != received.received_iq_sha256
        or prepare_receipt["checkpoint_sha256"] != checkpoint_sha
        or prepare_receipt["package_sha256"] != package_sha
        or prepare_receipt["asset_manifest_sha256"] != asset_manifest_sha
    ):
        raise NextR4Proxy24Error("prepare receipt/input authority drift")
    _require_file(args.checkpoint, checkpoint_sha, "checkpoint")
    package, plan = _load_package(args.package, package_sha, received=received, checkpoint_sha256=checkpoint_sha, asset_manifest_sha256=asset_manifest_sha)
    if prepare_receipt["matrix_sha256"] != package["matrix_sha256"]:
        raise NextR4Proxy24Error("prepare receipt/package matrix authority drift")
    # Re-check manifest with package classes; this avoids accepting a package
    # that names a different outer-cell asset set.
    expected_keys = tuple(f"{receiver}|{held}" for receiver in matrix.HELD_RECEIVERS for held in package["class_registry"])
    manifest = _load_asset_manifest(args.fa_asset_manifest, asset_manifest_sha, checkpoint_sha256=checkpoint_sha, expected_keys=expected_keys)
    for item in package["rows"]:
        pair_key = f"{item['held_receiver']}|{item['held_class']}"
        binding = item["physical_binding_receipt"]
        if binding.get("phase1_fit_physical_root_sha256") != manifest[pair_key]["phase1_fit_physical_root_sha256"]:
            raise NextR4Proxy24Error(f"predictor package/FA Phase1 root drift for {pair_key}")
    bridge = _load_checkpoint_bridge(args, received, checkpoint_sha)
    cache = _FeatureCache(bridge, received, checkpoint_sha)
    smoke = _checkpoint_smoke(cache)
    root = _new_root(args.run_root)
    _write_json_new(root / "plan.json", plan)
    _write_json_new(root / "preregistration.json", {"schema": RUNNER_SCHEMA, "run_id": args.run_id, "candidate_id": matrix.CANDIDATE_ID, "matrix_sha256": plan["matrix_sha256"], "row_count": matrix.ROW_COUNT, "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT, "artifact_arm_count": matrix.ARTIFACT_ARM_COUNT, "truth_loaded": False, "prepare_receipt_sha256": prepare_receipt_sha, "prepare_receipt_payload_sha256": prepare_receipt["prepare_receipt_sha256"], "package_sha256": package_sha, "truth_sha256": prepare_receipt["truth_sha256"], "received_iq_sha256": received.received_iq_sha256, "checkpoint_sha256": checkpoint_sha, "asset_manifest_sha256": asset_manifest_sha, "query_rows_used_for_fit": 0, "query_state_updates": 0, "query_selection_count": 0})
    _write_json_new(root / "smoke.json", smoke)
    rows_by_id = {str(item["row_id"]): item for item in package["rows"]}
    runtime_results: list[Mapping[str, Any]] = []
    resources: list[Mapping[str, Any]] = []
    row_receipts: list[Mapping[str, Any]] = []
    asset_cache: dict[str, fa.FARDCE3Phase1Asset] = {}
    for planned in plan["rows"]:
        row = rows_by_id[str(planned["row_id"])]
        pair_key = f"{row['held_receiver']}|{row['held_class']}"
        entry = manifest[pair_key]
        if pair_key not in asset_cache:
            asset_cache[pair_key] = fa.deserialize_fa_rdce3_phase1_asset(Path(entry["asset_path"]).read_bytes())
        binding = row["physical_binding_receipt"]
        reg0, reg1 = _registration_inputs(row=row, package=package, binding=binding, cache=cache)
        fa_binding = _fa_binding(row=row, package=package, support_ids=reg0.support_physical_ids)
        result = runtime.execute_next_r4_logical_row(row=matrix.outer_key_from_mapping(planned), binding_receipt=binding, fa_asset=asset_cache[pair_key], fa_binding=fa_binding, reg0=reg0, reg1=reg1, qknn_lock=_lock_for_k(package, int(row["active_k"])))
        checked_result = _validate_runtime_result(result=result, planned=planned, binding=binding)
        runtime_results.append(_plain(checked_result))
        resources.append({"row_id": row["row_id"], "resource_receipt": _plain(checked_result["resource_receipt"])})
        row_receipts.append({"row_id": row["row_id"], "resource_receipt_sha256": _sha(_canonical(checked_result["resource_receipt"])), "fa_state_reuse_receipt_sha256": _sha(_canonical(checked_result["fa_state_reuse_receipt"]))})
    prediction = artifact.build_next_r4_prediction_artifact(plan=plan, row_results=runtime_results)
    _write_json_new(root / "prediction.json", prediction)
    _write_json_new(root / "resource.json", {"schema": RESOURCE_SCHEMA, "rows": resources, "truth_loaded": False})
    manifest_doc = {"schema": MANIFEST_SCHEMA, "candidate_id": matrix.CANDIDATE_ID, "matrix_sha256": plan["matrix_sha256"], "row_count": matrix.ROW_COUNT, "rows": row_receipts, "all_rows_sealed": True, "sealed_before_scoring": True, "truth_loaded": False}
    manifest_doc["manifest_sha256"] = _sha(_canonical(manifest_doc))
    _write_json_new(root / "manifest.json", manifest_doc)
    completion = {"schema": COMPLETION_SCHEMA, "status": "ARTIFACTS_COMPLETE_NOT_SCORED", "run_id": args.run_id, "row_count": matrix.ROW_COUNT, "unique_prediction_count": matrix.UNIQUE_PREDICTION_COUNT, "artifact_arm_count": matrix.ARTIFACT_ARM_COUNT, "truth_loaded": False, "prediction_sha256": _sha_file(root / "prediction.json"), "manifest_sha256": _sha_file(root / "manifest.json"), "resource_sha256": _sha_file(root / "resource.json"), "plan_sha256": _sha_file(root / "plan.json"), "matrix_sha256": plan["matrix_sha256"], "prepare_receipt_sha256": prepare_receipt_sha, "prepare_receipt_payload_sha256": prepare_receipt["prepare_receipt_sha256"], "package_sha256": package_sha, "truth_sha256": prepare_receipt["truth_sha256"], "received_iq_sha256": received.received_iq_sha256, "checkpoint_sha256": checkpoint_sha, "asset_manifest_sha256": asset_manifest_sha}
    _write_json_new(root / "completion.json", completion)
    return completion


def run_score(args: argparse.Namespace) -> Mapping[str, Any]:
    root = args.run_root.resolve(strict=True)
    required = ("plan.json", "prediction.json", "manifest.json", "resource.json", "completion.json")
    if any(not (root / name).is_file() for name in required):
        raise NextR4Proxy24Error("score requires complete NEXT-R4 prediction artifacts")
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "ARTIFACTS_COMPLETE_NOT_SCORED" or completion.get("row_count") != matrix.ROW_COUNT or completion.get("truth_loaded") is not False:
        raise NextR4Proxy24Error("score refused incomplete prediction closure")
    prepare_receipt_sha = _require_sha(args.prepare_receipt_sha256, "prepare receipt SHA256")
    truth_sha = _require_sha(args.truth_sha256, "truth SHA256")
    prepare_receipt = _load_prepare_receipt(args.prepare_receipt, prepare_receipt_sha)
    if truth_sha != prepare_receipt["truth_sha256"]:
        raise NextR4Proxy24Error("score truth SHA does not match prepare authority")
    expected = {name: _sha_file(root / f"{name}.json") for name in ("prediction", "manifest", "resource", "plan")}
    if any(completion.get(f"{name}_sha256") != value for name, value in expected.items()):
        raise NextR4Proxy24Error("score refused hash-mismatched prediction closure")
    authority_expected = {
        "prepare_receipt_sha256": prepare_receipt_sha,
        "prepare_receipt_payload_sha256": prepare_receipt["prepare_receipt_sha256"],
        "package_sha256": prepare_receipt["package_sha256"],
        "truth_sha256": prepare_receipt["truth_sha256"],
        "received_iq_sha256": prepare_receipt["received_iq_sha256"],
        "checkpoint_sha256": prepare_receipt["checkpoint_sha256"],
        "asset_manifest_sha256": prepare_receipt["asset_manifest_sha256"],
        "matrix_sha256": prepare_receipt["matrix_sha256"],
    }
    if any(completion.get(field) != value for field, value in authority_expected.items()):
        raise NextR4Proxy24Error("score refused prepare authority/completion drift")
    plan = matrix.validate_next_r4_proxy24_plan(json.loads((root / "plan.json").read_text(encoding="utf-8")))
    if plan["matrix_sha256"] != prepare_receipt["matrix_sha256"]:
        raise NextR4Proxy24Error("score refused prepare/plan matrix authority drift")
    prediction = json.loads((root / "prediction.json").read_text(encoding="utf-8"))
    _reject_grouped_query(prediction, name="prediction")
    try:
        scorer._validate_prediction(prediction, plan)
    except Exception as error:
        raise NextR4Proxy24Error("score refused invalid prediction closure") from error
    sealed_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha = sealed_manifest.pop("manifest_sha256", None)
    if manifest_sha != _sha(_canonical(sealed_manifest)) or sealed_manifest.get("all_rows_sealed") is not True or sealed_manifest.get("sealed_before_scoring") is not True:
        raise NextR4Proxy24Error("score refused invalid sealed manifest")
    # Truth is intentionally opened only after every prediction-side check.
    truth_bytes = _require_file(args.truth, truth_sha, "truth sidecar")
    if _sha(truth_bytes) != prepare_receipt["truth_sha256"]:
        raise NextR4Proxy24Error("truth sidecar SHA differs from prepare authority")
    try:
        truth = json.loads(truth_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR4Proxy24Error("truth sidecar must be UTF-8 JSON") from error
    if not isinstance(truth, Mapping):
        raise NextR4Proxy24Error("truth sidecar must be a mapping")
    result = dict(scorer.score_next_r4_proxy24(prediction=prediction, plan=plan, truth_by_query_id=truth))
    result["prediction_sha256"] = expected["prediction"]
    result["truth_sha256"] = _sha_file(args.truth)
    result["prepare_receipt_sha256"] = prepare_receipt_sha
    result["prepare_receipt_payload_sha256"] = prepare_receipt["prepare_receipt_sha256"]
    output = args.output.resolve(strict=False)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise NextR4Proxy24Error("score output must be a new absolute file")
    _write_json_new(output, result)
    return {"score_sha256": _sha_file(output), "truth_opened_after_complete_prediction": True, "row_count": matrix.ROW_COUNT}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEXT-R4 FA-RDCE3 x CER-PLR160 prepare/predict/score runner")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="seal VALIDATED_ONCE capsule metadata and FA asset manifest")
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--received-iq", required=True, type=Path)
    prepare.add_argument("--received-iq-sha256", required=True)
    prepare.add_argument("--capsule-metadata", required=True, type=Path)
    prepare.add_argument("--capsule-metadata-sha256", required=True)
    prepare.add_argument("--fa-asset-manifest", required=True, type=Path)
    prepare.add_argument("--fa-asset-manifest-sha256", required=True)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.set_defaults(func=run_prepare)
    predict = commands.add_parser("predict", help="forward received IQ and publish truth-free prediction")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--run-root", required=True, type=Path)
    predict.add_argument("--received-iq", required=True, type=Path)
    predict.add_argument("--received-iq-sha256", required=True)
    predict.add_argument("--package", required=True, type=Path)
    predict.add_argument("--package-sha256", required=True)
    predict.add_argument("--fa-asset-manifest", required=True, type=Path)
    predict.add_argument("--fa-asset-manifest-sha256", required=True)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--checkpoint-sha256", required=True)
    predict.add_argument("--prepare-receipt", required=True, type=Path)
    predict.add_argument("--prepare-receipt-sha256", required=True)
    predict.add_argument("--device", default="cpu")
    predict.set_defaults(func=run_predict)
    score = commands.add_parser("score", help="open truth only after complete prediction closure")
    score.add_argument("--run-root", required=True, type=Path)
    score.add_argument("--truth", required=True, type=Path)
    score.add_argument("--truth-sha256", required=True)
    score.add_argument("--prepare-receipt", required=True, type=Path)
    score.add_argument("--prepare-receipt-sha256", required=True)
    score.add_argument("--output", required=True, type=Path)
    score.set_defaults(func=run_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
