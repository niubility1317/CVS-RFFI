"""Builder-only D106 ``L_s`` IQ join and strict same-IQ feature tap.

This module closes the data-side P1 items in the frozen D106 design.  It is
deliberately not a generic source-pool feature exporter: the only accepted
split is D104's exact 588/5292/2520 split, only the frozen 588-row ``L_s``
archive may expose TX labels, and the exported archive contains no IQ.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from types import CodeType
import uuid
import zipfile

import numpy as np

from .leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_STAGE,
    LEO_WEAK_CACHE_SCHEMA_V1,
    LEO_WEAK_CACHE_SCHEMA_V2,
    LEO_WEAK_CACHE_SET_SCHEMA_V1,
    LEO_WEAK_CACHE_SET_SCHEMA_V2,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
    post_channel_iq_sha256,
)
from .stage2_d104_source_split import (
    CANDIDATE_ID as D104_CANDIDATE_ID,
    EXCLUSION_MANIFEST_CONTENT_ROOT_SHA256,
    EXCLUSION_MANIFEST_FILE_SHA256,
    EXPECTED_COUNTS,
    HISTORICAL_QUERY_COUNT,
    SPLIT_ID,
)
from .stage2_d105_feature_tap import Z_DIM, extract_d105_feature_tap
from .stage2_d105_phase1_bundle import (
    D105_TAP_CACHE_SELECTION_DOMAIN,
    _tensor_from_d105_float32_c_iq,
    build_d105_exact_model_from_checkpoint,
    load_d105_exact_sha_bound_checkpoint,
    load_d105_tap_cache_selection_salt,
)


CANDIDATE_ID = "D106-RDCE/GTSM-r3-SCATTER02"
PROTOCOL_SCHEMA = "p2_min_v1"
EXPECTED_SOURCE_ROWS = 8400
UPSTREAM_SOURCE_POOL_CACHE_SCOPE = "source_validation"
D104_LEGACY_SOURCE_POOL_HASH_FIELD = "source_train_cache_set_sha256"
RHO_LABEL = 0.1
FORWARD_BATCH_CAPACITY = 256
EXPECTED_CHECKPOINT_SHA256 = (
    "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
)

SOURCE_SPLIT_SCHEMA = "cvs.d104_r1.source_split.archive.v2"
SOURCE_SPLIT_STATUS = "FORMAL_PHASE1_SOURCE_SPLIT_COMPLETE"
DISJOINT_RECEIPT_SCHEMA = "cvs.phase1.d106.train_held_disjoint_receipt.v1"
TAP_SCHEMA = "cvs.phase1.d106.ls_strict_tap.v1"
TAP_RECEIPT_SCHEMA = "cvs.phase1.d106.ls_strict_tap_receipt.v1"
LS_IQ_SCHEMA = "cvs.phase1.d106.ls_received_iq.v1"
LS_IQ_RECEIPT_SCHEMA = "cvs.phase1.d106.ls_received_iq_receipt.v1"
LS_IQ_VALIDATOR_SCHEMA = "cvs.phase1.d106.ls_received_iq_validator.v2"
LS_IQ_ARCHIVE_NAME = "d106_ls_received_iq.npz"
LS_IQ_RECEIPT_NAME = "d106_ls_received_iq.receipt.json"
LS_IQ_VALIDATOR_NAME = "d106_ls_received_iq.validator.json"
COMPLETION_MARKER_NAME = "COMPLETED.json"
COMPLETION_MARKER_SCHEMA = "cvs.phase1.d106.exact_member_completion.v1"
LS_IQ_MEMBERS = (
    "received_iq",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
TAP_ARCHIVE_NAME = "d106_ls_strict_tap.npz"
TAP_RECEIPT_NAME = "d106_ls_strict_tap.receipt.json"
TAP_MEMBERS = (
    "pre_relu",
    "z_dom",
    "tx_labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
LS_INPUT_MEMBERS = (
    "z_dom",
    "pre_relu",
    "receiver_ids",
    "day_ids",
    "tx_labels",
    "physical_ids",
)
US_INPUT_MEMBERS = ("z_dom", "receiver_ids", "day_ids", "physical_ids")
SOURCE_VAL_INPUT_MEMBERS = (
    "z_id",
    "z_dom",
    "pre_relu",
    "labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
    "class_ids",
)
SOURCE_CACHE_REQUIRED_MEMBERS_V1 = (
    "leo_weak_iq",
    "raw_labels",
    "domain_labels",
    "tx_ids",
    "rx_ids",
    "day_ids",
    "eq_ids",
    "sig_ids",
    "dataset_role",
    "channel_views",
    "sat_scenarios",
    "satellite_seeds",
    "overlay_applied",
    "sample_ids",
    "post_channel_iq_sha256",
    "overlay_ids",
    "manifest_json",
)
SOURCE_CACHE_REQUIRED_MEMBERS_V2 = (
    *SOURCE_CACHE_REQUIRED_MEMBERS_V1[:8],
    "source_dataset_sha256",
    "source_record_indices",
    *SOURCE_CACHE_REQUIRED_MEMBERS_V1[8:],
)
SOURCE_CACHE_MEMBERS_BY_SCHEMA = {
    LEO_WEAK_CACHE_SCHEMA_V1: SOURCE_CACHE_REQUIRED_MEMBERS_V1,
    LEO_WEAK_CACHE_SCHEMA_V2: SOURCE_CACHE_REQUIRED_MEMBERS_V2,
}
SOURCE_CACHE_PROVENANCE_BY_SCHEMA = {
    LEO_WEAK_CACHE_SCHEMA_V1: (
        "sample_ids",
        "sat_scenarios",
        "satellite_seeds",
        "post_channel_iq_sha256",
        "overlay_ids",
    ),
    LEO_WEAK_CACHE_SCHEMA_V2: (
        "sample_ids",
        "source_dataset_sha256",
        "source_record_indices",
        "sat_scenarios",
        "satellite_seeds",
        "post_channel_iq_sha256",
        "overlay_ids",
    ),
}
RESTRICTED_SOURCE_LABEL_MEMBERS = frozenset({"tx_ids", "raw_labels"})
OPTIONAL_SOURCE_CACHE_MEMBERS = frozenset({"split_partition", "split_rank"})
FORBIDDEN_SOURCE_CACHE_MEMBERS = frozenset(
    {
        "raw_iq",
        "features",
        "tx_logits",
        "logits",
        "prototypes",
        "fft_logmag_features",
        "rf_stat_features",
        "fft_rf_features",
    }
)
CONSTRUCTION_CLOSURE_SCHEMA = "cvs.phase1.d106.tap_construction_closure.v1"


class D106Phase1TapError(ValueError):
    """Raised when the D106 source split, join, tap, or lineage drifts."""


def _canonical_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(member) for key, member in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(member) for member in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(
        convert(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if (
        len(text) != 64
        or text != text.lower()
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise D106Phase1TapError(f"{name} must be a lowercase SHA256")
    return text


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D106Phase1TapError("expected a regular non-symlink file")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_bytes(
    path: str | Path, *, expected_sha256: str | None = None, name: str
) -> tuple[bytes, str]:
    """Read one regular file from one no-follow handle and bind its identity."""

    source = Path(path)
    if source.is_symlink():
        raise D106Phase1TapError(f"{name} open failed: symlink forbidden")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    if hasattr(os, "O_NOFOLLOW"):
        flags |= int(os.O_NOFOLLOW)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise D106Phase1TapError(f"{name} open failed") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise D106Phase1TapError(f"{name} must be a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise D106Phase1TapError(f"{name} changed during read")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != _require_sha256(
        expected_sha256, f"{name} SHA256"
    ):
        raise D106Phase1TapError(f"{name} path/SHA256 drift")
    return payload, observed


def _code_payload(code: CodeType) -> dict[str, Any]:
    def constant(value: Any) -> Any:
        if isinstance(value, CodeType):
            return {"code": _code_payload(value)}
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, tuple):
            return [constant(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {"type": type(value).__name__, "repr": repr(value)}

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "flags": code.co_flags,
        "bytecode": code.co_code.hex(),
        "constants": [constant(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _callable_record(value: Any) -> dict[str, str]:
    code = getattr(value, "__code__", None)
    if not callable(value) or not isinstance(code, CodeType):
        raise D106Phase1TapError("formal execution dependency is not a Python callable")
    try:
        source_text = inspect.getsource(value)
        source_path = Path(inspect.getsourcefile(value) or "").resolve(strict=True)
    except (OSError, TypeError, ValueError) as error:
        raise D106Phase1TapError("formal execution dependency source is unavailable") from error
    module_name = str(getattr(value, "__module__", ""))
    qualname = str(getattr(value, "__qualname__", ""))
    if not module_name or not qualname:
        raise D106Phase1TapError("formal execution dependency identity drift")
    return {
        "module": module_name,
        "qualname": qualname,
        "source_file_name": source_path.name,
        "source_file_sha256": sha256_file(source_path),
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "code_sha256": hashlib.sha256(
            _canonical_bytes(_code_payload(code))
        ).hexdigest(),
    }


EXTRACT_EXECUTION_CALLABLES = (
    "extract_d106_ls_received_iq",
    "load_d106_source_split_binding",
    "load_d106_train_held_disjoint_receipt",
    "load_d105_tap_cache_selection_salt",
    "_load_ids_only",
    "select_d106_ls_cache_observations",
    "_load_d106_source_cache_index",
    "_load_selected_cache_scenario",
    "_load_inner_cache_manifest",
    "_is_forbidden_source_cache_member",
    "_npz_safe_member",
    "_resolve_cache_artifact",
    "_resolve_manifest_artifact",
    "_validate_npz_member_names",
    "_validate_ls_iq_arrays",
    "_d106_selection_index",
    "_read_regular_bytes",
    "_load_json_exact",
    "_require_sha256",
    "_deterministic_npz_bytes",
    "_write_new",
    "_write_completion_marker",
    "_load_completion_marker",
    "_publish_new_directory",
    "_array_sha256",
    "_ordered_id_root",
    "_set_id_root",
    "_d104_canonical_sha256",
    "_canonical_bytes",
    "sha256_file",
    "canonical_json_sha256",
    "ids_sha256",
    "overlay_id",
    "post_channel_iq_sha256",
)
EXPORT_EXECUTION_CALLABLES = (
    "export_d106_phase1_ls_tap",
    "_read_regular_bytes",
    "_require_sha256",
    "load_d106_ls_received_iq",
    "load_d106_ls_storage_validator",
    "_validate_ls_iq_arrays",
    "_load_ls_join_metadata",
    "join_d106_ls_observations",
    "load_d105_exact_sha_bound_checkpoint",
    "build_d105_exact_model_from_checkpoint",
    "_forward_fixed256",
    "_tensor_from_d105_float32_c_iq",
    "extract_d105_feature_tap",
    "_validate_checkpoint_loader_receipt",
    "_validate_model_reconstruction_receipt",
    "_validate_forward_receipt",
    "_deterministic_npz_bytes",
    "_write_new",
    "_write_completion_marker",
    "_load_completion_marker",
    "_publish_new_directory",
    "_array_sha256",
    "_ordered_id_root",
    "_canonical_bytes",
    "sha256_file",
)
_FORMAL_EXECUTION_BASELINES: Mapping[str, Any] = MappingProxyType({})


def _execution_closure(stage: str) -> dict[str, Any]:
    names = {
        "extract": EXTRACT_EXECUTION_CALLABLES,
        "export": EXPORT_EXECUTION_CALLABLES,
    }.get(stage)
    if names is None:
        raise D106Phase1TapError("unknown formal execution stage")
    callables = {name: _callable_record(globals().get(name)) for name in names}
    payload = {
        "schema": "cvs.phase1.d106.actual_execution_closure.v1",
        "stage": stage,
        "callables": callables,
        "construction_closure": _construction_closure(),
    }
    return payload | {
        "execution_content_root_sha256": hashlib.sha256(
            _canonical_bytes(payload)
        ).hexdigest()
    }


def _assert_execution_closure(stage: str) -> dict[str, Any]:
    current = _execution_closure(stage)
    baseline = _FORMAL_EXECUTION_BASELINES.get(stage)
    if type(baseline) is not dict or current != baseline:
        raise D106Phase1TapError(f"D106 {stage} actual execution closure drift")
    return current


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise D106Phase1TapError("object arrays are forbidden")
    if array.dtype.kind in {"U", "S"}:
        descriptor = {"dtype": "utf8-string", "shape": list(array.shape)}
        body = _canonical_bytes(array.astype(str).tolist())
    else:
        array = np.ascontiguousarray(array)
        descriptor = {"dtype": array.dtype.str, "shape": list(array.shape)}
        body = array.tobytes(order="C")
    return hashlib.sha256(_canonical_bytes(descriptor) + b"\0" + body).hexdigest()


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize exact-order, no-pickle arrays with fixed ZIP metadata."""

    target = io.BytesIO()
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_STORED) as bundle:
        for name, raw in arrays.items():
            value = np.asarray(raw)
            if value.dtype.hasobject:
                raise D106Phase1TapError("object arrays are forbidden")
            member = io.BytesIO()
            np.lib.format.write_array(member, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            bundle.writestr(info, member.getvalue())
    return target.getvalue()


def _ordered_id_root(values: Sequence[str]) -> str:
    # D104 uses this exact JSON spelling for its partition roots.  Keep it
    # separate from D106's ASCII-only canonical receipt encoder.
    encoded = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _d104_canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _set_id_root(values: Sequence[str]) -> str:
    return _ordered_id_root(sorted(str(value) for value in values))


def _d106_selection_index(selection_salt_sha256: str, physical_id: str) -> int:
    salt = bytes.fromhex(_require_sha256(selection_salt_sha256, "selection salt"))
    identifier = str(physical_id)
    if not identifier:
        raise D106Phase1TapError("physical_id must be non-empty")
    digest = hashlib.sha256(
        D105_TAP_CACHE_SELECTION_DOMAIN + salt + identifier.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % len(
        FORMAL_LEO_WEAK_SCENARIOS
    )


def _construction_closure() -> dict[str, Any]:
    module_dir = Path(__file__).resolve().parent
    code_root = module_dir.parent
    files = {
        "d106_phase1_tap": Path(__file__).resolve(),
        "d106_selector_cache_parser": Path(__file__).resolve(),
        "d106_export_cli": code_root / "scripts" / "export_d106_phase1_ls_tap.py",
        "d105_feature_tap": module_dir / "stage2_d105_feature_tap.py",
        "d105_exact_checkpoint_loader": module_dir / "stage2_d105_phase1_bundle.py",
        "leo_weak_cache_primitives": module_dir / "leo_weak_cache.py",
        "d105_model_factory": code_root / "model_dual_cvsincnet.py",
        "d105_model_backbone": code_root / "model.py",
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    payload = {
        "schema": CONSTRUCTION_CLOSURE_SCHEMA,
        "files_sha256": hashes,
        "symbols": {
            "d106_selector": "select_d106_ls_cache_observations",
            "d106_cache_parser": "_load_d106_source_cache_index",
            "d105_feature_tap": "extract_d105_feature_tap",
            "d105_exact_checkpoint_loader": "load_d105_exact_sha_bound_checkpoint",
        },
    }
    return payload | {
        "construction_content_root_sha256": hashlib.sha256(
            _canonical_bytes(payload)
        ).hexdigest()
    }


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_completion_marker(
    staging: Path, *, artifact_kind: str, members: Sequence[str]
) -> None:
    """Write the canonical completion marker after every data member is final."""

    names = tuple(str(name) for name in members)
    if (
        not artifact_kind
        or not names
        or len(set(names)) != len(names)
        or COMPLETION_MARKER_NAME in names
    ):
        raise D106Phase1TapError("D106 completion marker member closure drift")
    hashes = {name: sha256_file(staging / name) for name in names}
    marker = {
        "schema": COMPLETION_MARKER_SCHEMA,
        "artifact_kind": artifact_kind,
        "member_order": list(names),
        "member_sha256": hashes,
        "publication_policy": "atomic_output_reservation_exact_members_marker_last",
        "directory_atomic_visibility_claimed": False,
        "partial_output_acceptable": False,
    }
    _write_new(staging / COMPLETION_MARKER_NAME, _canonical_bytes(marker))


def _load_completion_marker(
    parent: Path, *, artifact_kind: str, members: Sequence[str]
) -> Mapping[str, Any]:
    """Reject partial publication and verify every exact sibling member hash."""

    names = tuple(str(name) for name in members)
    marker_path = parent / COMPLETION_MARKER_NAME
    marker_bytes, _marker_sha = _read_regular_bytes(
        marker_path, expected_sha256=None, name="D106 completion marker"
    )
    try:
        marker = json.loads(marker_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Phase1TapError("D106 completion marker JSON drift") from error
    expected_keys = {
        "schema", "artifact_kind", "member_order", "member_sha256",
        "publication_policy", "directory_atomic_visibility_claimed",
        "partial_output_acceptable",
    }
    if (
        type(marker) is not dict
        or marker_bytes != _canonical_bytes(marker)
        or set(marker) != expected_keys
        or marker.get("schema") != COMPLETION_MARKER_SCHEMA
        or marker.get("artifact_kind") != artifact_kind
        or marker.get("member_order") != list(names)
        or type(marker.get("member_sha256")) is not dict
        or tuple(marker["member_sha256"]) != names
        or marker.get("publication_policy")
        != "atomic_output_reservation_exact_members_marker_last"
        or marker.get("directory_atomic_visibility_claimed") is not False
        or marker.get("partial_output_acceptable") is not False
    ):
        raise D106Phase1TapError("D106 completion marker semantic closure drift")
    for name in names:
        expected = _require_sha256(marker["member_sha256"].get(name), name)
        _read_regular_bytes(
            parent / name,
            expected_sha256=expected,
            name=f"D106 completed member {name}",
        )
    return MappingProxyType(dict(marker))


def _publish_new_directory(
    staging: Path, output: Path, *, members: Sequence[str]
) -> None:
    """Atomically reserve a directory, then publish exact members in order."""

    output.mkdir()
    moved: list[Path] = []
    try:
        for name in members:
            source = staging / name
            destination = output / name
            if not source.is_file() or source.is_symlink() or destination.exists():
                raise D106Phase1TapError("D106 publish member closure drift")
            os.rename(source, destination)
            moved.append(destination)
        if any(staging.iterdir()):
            raise D106Phase1TapError("D106 staging contains unexpected members")
        staging.rmdir()
    except Exception:
        for destination in reversed(moved):
            try:
                destination.unlink()
            except OSError:
                pass
        try:
            output.rmdir()
        except OSError:
            pass
        raise


def _load_json_exact(path: str | Path, *, expected_sha256: str, name: str) -> dict[str, Any]:
    payload, _observed = _read_regular_bytes(
        path, expected_sha256=expected_sha256, name=name
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Phase1TapError(f"{name} must be strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise D106Phase1TapError(f"{name} must be an exact JSON object")
    return value


def _resolve_manifest_artifact(manifest: Path, raw_path: Any) -> Path:
    relative = Path(str(raw_path))
    if relative.is_absolute() or not relative.parts:
        raise D106Phase1TapError("source split artifact path must be relative")
    base = manifest.parent.resolve(strict=True)
    lexical_candidate = base / relative
    if lexical_candidate.is_symlink():
        raise D106Phase1TapError("source split artifact symlinks are forbidden")
    candidate = lexical_candidate.resolve(strict=True)
    try:
        candidate.relative_to(base)
    except ValueError as error:
        raise D106Phase1TapError("source split artifact escapes manifest root") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise D106Phase1TapError("source split artifact must be a regular file")
    return candidate


@dataclass(frozen=True, slots=True)
class D106SourceSplitBinding:
    manifest_path: Path
    manifest_sha256: str
    checkpoint_sha256: str
    runtime_sha256: str
    source_pool_cache_set_sha256: str
    selection_salt_receipt_sha256: str
    ls_archive: Path
    us_archive: Path
    source_val_archive: Path
    archive_sha256_by_role: Mapping[str, str]
    physical_id_roots: Mapping[str, str]


def load_d106_source_split_binding(
    path: str | Path, expected_sha256: str
) -> D106SourceSplitBinding:
    """Validate and resolve the exact frozen D104 root manifest."""

    raw_manifest = Path(path)
    if raw_manifest.is_symlink() or not raw_manifest.is_file():
        raise D106Phase1TapError("D104 source split manifest must be a regular file")
    manifest_path = raw_manifest.resolve(strict=True)
    value = _load_json_exact(
        manifest_path,
        expected_sha256=expected_sha256,
        name="D104 source split manifest",
    )
    partition = value.get("partition")
    roles = value.get("roles")
    source_val = value.get("source_val")
    inputs = value.get("inputs")
    if not all(type(member) is dict for member in (partition, roles, source_val, inputs)):
        raise D106Phase1TapError("D104 source split manifest structure drift")
    roots = partition.get("physical_id_roots")
    exclusion = value.get("historical_exclusion_manifest")
    partition_body = dict(partition) if type(partition) is dict else {}
    partition_receipt_sha = partition_body.pop("receipt_sha256", None)
    if (
        value.get("schema") != SOURCE_SPLIT_SCHEMA
        or value.get("candidate_id") != D104_CANDIDATE_ID
        or value.get("split_id") != SPLIT_ID
        or value.get("status") != SOURCE_SPLIT_STATUS
        or value.get("artifact_stage")
        != "phase1_offline_before_new_source_held_truth_open"
        or value.get("protocol_schema") != PROTOCOL_SCHEMA
        or value.get("target25_authorized") is not False
        or value.get("target_access") is not False
        or value.get("formal_query_access") is not False
        or type(exclusion) is not dict
        or exclusion
        != {
            "sha256": EXCLUSION_MANIFEST_FILE_SHA256,
            "content_root_sha256": EXCLUSION_MANIFEST_CONTENT_ROOT_SHA256,
            "query_count": HISTORICAL_QUERY_COUNT,
        }
        or partition.get("schema") != "cvs.d104_r1.source_split.rows.v1"
        or partition.get("candidate_id") != D104_CANDIDATE_ID
        or partition.get("split_id") != SPLIT_ID
        or partition_receipt_sha != _d104_canonical_sha256(partition_body)
        or partition.get("counts") != EXPECTED_COUNTS
        or partition.get("overlap_count") != 0
        or partition.get("union_complete") is not True
        or partition.get("source_val_performance_computed") is not False
        or type(roots) is not dict
        or set(roots) != set(EXPECTED_COUNTS)
        or set(roles) != {"L_s", "U_s"}
    ):
        raise D106Phase1TapError("D104 source split semantic closure drift")
    for role in EXPECTED_COUNTS:
        _require_sha256(roots.get(role), f"{role} physical-ID root")
    if EXPECTED_COUNTS["L_s"] / (
        EXPECTED_COUNTS["L_s"] + EXPECTED_COUNTS["U_s"]
    ) != RHO_LABEL:
        raise D106Phase1TapError("D104 rho_label closure drift")

    checkpoint = _require_sha256(inputs.get("checkpoint_sha256"), "checkpoint")
    if checkpoint != EXPECTED_CHECKPOINT_SHA256:
        raise D106Phase1TapError("D106 frozen checkpoint SHA256 drift")
    runtime = _require_sha256(inputs.get("runtime_sha256"), "runtime")
    # D104 sealed this 8400-row upstream pool under a historically misleading
    # field name.  Preserve the immutable manifest while giving D106 the actual
    # authority semantics: it is the source-validation pool, not source-train.
    cache = _require_sha256(
        inputs.get(D104_LEGACY_SOURCE_POOL_HASH_FIELD),
        "D104 legacy-named upstream source-pool cache set",
    )
    salt = _require_sha256(
        inputs.get("selection_salt_receipt_sha256"), "selection-salt receipt"
    )

    resolved: dict[str, Path] = {}
    archive_hashes: dict[str, str] = {}
    for role, members in (("L_s", LS_INPUT_MEMBERS), ("U_s", US_INPUT_MEMBERS)):
        row = roles.get(role)
        if (
            type(row) is not dict
            or row.get("row_count") != EXPECTED_COUNTS[role]
            or not isinstance(row.get("archive"), str)
        ):
            raise D106Phase1TapError(f"D104 {role} artifact binding drift")
        expected = _require_sha256(row.get("archive_sha256"), f"{role} archive")
        archive = _resolve_manifest_artifact(manifest_path, row["archive"])
        _validate_npz_member_names(archive, members, expected_sha256=expected)
        resolved[role] = archive
        archive_hashes[role] = expected

    scorer = source_val.get("scorer_archive")
    if type(scorer) is not dict or set(scorer) != {"path", "sha256"}:
        raise D106Phase1TapError("D104 source_val scorer binding drift")
    scorer_sha = _require_sha256(scorer.get("sha256"), "source_val archive")
    scorer_path = _resolve_manifest_artifact(manifest_path, scorer.get("path"))
    _validate_npz_member_names(
        scorer_path, SOURCE_VAL_INPUT_MEMBERS, expected_sha256=scorer_sha
    )
    archive_hashes["source_val"] = scorer_sha
    return D106SourceSplitBinding(
        manifest_path=manifest_path,
        manifest_sha256=_require_sha256(expected_sha256, "source split manifest"),
        checkpoint_sha256=checkpoint,
        runtime_sha256=runtime,
        source_pool_cache_set_sha256=cache,
        selection_salt_receipt_sha256=salt,
        ls_archive=resolved["L_s"],
        us_archive=resolved["U_s"],
        source_val_archive=scorer_path,
        archive_sha256_by_role=MappingProxyType(dict(archive_hashes)),
        physical_id_roots=MappingProxyType(dict(roots)),
    )


def _validate_npz_member_names(
    path: Path, expected: Sequence[str], *, expected_sha256: str
) -> None:
    try:
        archive_bytes, _observed = _read_regular_bytes(
            path, expected_sha256=expected_sha256, name="trusted NPZ"
        )
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as payload:
            if tuple(payload.files) != tuple(expected):
                raise D106Phase1TapError(f"NPZ exact member closure drift: {path}")
    except (OSError, ValueError, TypeError) as error:
        raise D106Phase1TapError(f"cannot inspect trusted NPZ: {path}") from error


def _load_ids_only(
    path: Path,
    expected_members: Sequence[str],
    expected_count: int,
    *,
    expected_sha256: str | None = None,
) -> np.ndarray:
    try:
        archive_bytes, _observed = _read_regular_bytes(
            path, expected_sha256=expected_sha256, name="ID-only archive"
        )
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as payload:
            if tuple(payload.files) != tuple(expected_members):
                raise D106Phase1TapError("ID-only archive member closure drift")
            identifiers = np.asarray(payload["physical_ids"]).astype(str)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise D106Phase1TapError("ID-only archive read failed") from error
    if (
        identifiers.shape != (expected_count,)
        or len(set(identifiers.tolist())) != expected_count
        or any(not value for value in identifiers.tolist())
    ):
        raise D106Phase1TapError("physical-ID count/uniqueness closure drift")
    return identifiers


def build_d106_train_held_disjoint_receipt(
    *,
    source_split_manifest: str | Path,
    source_split_manifest_sha256: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Independently read IDs only and attest train/held disjointness."""

    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise D106Phase1TapError("disjoint receipt parent must be an existing directory")
    binding = load_d106_source_split_binding(
        source_split_manifest, source_split_manifest_sha256
    )
    ls_ids = _load_ids_only(
        binding.ls_archive,
        LS_INPUT_MEMBERS,
        EXPECTED_COUNTS["L_s"],
        expected_sha256=binding.archive_sha256_by_role["L_s"],
    )
    us_ids = _load_ids_only(
        binding.us_archive,
        US_INPUT_MEMBERS,
        EXPECTED_COUNTS["U_s"],
        expected_sha256=binding.archive_sha256_by_role["U_s"],
    )
    held_ids = _load_ids_only(
        binding.source_val_archive,
        SOURCE_VAL_INPUT_MEMBERS,
        EXPECTED_COUNTS["source_val"],
        expected_sha256=binding.archive_sha256_by_role["source_val"],
    )
    if _ordered_id_root(ls_ids.tolist()) != binding.physical_id_roots["L_s"]:
        raise D106Phase1TapError("L_s physical-ID root drift")
    if _ordered_id_root(us_ids.tolist()) != binding.physical_id_roots["U_s"]:
        raise D106Phase1TapError("U_s physical-ID root drift")
    if _ordered_id_root(held_ids.tolist()) != binding.physical_id_roots["source_val"]:
        raise D106Phase1TapError("source_val physical-ID root drift")
    train = set(ls_ids.tolist()) | set(us_ids.tolist())
    held = set(held_ids.tolist())
    overlap = train.intersection(held)
    if len(train) != EXPECTED_COUNTS["L_s"] + EXPECTED_COUNTS["U_s"] or overlap:
        raise D106Phase1TapError("D106 independent train/held disjointness failed")
    if len(train | held) != EXPECTED_SOURCE_ROWS:
        raise D106Phase1TapError("D106 independent source-pool union failed")
    receipt = {
        "schema": DISJOINT_RECEIPT_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "split_id": SPLIT_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "source_split_manifest_sha256": binding.manifest_sha256,
        "archive_sha256_by_role": dict(binding.archive_sha256_by_role),
        "counts": dict(EXPECTED_COUNTS),
        "rho_label": RHO_LABEL,
        "ordered_physical_id_roots": dict(binding.physical_id_roots),
        "set_physical_id_roots": {
            "L_s": _set_id_root(ls_ids.tolist()),
            "U_s": _set_id_root(us_ids.tolist()),
            "source_val": _set_id_root(held_ids.tolist()),
            "phase1_train": _set_id_root(sorted(train)),
        },
        "phase1_train_count": len(train),
        "train_held_intersection_count": 0,
        "source_pool_union_count": len(train | held),
        "id_only_validation": True,
        "tx_labels_read": False,
        "target_access": False,
        "formal_query_access": False,
    }
    _write_new(output, _canonical_bytes(receipt))
    return receipt | {"receipt_sha256": sha256_file(output), "path": str(output.resolve())}


def load_d106_train_held_disjoint_receipt(
    path: str | Path,
    expected_sha256: str,
    *,
    binding: D106SourceSplitBinding,
) -> dict[str, Any]:
    receipt = _load_json_exact(
        path,
        expected_sha256=expected_sha256,
        name="D106 train-held disjoint receipt",
    )
    source = Path(path)
    try:
        if source.read_bytes() != _canonical_bytes(receipt):
            raise D106Phase1TapError("D106 disjoint receipt is not canonical JSON")
    except OSError as error:
        raise D106Phase1TapError("cannot reread D106 disjoint receipt") from error
    expected_keys = {
        "schema", "candidate_id", "split_id", "protocol_schema",
        "source_split_manifest_sha256", "archive_sha256_by_role", "counts",
        "rho_label", "ordered_physical_id_roots", "set_physical_id_roots",
        "phase1_train_count", "train_held_intersection_count",
        "source_pool_union_count", "id_only_validation", "tx_labels_read",
        "target_access", "formal_query_access",
    }
    set_roots = receipt.get("set_physical_id_roots")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != DISJOINT_RECEIPT_SCHEMA
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("split_id") != SPLIT_ID
        or receipt.get("protocol_schema") != PROTOCOL_SCHEMA
        or receipt.get("source_split_manifest_sha256") != binding.manifest_sha256
        or receipt.get("archive_sha256_by_role") != dict(binding.archive_sha256_by_role)
        or receipt.get("counts") != EXPECTED_COUNTS
        or receipt.get("rho_label") != RHO_LABEL
        or receipt.get("ordered_physical_id_roots") != dict(binding.physical_id_roots)
        or type(set_roots) is not dict
        or set(set_roots) != {"L_s", "U_s", "source_val", "phase1_train"}
        or any(_require_sha256(value, "ID set root") != value for value in set_roots.values())
        or receipt.get("phase1_train_count") != 5880
        or receipt.get("train_held_intersection_count") != 0
        or receipt.get("source_pool_union_count") != EXPECTED_SOURCE_ROWS
        or receipt.get("id_only_validation") is not True
        or receipt.get("tx_labels_read") is not False
        or receipt.get("target_access") is not False
        or receipt.get("formal_query_access") is not False
    ):
        raise D106Phase1TapError("D106 disjoint receipt semantic closure drift")
    return receipt


def _resolve_cache_artifact(manifest_path: Path, raw_path: Any) -> Path:
    candidate = Path(str(raw_path))
    lexical = candidate if candidate.is_absolute() else manifest_path.parent / candidate
    if lexical.is_symlink() or not lexical.is_file():
        raise D106Phase1TapError("source cache artifact must be a regular non-symlink file")
    return lexical.resolve(strict=True)


def _load_d106_source_cache_index(
    path: str | Path, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    """Validate the cache-set envelope without materializing any cache label."""

    source = Path(path)
    payload, _observed = _read_regular_bytes(
        source,
        expected_sha256=expected_sha256,
        name="upstream source-pool cache set",
    )
    try:
        manifest = json.loads(payload.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Phase1TapError(
            "upstream source-pool cache set must be strict JSON"
        ) from error
    if type(manifest) is not dict:
        raise D106Phase1TapError("upstream source-pool cache set must be an object")
    observed_schema = manifest.get("schema")
    if (
        observed_schema not in {
            LEO_WEAK_CACHE_SET_SCHEMA_V1,
            LEO_WEAK_CACHE_SET_SCHEMA_V2,
        }
        or manifest.get("artifact_stage") != LEO_WEAK_CACHE_STAGE
        or manifest.get("phase2_sample_view_policy") != PHASE2_SAMPLE_VIEW_POLICY
        or manifest.get("clean_sample_access") is not False
        or manifest.get("clean_derived_signal_access") is not False
        or manifest.get("target_channel_view") != "leo_weak_only"
        or manifest.get("cache_scope") != UPSTREAM_SOURCE_POOL_CACHE_SCOPE
        or {str(value) for value in manifest.get("output_roles", [])} != {"source"}
    ):
        raise D106Phase1TapError("upstream source-pool cache-set semantic closure drift")
    scenario_map = manifest.get("cache_npz_by_scenario")
    hash_map = manifest.get("cache_sha256_by_scenario")
    if (
        type(scenario_map) is not dict
        or type(hash_map) is not dict
        or tuple(scenario_map) != FORMAL_LEO_WEAK_SCENARIOS
        or tuple(hash_map) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise D106Phase1TapError("source cache scenario map/order drift")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        expected_cache = _require_sha256(hash_map[scenario], f"{scenario} cache")
        cache_path = _resolve_cache_artifact(source, scenario_map[scenario])
        paths[scenario] = cache_path
        hashes[scenario] = expected_cache
    return manifest, paths, hashes


def _is_forbidden_source_cache_member(name: str) -> bool:
    normalized = str(name).strip().lower()
    return (
        normalized in FORBIDDEN_SOURCE_CACHE_MEMBERS
        or normalized.startswith("clean")
        or "clean_iq" in normalized
        or "clean_feature" in normalized
        or "clean_logit" in normalized
        or "clean_proto" in normalized
    )


def _npz_safe_member(
    archive: np.lib.npyio.NpzFile, name: str, *, rows: int | None = None
) -> np.ndarray:
    if name in RESTRICTED_SOURCE_LABEL_MEMBERS:
        raise D106Phase1TapError(f"restricted source label member access attempted: {name}")
    value = np.asarray(archive[name])
    if value.dtype.hasobject:
        raise D106Phase1TapError(f"safe source cache member is object dtype: {name}")
    if rows is not None and (value.ndim < 1 or len(value) != rows):
        raise D106Phase1TapError(f"source cache row count drift: {name}")
    return value


def _load_inner_cache_manifest(
    archive: np.lib.npyio.NpzFile, *, scenario: str
) -> tuple[dict[str, Any], str]:
    raw = _npz_safe_member(archive, "manifest_json")
    if raw.size != 1:
        raise D106Phase1TapError("source cache manifest_json must be scalar")
    value = raw.reshape(-1)[0]
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise D106Phase1TapError("source cache manifest_json is not UTF-8") from error
    try:
        manifest = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise D106Phase1TapError("source cache manifest_json is invalid") from error
    if type(manifest) is not dict:
        raise D106Phase1TapError("source cache manifest_json must be an object")
    schema = str(manifest.get("schema", ""))
    if schema not in SOURCE_CACHE_MEMBERS_BY_SCHEMA:
        raise D106Phase1TapError("source cache inner schema drift")
    required = {
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "contains_post_channel_iq_only": True,
        "contains_clean_rows": False,
        "target_channel_view": "leo_weak_only",
        "scenario": scenario,
        "iq_array_key": "leo_weak_iq",
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "star_ground_channel_impl": "simplified_leo_residual",
        "channel_model": "leo_residual",
    }
    if any(manifest.get(key) != expected for key, expected in required.items()):
        raise D106Phase1TapError("source cache inner manifest contract drift")
    if (
        tuple(str(item) for item in manifest.get("target_channel_scenarios", []))
        != (scenario,)
        or {str(item) for item in manifest.get("output_roles", [])} != {"source"}
        or tuple(
            str(item) for item in manifest.get("sample_overlay_provenance_fields", [])
        )
        != SOURCE_CACHE_PROVENANCE_BY_SCHEMA[schema]
    ):
        raise D106Phase1TapError("source cache inner role/provenance drift")
    _require_sha256(manifest.get("channel_config_sha256"), "channel config")
    _require_sha256(manifest.get("builder_sha256"), "cache builder")
    return manifest, schema


def _load_selected_cache_scenario(
    path: Path,
    *,
    scenario: str,
    wanted_ids: Sequence[str],
    expected_sha256: str,
) -> tuple[list[str], dict[str, tuple[str, str, str, np.ndarray]], dict[str, Any]]:
    """Read safe metadata and IQ; never materialize ``tx_ids``/``raw_labels``."""

    try:
        cache_bytes, observed_sha256 = _read_regular_bytes(
            path,
            expected_sha256=expected_sha256,
            name=f"{scenario} source cache",
        )
        with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as archive:
            members = tuple(str(name) for name in archive.files)
            if "manifest_json" not in members:
                raise D106Phase1TapError("source cache manifest_json missing")
            forbidden = [name for name in members if _is_forbidden_source_cache_member(name)]
            if forbidden:
                raise D106Phase1TapError(
                    f"source cache contains forbidden members: {sorted(forbidden)}"
                )
            manifest, schema = _load_inner_cache_manifest(archive, scenario=scenario)
            required = SOURCE_CACHE_MEMBERS_BY_SCHEMA[schema]
            observed_members = frozenset(members)
            required_members = set(required)
            if len(members) != len(observed_members) or observed_members not in {
                frozenset(required_members),
                frozenset(required_members | OPTIONAL_SOURCE_CACHE_MEMBERS),
            }:
                raise D106Phase1TapError("source cache exact member closure drift")

            raw_sample_ids = _npz_safe_member(archive, "sample_ids")
            if raw_sample_ids.dtype.kind not in {"U", "S"}:
                raise D106Phase1TapError("source cache sample-ID dtype drift")
            sample_ids = raw_sample_ids.astype(str)
            physical_root = ids_sha256(sample_ids.tolist())
            if (
                sample_ids.shape != (EXPECTED_SOURCE_ROWS,)
                or len(set(sample_ids.tolist())) != EXPECTED_SOURCE_ROWS
                or any(not item for item in sample_ids.tolist())
                or manifest.get("row_count") != EXPECTED_SOURCE_ROWS
                or manifest.get("physical_sample_ids_sha256") != physical_root
            ):
                raise D106Phase1TapError("source cache physical-ID registry drift")
            index = {value: row for row, value in enumerate(sample_ids.tolist())}
            try:
                selected_rows = np.asarray(
                    [index[str(value)] for value in wanted_ids], dtype=np.int64
                )
            except KeyError as error:
                raise D106Phase1TapError(
                    f"selected L_s physical ID missing from {scenario} cache"
                ) from error
            if len(set(selected_rows.tolist())) != len(wanted_ids):
                raise D106Phase1TapError("selected cache join is not one-to-one")

            string_members: dict[str, np.ndarray] = {}
            for member in (
                "dataset_role", "sat_scenarios", "channel_views", "rx_ids",
                "day_ids", "overlay_ids", "post_channel_iq_sha256",
            ):
                raw_member = _npz_safe_member(
                    archive, member, rows=EXPECTED_SOURCE_ROWS
                )
                if raw_member.dtype.kind not in {"U", "S"}:
                    raise D106Phase1TapError(
                        f"source cache string dtype drift: {member}"
                    )
                value = raw_member.astype(str)
                if any(not item for item in value.tolist()):
                    raise D106Phase1TapError(
                        f"source cache blank string drift: {member}"
                    )
                string_members[member] = value
            roles = string_members["dataset_role"]
            scenarios = string_members["sat_scenarios"]
            views = string_members["channel_views"]
            receiver = string_members["rx_ids"]
            day = string_members["day_ids"]
            overlay_ids = string_members["overlay_ids"]
            iq_hashes = string_members["post_channel_iq_sha256"]
            raw_applied = _npz_safe_member(
                archive, "overlay_applied", rows=EXPECTED_SOURCE_ROWS
            )
            if raw_applied.dtype != np.bool_:
                raise D106Phase1TapError("source cache overlay-applied dtype drift")
            applied = raw_applied.astype(bool)
            if (
                not np.all(roles == "source")
                or not np.all(scenarios == scenario)
                or not np.all(views == "rx_base")
                or not bool(np.all(applied))
            ):
                raise D106Phase1TapError("source cache full role/scenario/view closure drift")
            raw_seeds = _npz_safe_member(
                archive, "satellite_seeds", rows=EXPECTED_SOURCE_ROWS
            )
            if raw_seeds.dtype.kind not in {"i", "u"}:
                raise D106Phase1TapError("source cache satellite-seed dtype drift")
            seeds = raw_seeds.astype(np.int64)
            # NPZ member granularity requires materializing this complete IQ
            # member.  Only selected L_s rows are copied to the method surface.
            iq_storage = _npz_safe_member(
                archive, "leo_weak_iq", rows=EXPECTED_SOURCE_ROWS
            )
            if (
                iq_storage.dtype != np.float32
                or iq_storage.shape != (EXPECTED_SOURCE_ROWS, 2, 256)
                or not np.isfinite(iq_storage).all()
            ):
                raise D106Phase1TapError("source cache IQ storage contract drift")
            channel_hash = str(manifest["channel_config_sha256"])
            for row, physical_id in enumerate(sample_ids.tolist()):
                observed_iq_hash = post_channel_iq_sha256(iq_storage[row])
                if observed_iq_hash != iq_hashes[row]:
                    raise D106Phase1TapError("source cache full IQ digest drift")
                expected_overlay = overlay_id(
                    sample_id=physical_id,
                    scenario=scenario,
                    satellite_seed=int(seeds[row]),
                    channel_config_sha256=channel_hash,
                    iq_sha256=observed_iq_hash,
                )
                if overlay_ids[row] != expected_overlay:
                    raise D106Phase1TapError("source cache full overlay provenance drift")
            storage_semantics = {
                "physical_sample_ids_sha256": physical_root,
                "receiver_ids_sha256": _array_sha256(receiver),
                "day_ids_sha256": _array_sha256(day),
                "roles_sha256": _array_sha256(roles),
                "scenarios_sha256": _array_sha256(scenarios),
                "views_sha256": _array_sha256(views),
                "applied_sha256": _array_sha256(applied),
                "seeds_sha256": _array_sha256(seeds),
                "iq_hashes_sha256": _array_sha256(iq_hashes),
                "overlay_ids_sha256": _array_sha256(overlay_ids),
            }
            storage_semantic_root = hashlib.sha256(
                _canonical_bytes(storage_semantics)
            ).hexdigest()
            selected: dict[str, tuple[str, str, str, np.ndarray]] = {}
            for physical_id, row in zip(wanted_ids, selected_rows.tolist(), strict=True):
                selected_iq = np.ascontiguousarray(iq_storage[row], dtype=np.float32)
                selected[str(physical_id)] = (
                    receiver[row],
                    day[row],
                    overlay_ids[row],
                    selected_iq,
                )
    except (OSError, ValueError, TypeError, KeyError) as error:
        if isinstance(error, D106Phase1TapError):
            raise
        raise D106Phase1TapError(f"selective source cache read failed: {scenario}") from error
    return sample_ids.tolist(), selected, {
        "schema": schema,
        "row_count": EXPECTED_SOURCE_ROWS,
        "manifest_sha256": canonical_json_sha256(manifest),
        "physical_sample_ids_sha256": ids_sha256(sample_ids.tolist()),
        "storage_semantic_root_sha256": storage_semantic_root,
        "full_storage_semantics_verified": True,
        "full_iq_digest_rows_verified": EXPECTED_SOURCE_ROWS,
        "full_overlay_rows_verified": EXPECTED_SOURCE_ROWS,
        "iq_storage_rows_materialized": EXPECTED_SOURCE_ROWS,
        "method_visible_iq_rows": len(wanted_ids),
        "restricted_label_members_present": sorted(
            RESTRICTED_SOURCE_LABEL_MEMBERS.intersection(members)
        ),
        "restricted_label_members_read_or_materialized": False,
        "cache_sha256": observed_sha256,
    }


def select_d106_ls_cache_observations(
    path: str | Path,
    *,
    expected_sha256: str,
    ls_physical_ids: Sequence[str],
    selection_salt_sha256: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Two-stage selector whose selection domain is exactly frozen ``L_s``."""

    physical_ids = tuple(str(value) for value in ls_physical_ids)
    if (
        len(physical_ids) != EXPECTED_COUNTS["L_s"]
        or len(set(physical_ids)) != EXPECTED_COUNTS["L_s"]
        or any(not value for value in physical_ids)
    ):
        raise D106Phase1TapError("D106 selection requires exactly 588 unique L_s IDs")
    # Stage one: determine the unique scenario from only L_s IDs plus salt.
    scenario_by_id = {
        physical_id: FORMAL_LEO_WEAK_SCENARIOS[
            _d106_selection_index(selection_salt_sha256, physical_id)
        ]
        for physical_id in physical_ids
    }
    wanted_by_scenario = {
        scenario: [
            physical_id
            for physical_id in physical_ids
            if scenario_by_id[physical_id] == scenario
        ]
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    outer, paths, hashes = _load_d106_source_cache_index(
        path, expected_sha256=expected_sha256
    )
    selected_by_id: dict[str, tuple[str, str, str, np.ndarray]] = {}
    scenario_audits: dict[str, Any] = {}
    reference_ids: list[str] | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        full_ids, selected, audit = _load_selected_cache_scenario(
            paths[scenario],
            scenario=scenario,
            wanted_ids=wanted_by_scenario[scenario],
            expected_sha256=hashes[scenario],
        )
        if reference_ids is None:
            reference_ids = full_ids
        elif full_ids != reference_ids:
            raise D106Phase1TapError("source cache scenarios physical-ID order drift")
        selected_by_id.update(selected)
        scenario_audits[scenario] = audit
    assert reference_ids is not None
    if (
        len(selected_by_id) != EXPECTED_COUNTS["L_s"]
        or not set(physical_ids).issubset(set(reference_ids))
        or str(outer.get("physical_sample_ids_sha256", ""))
        != ids_sha256(reference_ids)
    ):
        raise D106Phase1TapError("D106 selected/full source cache closure drift")
    metadata = {
        "receiver_ids": np.asarray(
            [selected_by_id[value][0] for value in physical_ids], dtype=np.str_
        ),
        "day_ids": np.asarray(
            [selected_by_id[value][1] for value in physical_ids], dtype=np.str_
        ),
        "physical_ids": np.asarray(physical_ids, dtype=np.str_),
        "scenario_names": np.asarray(
            [scenario_by_id[value] for value in physical_ids], dtype=np.str_
        ),
        "observation_ids": np.asarray(
            [selected_by_id[value][2] for value in physical_ids], dtype=np.str_
        ),
    }
    selected_iq = np.ascontiguousarray(
        np.stack([selected_by_id[value][3] for value in physical_ids]),
        dtype=np.float32,
    )
    audit = {
        "cache_set_sha256": _require_sha256(expected_sha256, "source cache set"),
        "cache_scope": UPSTREAM_SOURCE_POOL_CACHE_SCOPE,
        "physical_sample_count": EXPECTED_SOURCE_ROWS,
        "physical_sample_observation_count": EXPECTED_SOURCE_ROWS
        * len(FORMAL_LEO_WEAK_SCENARIOS),
        "physical_sample_count_by_scenario": {
            scenario: EXPECTED_SOURCE_ROWS for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "selection_domain_physical_rows": EXPECTED_COUNTS["L_s"],
        "method_visible_physical_rows": EXPECTED_COUNTS["L_s"],
        "method_visible_iq_rows": EXPECTED_COUNTS["L_s"],
        "storage_iq_rows_materialized": sum(
            int(value["iq_storage_rows_materialized"])
            for value in scenario_audits.values()
        ),
        "u_s_tx_ids_or_raw_labels_read_or_materialized": False,
        "source_val_tx_ids_or_raw_labels_read_or_materialized": False,
        "restricted_label_members_read_or_materialized": False,
        "scenario_audits": scenario_audits,
    }
    return metadata, selected_iq, audit


@dataclass(frozen=True, slots=True)
class D106SelectedLSIQ:
    received_iq: np.ndarray
    receiver_ids: np.ndarray
    day_ids: np.ndarray
    physical_ids: np.ndarray
    scenario_names: np.ndarray
    observation_ids: np.ndarray
    receipt: Mapping[str, Any]


def _validate_ls_iq_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    if tuple(arrays) != LS_IQ_MEMBERS:
        raise D106Phase1TapError("D106 selected L_s IQ exact member closure drift")
    iq = np.asarray(arrays["received_iq"])
    if (
        iq.dtype != np.float32
        or iq.shape != (EXPECTED_COUNTS["L_s"], 2, 256)
        or not np.isfinite(iq).all()
    ):
        raise D106Phase1TapError("D106 selected L_s IQ array contract drift")
    for name in LS_IQ_MEMBERS[1:]:
        value = np.asarray(arrays[name])
        if value.dtype.kind not in {"U", "S"} or value.shape != (
            EXPECTED_COUNTS["L_s"],
        ):
            raise D106Phase1TapError(f"D106 selected L_s {name} contract drift")
        if any(not row for row in value.astype(str).tolist()):
            raise D106Phase1TapError(f"D106 selected L_s {name} contains blanks")
    if (
        len(set(np.asarray(arrays["physical_ids"]).astype(str).tolist()))
        != EXPECTED_COUNTS["L_s"]
        or len(set(np.asarray(arrays["observation_ids"]).astype(str).tolist()))
        != EXPECTED_COUNTS["L_s"]
        or not set(np.asarray(arrays["scenario_names"]).astype(str).tolist()).issubset(
            set(FORMAL_LEO_WEAK_SCENARIOS)
        )
    ):
        raise D106Phase1TapError("D106 selected L_s identity/scenario closure drift")


def extract_d106_ls_received_iq(
    *,
    source_split_manifest: str | Path,
    source_split_manifest_sha256: str,
    disjoint_receipt: str | Path,
    disjoint_receipt_sha256: str,
    upstream_source_pool_cache_set: str | Path,
    selection_salt_receipt: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """One-time 8400x3 validator/extractor publishing only sealed 588 L_s rows."""

    execution_pre = _assert_execution_closure("extract")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise D106Phase1TapError("L_s IQ output parent must be an existing directory")
    binding = load_d106_source_split_binding(
        source_split_manifest, source_split_manifest_sha256
    )
    load_d106_train_held_disjoint_receipt(
        disjoint_receipt, disjoint_receipt_sha256, binding=binding
    )
    salt_bytes, _salt_sha = _read_regular_bytes(
        selection_salt_receipt,
        expected_sha256=binding.selection_salt_receipt_sha256,
        name="D106 selection salt receipt",
    )
    salt_snapshot = output.parent / f".d106-salt-{uuid.uuid4().hex}.json"
    _write_new(salt_snapshot, salt_bytes)
    try:
        salt = load_d105_tap_cache_selection_salt(
            salt_snapshot,
            binding.selection_salt_receipt_sha256,
            checkpoint_sha256=binding.checkpoint_sha256,
        )
    finally:
        salt_snapshot.unlink(missing_ok=True)
    ls_physical_ids = _load_ids_only(
        binding.ls_archive,
        LS_INPUT_MEMBERS,
        EXPECTED_COUNTS["L_s"],
        expected_sha256=binding.archive_sha256_by_role["L_s"],
    )
    metadata, selected_iq, audit = select_d106_ls_cache_observations(
        upstream_source_pool_cache_set,
        expected_sha256=binding.source_pool_cache_set_sha256,
        ls_physical_ids=ls_physical_ids.tolist(),
        selection_salt_sha256=salt["selection_salt_sha256"],
    )
    arrays = {"received_iq": selected_iq, **metadata}
    _validate_ls_iq_arrays(arrays)
    if (
        audit.get("physical_sample_count") != EXPECTED_SOURCE_ROWS
        or audit.get("selection_domain_physical_rows") != EXPECTED_COUNTS["L_s"]
        or audit.get("method_visible_iq_rows") != EXPECTED_COUNTS["L_s"]
        or audit.get("storage_iq_rows_materialized")
        != EXPECTED_SOURCE_ROWS * len(FORMAL_LEO_WEAK_SCENARIOS)
        or audit.get("restricted_label_members_read_or_materialized") is not False
    ):
        raise D106Phase1TapError("D106 extraction storage audit closure drift")
    execution_post = _assert_execution_closure("extract")
    if execution_post != execution_pre:
        raise D106Phase1TapError("D106 extraction execution changed in flight")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise D106Phase1TapError("D106 extraction staging path collision")
    staging.mkdir()
    try:
        archive = staging / LS_IQ_ARCHIVE_NAME
        _write_new(archive, _deterministic_npz_bytes(arrays))
        archive_sha = sha256_file(archive)
        array_hashes = {
            name: _array_sha256(value) for name, value in arrays.items()
        }
        physical_root = _ordered_id_root(arrays["physical_ids"].astype(str).tolist())
        selected_content = {
            "array_sha256": array_hashes,
            "row_count": EXPECTED_COUNTS["L_s"],
            "physical_id_root_sha256": physical_root,
            "selection_salt_sha256": salt["selection_salt_sha256"],
            "input_ls_archive_sha256": binding.archive_sha256_by_role["L_s"],
        }
        selected_content_root = hashlib.sha256(
            _canonical_bytes(selected_content)
        ).hexdigest()
        receipt = {
            "schema": LS_IQ_RECEIPT_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "protocol_schema": PROTOCOL_SCHEMA,
            "tap_input_schema": LS_IQ_SCHEMA,
            "archive_name": LS_IQ_ARCHIVE_NAME,
            "archive_sha256": archive_sha,
            "archive_members": list(LS_IQ_MEMBERS),
            "array_sha256": array_hashes,
            "selected_content_root_sha256": selected_content_root,
            "row_count": EXPECTED_COUNTS["L_s"],
            "rho_label": RHO_LABEL,
            "physical_id_root_sha256": physical_root,
            "selection_salt_sha256": salt["selection_salt_sha256"],
            "input_ls_archive_sha256": binding.archive_sha256_by_role["L_s"],
            "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
            "execution_closure": execution_pre,
            "execution_pre_root_sha256": execution_pre[
                "execution_content_root_sha256"
            ],
            "execution_post_root_sha256": execution_post[
                "execution_content_root_sha256"
            ],
            "contains_only_selected_ls_rows": True,
            "source_pool_labels_persisted": False,
            "clean_iq_access": False,
            "target_access": False,
            "formal_query_access": False,
        }
        receipt_path = staging / LS_IQ_RECEIPT_NAME
        _write_new(receipt_path, _canonical_bytes(receipt))
        receipt_sha = sha256_file(receipt_path)
        scenario_validation = {
            scenario: {
                "cache_sha256": audit["scenario_audits"][scenario]["cache_sha256"],
                "manifest_sha256": audit["scenario_audits"][scenario][
                    "manifest_sha256"
                ],
                "physical_sample_ids_sha256": audit["scenario_audits"][scenario][
                    "physical_sample_ids_sha256"
                ],
                "storage_semantic_root_sha256": audit["scenario_audits"][scenario][
                    "storage_semantic_root_sha256"
                ],
                "row_count": audit["scenario_audits"][scenario]["row_count"],
                "full_storage_semantics_verified": audit["scenario_audits"][scenario][
                    "full_storage_semantics_verified"
                ],
                "full_iq_digest_rows_verified": audit["scenario_audits"][scenario][
                    "full_iq_digest_rows_verified"
                ],
                "full_overlay_rows_verified": audit["scenario_audits"][scenario][
                    "full_overlay_rows_verified"
                ],
            }
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        }
        storage_validation_root = hashlib.sha256(
            _canonical_bytes(scenario_validation)
        ).hexdigest()
        validator = {
            "schema": LS_IQ_VALIDATOR_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "protocol_schema": PROTOCOL_SCHEMA,
            "rho_label": RHO_LABEL,
            "source_split_manifest_sha256": binding.manifest_sha256,
            "disjoint_receipt_sha256": _require_sha256(
                disjoint_receipt_sha256, "disjoint receipt"
            ),
            "upstream_source_pool_cache_set_sha256": (
                binding.source_pool_cache_set_sha256
            ),
            "upstream_source_pool_cache_scope": UPSTREAM_SOURCE_POOL_CACHE_SCOPE,
            "d104_legacy_source_pool_hash_field": (
                D104_LEGACY_SOURCE_POOL_HASH_FIELD
            ),
            "selection_salt_receipt_sha256": binding.selection_salt_receipt_sha256,
            "selected_archive_sha256": archive_sha,
            "selected_receipt_sha256": receipt_sha,
            "selected_content_root_sha256": selected_content_root,
            "storage_iq_rows_read": audit["storage_iq_rows_materialized"],
            "storage_physical_rows_validated": EXPECTED_SOURCE_ROWS,
            "selected_iq_rows_persisted": EXPECTED_COUNTS["L_s"],
            "source_cache_label_members_read": False,
            "scenario_validation": scenario_validation,
            "storage_validation_root_sha256": storage_validation_root,
            "all_8400x3_storage_semantics_verified": True,
            "validator_only_not_method_input": True,
        }
        validator_path = staging / LS_IQ_VALIDATOR_NAME
        _write_new(validator_path, _canonical_bytes(validator))
        _write_completion_marker(
            staging,
            artifact_kind="d106_ls_received_iq",
            members=(
                LS_IQ_ARCHIVE_NAME,
                LS_IQ_RECEIPT_NAME,
                LS_IQ_VALIDATOR_NAME,
            ),
        )
        _publish_new_directory(
            staging,
            output,
            members=(
                LS_IQ_ARCHIVE_NAME,
                LS_IQ_RECEIPT_NAME,
                LS_IQ_VALIDATOR_NAME,
                COMPLETION_MARKER_NAME,
            ),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "D106_LS_RECEIVED_IQ_EXTRACTED",
        "output_dir": str(output.resolve()),
        "archive": str((output / LS_IQ_ARCHIVE_NAME).resolve()),
        "archive_sha256": archive_sha,
        "receipt": str((output / LS_IQ_RECEIPT_NAME).resolve()),
        "receipt_sha256": receipt_sha,
        "validator_receipt": str((output / LS_IQ_VALIDATOR_NAME).resolve()),
        "validator_receipt_sha256": sha256_file(output / LS_IQ_VALIDATOR_NAME),
        "row_count": EXPECTED_COUNTS["L_s"],
    }


def load_d106_ls_received_iq(
    archive_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_archive_sha256: str,
    expected_receipt_sha256: str,
) -> D106SelectedLSIQ:
    """Load only the sealed 588-row method input from same-handle snapshots."""

    archive_source = Path(archive_path)
    receipt_source = Path(receipt_path)
    if (
        archive_source.name != LS_IQ_ARCHIVE_NAME
        or receipt_source.name != LS_IQ_RECEIPT_NAME
        or archive_source.parent.resolve() != receipt_source.parent.resolve()
    ):
        raise D106Phase1TapError("D106 selected L_s completed path closure drift")
    _load_completion_marker(
        archive_source.parent,
        artifact_kind="d106_ls_received_iq",
        members=(LS_IQ_ARCHIVE_NAME, LS_IQ_RECEIPT_NAME, LS_IQ_VALIDATOR_NAME),
    )
    archive_bytes, archive_sha = _read_regular_bytes(
        archive_source,
        expected_sha256=expected_archive_sha256,
        name="D106 selected L_s IQ archive",
    )
    receipt_bytes, _receipt_sha = _read_regular_bytes(
        receipt_source,
        expected_sha256=expected_receipt_sha256,
        name="D106 selected L_s IQ receipt",
    )
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Phase1TapError("D106 selected L_s IQ receipt JSON drift") from error
    if receipt_bytes != _canonical_bytes(receipt):
        raise D106Phase1TapError("D106 selected L_s IQ receipt is not canonical")
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as payload:
            if tuple(payload.files) != LS_IQ_MEMBERS:
                raise D106Phase1TapError("D106 selected L_s IQ member drift")
            arrays = {name: np.array(payload[name], copy=True) for name in LS_IQ_MEMBERS}
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise D106Phase1TapError("D106 selected L_s IQ archive load failed") from error
    _validate_ls_iq_arrays(arrays)
    current_execution = _assert_execution_closure("extract")
    expected_fields = {
        "schema", "candidate_id", "split_id", "protocol_schema",
        "tap_input_schema", "archive_name", "archive_sha256",
        "archive_members", "array_sha256", "row_count", "rho_label",
        "selected_content_root_sha256",
        "physical_id_root_sha256", "selection_salt_sha256",
        "input_ls_archive_sha256", "scenario_order", "execution_closure",
        "execution_pre_root_sha256", "execution_post_root_sha256",
        "contains_only_selected_ls_rows", "source_pool_labels_persisted",
        "clean_iq_access", "target_access", "formal_query_access",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != expected_fields
        or receipt.get("schema") != LS_IQ_RECEIPT_SCHEMA
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("split_id") != SPLIT_ID
        or receipt.get("protocol_schema") != PROTOCOL_SCHEMA
        or receipt.get("tap_input_schema") != LS_IQ_SCHEMA
        or receipt.get("archive_name") != LS_IQ_ARCHIVE_NAME
        or receipt.get("archive_sha256") != archive_sha
        or receipt.get("archive_members") != list(LS_IQ_MEMBERS)
        or receipt.get("array_sha256")
        != {name: _array_sha256(value) for name, value in arrays.items()}
        or receipt.get("selected_content_root_sha256")
        != hashlib.sha256(
            _canonical_bytes(
                {
                    "array_sha256": {
                        name: _array_sha256(value) for name, value in arrays.items()
                    },
                    "row_count": EXPECTED_COUNTS["L_s"],
                    "physical_id_root_sha256": _ordered_id_root(
                        arrays["physical_ids"].astype(str).tolist()
                    ),
                    "selection_salt_sha256": receipt.get("selection_salt_sha256"),
                    "input_ls_archive_sha256": receipt.get(
                        "input_ls_archive_sha256"
                    ),
                }
            )
        ).hexdigest()
        or receipt.get("row_count") != EXPECTED_COUNTS["L_s"]
        or receipt.get("rho_label") != RHO_LABEL
        or receipt.get("physical_id_root_sha256")
        != _ordered_id_root(arrays["physical_ids"].astype(str).tolist())
        or receipt.get("scenario_order") != list(FORMAL_LEO_WEAK_SCENARIOS)
        or receipt.get("execution_closure") != current_execution
        or receipt.get("execution_pre_root_sha256")
        != current_execution["execution_content_root_sha256"]
        or receipt.get("execution_post_root_sha256")
        != current_execution["execution_content_root_sha256"]
        or receipt.get("contains_only_selected_ls_rows") is not True
        or receipt.get("source_pool_labels_persisted") is not False
        or receipt.get("clean_iq_access") is not False
        or receipt.get("target_access") is not False
        or receipt.get("formal_query_access") is not False
    ):
        raise D106Phase1TapError("D106 selected L_s IQ receipt closure drift")
    _require_sha256(receipt.get("selection_salt_sha256"), "selection salt")
    _require_sha256(receipt.get("input_ls_archive_sha256"), "L_s archive")
    _require_sha256(receipt.get("selected_content_root_sha256"), "selected content")
    normalized = {
        "received_iq": np.ascontiguousarray(arrays["received_iq"], dtype=np.float32),
        **{
            name: arrays[name].astype(str)
            for name in LS_IQ_MEMBERS[1:]
        },
    }
    for value in normalized.values():
        value.setflags(write=False)
    return D106SelectedLSIQ(
        **normalized, receipt=MappingProxyType(dict(receipt))
    )


def load_d106_ls_storage_validator(
    path: str | Path,
    *,
    expected_sha256: str,
    selected_archive_sha256: str,
    selected_receipt_sha256: str,
    selected_content_root_sha256: str,
) -> Mapping[str, Any]:
    """Load the small read-only 8400x3 validation receipt, never the caches."""

    payload, _observed = _read_regular_bytes(
        path,
        expected_sha256=expected_sha256,
        name="D106 L_s storage validator receipt",
    )
    try:
        validator = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Phase1TapError("D106 storage validator JSON drift") from error
    if type(validator) is not dict or payload != _canonical_bytes(validator):
        raise D106Phase1TapError("D106 storage validator canonical JSON drift")
    expected_fields = {
        "schema", "candidate_id", "split_id", "protocol_schema", "rho_label",
        "source_split_manifest_sha256", "disjoint_receipt_sha256",
        "upstream_source_pool_cache_set_sha256",
        "upstream_source_pool_cache_scope",
        "d104_legacy_source_pool_hash_field",
        "selection_salt_receipt_sha256",
        "selected_archive_sha256", "selected_receipt_sha256",
        "selected_content_root_sha256", "storage_iq_rows_read",
        "storage_physical_rows_validated", "selected_iq_rows_persisted",
        "source_cache_label_members_read", "scenario_validation",
        "storage_validation_root_sha256",
        "all_8400x3_storage_semantics_verified",
        "validator_only_not_method_input",
    }
    scenario_fields = {
        "cache_sha256", "manifest_sha256", "physical_sample_ids_sha256",
        "storage_semantic_root_sha256", "row_count",
        "full_storage_semantics_verified", "full_iq_digest_rows_verified",
        "full_overlay_rows_verified",
    }
    scenarios = validator.get("scenario_validation")
    if (
        set(validator) != expected_fields
        or validator.get("schema") != LS_IQ_VALIDATOR_SCHEMA
        or validator.get("candidate_id") != CANDIDATE_ID
        or validator.get("split_id") != SPLIT_ID
        or validator.get("protocol_schema") != PROTOCOL_SCHEMA
        or validator.get("rho_label") != RHO_LABEL
        or validator.get("selected_archive_sha256")
        != _require_sha256(selected_archive_sha256, "selected archive")
        or validator.get("selected_receipt_sha256")
        != _require_sha256(selected_receipt_sha256, "selected receipt")
        or validator.get("selected_content_root_sha256")
        != _require_sha256(selected_content_root_sha256, "selected content")
        or validator.get("storage_iq_rows_read")
        != EXPECTED_SOURCE_ROWS * len(FORMAL_LEO_WEAK_SCENARIOS)
        or validator.get("storage_physical_rows_validated") != EXPECTED_SOURCE_ROWS
        or validator.get("selected_iq_rows_persisted") != EXPECTED_COUNTS["L_s"]
        or validator.get("source_cache_label_members_read") is not False
        or validator.get("upstream_source_pool_cache_scope")
        != UPSTREAM_SOURCE_POOL_CACHE_SCOPE
        or validator.get("d104_legacy_source_pool_hash_field")
        != D104_LEGACY_SOURCE_POOL_HASH_FIELD
        or type(scenarios) is not dict
        or tuple(scenarios) != FORMAL_LEO_WEAK_SCENARIOS
        or validator.get("all_8400x3_storage_semantics_verified") is not True
        or validator.get("validator_only_not_method_input") is not True
    ):
        raise D106Phase1TapError("D106 storage validator semantic closure drift")
    for name in (
        "source_split_manifest_sha256", "disjoint_receipt_sha256",
        "upstream_source_pool_cache_set_sha256",
        "selection_salt_receipt_sha256",
        "selected_archive_sha256", "selected_receipt_sha256",
        "selected_content_root_sha256", "storage_validation_root_sha256",
    ):
        _require_sha256(validator.get(name), name)
    physical_roots: list[str] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = scenarios[scenario]
        if (
            type(row) is not dict
            or set(row) != scenario_fields
            or row.get("row_count") != EXPECTED_SOURCE_ROWS
            or row.get("full_storage_semantics_verified") is not True
            or row.get("full_iq_digest_rows_verified") != EXPECTED_SOURCE_ROWS
            or row.get("full_overlay_rows_verified") != EXPECTED_SOURCE_ROWS
        ):
            raise D106Phase1TapError(
                f"D106 storage validator {scenario} closure drift"
            )
        for name in (
            "cache_sha256", "manifest_sha256", "physical_sample_ids_sha256",
            "storage_semantic_root_sha256",
        ):
            _require_sha256(row.get(name), f"{scenario} {name}")
        physical_roots.append(row["physical_sample_ids_sha256"])
    if (
        len(set(physical_roots)) != 1
        or validator.get("storage_validation_root_sha256")
        != hashlib.sha256(_canonical_bytes(scenarios)).hexdigest()
    ):
        raise D106Phase1TapError("D106 storage validator cross-scenario drift")
    return MappingProxyType(dict(validator))


def _load_ls_join_metadata(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, np.ndarray]:
    try:
        archive_bytes, _observed = _read_regular_bytes(
            path,
            expected_sha256=expected_sha256,
            name="D106 L_s metadata archive",
        )
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as payload:
            if tuple(payload.files) != LS_INPUT_MEMBERS:
                raise D106Phase1TapError("L_s input member closure drift")
            arrays = {
                name: np.asarray(payload[name]).astype(str)
                for name in ("receiver_ids", "day_ids", "tx_labels", "physical_ids")
            }
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise D106Phase1TapError("L_s join metadata read failed") from error
    for name, value in arrays.items():
        if value.shape != (EXPECTED_COUNTS["L_s"],) or any(not row for row in value.tolist()):
            raise D106Phase1TapError(f"L_s {name} contract drift")
    cells: dict[tuple[str, str, str], int] = {}
    receiver_tx: dict[tuple[str, str], int] = {}
    for tx, receiver, day in zip(
        arrays["tx_labels"].tolist(),
        arrays["receiver_ids"].tolist(),
        arrays["day_ids"].tolist(),
        strict=True,
    ):
        cells[(tx, receiver, day)] = cells.get((tx, receiver, day), 0) + 1
        receiver_tx[(receiver, tx)] = receiver_tx.get((receiver, tx), 0) + 1
    if (
        len(set(arrays["physical_ids"].tolist())) != EXPECTED_COUNTS["L_s"]
        or len(set(arrays["tx_labels"].tolist())) != 6
        or len(set(arrays["receiver_ids"].tolist())) != 7
        or len(set(arrays["day_ids"].tolist())) != 4
        or len(cells) != 168
        or set(cells.values()).difference({2, 3, 4})
        or len(receiver_tx) != 42
        or set(receiver_tx.values()) != {14}
    ):
        raise D106Phase1TapError("L_s frozen 588/6x7x4/cell2-4/group14 closure drift")
    return arrays


@dataclass(frozen=True, slots=True)
class D106JoinedLSRows:
    received_iq: np.ndarray
    tx_labels: np.ndarray
    receiver_ids: np.ndarray
    day_ids: np.ndarray
    physical_ids: np.ndarray
    scenario_names: np.ndarray
    observation_ids: np.ndarray

    def __post_init__(self) -> None:
        iq = np.asarray(self.received_iq)
        count = EXPECTED_COUNTS["L_s"]
        if (
            iq.dtype != np.float32
            or iq.shape != (count, 2, 256)
            or not iq.flags.c_contiguous
            or not np.isfinite(iq).all()
        ):
            raise D106Phase1TapError(
                "joined L_s IQ must be finite C-order float32 [588,2,256]"
            )
        object.__setattr__(self, "received_iq", iq.copy(order="C"))
        for name in (
            "tx_labels", "receiver_ids", "day_ids", "physical_ids",
            "scenario_names", "observation_ids",
        ):
            array = np.asarray(getattr(self, name)).astype(str)
            if array.shape != (count,) or any(not value for value in array.tolist()):
                raise D106Phase1TapError(f"joined L_s {name} contract drift")
            copied = np.array(array, dtype=np.str_, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if (
            len(set(self.physical_ids.tolist())) != count
            or len(set(self.observation_ids.tolist())) != count
            or not set(self.scenario_names.tolist()).issubset(set(FORMAL_LEO_WEAK_SCENARIOS))
        ):
            raise D106Phase1TapError("joined L_s identity/scenario closure drift")
        self.received_iq.setflags(write=False)


def join_d106_ls_observations(
    selected_metadata: Mapping[str, np.ndarray],
    selected_iq: np.ndarray,
    *,
    ls_archive: str | Path,
    expected_ls_archive_sha256: str | None = None,
) -> D106JoinedLSRows:
    """Perform the exact D104 ``L_s`` inner join before any forward."""

    required = {
        "receiver_ids", "day_ids", "physical_ids", "scenario_names",
        "observation_ids",
    }
    if set(selected_metadata) != required:
        raise D106Phase1TapError("selected source metadata closure drift")
    selected = {name: np.asarray(selected_metadata[name]).astype(str) for name in required}
    if any(value.shape != (EXPECTED_COUNTS["L_s"],) for value in selected.values()):
        raise D106Phase1TapError("selected source metadata must cover exactly 588 L_s rows")
    iq = np.asarray(selected_iq)
    if (
        iq.dtype != np.float32
        or iq.shape != (EXPECTED_COUNTS["L_s"], 2, 256)
        or not np.isfinite(iq).all()
    ):
        raise D106Phase1TapError(
            "selected source IQ must be finite float32 [588,2,256]"
        )
    selected_ids = selected["physical_ids"].tolist()
    if len(set(selected_ids)) != EXPECTED_COUNTS["L_s"]:
        raise D106Phase1TapError("selected L_s physical IDs are not an exact registry")
    ls = _load_ls_join_metadata(
        Path(ls_archive), expected_sha256=expected_ls_archive_sha256
    )
    for selected_name, ls_name in (
        ("receiver_ids", "receiver_ids"),
        ("day_ids", "day_ids"),
        ("physical_ids", "physical_ids"),
    ):
        if selected[selected_name].tolist() != ls[ls_name].tolist():
            raise D106Phase1TapError(f"L_s exact inner join {selected_name} drift")
    return D106JoinedLSRows(
        received_iq=np.ascontiguousarray(iq, dtype=np.float32),
        tx_labels=ls["tx_labels"],
        receiver_ids=ls["receiver_ids"],
        day_ids=ls["day_ids"],
        physical_ids=ls["physical_ids"],
        scenario_names=selected["scenario_names"],
        observation_ids=selected["observation_ids"],
    )


def _forward_fixed256(
    model: Any, received_iq: np.ndarray, *, device: Any
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    received = np.asarray(received_iq)
    if (
        received.dtype != np.float32
        or received.shape != (EXPECTED_COUNTS["L_s"], 2, 256)
        or not np.isfinite(received).all()
    ):
        raise D106Phase1TapError(
            "D106 forward input must be finite float32 [588,2,256]"
        )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - deployment dependency.
        raise D106Phase1TapError("D106 strict tap requires PyTorch") from error
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise D106Phase1TapError("D106 strict tap device is invalid") from error
    if torch_device.type == "cuda" and (
        not torch.cuda.is_available()
        or torch_device.index is None
        or torch_device.index >= torch.cuda.device_count()
    ):
        raise D106Phase1TapError("requested D106 CUDA device is unavailable")
    if bool(getattr(model, "training", True)):
        raise D106Phase1TapError("D106 strict tap model must be in eval mode")
    pre_rows: list[np.ndarray] = []
    dom_rows: list[np.ndarray] = []
    invocations = 0
    last_real = 0
    for start in range(0, len(received), FORWARD_BATCH_CAPACITY):
        batch = np.ascontiguousarray(
            received[start : start + FORWARD_BATCH_CAPACITY], dtype=np.float32
        )
        last_real = len(batch)
        padded = np.zeros((FORWARD_BATCH_CAPACITY, *batch.shape[1:]), dtype=np.float32)
        padded[:last_real] = batch
        tensor = _tensor_from_d105_float32_c_iq(
            padded,
            torch_module=torch,
            device=torch_device,
            error_type=D106Phase1TapError,
            name="D106 strict tap batch",
        )
        tap = extract_d105_feature_tap(model, tensor)
        pre_rows.append(np.ascontiguousarray(tap.pre_relu[:last_real], dtype=np.float32))
        dom_rows.append(np.ascontiguousarray(tap.z_dom[:last_real], dtype=np.float32))
        invocations += 1
    return (
        np.ascontiguousarray(np.concatenate(pre_rows), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(dom_rows), dtype=np.float32),
        {
            "forward_batch_capacity": FORWARD_BATCH_CAPACITY,
            "forward_invocation_count": invocations,
            "last_batch_real_rows": last_real,
            "last_batch_padding_rows": FORWARD_BATCH_CAPACITY - last_real,
            "same_iq_dual_forward": True,
            "fixed256_zero_pad_then_slice": True,
        },
    )


@dataclass(frozen=True, slots=True)
class D106Phase1TapRows:
    pre_relu: np.ndarray
    z_dom: np.ndarray
    tx_labels: np.ndarray
    receiver_ids: np.ndarray
    day_ids: np.ndarray
    physical_ids: np.ndarray
    scenario_names: np.ndarray
    observation_ids: np.ndarray
    z_id: np.ndarray
    receipt: Mapping[str, Any]


def _validate_checkpoint_loader_receipt(
    value: Any, *, checkpoint_sha256: str
) -> None:
    expected_keys = {
        "policy", "torch_version", "safe_globals_available", "weights_only",
        "exact_frozen_checkpoint_sha256_required",
        "caller_selected_checkpoint_allowed",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("policy") not in {
            "weights_only_with_explicit_safe_globals",
            "legacy_pickle_exact_frozen_sha_only",
        }
        or not isinstance(value.get("torch_version"), str)
        or not value.get("torch_version")
        or type(value.get("safe_globals_available")) is not bool
        or type(value.get("weights_only")) is not bool
        or value.get("exact_frozen_checkpoint_sha256_required")
        != checkpoint_sha256
        or value.get("caller_selected_checkpoint_allowed") is not False
        or (
            value.get("policy") == "weights_only_with_explicit_safe_globals"
            and (
                value.get("safe_globals_available") is not True
                or value.get("weights_only") is not True
            )
        )
        or (
            value.get("policy") == "legacy_pickle_exact_frozen_sha_only"
            and value.get("weights_only") is not False
        )
    ):
        raise D106Phase1TapError("D106 checkpoint-loader receipt closure drift")


def _validate_model_reconstruction_receipt(value: Any) -> None:
    expected_keys = {
        "loader", "model_factory", "backbone_factory",
        "checkpoint_load_strict", "missing_keys", "unexpected_keys",
        "skipped_mismatch", "state_tensor_count", "num_domains_from_state",
        "input_len", "eval_mode",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("loader")
        != "d105_minimal_cvsincnet_checkpoint_reconstruction_v1"
        or value.get("model_factory")
        != "model_dual_cvsincnet.build_dual_model"
        or value.get("backbone_factory") != "model.build_model"
        or value.get("checkpoint_load_strict") is not True
        or value.get("missing_keys") != 0
        or value.get("unexpected_keys") != 0
        or value.get("skipped_mismatch") != 0
        or type(value.get("state_tensor_count")) is not int
        or value.get("state_tensor_count", 0) <= 0
        or type(value.get("num_domains_from_state")) is not int
        or value.get("num_domains_from_state", 0) <= 0
        or value.get("input_len") != 256
        or value.get("eval_mode") is not True
    ):
        raise D106Phase1TapError("D106 model-reconstruction receipt closure drift")


def _validate_forward_receipt(value: Any) -> None:
    expected = {
        "forward_batch_capacity": 256,
        "forward_invocation_count": 3,
        "last_batch_real_rows": 76,
        "last_batch_padding_rows": 180,
        "same_iq_dual_forward": True,
        "fixed256_zero_pad_then_slice": True,
    }
    if type(value) is not dict or value != expected:
        raise D106Phase1TapError("D106 forward receipt closure drift")


def export_d106_phase1_ls_tap(
    *,
    selected_iq_archive: str | Path,
    selected_iq_archive_sha256: str,
    selected_iq_receipt: str | Path,
    selected_iq_receipt_sha256: str,
    storage_validator_receipt: str | Path,
    storage_validator_receipt_sha256: str,
    ls_archive: str | Path,
    ls_archive_sha256: str,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    runtime_manifest: str | Path,
    runtime_sha256: str,
    output_dir: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build features from only the sealed 588-row IQ artifact and L_s labels."""

    execution_pre = _assert_execution_closure("export")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise D106Phase1TapError("tap output parent must be an existing directory")
    checkpoint_hash = _require_sha256(checkpoint_sha256, "checkpoint")
    runtime_hash = _require_sha256(runtime_sha256, "runtime")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise D106Phase1TapError("D106 frozen checkpoint authority drift")
    _runtime_bytes, observed_runtime = _read_regular_bytes(
        runtime_manifest,
        expected_sha256=runtime_hash,
        name="D106 runtime manifest",
    )
    if observed_runtime != runtime_hash:
        raise D106Phase1TapError("D106 runtime manifest binding drift")
    selected = load_d106_ls_received_iq(
        selected_iq_archive,
        selected_iq_receipt,
        expected_archive_sha256=selected_iq_archive_sha256,
        expected_receipt_sha256=selected_iq_receipt_sha256,
    )
    validator_source = Path(storage_validator_receipt)
    if (
        validator_source.name != LS_IQ_VALIDATOR_NAME
        or validator_source.parent.resolve()
        != Path(selected_iq_archive).parent.resolve()
    ):
        raise D106Phase1TapError("D106 storage validator completed path drift")
    storage_validator = load_d106_ls_storage_validator(
        validator_source,
        expected_sha256=storage_validator_receipt_sha256,
        selected_archive_sha256=selected_iq_archive_sha256,
        selected_receipt_sha256=selected_iq_receipt_sha256,
        selected_content_root_sha256=selected.receipt[
            "selected_content_root_sha256"
        ],
    )
    ls_hash = _require_sha256(ls_archive_sha256, "L_s archive")
    if selected.receipt.get("input_ls_archive_sha256") != ls_hash:
        raise D106Phase1TapError("D106 selected IQ/L_s archive binding drift")
    selected_metadata = {
        name: getattr(selected, name) for name in LS_IQ_MEMBERS[1:]
    }
    joined = join_d106_ls_observations(
        selected_metadata,
        selected.received_iq,
        ls_archive=ls_archive,
        expected_ls_archive_sha256=ls_hash,
    )
    checkpoint_bytes, _observed_checkpoint = _read_regular_bytes(
        checkpoint, expected_sha256=checkpoint_hash, name="D106 checkpoint"
    )
    checkpoint_snapshot = output.parent / f".d106-checkpoint-{uuid.uuid4().hex}.pth"
    _write_new(checkpoint_snapshot, checkpoint_bytes)
    try:
        checkpoint_payload, checkpoint_loader = load_d105_exact_sha_bound_checkpoint(
            checkpoint_snapshot, checkpoint_hash
        )
        model, model_receipt = build_d105_exact_model_from_checkpoint(
            checkpoint_payload, input_len=256, device=device
        )
        pre_relu, z_dom, forward_receipt = _forward_fixed256(
            model, joined.received_iq, device=device
        )
    finally:
        checkpoint_snapshot.unlink(missing_ok=True)
    _validate_checkpoint_loader_receipt(
        checkpoint_loader, checkpoint_sha256=checkpoint_hash
    )
    _validate_model_reconstruction_receipt(model_receipt)
    _validate_forward_receipt(forward_receipt)
    expected_feature_shape = (EXPECTED_COUNTS["L_s"], Z_DIM)
    if (
        pre_relu.shape != expected_feature_shape
        or z_dom.shape != expected_feature_shape
        or pre_relu.dtype != np.float32
        or z_dom.dtype != np.float32
        or not np.isfinite(pre_relu).all()
        or not np.isfinite(z_dom).all()
    ):
        raise D106Phase1TapError("D106 strict tap feature contract drift")
    arrays = {
        "pre_relu": pre_relu,
        "z_dom": z_dom,
        "tx_labels": joined.tx_labels,
        "receiver_ids": joined.receiver_ids,
        "day_ids": joined.day_ids,
        "physical_ids": joined.physical_ids,
        "scenario_names": joined.scenario_names,
        "observation_ids": joined.observation_ids,
    }
    if tuple(arrays) != TAP_MEMBERS:
        raise D106Phase1TapError("D106 tap member order drift")
    execution_post = _assert_execution_closure("export")
    if execution_post != execution_pre:
        raise D106Phase1TapError("D106 export execution changed in flight")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise D106Phase1TapError("D106 tap staging path collision")
    staging.mkdir()
    try:
        archive = staging / TAP_ARCHIVE_NAME
        _write_new(archive, _deterministic_npz_bytes(arrays))
        archive_sha = sha256_file(archive)
        receipt = {
            "schema": TAP_RECEIPT_SCHEMA,
            "candidate_id": CANDIDATE_ID,
            "split_id": SPLIT_ID,
            "protocol_schema": PROTOCOL_SCHEMA,
            "selected_iq_archive_sha256": _require_sha256(
                selected_iq_archive_sha256, "selected IQ archive"
            ),
            "selected_iq_receipt_sha256": _require_sha256(
                selected_iq_receipt_sha256, "selected IQ receipt"
            ),
            "storage_validator_receipt_sha256": _require_sha256(
                storage_validator_receipt_sha256, "storage validator receipt"
            ),
            "storage_validation_binding": {
                "schema": storage_validator["schema"],
                "storage_validation_root_sha256": storage_validator[
                    "storage_validation_root_sha256"
                ],
                "selected_content_root_sha256": storage_validator[
                    "selected_content_root_sha256"
                ],
                "all_8400x3_storage_semantics_verified": storage_validator[
                    "all_8400x3_storage_semantics_verified"
                ],
            },
            "extraction_binding": {
                "schema": selected.receipt["schema"],
                "row_count": selected.receipt["row_count"],
                "selection_salt_sha256": selected.receipt["selection_salt_sha256"],
                "selected_content_root_sha256": selected.receipt[
                    "selected_content_root_sha256"
                ],
                "input_ls_archive_sha256": selected.receipt[
                    "input_ls_archive_sha256"
                ],
                "execution_root_sha256": selected.receipt[
                    "execution_pre_root_sha256"
                ],
            },
            "input_ls_archive_sha256": ls_hash,
            "checkpoint_sha256": checkpoint_hash,
            "runtime_sha256": runtime_hash,
            "execution_closure": execution_pre,
            "execution_pre_root_sha256": execution_pre[
                "execution_content_root_sha256"
            ],
            "execution_post_root_sha256": execution_post[
                "execution_content_root_sha256"
            ],
            "tap_archive_name": TAP_ARCHIVE_NAME,
            "tap_archive_sha256": archive_sha,
            "tap_archive_members": list(TAP_MEMBERS),
            "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
            "row_count": EXPECTED_COUNTS["L_s"],
            "source_split_counts": dict(EXPECTED_COUNTS),
            "rho_label": RHO_LABEL,
            "physical_id_root_sha256": _ordered_id_root(joined.physical_ids.tolist()),
            "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
            "method_visible_received_iq_rows": EXPECTED_COUNTS["L_s"],
            "method_visible_tx_label_rows": EXPECTED_COUNTS["L_s"],
            "checkpoint_loader": checkpoint_loader,
            "model_reconstruction": model_receipt,
            "forward": forward_receipt,
            "exact_inner_join": True,
            "same_received_iq_for_zid_zdom": True,
            "z_id_storage_policy": "derive_relu_pre_relu",
            "feature_stage_source_pool_access": False,
            "received_iq_persisted": False,
            "raw_iq_persisted": False,
            "clean_iq_access": False,
            "target_access": False,
            "formal_query_access": False,
        }
        _write_new(staging / TAP_RECEIPT_NAME, _canonical_bytes(receipt))
        _write_completion_marker(
            staging,
            artifact_kind="d106_ls_strict_tap",
            members=(TAP_ARCHIVE_NAME, TAP_RECEIPT_NAME),
        )
        _publish_new_directory(
            staging,
            output,
            members=(TAP_ARCHIVE_NAME, TAP_RECEIPT_NAME, COMPLETION_MARKER_NAME),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "D106_LS_STRICT_TAP_COMPLETE",
        "output_dir": str(output.resolve()),
        "archive": str((output / TAP_ARCHIVE_NAME).resolve()),
        "archive_sha256": archive_sha,
        "receipt": str((output / TAP_RECEIPT_NAME).resolve()),
        "receipt_sha256": sha256_file(output / TAP_RECEIPT_NAME),
        "row_count": EXPECTED_COUNTS["L_s"],
        "received_iq_persisted": False,
    }


def load_d106_phase1_ls_tap(
    archive_path: str | Path,
    receipt_path: str | Path,
    *,
    expected_archive_sha256: str,
    expected_receipt_sha256: str,
) -> D106Phase1TapRows:
    """Load the sealed tap and derive ``z_id = ReLU(pre_relu)`` in memory."""

    archive_source = Path(archive_path)
    receipt_source = Path(receipt_path)
    if (
        archive_source.name != TAP_ARCHIVE_NAME
        or receipt_source.name != TAP_RECEIPT_NAME
        or archive_source.parent.resolve() != receipt_source.parent.resolve()
    ):
        raise D106Phase1TapError("D106 tap completed path closure drift")
    _load_completion_marker(
        archive_source.parent,
        artifact_kind="d106_ls_strict_tap",
        members=(TAP_ARCHIVE_NAME, TAP_RECEIPT_NAME),
    )
    archive_bytes, expected_archive = _read_regular_bytes(
        archive_source,
        expected_sha256=expected_archive_sha256,
        name="D106 tap archive",
    )
    receipt_bytes, _receipt_sha = _read_regular_bytes(
        receipt_source,
        expected_sha256=expected_receipt_sha256,
        name="D106 tap receipt",
    )
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D106Phase1TapError("D106 tap receipt JSON drift") from error
    if type(receipt) is not dict:
        raise D106Phase1TapError("D106 tap receipt must be an object")
    if receipt_bytes != _canonical_bytes(receipt):
        raise D106Phase1TapError("D106 tap receipt is not canonical JSON")
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as payload:
            if tuple(payload.files) != TAP_MEMBERS:
                raise D106Phase1TapError("D106 tap exact member closure drift")
            arrays = {name: np.array(payload[name], copy=True) for name in TAP_MEMBERS}
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise D106Phase1TapError("D106 tap archive load failed") from error
    count = EXPECTED_COUNTS["L_s"]
    for name in ("pre_relu", "z_dom"):
        value = arrays[name]
        if (
            value.dtype != np.float32
            or value.shape != (count, Z_DIM)
            or not np.isfinite(value).all()
        ):
            raise D106Phase1TapError(f"D106 tap {name} contract drift")
    for name in TAP_MEMBERS[2:]:
        value = arrays[name]
        if value.dtype.kind not in {"U", "S"} or value.shape != (count,):
            raise D106Phase1TapError(f"D106 tap {name} contract drift")
        arrays[name] = value.astype(str)
        if any(not row for row in arrays[name].tolist()):
            raise D106Phase1TapError(f"D106 tap {name} contains blanks")
    if (
        len(set(arrays["physical_ids"].tolist())) != count
        or len(set(arrays["observation_ids"].tolist())) != count
        or not set(arrays["scenario_names"].tolist()).issubset(
            set(FORMAL_LEO_WEAK_SCENARIOS)
        )
    ):
        raise D106Phase1TapError("D106 tap identity/scenario closure drift")
    expected_receipt_fields = {
        "schema", "candidate_id", "split_id", "protocol_schema",
        "selected_iq_archive_sha256", "selected_iq_receipt_sha256",
        "storage_validator_receipt_sha256", "storage_validation_binding",
        "extraction_binding", "input_ls_archive_sha256", "checkpoint_sha256",
        "runtime_sha256", "execution_closure", "execution_pre_root_sha256",
        "execution_post_root_sha256", "tap_archive_name", "tap_archive_sha256",
        "tap_archive_members", "array_sha256", "row_count", "source_split_counts",
        "rho_label", "physical_id_root_sha256", "scenario_order",
        "method_visible_received_iq_rows",
        "method_visible_tx_label_rows", "checkpoint_loader",
        "model_reconstruction", "forward", "exact_inner_join",
        "same_received_iq_for_zid_zdom", "z_id_storage_policy",
        "feature_stage_source_pool_access",
        "received_iq_persisted", "raw_iq_persisted", "clean_iq_access",
        "target_access", "formal_query_access",
    }
    current_execution = _assert_execution_closure("export")
    current_extraction = _assert_execution_closure("extract")
    extraction = receipt.get("extraction_binding")
    storage_validation = receipt.get("storage_validation_binding")
    extraction_keys = {
        "schema", "row_count", "selection_salt_sha256",
        "selected_content_root_sha256", "input_ls_archive_sha256",
        "execution_root_sha256",
    }
    if (
        set(receipt) != expected_receipt_fields
        or receipt.get("schema") != TAP_RECEIPT_SCHEMA
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("split_id") != SPLIT_ID
        or receipt.get("protocol_schema") != PROTOCOL_SCHEMA
        or type(extraction) is not dict
        or set(extraction) != extraction_keys
        or type(storage_validation) is not dict
        or set(storage_validation) != {
            "schema", "storage_validation_root_sha256",
            "selected_content_root_sha256",
            "all_8400x3_storage_semantics_verified",
        }
        or storage_validation.get("schema") != LS_IQ_VALIDATOR_SCHEMA
        or storage_validation.get("selected_content_root_sha256")
        != extraction.get("selected_content_root_sha256")
        or storage_validation.get("all_8400x3_storage_semantics_verified")
        is not True
        or extraction.get("schema") != LS_IQ_RECEIPT_SCHEMA
        or extraction.get("row_count") != count
        or extraction.get("input_ls_archive_sha256")
        != receipt.get("input_ls_archive_sha256")
        or extraction.get("execution_root_sha256")
        != current_extraction["execution_content_root_sha256"]
        or receipt.get("tap_archive_name") != TAP_ARCHIVE_NAME
        or receipt.get("tap_archive_sha256") != expected_archive
        or receipt.get("tap_archive_members") != list(TAP_MEMBERS)
        or receipt.get("array_sha256")
        != {name: _array_sha256(value) for name, value in arrays.items()}
        or receipt.get("row_count") != count
        or receipt.get("source_split_counts") != EXPECTED_COUNTS
        or receipt.get("rho_label") != RHO_LABEL
        or receipt.get("physical_id_root_sha256")
        != _ordered_id_root(arrays["physical_ids"].tolist())
        or receipt.get("scenario_order") != list(FORMAL_LEO_WEAK_SCENARIOS)
        or receipt.get("method_visible_received_iq_rows") != count
        or receipt.get("method_visible_tx_label_rows") != count
        or receipt.get("execution_closure") != current_execution
        or receipt.get("execution_pre_root_sha256")
        != current_execution["execution_content_root_sha256"]
        or receipt.get("execution_post_root_sha256")
        != current_execution["execution_content_root_sha256"]
        or receipt.get("exact_inner_join") is not True
        or receipt.get("same_received_iq_for_zid_zdom") is not True
        or receipt.get("z_id_storage_policy") != "derive_relu_pre_relu"
        or receipt.get("feature_stage_source_pool_access") is not False
        or receipt.get("received_iq_persisted") is not False
        or receipt.get("raw_iq_persisted") is not False
        or receipt.get("clean_iq_access") is not False
        or receipt.get("target_access") is not False
        or receipt.get("formal_query_access") is not False
    ):
        raise D106Phase1TapError("D106 tap receipt semantic closure drift")
    for name in (
        "selected_iq_archive_sha256", "selected_iq_receipt_sha256",
        "storage_validator_receipt_sha256",
        "checkpoint_sha256", "runtime_sha256", "input_ls_archive_sha256",
    ):
        _require_sha256(receipt.get(name), name)
    _require_sha256(extraction.get("selection_salt_sha256"), "selection salt")
    _require_sha256(extraction.get("selected_content_root_sha256"), "selected content")
    _require_sha256(extraction.get("execution_root_sha256"), "extraction root")
    _require_sha256(
        storage_validation.get("storage_validation_root_sha256"),
        "storage validation root",
    )
    _validate_checkpoint_loader_receipt(
        receipt.get("checkpoint_loader"),
        checkpoint_sha256=receipt["checkpoint_sha256"],
    )
    _validate_model_reconstruction_receipt(receipt.get("model_reconstruction"))
    _validate_forward_receipt(receipt.get("forward"))
    if receipt["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise D106Phase1TapError("D106 tap checkpoint lineage drift")
    z_id = np.maximum(arrays["pre_relu"], np.float32(0.0)).astype(np.float32, copy=False)
    for value in (*arrays.values(), z_id):
        value.setflags(write=False)
    return D106Phase1TapRows(
        **arrays,
        z_id=z_id,
        receipt=MappingProxyType(dict(receipt)),
    )


_FORMAL_EXECUTION_BASELINES = MappingProxyType(
    {
        "extract": _execution_closure("extract"),
        "export": _execution_closure("export"),
    }
)


__all__ = [
    "CANDIDATE_ID",
    "COMPLETION_MARKER_NAME",
    "DISJOINT_RECEIPT_SCHEMA",
    "D106JoinedLSRows",
    "D106Phase1TapError",
    "D106Phase1TapRows",
    "D106SelectedLSIQ",
    "EXPECTED_CHECKPOINT_SHA256",
    "FORWARD_BATCH_CAPACITY",
    "LS_IQ_ARCHIVE_NAME",
    "LS_IQ_MEMBERS",
    "LS_IQ_RECEIPT_NAME",
    "LS_IQ_VALIDATOR_NAME",
    "RHO_LABEL",
    "TAP_ARCHIVE_NAME",
    "TAP_MEMBERS",
    "TAP_RECEIPT_NAME",
    "build_d106_train_held_disjoint_receipt",
    "extract_d106_ls_received_iq",
    "export_d106_phase1_ls_tap",
    "join_d106_ls_observations",
    "load_d106_phase1_ls_tap",
    "load_d106_ls_received_iq",
    "load_d106_ls_storage_validator",
    "load_d106_source_split_binding",
    "load_d106_train_held_disjoint_receipt",
    "select_d106_ls_cache_observations",
]
