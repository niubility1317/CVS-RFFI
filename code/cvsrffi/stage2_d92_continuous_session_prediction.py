"""Truth-free publication for the frozen D92 cumulative-session screen.

The module intentionally owns only the narrow prediction boundary.  It accepts
sealed-package identities and a session builder, opens no truth/scorer surface,
and publishes one immutable five-file receipt set per completed session.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cvsrffi.somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_diag_cosine_exploration import (
    _sha256_file,
    _write_json_new,
    _write_npz_new,
)


METHOD_ID = "D92_E0_CUMULATIVE_REPLAY_SESSION_V1"
SCHEMA = "cvs.phase2.d92_e0_continuous_session.truth_free_prediction.v1"
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_ZERO_QUERY_FIELDS = (
    "query_truth_access",
    "query_fit_access",
    "query_update_access",
    "query_selection_access",
    "query_role_oracle_access",
    "query_class_quota_access",
    "query_global_reassignment",
)
_RESOURCE_FIELDS = (
    "registration_wall_time_ns",
    "registration_incremental_peak_working_set_bytes",
    "support_bytes",
    "state_bytes",
    "query_macs",
    "head_latency_ns",
)
_REGISTRATION_WALL_HARD_MAX_NS = 300_000_000
_REGISTRATION_PEAK_HARD_MAX_BYTES = 4 * 1024 * 1024


class ContinuousSessionPredictionError(ValueError):
    """Raised when the truth-free continuous-session boundary drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_values(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_bytes(list(values))).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuousSessionPredictionError(f"{label} must be a mapping")
    return value


def _field(value: Any, name: str, *, label: str) -> Any:
    if isinstance(value, Mapping):
        try:
            return value[name]
        except KeyError as error:
            raise ContinuousSessionPredictionError(f"{label} missing {name}") from error
    try:
        return getattr(value, name)
    except AttributeError as error:
        raise ContinuousSessionPredictionError(f"{label} missing {name}") from error


def _sha256_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContinuousSessionPredictionError(f"{name} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ContinuousSessionPredictionError(f"{name} must be a SHA256 digest") from error
    if value != value.lower():
        raise ContinuousSessionPredictionError(f"{name} must be lowercase")
    return value


def _observed_seal_sha256(name: str, path: str | Path, expected: str | None) -> str:
    """Use a frozen digest when supplied, otherwise bind the on-disk detached seal."""

    if expected is not None:
        return _sha256_text(name, expected)
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise ContinuousSessionPredictionError(f"{name} is unavailable for observation") from error
    if candidate.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ContinuousSessionPredictionError(f"{name} must be a regular detached seal")
    return _sha256_file(candidate)


def _string_vector(
    value: Any, *, label: str, allow_empty: bool = False
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or (array.size == 0 and not allow_empty):
        qualifier = "a vector" if allow_empty else "a nonempty vector"
        raise ContinuousSessionPredictionError(f"{label} must be {qualifier}")
    if array.size == 0:
        return ()
    if array.dtype.kind not in {"U", "S", "O"}:
        raise ContinuousSessionPredictionError(f"{label} must contain text")
    result = tuple(str(item) for item in array.tolist())
    if any(not item or "\x00" in item for item in result):
        raise ContinuousSessionPredictionError(f"{label} contains an invalid token")
    return result


def _state_fingerprint(state: Any) -> str:
    """Fingerprint the exact deployed affine state without retaining a sidecar."""

    supplied = getattr(state, "persistent_state_sha256", None)
    if isinstance(supplied, str):
        return _sha256_text("persistent state", supplied)

    digest = hashlib.sha256()
    schema = _field(state, "schema", label="state") if hasattr(state, "schema") or (isinstance(state, Mapping) and "schema" in state) else ""
    for text in (
        str(schema),
        "\x1f".join(_string_vector(_field(state, "classes", label="state"), label="state classes")),
        str(int(_field(state, "old_class_count", label="state"))),
        str(_field(state, "covariance_policy", label="state")),
    ):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    for name in (
        "log_diag_fp32",
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
    ):
        array = np.ascontiguousarray(np.asarray(_field(state, name, label="state")))
        if not np.isfinite(array.astype(np.float64, copy=False)).all():
            raise ContinuousSessionPredictionError(f"state {name} is non-finite")
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _normalized_schedules(schedules: Mapping[str, Any]) -> dict[str, dict[str, tuple[int, ...]]]:
    source = _mapping(schedules, label="schedules")
    if not source:
        raise ContinuousSessionPredictionError("schedules must be nonempty")
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for name, raw in source.items():
        if not isinstance(name, str) or not name:
            raise ContinuousSessionPredictionError("schedule name is invalid")
        item = _mapping(raw, label=f"schedule {name}")
        if set(item) != {"increments", "arrival_order"}:
            raise ContinuousSessionPredictionError(f"schedule {name} exact schema drift")
        increments = tuple(int(value) for value in item["increments"])
        arrival_order = tuple(int(value) for value in item["arrival_order"])
        if (
            not increments
            or any(value < 1 for value in increments)
            or sum(increments) != len(arrival_order)
            or len(set(arrival_order)) != len(arrival_order)
            or set(arrival_order) != set(range(len(arrival_order)))
        ):
            raise ContinuousSessionPredictionError(f"schedule {name} increment/order drift")
        result[name] = {"increments": increments, "arrival_order": arrival_order}
    return result


def _packages(
    *,
    before_enrollment_package_root: str | Path,
    before_enrollment_seal_path: str | Path,
    before_enrollment_seal_sha256: str | None,
    before_apply_package_root: str | Path,
    before_apply_seal_path: str | Path,
    before_apply_seal_sha256: str | None,
    after_apply_package_root: str | Path,
    after_apply_seal_path: str | Path,
    after_apply_seal_sha256: str | None,
) -> dict[str, dict[str, str]]:
    values = {
        "before_enrollment": (
            before_enrollment_package_root,
            before_enrollment_seal_path,
            before_enrollment_seal_sha256,
        ),
        "before_apply": (
            before_apply_package_root,
            before_apply_seal_path,
            before_apply_seal_sha256,
        ),
        "after_apply": (
            after_apply_package_root,
            after_apply_seal_path,
            after_apply_seal_sha256,
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (root, seal_path, digest) in values.items():
        if not isinstance(root, (str, Path)) or not str(root):
            raise ContinuousSessionPredictionError(f"{name} package root is invalid")
        if not isinstance(seal_path, (str, Path)) or not str(seal_path):
            raise ContinuousSessionPredictionError(f"{name} seal path is invalid")
        result[name] = {
            "package_root": str(root),
            "seal_path": str(seal_path),
            "seal_sha256": _observed_seal_sha256(f"{name} seal", seal_path, digest),
        }
    return result


def _registered_handles(manifest: Mapping[str, Any], *, label: str) -> tuple[str, ...]:
    raw = manifest.get("registered_classes")
    if not isinstance(raw, list):
        raise ContinuousSessionPredictionError(f"{label} registered classes drift")
    handles = tuple(
        str(item.get("class_handle", "")) if isinstance(item, Mapping) else ""
        for item in raw
    )
    if not handles or any(not value for value in handles) or len(set(handles)) != len(handles):
        raise ContinuousSessionPredictionError(f"{label} registered class handle drift")
    return handles


def _support_entries(
    payload: Mapping[str, Any],
    *,
    classes: Sequence[str],
    k_shot: int,
    label: str,
) -> dict[tuple[str, int], tuple[str, str, np.ndarray]]:
    """Map canonical support class/rank keys to token, IQ SHA and fixed IQ row."""

    required = (
        "support_leo_weak_iq",
        "support_class_indices",
        "support_rank_within_class",
        "support_tokens",
        "support_post_channel_iq_sha256",
    )
    if any(name not in payload for name in required):
        raise ContinuousSessionPredictionError(f"{label} support payload member drift")
    iq = np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)
    class_indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    tokens = np.asarray(payload["support_tokens"]).astype(str)
    row_hashes = np.asarray(payload["support_post_channel_iq_sha256"]).astype(str)
    count = len(class_indices)
    if (
        iq.ndim < 2
        or class_indices.shape != (count,)
        or ranks.shape != (count,)
        or tokens.shape != (count,)
        or row_hashes.shape != (count,)
    ):
        raise ContinuousSessionPredictionError(f"{label} support shape drift")
    mask = ranks < int(k_shot)
    expected = len(classes) * int(k_shot)
    if int(np.sum(mask)) != expected or np.any(ranks[mask] < 0):
        raise ContinuousSessionPredictionError(f"{label} K-shot support prefix drift")
    result: dict[tuple[str, int], tuple[str, str, np.ndarray]] = {}
    for index in np.flatnonzero(mask).tolist():
        class_index, rank = int(class_indices[index]), int(ranks[index])
        if class_index < 0 or class_index >= len(classes):
            raise ContinuousSessionPredictionError(f"{label} support class index drift")
        token, row_hash = str(tokens[index]), str(row_hashes[index])
        if not token or not row_hash:
            raise ContinuousSessionPredictionError(f"{label} support identity drift")
        key = (str(classes[class_index]), rank)
        if key in result:
            raise ContinuousSessionPredictionError(f"{label} duplicate support class/rank")
        result[key] = (token, row_hash, np.asarray(iq[index], dtype=np.float32))
    if len(result) != expected or len({entry[0] for entry in result.values()}) != expected:
        raise ContinuousSessionPredictionError(f"{label} support token uniqueness drift")
    return result


def _write_delta_npz(path: Path, **arrays: np.ndarray) -> str:
    return _write_npz_new(path, **arrays)


def prepare_continuous_session_support_deltas(
    *,
    before_enrollment_package_root: str | Path,
    before_enrollment_seal_path: str | Path,
    before_enrollment_seal_sha256: str | None,
    after_enrollment_package_root: str | Path,
    after_enrollment_seal_path: str | Path,
    after_enrollment_seal_sha256: str | None,
    prepared_delta_root: str | Path,
) -> dict[str, Any]:
    """Create sealed one-class support deltas outside the candidate process.

    Only the two enrollment packages are opened here.  The later prediction
    process receives the immutable old package plus this directory, never the
    complete after-enrollment support pool.
    """

    root = Path(prepared_delta_root)
    if root.is_symlink():
        raise ContinuousSessionPredictionError("prepared delta destination must not be a symbolic link")
    create_root = not root.exists()
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise FileExistsError("prepared continuous-session delta output already exists")
    before_seal = _observed_seal_sha256(
        "before enrollment seal", before_enrollment_seal_path, before_enrollment_seal_sha256
    )
    after_seal = _observed_seal_sha256(
        "after enrollment seal", after_enrollment_seal_path, after_enrollment_seal_sha256
    )
    before_payloads, before_manifest, before_audit = load_verified_somph_predictor_bundle(
        before_enrollment_package_root,
        detached_seal_path=before_enrollment_seal_path,
        expected_seal_sha256=before_seal,
    )
    after_payloads, after_manifest, after_audit = load_verified_somph_predictor_bundle(
        after_enrollment_package_root,
        detached_seal_path=after_enrollment_seal_path,
        expected_seal_sha256=after_seal,
    )
    old_classes = _registered_handles(before_manifest, label="before enrollment")
    all_classes = _registered_handles(after_manifest, label="after enrollment")
    if all_classes[: len(old_classes)] != old_classes:
        raise ContinuousSessionPredictionError("old registered prefix drift")
    new_classes = tuple(sorted(all_classes[len(old_classes) :]))
    k_shot = int(after_manifest.get("k_shot", -1))
    if len(old_classes) != 6 or len(new_classes) != 5 or k_shot != 10:
        raise ContinuousSessionPredictionError("continuous session requires old6/new5/K10 enrollment")
    if int(before_manifest.get("k_shot", -1)) != k_shot:
        raise ContinuousSessionPredictionError("before/after K-shot drift")
    prepared: list[dict[str, Any]] = []
    delta_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        try:
            before_entries = _support_entries(
                _mapping(before_payloads[scenario], label="before support payload"),
                classes=old_classes,
                k_shot=k_shot,
                label=f"before {scenario}",
            )
            after_entries = _support_entries(
                _mapping(after_payloads[scenario], label="after support payload"),
                classes=all_classes,
                k_shot=k_shot,
                label=f"after {scenario}",
            )
        except KeyError as error:
            raise ContinuousSessionPredictionError("formal support scenario coverage drift") from error
        for handle in old_classes:
            for rank in range(k_shot):
                before_entry = before_entries[(handle, rank)]
                after_entry = after_entries[(handle, rank)]
                if (
                    before_entry[0] != after_entry[0]
                    or before_entry[1] != after_entry[1]
                    or not np.array_equal(before_entry[2], after_entry[2])
                ):
                    raise ContinuousSessionPredictionError("old support/token drift")
        for handle in new_classes:
            records = [after_entries[(handle, rank)] for rank in range(k_shot)]
            delta_arrays.setdefault(handle, {})[scenario] = {
                "support_leo_weak_iq": np.stack([record[2] for record in records]).astype(np.float32),
                "support_tokens": np.asarray([record[0] for record in records]).astype(str),
                "support_post_channel_iq_sha256": np.asarray(
                    [record[1] for record in records]
                ).astype(str),
                "support_rank_within_class": np.arange(k_shot, dtype=np.int64),
                "class_handle": np.asarray([handle]),
            }
    if create_root:
        root.mkdir(parents=True, exist_ok=False)
    for session_index, handle in enumerate(new_classes, start=1):
        delta_root = root / f"delta_{session_index:02d}"
        delta_root.mkdir()
        members: list[dict[str, Any]] = []
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            payload_path = delta_root / f"{scenario}.npz"
            payload_sha256 = _write_delta_npz(payload_path, **delta_arrays[handle][scenario])
            members.append(
                {
                    "relative_path": payload_path.name,
                    "sha256": payload_sha256,
                    "size_bytes": payload_path.stat().st_size,
                    "scenario": scenario,
                    "support_token_sha256": _sha256_values(
                        tuple(delta_arrays[handle][scenario]["support_tokens"].astype(str).tolist())
                    ),
                }
            )
        delta_manifest = {
            "schema": "cvs.phase2.d92_e0_continuous_session.support_delta.v1",
            "session_index": session_index,
            "class_handle": handle,
            "k_shot": k_shot,
            "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
            "source_after_package_root_sha256": str(after_manifest["package_root_sha256"]),
            "source_after_seal_sha256": after_seal,
            "members": members,
            "artifact_root_sha256": hashlib.sha256(_canonical_bytes(members)).hexdigest(),
        }
        manifest_sha256 = _write_json_new(delta_root / "manifest.json", delta_manifest)
        if not all(_readonly(path) for path in delta_root.iterdir()):
            raise ContinuousSessionPredictionError("prepared delta is not immutable")
        prepared.append(
            {
                "session_index": session_index,
                "class_handle": handle,
                "root": str(delta_root.resolve()),
                "manifest_sha256": manifest_sha256,
            }
        )
    receipt = {
        "schema": "cvs.phase2.d92_e0_continuous_session.prepared_deltas.v1",
        "status": "PREPARED_DELTA_SUPPORT_COMPLETE",
        "method_id": METHOD_ID,
        "old_class_handles": list(old_classes),
        "new_class_handles": list(new_classes),
        "k_shot": k_shot,
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "before_package_root_sha256": str(before_manifest["package_root_sha256"]),
        "before_seal_sha256": before_seal,
        "after_package_root_sha256": str(after_manifest["package_root_sha256"]),
        "after_seal_sha256": after_seal,
        "before_preopen_audit": dict(before_audit),
        "after_preopen_audit": dict(after_audit),
        "future_support_open_sentinel": 0,
        "deltas": prepared,
    }
    receipt_sha256 = _write_json_new(root / "prepared_manifest.json", receipt)
    commit_members = [
        {
            "relative_path": "prepared_manifest.json",
            "sha256": receipt_sha256,
            "size_bytes": (root / "prepared_manifest.json").stat().st_size,
        },
        *[
            {
                "relative_path": f"delta_{entry['session_index']:02d}/manifest.json",
                "sha256": entry["manifest_sha256"],
                "size_bytes": (Path(entry["root"]) / "manifest.json").stat().st_size,
            }
            for entry in prepared
        ],
    ]
    commit_sha256 = _write_json_new(
        root / "COMMIT.json",
        {
            "schema": "cvs.phase2.d92_e0_continuous_session.prepared_deltas_commit.v1",
            "members": commit_members,
            "artifact_root_sha256": hashlib.sha256(_canonical_bytes(commit_members)).hexdigest(),
        },
    )
    return {
        **receipt,
        "prepared_delta_root": str(root.resolve()),
        "prepared_manifest_sha256": receipt_sha256,
        "commit_sha256": commit_sha256,
    }


def _required_audit(value: Any, *, session_index: int) -> dict[str, Any]:
    audit = dict(_mapping(value, label="session audit"))
    expected_lifecycle = "DA1_REG0" if session_index == 0 else f"DA1_REG1_S{session_index}"
    if audit.get("lifecycle_state") != expected_lifecycle:
        raise ContinuousSessionPredictionError("session lifecycle drift")
    if int(audit.get("session_index", -1)) != session_index:
        raise ContinuousSessionPredictionError("session index drift")
    if int(audit.get("future_support_open_sentinel", -1)) != 0:
        raise ContinuousSessionPredictionError("future support open sentinel drift")
    if int(audit.get("past_token_duplicate_count", -1)) != 0:
        raise ContinuousSessionPredictionError("past support token duplicate drift")
    if int(audit.get("full_solve_count", -1)) != 1:
        raise ContinuousSessionPredictionError("session must have exactly one full solve")
    if int(audit.get("d42_codec_count", -1)) != 1:
        raise ContinuousSessionPredictionError("session must have exactly one D42 codec")
    if audit.get("query_decision_policy") != "per_sample_all_registered_classes":
        raise ContinuousSessionPredictionError("query decision policy drift")
    if any(audit.get(name) is not False for name in _ZERO_QUERY_FIELDS):
        raise ContinuousSessionPredictionError("query access closure drift")
    return audit


def _required_resource(value: Any) -> dict[str, Any]:
    source = _mapping(value, label="session resource audit")
    result: dict[str, Any] = dict(source)
    for name in _RESOURCE_FIELDS:
        try:
            number = int(source[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ContinuousSessionPredictionError(f"session resource missing {name}") from error
        if number < 0:
            raise ContinuousSessionPredictionError(f"session resource {name} is negative")
        result[name] = number
    return result


def _readonly(path: Path) -> bool:
    info = path.lstat()
    return path.is_file() and not path.is_symlink() and not (stat.S_IMODE(info.st_mode) & _WRITE_BITS)


def _session_destination(root: Path, row: Mapping[str, Any]) -> Path:
    """Return the immutable per-state directory, optionally below one scene."""

    scene = row.get("scene")
    prefix = root
    if scene is not None:
        if not isinstance(scene, str) or scene not in FORMAL_LEO_WEAK_SCENARIOS:
            raise ContinuousSessionPredictionError("session scene is invalid")
        prefix = root / scene
    return (
        prefix / "DA1_REG0"
        if int(row["session_index"]) == 0
        else prefix / str(row["schedule"]) / f"session_{int(row['session_index']):02d}"
    )


@dataclass(frozen=True)
class _PreparedDeltaStore:
    """Metadata-only view of sealed deltas; payloads open only on arrival."""

    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    commit_members: Mapping[str, Mapping[str, Any]]
    old_handles: tuple[str, ...]
    new_handles: tuple[str, ...]
    k_shot: int


def _read_regular_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ContinuousSessionPredictionError(f"{label} is unavailable") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ContinuousSessionPredictionError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContinuousSessionPredictionError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise ContinuousSessionPredictionError(f"{label} must be an object")
    return value


def _verify_regular_member(
    path: Path, expected: Mapping[str, Any], *, label: str
) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise ContinuousSessionPredictionError(f"{label} is unavailable") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ContinuousSessionPredictionError(f"{label} must be a regular file")
    if int(expected.get("size_bytes", -1)) != int(info.st_size):
        raise ContinuousSessionPredictionError(f"{label} size drift")
    if _sha256_file(path) != _sha256_text(f"{label} SHA", expected.get("sha256")):
        raise ContinuousSessionPredictionError(f"{label} SHA drift")


def _prepared_delta_store(path: str | Path) -> _PreparedDeltaStore:
    root = Path(path)
    try:
        info = root.lstat()
    except FileNotFoundError as error:
        raise ContinuousSessionPredictionError("prepared delta root is unavailable") from error
    if root.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ContinuousSessionPredictionError("prepared delta root must be a regular directory")
    manifest_path = root / "prepared_manifest.json"
    commit_path = root / "COMMIT.json"
    commit = _read_regular_json(commit_path, label="prepared delta commit")
    members = commit.get("members")
    if not isinstance(members, list) or not members:
        raise ContinuousSessionPredictionError("prepared delta commit member drift")
    if commit.get("artifact_root_sha256") != hashlib.sha256(_canonical_bytes(members)).hexdigest():
        raise ContinuousSessionPredictionError("prepared delta commit root SHA drift")
    member_index: dict[str, Mapping[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise ContinuousSessionPredictionError("prepared delta commit member schema drift")
        relative = item.get("relative_path")
        if not isinstance(relative, str) or not relative or relative in member_index:
            raise ContinuousSessionPredictionError("prepared delta commit member path drift")
        member_index[relative] = item
    prepared_member = member_index.get("prepared_manifest.json")
    if prepared_member is None:
        raise ContinuousSessionPredictionError("prepared delta manifest is not committed")
    _verify_regular_member(manifest_path, prepared_member, label="prepared delta manifest")
    manifest = _read_regular_json(manifest_path, label="prepared delta manifest")
    if (
        manifest.get("schema")
        != "cvs.phase2.d92_e0_continuous_session.prepared_deltas.v1"
        or manifest.get("status") != "PREPARED_DELTA_SUPPORT_COMPLETE"
        or manifest.get("method_id") != METHOD_ID
        or manifest.get("scenarios") != list(FORMAL_LEO_WEAK_SCENARIOS)
        or int(manifest.get("k_shot", -1)) != 10
    ):
        raise ContinuousSessionPredictionError("prepared delta manifest identity drift")
    old_handles = tuple(str(item) for item in manifest.get("old_class_handles", ()))
    new_handles = tuple(str(item) for item in manifest.get("new_class_handles", ()))
    if (
        len(old_handles) != 6
        or len(new_handles) != 5
        or len(set(old_handles)) != 6
        or len(set(new_handles)) != 5
        or set(old_handles).intersection(new_handles)
        or new_handles != tuple(sorted(new_handles))
    ):
        raise ContinuousSessionPredictionError("prepared delta class registry drift")
    deltas = manifest.get("deltas")
    if not isinstance(deltas, list) or len(deltas) != len(new_handles):
        raise ContinuousSessionPredictionError("prepared delta entry count drift")
    for index, (entry, handle) in enumerate(zip(deltas, new_handles), start=1):
        if not isinstance(entry, Mapping):
            raise ContinuousSessionPredictionError("prepared delta entry schema drift")
        if (
            int(entry.get("session_index", -1)) != index
            or entry.get("class_handle") != handle
            or _sha256_text("prepared delta manifest SHA", entry.get("manifest_sha256"))
            != entry.get("manifest_sha256")
            or Path(str(entry.get("root", ""))).name != f"delta_{index:02d}"
            or f"delta_{index:02d}/manifest.json" not in member_index
        ):
            raise ContinuousSessionPredictionError("prepared delta entry identity drift")
    return _PreparedDeltaStore(
        root=root,
        manifest=manifest,
        manifest_sha256=str(prepared_member["sha256"]),
        commit_members=member_index,
        old_handles=old_handles,
        new_handles=new_handles,
        k_shot=10,
    )


def _open_arriving_delta(
    store: _PreparedDeltaStore, *, canonical_index: int, scenario: str
) -> tuple[str, np.ndarray, tuple[str, ...]]:
    """Open exactly one sealed delta payload at its scheduled arrival."""

    if scenario not in FORMAL_LEO_WEAK_SCENARIOS:
        raise ContinuousSessionPredictionError("prepared delta scenario drift")
    if canonical_index < 0 or canonical_index >= len(store.new_handles):
        raise ContinuousSessionPredictionError("prepared delta arrival index drift")
    session_index = canonical_index + 1
    delta_root = store.root / f"delta_{session_index:02d}"
    try:
        delta_info = delta_root.lstat()
    except FileNotFoundError as error:
        raise ContinuousSessionPredictionError("arriving delta is unavailable") from error
    if delta_root.is_symlink() or not stat.S_ISDIR(delta_info.st_mode):
        raise ContinuousSessionPredictionError("arriving delta directory drift")
    manifest_path = delta_root / "manifest.json"
    committed = store.commit_members[f"delta_{session_index:02d}/manifest.json"]
    _verify_regular_member(manifest_path, committed, label="arriving delta manifest")
    manifest = _read_regular_json(manifest_path, label="arriving delta manifest")
    expected_handle = store.new_handles[canonical_index]
    if (
        manifest.get("schema")
        != "cvs.phase2.d92_e0_continuous_session.support_delta.v1"
        or int(manifest.get("session_index", -1)) != session_index
        or manifest.get("class_handle") != expected_handle
        or int(manifest.get("k_shot", -1)) != store.k_shot
        or manifest.get("scenarios") != list(FORMAL_LEO_WEAK_SCENARIOS)
    ):
        raise ContinuousSessionPredictionError("arriving delta manifest identity drift")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(FORMAL_LEO_WEAK_SCENARIOS):
        raise ContinuousSessionPredictionError("arriving delta member count drift")
    matches = [item for item in members if isinstance(item, Mapping) and item.get("scenario") == scenario]
    if len(matches) != 1:
        raise ContinuousSessionPredictionError("arriving delta scenario member drift")
    member = matches[0]
    relative = member.get("relative_path")
    if relative != f"{scenario}.npz" or Path(str(relative)).name != relative:
        raise ContinuousSessionPredictionError("arriving delta payload path drift")
    payload_path = delta_root / str(relative)
    _verify_regular_member(payload_path, member, label="arriving delta payload")
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            required = {
                "support_leo_weak_iq",
                "support_tokens",
                "support_post_channel_iq_sha256",
                "support_rank_within_class",
                "class_handle",
            }
            if set(archive.files) != required:
                raise ContinuousSessionPredictionError("arriving delta payload schema drift")
            iq = np.array(archive["support_leo_weak_iq"], copy=True)
            tokens = np.array(archive["support_tokens"], copy=True)
            hashes = np.array(archive["support_post_channel_iq_sha256"], copy=True)
            ranks = np.array(archive["support_rank_within_class"], copy=True)
            handles = np.array(archive["class_handle"], copy=True)
    except (OSError, ValueError) as error:
        raise ContinuousSessionPredictionError("arriving delta payload is unreadable") from error
    token_values = _string_vector(tokens, label="arriving support tokens")
    hash_values = _string_vector(hashes, label="arriving support hashes")
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or iq.shape[0] != store.k_shot
        or not np.isfinite(iq).all()
        or len(token_values) != store.k_shot
        or len(set(token_values)) != store.k_shot
        or len(hash_values) != store.k_shot
        or any(_sha256_text("arriving support hash", value) != value for value in hash_values)
        or ranks.dtype.kind not in {"i", "u"}
        or not np.array_equal(ranks, np.arange(store.k_shot, dtype=ranks.dtype))
        or _string_vector(handles, label="arriving class handle") != (expected_handle,)
    ):
        raise ContinuousSessionPredictionError("arriving delta payload value drift")
    return expected_handle, np.asarray(iq, dtype=np.float32), token_values


def _state_bytes(state: Any) -> int:
    supplied = getattr(state, "persistent_state_bytes", None)
    if isinstance(supplied, (int, np.integer)) and int(supplied) > 0:
        return int(supplied)
    return int(
        sum(
            np.asarray(_field(state, name, label="state")).nbytes
            for name in (
                "log_diag_fp32",
                "coef1_qint8",
                "coef2_qint8",
                "scale1_fp16",
                "scale2_fp16",
                "intercept_fp16",
            )
        )
    )


def _registration_resource(
    measured: Mapping[str, Any],
    *,
    state: Any,
    support_bytes: int,
    registered_class_count: int,
    enforce_hard_gate: bool = True,
) -> dict[str, Any]:
    resource = dict(measured)
    wall = int(resource.get("registration_wall_time_ns", -1))
    peak = int(resource.get("registration_incremental_peak_working_set_bytes", -1))
    if wall < 0 or peak < 0:
        raise ContinuousSessionPredictionError("registration resource receipt drift")
    if enforce_hard_gate and wall > _REGISTRATION_WALL_HARD_MAX_NS:
        raise ContinuousSessionPredictionError("registration wall hard gate failed")
    if enforce_hard_gate and peak > _REGISTRATION_PEAK_HARD_MAX_BYTES:
        raise ContinuousSessionPredictionError("registration peak hard gate failed")
    resource.update(
        {
            "registration_wall_hard_max_ns": _REGISTRATION_WALL_HARD_MAX_NS,
            "registration_incremental_peak_hard_max_bytes": _REGISTRATION_PEAK_HARD_MAX_BYTES,
            "registration_hard_gate_enforced": bool(enforce_hard_gate),
            "registration_resource_scope": (
                "cumulative_new_class_registration"
                if enforce_hard_gate
                else "frozen_da1_reg0_baseline_rebuild"
            ),
            "support_bytes": int(support_bytes),
            "state_bytes": _state_bytes(state),
            "query_macs": int(registered_class_count * 288),
            "head_latency_ns": 0,
        }
    )
    return _required_resource(resource)


def _unlabeled_query_features(
    payload: Mapping[str, Any], *, feature_rows: Callable[[np.ndarray, int], np.ndarray]
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    iq = np.asarray(payload.get("query_leo_weak_iq"))
    tokens = _string_vector(payload.get("query_tokens"), label="query tokens")
    hashes = _string_vector(
        payload.get("query_post_channel_iq_sha256"), label="query IQ hashes"
    )
    if (
        iq.dtype != np.float32
        or iq.ndim != 3
        or len(iq) != len(tokens)
        or len(hashes) != len(tokens)
        or len(set(tokens)) != len(tokens)
        or not np.isfinite(iq).all()
        or any(_sha256_text("query IQ hash", value) != value for value in hashes)
    ):
        raise ContinuousSessionPredictionError("unlabeled query payload drift")
    features = np.asarray(feature_rows(iq, 1), dtype=np.float32)
    if features.shape != (len(iq), 288) or not np.isfinite(features).all():
        raise ContinuousSessionPredictionError("unlabeled query feature drift")
    return features, tokens, hashes


def _old_support_packets(
    payload: Mapping[str, Any],
    *,
    class_handles: tuple[str, ...],
    feature_rows: Callable[[np.ndarray, int], np.ndarray],
    package_identity: str,
    continuous: Any,
) -> tuple[tuple[Any, ...], np.ndarray, np.ndarray]:
    entries = _support_entries(
        payload,
        classes=class_handles,
        k_shot=10,
        label="old support payload",
    )
    packets: list[Any] = []
    raw_rows: list[np.ndarray] = []
    for handle in class_handles:
        records = [entries[(handle, rank)] for rank in range(10)]
        iq = np.stack([item[2] for item in records]).astype(np.float32)
        features = np.asarray(feature_rows(iq, 64), dtype=np.float32)
        if features.shape != (10, 288) or not np.isfinite(features).all():
            raise ContinuousSessionPredictionError("old support feature drift")
        tokens = tuple(item[0] for item in records)
        packets.append(
            continuous.SupportPacket(
                handle=handle,
                rows=features,
                physical_tokens=tokens,
                package_id=f"{package_identity}:{handle}",
                arrival_session=0,
            )
        )
        raw_rows.append(features)
    return (
        tuple(packets),
        np.concatenate(raw_rows, axis=0).astype(np.float32),
        np.repeat(np.arange(len(class_handles), dtype=np.int64), 10),
    )


def _support_transform(
    d42: Any,
    d81: Any,
    *,
    log_diag_fp32: np.ndarray,
    basis: np.ndarray,
    spectral_weights: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]:
    """The frozen E0 D42-to-D81 support-only closure; no identity fallback."""

    def transform(
        canonical_raw_rows: np.ndarray,
        targets: np.ndarray,
        class_count: int,
        k_shot: int,
    ) -> np.ndarray:
        scaled = d42._transform(
            np.asarray(canonical_raw_rows, dtype=np.float32), log_diag_fp32
        )
        translated, _audit = d81.translate_to_robust_centers(
            scaled,
            np.asarray(targets, dtype=np.int64),
            int(class_count),
            int(k_shot),
            basis,
            spectral_weights,
        )
        return np.asarray(translated, dtype=np.float32)

    return transform


def _baseline_state(
    d42: Any,
    anchor: Any,
    transform: Callable[[np.ndarray, np.ndarray, int, int], np.ndarray],
) -> tuple[Any, dict[str, Any]]:
    rows = np.concatenate([record.rows for record in anchor.old_records], axis=0)
    targets = np.repeat(np.arange(len(anchor.old_records), dtype=np.int64), 10)
    transformed = np.asarray(
        transform(rows, targets, len(anchor.old_records), 10), dtype=np.float32
    )
    coefficient, intercept, fit_audit = d42._fit_equal_prior_lda(
        transformed, targets, len(anchor.old_records), 10
    )
    state, codec_audit = d42._compile_state(
        tuple(record.handle for record in anchor.old_records),
        len(anchor.old_records),
        anchor.log_diag_fp32,
        coefficient,
        intercept,
        str(fit_audit["covariance_policy"]),
        precision="int8",
    )
    return state, {**fit_audit, **codec_audit}


def _original_d42_f0_prediction(d42: Any, state: Any, features: np.ndarray) -> np.ndarray:
    """Use only the pre-existing D42 F0 scorer and its bound log-diagonal."""

    convert = getattr(state, "to_d42_unified_state", None)
    deployed = convert() if callable(convert) else state
    if not isinstance(deployed, d42.D42UnifiedShrinkageLDAState):
        raise ContinuousSessionPredictionError("continuous state cannot enter original D42 F0 scorer")
    try:
        prediction = d42.predict_d42_unified_shrinkage_lda(
            deployed, np.asarray(features, dtype=np.float32)
        )
    except Exception as error:
        raise ContinuousSessionPredictionError("original D42 F0 query scorer failed") from error
    values = _string_vector(prediction, label="original D42 F0 prediction")
    if any(value not in deployed.classes for value in values):
        raise ContinuousSessionPredictionError("original D42 F0 class closure drift")
    return np.asarray(values).astype(str)


def _claim_empty_output_root(path: Path) -> Path:
    if path.is_symlink():
        raise ContinuousSessionPredictionError("continuous-session output must not be a symbolic link")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError("continuous-session prediction output already exists")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def _scene_names(job: Mapping[str, Any] | None) -> tuple[str, ...]:
    if job is None:
        return tuple(FORMAL_LEO_WEAK_SCENARIOS)
    raw = job.get("scenes")
    if not isinstance(raw, list) or tuple(raw) != tuple(FORMAL_LEO_WEAK_SCENARIOS):
        raise ContinuousSessionPredictionError("frozen scene order drift")
    return tuple(raw)


def _after_apply_manifest_lock(
    before: Mapping[str, Any], after_apply: Mapping[str, Any], store: _PreparedDeltaStore
) -> None:
    if after_apply.get("profile") != "apply_only" or after_apply.get("registration_state") != "after":
        raise ContinuousSessionPredictionError("after apply profile/state drift")
    for field in (
        "stage",
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
    ):
        if before.get(field) != after_apply.get(field):
            raise ContinuousSessionPredictionError(f"before/after apply {field} drift")
    after_handles = _registered_handles(after_apply, label="after apply")
    if (
        after_handles[: len(store.old_handles)] != store.old_handles
        or len(after_handles) != len(store.old_handles) + len(store.new_handles)
        or set(after_handles[len(store.old_handles) :]) != set(store.new_handles)
    ):
        raise ContinuousSessionPredictionError("after apply registered class drift")


def _strict_s5_equivalence(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    required = {"batch_5", "singleton_forward", "singleton_reverse", "chunk_2_2_1"}
    if set(rows) != required or any(not rows[name] for name in required):
        raise ContinuousSessionPredictionError("terminal schedule closure drift")
    reference = rows["batch_5"][-1]
    reference_state = reference["state"]
    reference_prediction = tuple(reference["predictions"])
    fields = (
        "coef1_qint8",
        "coef2_qint8",
        "scale1_fp16",
        "scale2_fp16",
        "intercept_fp16",
    )
    for name in sorted(required):
        candidate = rows[name][-1]
        if (
            tuple(candidate["registered_classes"]) != tuple(reference["registered_classes"])
            or candidate["state_sha256"] != reference["state_sha256"]
            or tuple(candidate["predictions"]) != reference_prediction
        ):
            raise ContinuousSessionPredictionError("S5 batch/continuous state or prediction drift")
        for field in fields:
            if not np.array_equal(
                np.asarray(_field(candidate["state"], field, label="terminal state")),
                np.asarray(_field(reference_state, field, label="terminal state")),
            ):
                raise ContinuousSessionPredictionError("S5 D42 codec field drift")
    return {
        "status": "STRICT_EQUAL",
        "reference_schedule": "batch_5",
        "state_sha256": reference["state_sha256"],
        "prediction_sha256": hashlib.sha256(
            _canonical_bytes(list(reference_prediction))
        ).hexdigest(),
        "registered_classes": list(reference["registered_classes"]),
        "checked_schedules": sorted(required),
        "checked_codec_fields": list(fields),
    }


def _run_real_continuous_session_prediction(
    *,
    root: Path,
    package_rows: Mapping[str, Mapping[str, str]],
    prepared_delta_root: str | Path,
    ground_component_dir: str | Path,
    ground_manifest_sha256: str,
    schedules: Mapping[str, dict[str, tuple[int, ...]]],
    device: str,
    job: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run support-state Phase A, then the original D42 F0 Phase B scorer."""

    from cvsrffi import stage2_d42_unified_shrinkage_lda as d42
    from cvsrffi import stage2_d81_ground_nuisance_cauchy_center as d81
    from cvsrffi import stage2_d92_continuous_session as continuous
    from cvsrffi.stage2_diag_cosine_exploration import (
        _descriptor,
        _device,
        _validate_matched_packages,
        forward_zid160,
        registered_feature,
    )
    from cvsrffi.stage2_predictor_runtime import load_torchscript_backbone_same_fd
    from cvsrffi.stage2_registration_resource_probe import measure_registration_call
    from scripts.probe_d81_ground_nuisance_cauchy_center import load_ground_basis

    store = _prepared_delta_store(prepared_delta_root)
    before_payloads, before_manifest, before_preopen_audit = load_verified_somph_predictor_bundle(
        package_rows["before_enrollment"]["package_root"],
        detached_seal_path=package_rows["before_enrollment"]["seal_path"],
        expected_seal_sha256=package_rows["before_enrollment"]["seal_sha256"],
    )
    old_handles = _registered_handles(before_manifest, label="before enrollment")
    if (
        old_handles != store.old_handles
        or len(old_handles) != 6
        or int(before_manifest.get("k_shot", -1)) != store.k_shot
        or before_manifest.get("package_root_sha256") != store.manifest.get("before_package_root_sha256")
        or package_rows["before_enrollment"]["seal_sha256"]
        != store.manifest.get("before_seal_sha256")
    ):
        raise ContinuousSessionPredictionError("before enrollment/delta anchor drift")
    runtime_device = _device(device)
    model = load_torchscript_backbone_same_fd(
        package_rows["before_enrollment"]["package_root"],
        _descriptor(before_manifest, "feature_runtime"),
        device=runtime_device,
    )
    basis, spectral_weights, ground_audit = load_ground_basis(
        Path(ground_component_dir), ground_manifest_sha256, 288
    )

    def feature_rows(iq: np.ndarray, batch_size: int) -> np.ndarray:
        zid = forward_zid160(model, iq, device=runtime_device, batch_size=batch_size)
        values = np.asarray(registered_feature(iq, zid), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 288 or not np.isfinite(values).all():
            raise ContinuousSessionPredictionError("sealed backbone feature drift")
        return values

    package_digest = hashlib.sha256(
        _canonical_bytes(
            {
                "packages": package_rows,
                "prepared_delta_manifest_sha256": store.manifest_sha256,
                "ground_manifest_sha256": ground_manifest_sha256,
            }
        )
    ).hexdigest()
    phase_a: dict[str, dict[str, Any]] = {}
    for scene_index, scene in enumerate(_scene_names(job)):
        try:
            scene_payload = _mapping(before_payloads[scene], label="before support scene")
        except KeyError as error:
            raise ContinuousSessionPredictionError("before enrollment scene coverage drift") from error
        old_packets, old_raw_rows, old_targets = _old_support_packets(
            scene_payload,
            class_handles=old_handles,
            feature_rows=feature_rows,
            package_identity=f"{before_manifest['package_root_sha256']}:{scene}",
            continuous=continuous,
        )
        log_diag, metric_trace, metric_resource = d42._fit_old_only_b3_metric(
            old_raw_rows,
            old_targets,
            len(old_handles),
            seed=int(before_manifest["seed"]) + scene_index,
            device=runtime_device,
        )
        transform = _support_transform(
            d42,
            d81,
            log_diag_fp32=np.asarray(log_diag, dtype=np.float32),
            basis=basis,
            spectral_weights=spectral_weights,
        )
        transform_identity = hashlib.sha256(
            _canonical_bytes(
                {
                    "d81": "ground_nuisance_cauchy_center",
                    "ground_basis_sha256": ground_audit.get("basis_sha256"),
                    "ground_spectral_weight_sha256": ground_audit.get("spectral_weight_sha256"),
                    "log_diag_sha256": hashlib.sha256(
                        np.ascontiguousarray(log_diag, dtype=np.float32).tobytes(order="C")
                    ).hexdigest(),
                }
            )
        ).hexdigest()
        anchor_id = hashlib.sha256(
            _canonical_bytes(
                {
                    "before_package_root_sha256": before_manifest["package_root_sha256"],
                    "before_seal_sha256": package_rows["before_enrollment"]["seal_sha256"],
                    "prepared_delta_manifest_sha256": store.manifest_sha256,
                    "scene": scene,
                    "support_transform_identity": transform_identity,
                }
            )
        ).hexdigest()
        anchor = continuous.FrozenDAAnchor.from_old_support(
            old_packets,
            da_anchor_id=anchor_id,
            log_diag_fp32=np.asarray(log_diag, dtype=np.float32),
            support_transform=transform,
            support_transform_identity=transform_identity,
        )
        (baseline_state, baseline_fit), baseline_measure = measure_registration_call(
            lambda: _baseline_state(d42, anchor, transform)
        )
        baseline_audit = {
            **baseline_fit,
            "method_id": METHOD_ID,
            "lifecycle_state": "DA1_REG0",
            "session_index": 0,
            "full_solve_count": 1,
            "d42_codec_count": 1,
            "d81_transform_count": 1,
            "frozen_da_anchor_id": anchor_id,
            "support_transform_identity": transform_identity,
            "old_metric_optimizer_steps": int(metric_resource["optimizer_steps"]),
            "old_metric_query_rows_used": 0,
            "old_metric_trace_count": len(metric_trace),
            "future_support_open_sentinel": 0,
            "past_token_duplicate_count": 0,
            "query_decision_policy": "per_sample_all_registered_classes",
            **{name: False for name in _ZERO_QUERY_FIELDS},
        }
        _required_audit(baseline_audit, session_index=0)
        baseline_row = {
            "scene": scene,
            "schedule": "DA1_REG0",
            "session_index": 0,
            "increment": 0,
            "arrival_order": [],
            "lifecycle_state": "DA1_REG0",
            "state": baseline_state,
            "state_sha256": _state_fingerprint(baseline_state),
            "registered_classes": list(old_handles),
            "old_class_count": len(old_handles),
            "arriving_support_tokens": (),
            "cumulative_support_tokens": (),
            "audit": baseline_audit,
            "resource": _registration_resource(
                baseline_measure,
                state=baseline_state,
                support_bytes=sum(record.rows.nbytes for record in anchor.old_records),
                registered_class_count=len(old_handles),
                enforce_hard_gate=False,
            ),
            "ledger": continuous.SessionLedger.start(anchor),
        }
        sealed_baseline = _seal_session_state(root, baseline_row, package_digest=package_digest)
        scene_rows: dict[str, list[dict[str, Any]]] = {}
        for schedule_name, schedule in schedules.items():
            ledger = continuous.SessionLedger.start(anchor)
            rows: list[dict[str, Any]] = []
            cursor = 0
            for session_index, increment in enumerate(schedule["increments"], start=1):
                arrival_indices = schedule["arrival_order"][cursor : cursor + increment]
                cursor += increment
                packets: list[Any] = []
                for canonical_index in arrival_indices:
                    handle, iq, tokens = _open_arriving_delta(
                        store, canonical_index=canonical_index, scenario=scene
                    )
                    features = feature_rows(iq, 64)
                    packets.append(
                        continuous.SupportPacket(
                            handle=handle,
                            rows=features,
                            physical_tokens=tokens,
                            package_id=f"{store.manifest_sha256}:{canonical_index}",
                            arrival_session=session_index,
                        )
                    )
                result, measured = measure_registration_call(
                    lambda: continuous.advance_session(ledger, tuple(packets))
                )
                ledger = result.ledger
                core_audit = dict(result.audit)
                audit = {
                    **core_audit,
                    "lifecycle_state": f"DA1_REG1_S{session_index}",
                    "session_index": session_index,
                    "full_solve_count": int(core_audit["d92_continuous_full_solve_count"]),
                    "d42_codec_count": int(core_audit["d92_continuous_d42_codec_count"]),
                    "d81_transform_count": int(core_audit["d92_continuous_d81_transform_count"]),
                    "query_decision_policy": "per_sample_all_registered_classes",
                    **{name: False for name in _ZERO_QUERY_FIELDS},
                }
                _required_audit(audit, session_index=session_index)
                arriving_tokens = tuple(
                    sorted(token for packet in packets for token in packet.physical_tokens)
                )
                cumulative_tokens = tuple(
                    token
                    for record in ledger.arrived_records
                    for token in record.physical_tokens
                )
                if len(set(cumulative_tokens)) != len(cumulative_tokens):
                    raise ContinuousSessionPredictionError("continuous ledger token duplication")
                row = {
                    "scene": scene,
                    "schedule": schedule_name,
                    "session_index": session_index,
                    "increment": increment,
                    "arrival_order": list(arrival_indices),
                    "lifecycle_state": audit["lifecycle_state"],
                    "state": result.state,
                    "state_sha256": _state_fingerprint(result.state),
                    "registered_classes": list(result.state.classes),
                    "old_class_count": int(result.state.old_class_count),
                    "arriving_support_tokens": arriving_tokens,
                    "cumulative_support_tokens": cumulative_tokens,
                    "audit": audit,
                    "resource": _registration_resource(
                        measured,
                        state=result.state,
                        support_bytes=sum(
                            record.rows.nbytes
                            for record in ledger.anchor.old_records + ledger.arrived_records
                        ),
                        registered_class_count=len(result.state.classes),
                    ),
                    "ledger": ledger,
                }
                rows.append(_seal_session_state(root, row, package_digest=package_digest))
            scene_rows[schedule_name] = rows
        phase_a[scene] = {"baseline": sealed_baseline, "schedules": scene_rows}

    # Phase B begins only once every state receipt has been sealed above.
    before_apply_payloads, before_apply_manifest, before_apply_audit = load_verified_somph_predictor_bundle(
        package_rows["before_apply"]["package_root"],
        detached_seal_path=package_rows["before_apply"]["seal_path"],
        expected_seal_sha256=package_rows["before_apply"]["seal_sha256"],
    )
    after_apply_payloads, after_apply_manifest, after_apply_audit = load_verified_somph_predictor_bundle(
        package_rows["after_apply"]["package_root"],
        detached_seal_path=package_rows["after_apply"]["seal_path"],
        expected_seal_sha256=package_rows["after_apply"]["seal_sha256"],
    )
    _validate_matched_packages(before_manifest, before_apply_manifest)
    _after_apply_manifest_lock(before_manifest, after_apply_manifest, store)
    result_scenes: dict[str, Any] = {}
    equivalence: dict[str, Any] = {}
    for scene, phase in phase_a.items():
        before_features, before_tokens, before_hashes = _unlabeled_query_features(
            _mapping(before_apply_payloads[scene], label="before apply scene"),
            feature_rows=feature_rows,
        )
        after_features, after_tokens, after_hashes = _unlabeled_query_features(
            _mapping(after_apply_payloads[scene], label="after apply scene"),
            feature_rows=feature_rows,
        )
        if before_tokens != after_tokens or before_hashes != after_hashes:
            raise ContinuousSessionPredictionError("fixed apply query token/IQ identity drift")

        def attach_prediction(row: Mapping[str, Any], features: np.ndarray) -> dict[str, Any]:
            started = time.perf_counter_ns()
            prediction = _original_d42_f0_prediction(d42, row["state"], features)
            elapsed = time.perf_counter_ns() - started
            return {
                **dict(row),
                "query_tokens": before_tokens,
                "scenarios": tuple(scene for _ in before_tokens),
                "predictions": tuple(prediction.tolist()),
                "query_prediction_wall_time_ns": int(elapsed),
            }

        enriched_baseline = attach_prediction(phase["baseline"], before_features)
        enriched_schedules = {
            name: [attach_prediction(row, after_features) for row in rows]
            for name, rows in phase["schedules"].items()
        }
        equivalence[scene] = _strict_s5_equivalence(enriched_schedules)
        baseline_receipt = _finalize_session_prediction(
            enriched_baseline, package_digest=package_digest
        )
        schedules_receipt: dict[str, Any] = {}
        for name, rows in enriched_schedules.items():
            schedules_receipt[name] = {
                "increments": list(schedules[name]["increments"]),
                "arrival_order": list(schedules[name]["arrival_order"]),
                "sessions": [
                    _finalize_session_prediction(row, package_digest=package_digest)
                    for row in rows
                ],
            }
        result_scenes[scene] = {
            "DA1_REG0": baseline_receipt,
            "schedules": schedules_receipt,
        }
    manifest = {
        "schema": SCHEMA,
        "status": "D92_E0_CONTINUOUS_SESSION_TRUTH_FREE_PREDICTIONS_COMPLETE",
        "method_id": METHOD_ID,
        "output_root": str(root.resolve()),
        "package_identity_sha256": package_digest,
        "prepared_delta_manifest_sha256": store.manifest_sha256,
        "before_preopen_audit": dict(before_preopen_audit),
        "before_apply_preopen_audit": dict(before_apply_audit),
        "after_apply_preopen_audit": dict(after_apply_audit),
        "future_support_open_sentinel": 0,
        **{name: False for name in _ZERO_QUERY_FIELDS},
        "scenes": result_scenes,
        "final_s5_equivalence": equivalence,
    }
    manifest_sha256 = _write_json_new(root / "prediction_manifest.json", manifest)
    return {**manifest, "prediction_manifest_sha256": manifest_sha256}


def _prepare_sessions(
    *,
    schedules: Mapping[str, dict[str, tuple[int, ...]]],
    packages: Mapping[str, Mapping[str, str]],
    ground_component_dir: str | Path,
    ground_manifest_sha256: str,
    device: str,
    session_builder: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    prepared: dict[str, list[dict[str, Any]]] = {}
    baseline_raw = session_builder(
        schedule_name="DA1_REG0",
        schedule={"increments": (), "arrival_order": ()},
        session_index=0,
        increment=0,
        arrival_order=(),
        previous_ledger=None,
        packages=packages,
        ground_component_dir=str(ground_component_dir),
        ground_manifest_sha256=ground_manifest_sha256,
        device=device,
    )

    def normalize(raw: Any, *, schedule_name: str, session_index: int, increment: int, arrival_indices: tuple[int, ...], expected_seen_tokens: set[str]) -> dict[str, Any]:
        audit = _required_audit(
            _field(raw, "audit", label="session result"), session_index=session_index
        )
        state = _field(raw, "state", label="session result")
        classes = _string_vector(
            _field(state, "classes", label="state"), label="state classes"
        )
        old_class_count = int(_field(state, "old_class_count", label="state"))
        if (
            len(set(classes)) != len(classes)
            or old_class_count < 1
            or (session_index == 0 and old_class_count != len(classes))
            or (session_index > 0 and old_class_count >= len(classes))
        ):
            raise ContinuousSessionPredictionError("registered class state drift")
        arriving_tokens = _string_vector(
            _field(raw, "arriving_support_tokens", label="session result"),
            label="arriving support tokens",
            allow_empty=session_index == 0,
        )
        cumulative_tokens = _string_vector(
            _field(raw, "cumulative_support_tokens", label="session result"),
            label="cumulative support tokens",
            allow_empty=session_index == 0,
        )
        if len(set(arriving_tokens)) != len(arriving_tokens) or expected_seen_tokens.intersection(arriving_tokens):
            raise ContinuousSessionPredictionError("duplicate support token across sessions")
        expected_support_tokens = expected_seen_tokens | set(arriving_tokens)
        if len(set(cumulative_tokens)) != len(cumulative_tokens) or set(cumulative_tokens) != expected_support_tokens:
            raise ContinuousSessionPredictionError("cumulative support token ledger drift")
        query_tokens = _string_vector(
            _field(raw, "query_tokens", label="session result"), label="query tokens"
        )
        scenarios = _string_vector(
            _field(raw, "scenarios", label="session result"), label="query scenarios"
        )
        predictions = _string_vector(
            _field(raw, "predicted_class_handles", label="session result"),
            label="predicted class handles",
        )
        if len(query_tokens) != len(scenarios) or len(query_tokens) != len(predictions):
            raise ContinuousSessionPredictionError("query prediction length drift")
        if any(prediction not in classes for prediction in predictions):
            raise ContinuousSessionPredictionError("prediction is outside all registered classes")
        resource = _required_resource(_field(raw, "resource_audit", label="session result"))
        return {
            "schedule": schedule_name,
            "session_index": session_index,
            "increment": increment,
            "arrival_order": list(arrival_indices),
            "lifecycle_state": str(audit["lifecycle_state"]),
            "state": state,
            "state_sha256": _state_fingerprint(state),
            "registered_classes": list(classes),
            "old_class_count": old_class_count,
            "arriving_support_tokens": arriving_tokens,
            "cumulative_support_tokens": cumulative_tokens,
            "query_tokens": query_tokens,
            "scenarios": scenarios,
            "predictions": predictions,
            "audit": audit,
            "resource": resource,
            "ledger": _field(raw, "ledger", label="session result"),
        }

    baseline = normalize(
        baseline_raw,
        schedule_name="DA1_REG0",
        session_index=0,
        increment=0,
        arrival_indices=(),
        expected_seen_tokens=set(),
    )
    query_identity: tuple[tuple[str, ...], tuple[str, ...]] = (
        tuple(baseline["query_tokens"]),
        tuple(baseline["scenarios"]),
    )
    for schedule_name, schedule in schedules.items():
        previous_ledger: Any = baseline["ledger"]
        seen_support_tokens: set[str] = set()
        rows: list[dict[str, Any]] = []
        cursor = 0
        for session_index, increment in enumerate(schedule["increments"], start=1):
            arrival_indices = schedule["arrival_order"][cursor : cursor + increment]
            cursor += increment
            raw = session_builder(
                schedule_name=schedule_name,
                schedule={
                    "increments": schedule["increments"],
                    "arrival_order": schedule["arrival_order"],
                },
                session_index=session_index,
                increment=increment,
                arrival_order=arrival_indices,
                previous_ledger=previous_ledger,
                packages=packages,
                ground_component_dir=str(ground_component_dir),
                ground_manifest_sha256=ground_manifest_sha256,
                device=device,
            )
            row = normalize(
                raw,
                schedule_name=schedule_name,
                session_index=session_index,
                increment=increment,
                arrival_indices=arrival_indices,
                expected_seen_tokens=seen_support_tokens,
            )
            identity = (tuple(row["query_tokens"]), tuple(row["scenarios"]))
            if identity != query_identity:
                raise ContinuousSessionPredictionError("fixed query token identity drift")
            seen_support_tokens = set(row["cumulative_support_tokens"])
            previous_ledger = row["ledger"]
            rows.append(row)
        prepared[schedule_name] = rows
    return baseline, prepared


def _publish_session(root: Path, row: Mapping[str, Any], *, package_digest: str) -> dict[str, Any]:
    destination = _session_destination(root, row)
    destination.mkdir(parents=True, exist_ok=False)
    artifact_path = destination / "prediction_artifact.npz"
    prediction_sha256 = _write_npz_new(
        artifact_path,
        query_tokens=np.asarray(row["query_tokens"]).astype(str),
        scenarios=np.asarray(row["scenarios"]).astype(str),
        predicted_class_handles=np.asarray(row["predictions"]).astype(str),
    )
    fit = {
        **dict(row["audit"]),
        "state_sha256": row["state_sha256"],
        "registered_classes": list(row["registered_classes"]),
        "old_class_count": int(row["old_class_count"]),
        "arriving_support_token_sha256": _sha256_values(row["arriving_support_tokens"]),
        "cumulative_support_token_sha256": _sha256_values(row["cumulative_support_tokens"]),
        "query_token_sha256": _sha256_values(row["query_tokens"]),
        "query_scenario_sha256": _sha256_values(row["scenarios"]),
    }
    fit_sha256 = _write_json_new(destination / "fit_audit.json", fit)
    resource = dict(row["resource"])
    resource_sha256 = _write_json_new(destination / "resource_audit.json", resource)
    receipt = {
        "schema": SCHEMA,
        "status": "D92_E0_CONTINUOUS_SESSION_TRUTH_FREE_PREDICTION_COMPLETE",
        "method_id": METHOD_ID,
        "schedule": row["schedule"],
        "session_index": int(row["session_index"]),
        "lifecycle_state": row["lifecycle_state"],
        "state_sha256": row["state_sha256"],
        "registered_class_count": len(row["registered_classes"]),
        "registered_classes": list(row["registered_classes"]),
        "cumulative_support_token_sha256": _sha256_values(row["cumulative_support_tokens"]),
        "query_token_sha256": _sha256_values(row["query_tokens"]),
        "query_scenario_sha256": _sha256_values(row["scenarios"]),
        "package_identity_sha256": package_digest,
        "prediction_artifact_sha256": prediction_sha256,
        "fit_audit_sha256": fit_sha256,
        "resource_audit_sha256": resource_sha256,
        **{name: False for name in _ZERO_QUERY_FIELDS},
    }
    receipt_sha256 = _write_json_new(destination / "execution_receipt.json", receipt)
    members = [
        {
            "relative_path": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(destination.iterdir(), key=lambda item: item.name)
    ]
    commit = {
        "schema": "cvs.phase2.d92_e0_continuous_session.commit.v1",
        "members": members,
        "artifact_root_sha256": hashlib.sha256(_canonical_bytes(members)).hexdigest(),
        "prediction_artifact_sha256": prediction_sha256,
        "execution_receipt_sha256": receipt_sha256,
    }
    commit_sha256 = _write_json_new(destination / "COMMIT.json", commit)
    for path in destination.iterdir():
        if not _readonly(path):
            raise ContinuousSessionPredictionError("session output is not immutable")
    return {
        "schedule": row["schedule"],
        "session_index": int(row["session_index"]),
        "lifecycle_state": row["lifecycle_state"],
        "output_root": str(destination.resolve()),
        "state_sha256": row["state_sha256"],
        "prediction_artifact_sha256": prediction_sha256,
        "fit_audit_sha256": fit_sha256,
        "resource_audit_sha256": resource_sha256,
        "execution_receipt_sha256": receipt_sha256,
        "commit_sha256": commit_sha256,
        "query_token_sha256": _sha256_values(row["query_tokens"]),
        "cumulative_support_token_count": len(row["cumulative_support_tokens"]),
        "registered_class_count": len(row["registered_classes"]),
        **dict(row["resource"]),
    }


def _seal_session_state(
    root: Path, row: Mapping[str, Any], *, package_digest: str
) -> dict[str, Any]:
    """Persist support-only fit/state receipts before any apply package opens."""

    destination = _session_destination(root, row)
    destination.mkdir(parents=True, exist_ok=False)
    fit = {
        **dict(row["audit"]),
        "state_sha256": row["state_sha256"],
        "registered_classes": list(row["registered_classes"]),
        "old_class_count": int(row["old_class_count"]),
        "arriving_support_token_sha256": _sha256_values(row["arriving_support_tokens"]),
        "cumulative_support_token_sha256": _sha256_values(row["cumulative_support_tokens"]),
        "package_identity_sha256": package_digest,
        "phase": "SUPPORT_STATE_SEALED_BEFORE_APPLY_OPEN",
    }
    fit_sha256 = _write_json_new(destination / "fit_audit.json", fit)
    resource_sha256 = _write_json_new(
        destination / "resource_audit.json", dict(row["resource"])
    )
    if not _readonly(destination / "fit_audit.json") or not _readonly(
        destination / "resource_audit.json"
    ):
        raise ContinuousSessionPredictionError("support state receipt is not immutable")
    return {
        **dict(row),
        "_phase_a_destination": destination,
        "_phase_a_fit_sha256": fit_sha256,
        "_phase_a_resource_sha256": resource_sha256,
    }


def _finalize_session_prediction(
    row: Mapping[str, Any], *, package_digest: str
) -> dict[str, Any]:
    """Add only unlabeled F0 predictions to a previously sealed state receipt."""

    destination = row.get("_phase_a_destination")
    if not isinstance(destination, Path) or destination.is_symlink() or not destination.is_dir():
        raise ContinuousSessionPredictionError("support state destination drift")
    fit_path = destination / "fit_audit.json"
    resource_path = destination / "resource_audit.json"
    if (
        not _readonly(fit_path)
        or not _readonly(resource_path)
        or _sha256_file(fit_path) != row.get("_phase_a_fit_sha256")
        or _sha256_file(resource_path) != row.get("_phase_a_resource_sha256")
    ):
        raise ContinuousSessionPredictionError("support state receipt changed before prediction")
    query_tokens = _string_vector(row.get("query_tokens"), label="query tokens")
    scenarios = _string_vector(row.get("scenarios"), label="query scenarios")
    predictions = _string_vector(
        row.get("predictions"), label="predicted class handles"
    )
    if len(query_tokens) != len(scenarios) or len(query_tokens) != len(predictions):
        raise ContinuousSessionPredictionError("F0 prediction length drift")
    if any(value not in row["registered_classes"] for value in predictions):
        raise ContinuousSessionPredictionError("F0 prediction is outside registered classes")
    prediction_sha256 = _write_npz_new(
        destination / "prediction_artifact.npz",
        query_tokens=np.asarray(query_tokens).astype(str),
        scenarios=np.asarray(scenarios).astype(str),
        predicted_class_handles=np.asarray(predictions).astype(str),
    )
    receipt = {
        "schema": SCHEMA,
        "status": "D92_E0_CONTINUOUS_SESSION_TRUTH_FREE_PREDICTION_COMPLETE",
        "method_id": METHOD_ID,
        "schedule": row["schedule"],
        "session_index": int(row["session_index"]),
        "lifecycle_state": row["lifecycle_state"],
        "state_sha256": row["state_sha256"],
        "registered_class_count": len(row["registered_classes"]),
        "registered_classes": list(row["registered_classes"]),
        "cumulative_support_token_sha256": _sha256_values(row["cumulative_support_tokens"]),
        "query_token_sha256": _sha256_values(query_tokens),
        "query_scenario_sha256": _sha256_values(scenarios),
        "package_identity_sha256": package_digest,
        "prediction_artifact_sha256": prediction_sha256,
        "fit_audit_sha256": row["_phase_a_fit_sha256"],
        "resource_audit_sha256": row["_phase_a_resource_sha256"],
        "query_prediction_wall_time_ns": int(row.get("query_prediction_wall_time_ns", 0)),
        **({"scene": row["scene"]} if row.get("scene") is not None else {}),
        **{name: False for name in _ZERO_QUERY_FIELDS},
    }
    receipt_sha256 = _write_json_new(destination / "execution_receipt.json", receipt)
    members = [
        {
            "relative_path": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(destination.iterdir(), key=lambda item: item.name)
    ]
    commit = {
        "schema": "cvs.phase2.d92_e0_continuous_session.commit.v1",
        "members": members,
        "artifact_root_sha256": hashlib.sha256(_canonical_bytes(members)).hexdigest(),
        "prediction_artifact_sha256": prediction_sha256,
        "execution_receipt_sha256": receipt_sha256,
    }
    commit_sha256 = _write_json_new(destination / "COMMIT.json", commit)
    if any(not _readonly(path) for path in destination.iterdir()):
        raise ContinuousSessionPredictionError("completed session output is not immutable")
    return {
        "scene": row.get("scene"),
        "schedule": row["schedule"],
        "session_index": int(row["session_index"]),
        "lifecycle_state": row["lifecycle_state"],
        "output_root": str(destination.resolve()),
        "state_sha256": row["state_sha256"],
        "prediction_artifact_sha256": prediction_sha256,
        "fit_audit_sha256": row["_phase_a_fit_sha256"],
        "resource_audit_sha256": row["_phase_a_resource_sha256"],
        "execution_receipt_sha256": receipt_sha256,
        "commit_sha256": commit_sha256,
        "query_token_sha256": _sha256_values(query_tokens),
        "cumulative_support_token_count": len(row["cumulative_support_tokens"]),
        "registered_class_count": len(row["registered_classes"]),
        **dict(row["resource"]),
    }


def run_continuous_session_prediction(
    *,
    before_enrollment_package_root: str | Path,
    before_enrollment_seal_path: str | Path,
    before_enrollment_seal_sha256: str | None,
    before_apply_package_root: str | Path,
    before_apply_seal_path: str | Path,
    before_apply_seal_sha256: str | None,
    after_apply_package_root: str | Path,
    after_apply_seal_path: str | Path,
    after_apply_seal_sha256: str | None,
    prepared_delta_root: str | Path,
    ground_component_dir: str | Path,
    ground_manifest_sha256: str,
    schedules: Mapping[str, Any],
    output_root: str | Path,
    device: str,
    job: Mapping[str, Any] | None = None,
    session_builder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build and seal one truth-free prediction trajectory per frozen schedule.

    ``prepared_delta_root`` contains independently sealed one-class support
    deltas.  The complete after-enrollment package is deliberately absent from
    this runtime API.  ``session_builder`` remains a narrow test seam only.
    """

    root = Path(output_root)
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise FileExistsError("continuous-session prediction output already exists")
    normalized = _normalized_schedules(schedules)
    package_rows = _packages(
        before_enrollment_package_root=before_enrollment_package_root,
        before_enrollment_seal_path=before_enrollment_seal_path,
        before_enrollment_seal_sha256=before_enrollment_seal_sha256,
        before_apply_package_root=before_apply_package_root,
        before_apply_seal_path=before_apply_seal_path,
        before_apply_seal_sha256=before_apply_seal_sha256,
        after_apply_package_root=after_apply_package_root,
        after_apply_seal_path=after_apply_seal_path,
        after_apply_seal_sha256=after_apply_seal_sha256,
    )
    _sha256_text("ground manifest", ground_manifest_sha256)
    if not isinstance(device, str) or not device:
        raise ContinuousSessionPredictionError("device is invalid")
    if not isinstance(prepared_delta_root, (str, Path)) or not str(prepared_delta_root):
        raise ContinuousSessionPredictionError("prepared delta root is invalid")
    if session_builder is None:
        return _run_real_continuous_session_prediction(
            root=_claim_empty_output_root(root),
            package_rows=package_rows,
            prepared_delta_root=prepared_delta_root,
            ground_component_dir=ground_component_dir,
            ground_manifest_sha256=ground_manifest_sha256,
            schedules=normalized,
            device=device,
            job=job,
        )
    builder = session_builder
    if not callable(builder):
        raise ContinuousSessionPredictionError("session_builder must be callable")
    baseline_row, prepared = _prepare_sessions(
        schedules=normalized,
        packages=package_rows,
        ground_component_dir=ground_component_dir,
        ground_manifest_sha256=ground_manifest_sha256,
        device=device,
        session_builder=builder,
    )
    root = _claim_empty_output_root(root)
    package_digest = hashlib.sha256(_canonical_bytes(package_rows)).hexdigest()
    baseline = _publish_session(root, baseline_row, package_digest=package_digest)
    published: dict[str, dict[str, Any]] = {}
    for schedule_name, rows in prepared.items():
        sessions = [
            _publish_session(root, row, package_digest=package_digest)
            for row in rows
        ]
        published[schedule_name] = {
            "increments": list(normalized[schedule_name]["increments"]),
            "arrival_order": list(normalized[schedule_name]["arrival_order"]),
            "sessions": sessions,
        }
    manifest = {
        "schema": SCHEMA,
        "status": "D92_E0_CONTINUOUS_SESSION_TRUTH_FREE_PREDICTIONS_COMPLETE",
        "method_id": METHOD_ID,
        "output_root": str(root.resolve()),
        "package_identity_sha256": package_digest,
        "future_support_open_sentinel": 0,
        **{name: False for name in _ZERO_QUERY_FIELDS},
        "DA1_REG0": baseline,
        "schedules": published,
    }
    manifest_sha256 = _write_json_new(root / "prediction_manifest.json", manifest)
    return {**manifest, "prediction_manifest_sha256": manifest_sha256}


__all__ = [
    "ContinuousSessionPredictionError",
    "METHOD_ID",
    "SCHEMA",
    "prepare_continuous_session_support_deltas",
    "run_continuous_session_prediction",
]
