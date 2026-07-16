"""External, byte-grounded LEO_weak lineage receipts for SOMP-H.

This module runs outside the Phase2 predictor boundary.  It reads the Phase1
cache-set manifest, the three cache archives, the exporter source, the build
spec, and the channel-code closure from caller-supplied paths.  Every digest in
the receipt is either recomputed from those bytes or from sample-level cache
content; a 64-hex manifest declaration is never sufficient by itself.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping

import numpy as np

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA,
    LEO_WEAK_CACHE_SET_SCHEMA,
    LEO_WEAK_CACHE_STAGE,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    overlay_id,
    physical_sample_id,
    post_channel_iq_sha256,
)
from cvsrffi.stage2_predictor_bundle import (
    PredictorPackageError,
    _hash_handle,
    _json_from_handle,
    _zip_members_from_handle,
    canonical_json_bytes,
    open_regular_member_same_fd,
    sha256_bytes,
)


LINEAGE_RECEIPT_SCHEMA = "cvs.phase2.somph_leo_weak_lineage_receipt.v2"
LINEAGE_SEAL_SCHEMA = "cvs.phase2.somph_leo_weak_lineage_detached_seal.v1"
CHANNEL_CODE_CLOSURE_SCHEMA = "cvs.phase1.leo_weak_channel_code_closure.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_LOGICAL_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_CACHE_MEMBERS = (
    "leo_weak_iq",
    "raw_labels",
    "domain_labels",
    "tx_ids",
    "rx_ids",
    "day_ids",
    "eq_ids",
    "sig_ids",
    "source_dataset_sha256",
    "source_record_indices",
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
_OPTIONAL_SPLIT_MEMBERS = ("split_partition", "split_rank")
_SCOPE_ROLES = {
    "source_train": {"source"},
    "source_validation": {"source"},
    "stage2_target_old": {"target_old"},
    "stage2_registered": {"target_old", "target_new"},
}
_CACHE_SET_KEYS = {
    "schema",
    "artifact_stage",
    "cache_set_id",
    "cache_scope",
    "phase2_sample_view_policy",
    "clean_sample_access",
    "clean_derived_signal_access",
    "target_channel_view",
    "target_channel_scenarios",
    "output_roles",
    "cache_npz_by_scenario",
    "cache_sha256_by_scenario",
    "cache_audits",
    "phase2_physical_sample_observation_policy",
    "phase2_cross_scenario_physical_sample_reuse",
    "phase2_additional_leo_channel_state_generation",
    "phase2_post_reception_equalization_augmentation_transform_allowed",
    "phase2_post_reception_view_from_fixed_received_iq_only",
    "phase2_post_reception_view_counts_as_additional_physical_sample",
    "phase2_physical_sample_root_id_policy",
    "phase2_query_post_reception_view_fit_access",
    "physical_sample_scenario_assignment_policy",
    "physical_sample_ids_sha256_by_scenario",
    "physical_sample_scenario_assignment_sha256",
    "builder_sha256",
    "build_spec_sha256",
    "build_spec_path_exposed_to_phase2",
}
_CACHE_MANIFEST_KEYS = {
    "schema",
    "artifact_stage",
    "phase2_sample_view_policy",
    "clean_sample_access",
    "clean_derived_signal_access",
    "contains_post_channel_iq_only",
    "contains_clean_rows",
    "target_channel_view",
    "target_channel_scenarios",
    "scenario",
    "iq_array_key",
    "raw_or_clean_iq_key_present",
    "overlay_applied_before_phase2",
    "star_ground_channel_impl",
    "channel_model",
    "channel_config",
    "channel_config_sha256",
    "builder_sha256",
    "build_spec_sha256",
    "output_roles",
    "role_satellite_seeds",
    "role_inputs",
    "row_count",
    "physical_sample_ids_sha256",
    "post_channel_iq_sha256_root",
    "overlay_ids_sha256",
    "channel_meta_keys",
    "sample_overlay_provenance_fields",
    "phase2_physical_sample_observation_policy",
    "phase2_cross_scenario_physical_sample_reuse",
    "phase2_additional_leo_channel_state_generation",
    "phase2_post_reception_equalization_augmentation_transform_allowed",
    "phase2_post_reception_view_from_fixed_received_iq_only",
    "phase2_post_reception_view_counts_as_additional_physical_sample",
    "phase2_physical_sample_root_id_policy",
    "phase2_query_post_reception_view_fit_access",
    "physical_sample_scenario_assignment_policy",
}
_CACHE_CONTRACT = {
    "schema": LEO_WEAK_CACHE_SCHEMA,
    "artifact_stage": LEO_WEAK_CACHE_STAGE,
    "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
    "clean_sample_access": False,
    "clean_derived_signal_access": False,
    "contains_post_channel_iq_only": True,
    "contains_clean_rows": False,
    "target_channel_view": "leo_weak_only",
    "iq_array_key": "leo_weak_iq",
    "raw_or_clean_iq_key_present": False,
    "overlay_applied_before_phase2": True,
    "star_ground_channel_impl": "simplified_leo_residual",
    "channel_model": "leo_residual",
}


class SomphLineageError(PredictorPackageError):
    """Raised when external LEO_weak lineage cannot be byte-verified."""


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SomphLineageError(f"{field} must be a lowercase SHA256")
    return value


def _require_scenario_map(value: Mapping[str, Any], *, field: str) -> dict[str, str]:
    result = dict(value)
    if tuple(result) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphLineageError(f"{field} must use the exact formal scenario order")
    return {
        scenario: _require_sha256(result[scenario], field=f"{field}.{scenario}")
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }


@contextmanager
def _open_external_same_fd(path: str | Path) -> Iterator[BinaryIO]:
    candidate = Path(path)
    if candidate.name in {"", ".", ".."}:
        raise SomphLineageError("external artifact path must name one file")
    parent = candidate.parent.resolve()
    try:
        with open_regular_member_same_fd(parent, candidate.name) as handle:
            yield handle
    except PredictorPackageError as exc:
        raise SomphLineageError(str(exc)) from exc


def _hash_external(path: str | Path) -> tuple[str, int]:
    with _open_external_same_fd(path) as handle:
        return _hash_handle(handle)


def _json_external(path: str | Path, *, context: str) -> tuple[dict[str, Any], str, int]:
    with _open_external_same_fd(path) as handle:
        digest, size = _hash_handle(handle)
        payload = _json_from_handle(handle, context=context)
    return payload, digest, size


def _resolve_cache(manifest_path: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SomphLineageError("cache-set path entry must be a nonempty string")
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate


def _embedded_manifest(value: np.ndarray, *, scenario: str) -> dict[str, Any]:
    array = np.asarray(value)
    if array.size != 1 or array.dtype == object:
        raise SomphLineageError(f"cache manifest_json scalar drift for {scenario}")
    raw = array.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        payload = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise SomphLineageError(
            f"cache manifest_json invalid for {scenario}"
        ) from exc
    if not isinstance(payload, dict):
        raise SomphLineageError(f"cache manifest_json object drift for {scenario}")
    return payload


def _channel_code_closure(
    members: Mapping[str, str | Path],
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(members, Mapping) or not members:
        raise SomphLineageError("channel_code_members must be a nonempty mapping")
    descriptors: list[dict[str, Any]] = []
    for logical_name in sorted(members):
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or _SAFE_LOGICAL_NAME_RE.fullmatch(logical_name) is None
            or any(
                token in logical_name.lower()
                for token in ("clean", "raw", "dataset", "truth", "scorer")
            )
        ):
            raise SomphLineageError("channel code logical name is unsafe")
        digest, size = _hash_external(members[logical_name])
        descriptors.append(
            {
                "logical_name": logical_name,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    payload = {
        "schema": CHANNEL_CODE_CLOSURE_SCHEMA,
        "members": descriptors,
    }
    return sha256_bytes(canonical_json_bytes(payload)), descriptors


def _validate_cache_set(payload: dict[str, Any], *, expected_scope: str) -> set[str]:
    if set(payload) != _CACHE_SET_KEYS:
        raise SomphLineageError("cache-set manifest exact schema drift")
    expected = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_scope": expected_scope,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "build_spec_path_exposed_to_phase2": False,
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_equalization_augmentation_transform_allowed": True,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_physical_sample_root_id_policy": (
            "immutable_preoverlay_lineage_token"
        ),
        "phase2_query_post_reception_view_fit_access": False,
        "physical_sample_scenario_assignment_policy": (
            "disjoint_preoverlay_tx_day_stratified_v1"
        ),
    }
    failed = [
        key
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if failed:
        raise SomphLineageError(f"cache-set contract failed: {failed}")
    expected_roles = _SCOPE_ROLES.get(expected_scope)
    if expected_roles is None:
        raise SomphLineageError("cache-set scope is not a supported formal scope")
    if (
        not isinstance(payload.get("output_roles"), list)
        or set(str(value) for value in payload["output_roles"]) != expected_roles
        or len(payload["output_roles"]) != len(expected_roles)
    ):
        raise SomphLineageError("cache-set output role contract drift")
    for field in ("cache_npz_by_scenario", "cache_sha256_by_scenario"):
        values = payload.get(field)
        if not isinstance(values, dict) or tuple(values) != FORMAL_LEO_WEAK_SCENARIOS:
            raise SomphLineageError(f"cache-set {field} scenario order drift")
    return expected_roles


def _load_and_verify_cache(
    path: Path,
    *,
    scenario: str,
    expected_cache_sha256: str,
    exporter_sha256: str,
    build_spec_sha256: str,
    expected_channel_config_sha256: str,
    expected_physical_sample_ids_sha256: str,
    expected_post_channel_iq_sha256_root: str,
    expected_overlay_ids_sha256: str,
    expected_roles: set[str],
) -> tuple[list[str], dict[str, Any]]:
    with _open_external_same_fd(path) as handle:
        actual_sha256, size_bytes = _hash_handle(handle)
        if actual_sha256 != expected_cache_sha256:
            raise SomphLineageError(f"external cache SHA mismatch for {scenario}")
        members = _zip_members_from_handle(handle, context=f"LEO cache:{scenario}")
        allowed_members = {_CACHE_MEMBERS, _CACHE_MEMBERS + _OPTIONAL_SPLIT_MEMBERS}
        if members not in allowed_members:
            raise SomphLineageError(f"cache NPZ exact member allowlist drift for {scenario}")
        handle.seek(0)
        with np.load(handle, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in members}
        ending_sha256, ending_size = _hash_handle(handle)
        if (ending_sha256, ending_size) != (actual_sha256, size_bytes):
            raise SomphLineageError(
                f"external cache bytes changed during verification for {scenario}"
            )

    manifest = _embedded_manifest(arrays.pop("manifest_json"), scenario=scenario)
    expected_manifest_keys = set(_CACHE_MANIFEST_KEYS)
    if members == _CACHE_MEMBERS + _OPTIONAL_SPLIT_MEMBERS:
        expected_manifest_keys.add("offline_split_partition_policy")
    if set(manifest) != expected_manifest_keys:
        raise SomphLineageError(f"cache embedded manifest exact schema drift for {scenario}")
    failed = [key for key, expected in _CACHE_CONTRACT.items() if manifest.get(key) != expected]
    if failed:
        raise SomphLineageError(f"cache contract failed for {scenario}: {failed}")
    if (
        manifest.get("scenario") != scenario
        or manifest.get("target_channel_scenarios") != [scenario]
    ):
        raise SomphLineageError(f"cache scenario binding drift for {scenario}")
    if manifest.get("builder_sha256") != exporter_sha256:
        raise SomphLineageError(f"cache exporter SHA mismatch for {scenario}")
    if manifest.get("build_spec_sha256") != build_spec_sha256:
        raise SomphLineageError(f"cache build-spec SHA mismatch for {scenario}")
    channel_config = manifest.get("channel_config")
    if (
        not isinstance(channel_config, dict)
        or canonical_json_sha256(channel_config) != expected_channel_config_sha256
        or manifest.get("channel_config_sha256") != expected_channel_config_sha256
    ):
        raise SomphLineageError(f"cache channel-config trust root mismatch for {scenario}")

    iq = np.asarray(arrays["leo_weak_iq"])
    if iq.dtype != np.float32 or iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[0] < 1:
        raise SomphLineageError(f"cache IQ shape/dtype drift for {scenario}")
    row_count = int(iq.shape[0])
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype == object or array.ndim < 1 or int(array.shape[0]) != row_count:
            raise SomphLineageError(f"cache row/dtype drift for {scenario}:{name}")
    if manifest.get("row_count") != row_count:
        raise SomphLineageError(f"cache row-count manifest drift for {scenario}")
    if "split_partition" in arrays:
        if set(np.asarray(arrays["split_partition"]).astype(str).tolist()) != {
            "support_pool",
            "query",
        }:
            raise SomphLineageError(f"cache split partition drift for {scenario}")
        if np.any(np.asarray(arrays["split_rank"], dtype=np.int64) < 0):
            raise SomphLineageError(f"cache split rank drift for {scenario}")
        if manifest.get("offline_split_partition_policy") != "legacy_seeded_nested_exact":
            raise SomphLineageError(f"cache split policy drift for {scenario}")

    scenarios = np.asarray(arrays["sat_scenarios"]).astype(str)
    views = np.asarray(arrays["channel_views"]).astype(str)
    applied = np.asarray(arrays["overlay_applied"])
    if (
        applied.dtype != np.bool_
        or not bool(np.all(applied))
        or not bool(np.all(scenarios == scenario))
        or not bool(np.all(views == "rx_base"))
    ):
        raise SomphLineageError(f"cache row-level LEO overlay contract drift for {scenario}")
    seeds = np.asarray(arrays["satellite_seeds"])
    if seeds.dtype != np.int64:
        raise SomphLineageError(f"cache satellite seed dtype drift for {scenario}")
    roles = np.asarray(arrays["dataset_role"]).astype(str)
    observed_roles = set(roles.tolist())
    if (
        observed_roles != expected_roles
        or set(str(value) for value in manifest.get("output_roles", []))
        != expected_roles
    ):
        raise SomphLineageError(f"cache output role drift for {scenario}")
    role_seed_map = manifest.get("role_satellite_seeds")
    if (
        not isinstance(role_seed_map, dict)
        or set(str(value) for value in role_seed_map) != expected_roles
    ):
        raise SomphLineageError(f"cache role seed map drift for {scenario}")
    for role in expected_roles:
        role_seeds = set(int(value) for value in seeds[roles == role].tolist())
        if role_seeds != {int(role_seed_map[role])}:
            raise SomphLineageError(f"cache role satellite seed drift for {scenario}")
    if manifest.get("sample_overlay_provenance_fields") != [
        "sample_ids",
        "source_dataset_sha256",
        "source_record_indices",
        "sat_scenarios",
        "satellite_seeds",
        "post_channel_iq_sha256",
        "overlay_ids",
    ]:
        raise SomphLineageError(f"cache provenance field order drift for {scenario}")

    physical_ids: list[str] = []
    iq_hashes: list[str] = []
    overlay_ids: list[str] = []
    for index in range(row_count):
        sample_id = physical_sample_id(arrays, index)
        iq_digest = post_channel_iq_sha256(iq[index])
        overlay_digest = overlay_id(
            sample_id=sample_id,
            scenario=scenario,
            satellite_seed=int(seeds[index]),
            channel_config_sha256=expected_channel_config_sha256,
            iq_sha256=iq_digest,
        )
        physical_ids.append(sample_id)
        iq_hashes.append(iq_digest)
        overlay_ids.append(overlay_digest)
    if len(set(physical_ids)) != row_count:
        raise SomphLineageError(f"duplicate physical sample IDs for {scenario}")
    if np.asarray(arrays["sample_ids"]).astype(str).tolist() != physical_ids:
        raise SomphLineageError(f"physical sample ID row mismatch for {scenario}")
    if np.asarray(arrays["post_channel_iq_sha256"]).astype(str).tolist() != iq_hashes:
        raise SomphLineageError(f"post-channel IQ row digest mismatch for {scenario}")
    if np.asarray(arrays["overlay_ids"]).astype(str).tolist() != overlay_ids:
        raise SomphLineageError(f"overlay row lineage mismatch for {scenario}")

    physical_root = ids_sha256(physical_ids)
    iq_root = ids_sha256(iq_hashes)
    overlay_root = ids_sha256(overlay_ids)
    roots = {
        "physical_sample_ids_sha256": physical_root,
        "post_channel_iq_sha256_root": iq_root,
        "overlay_ids_sha256": overlay_root,
    }
    expected_roots = {
        "physical_sample_ids_sha256": expected_physical_sample_ids_sha256,
        "post_channel_iq_sha256_root": expected_post_channel_iq_sha256_root,
        "overlay_ids_sha256": expected_overlay_ids_sha256,
    }
    for field, expected in expected_roots.items():
        if roots[field] != expected or manifest.get(field) != expected:
            raise SomphLineageError(f"cache sample root mismatch for {scenario}:{field}")
    return physical_ids, {
        "cache_sha256": actual_sha256,
        "cache_size_bytes": size_bytes,
        "cache_manifest_sha256": canonical_json_sha256(manifest),
        "channel_config_sha256": expected_channel_config_sha256,
        **roots,
        "row_count": row_count,
        "zip_member_crc_and_bounds_check": "PASS",
        "sample_level_overlay_recompute": "PASS",
    }


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite lineage artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite lineage artifact: {path}")
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_somph_leo_weak_lineage_seal(
    cache_set_manifest_path: str | Path,
    *,
    expected_scope: str,
    expected_cache_set_manifest_sha256: str,
    expected_cache_sha256_by_scenario: Mapping[str, str],
    exporter_path: str | Path,
    expected_exporter_sha256: str,
    build_spec_path: str | Path,
    expected_build_spec_sha256: str,
    channel_code_members: Mapping[str, str | Path],
    expected_channel_code_closure_sha256: str,
    expected_channel_config_sha256_by_scenario: Mapping[str, str],
    expected_physical_sample_ids_sha256_by_scenario: Mapping[str, str],
    expected_physical_sample_scenario_assignment_sha256: str,
    expected_post_channel_iq_sha256_root_by_scenario: Mapping[str, str],
    expected_overlay_ids_sha256_by_scenario: Mapping[str, str],
    receipt_path: str | Path,
    detached_seal_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify external Phase1 bytes and atomically create a detached receipt."""

    expected_set_sha = _require_sha256(
        expected_cache_set_manifest_sha256,
        field="expected_cache_set_manifest_sha256",
    )
    expected_caches = _require_scenario_map(
        expected_cache_sha256_by_scenario,
        field="expected_cache_sha256_by_scenario",
    )
    expected_exporter = _require_sha256(
        expected_exporter_sha256, field="expected_exporter_sha256"
    )
    expected_build_spec = _require_sha256(
        expected_build_spec_sha256, field="expected_build_spec_sha256"
    )
    expected_channel_closure = _require_sha256(
        expected_channel_code_closure_sha256,
        field="expected_channel_code_closure_sha256",
    )
    expected_channel_configs = _require_scenario_map(
        expected_channel_config_sha256_by_scenario,
        field="expected_channel_config_sha256_by_scenario",
    )
    expected_physical_roots = _require_scenario_map(
        expected_physical_sample_ids_sha256_by_scenario,
        field="expected_physical_sample_ids_sha256_by_scenario",
    )
    expected_assignment_root = _require_sha256(
        expected_physical_sample_scenario_assignment_sha256,
        field="expected_physical_sample_scenario_assignment_sha256",
    )
    expected_iq_roots = _require_scenario_map(
        expected_post_channel_iq_sha256_root_by_scenario,
        field="expected_post_channel_iq_sha256_root_by_scenario",
    )
    expected_overlay_roots = _require_scenario_map(
        expected_overlay_ids_sha256_by_scenario,
        field="expected_overlay_ids_sha256_by_scenario",
    )

    manifest_path = Path(cache_set_manifest_path)
    cache_set, set_sha, set_size = _json_external(
        manifest_path, context="SOMP-H LEO_weak cache-set manifest"
    )
    if set_sha != expected_set_sha:
        raise SomphLineageError("external cache-set manifest SHA mismatch")
    expected_roles = _validate_cache_set(cache_set, expected_scope=expected_scope)

    exporter_sha, exporter_size = _hash_external(exporter_path)
    if exporter_sha != expected_exporter:
        raise SomphLineageError("external exporter SHA mismatch")
    build_spec, _build_spec_file_sha, build_spec_size = _json_external(
        build_spec_path, context="LEO_weak build spec"
    )
    actual_build_spec_sha = canonical_json_sha256(build_spec)
    if actual_build_spec_sha != expected_build_spec:
        raise SomphLineageError("external canonical build-spec SHA mismatch")
    channel_closure_sha, channel_descriptors = _channel_code_closure(
        channel_code_members
    )
    if channel_closure_sha != expected_channel_closure:
        raise SomphLineageError("external channel-code closure SHA mismatch")
    if (
        cache_set.get("builder_sha256") != exporter_sha
        or cache_set.get("build_spec_sha256") != actual_build_spec_sha
    ):
        raise SomphLineageError("cache-set exporter/build-spec lineage mismatch")

    declared_caches = dict(cache_set["cache_sha256_by_scenario"])
    cache_paths = dict(cache_set["cache_npz_by_scenario"])
    scenario_receipts: dict[str, Any] = {}
    physical_ids_by_scenario: dict[str, list[str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        if declared_caches.get(scenario) != expected_caches[scenario]:
            raise SomphLineageError(
                f"cache-set declared SHA does not match external trust root for {scenario}"
            )
        physical_ids, cache_receipt = _load_and_verify_cache(
            _resolve_cache(manifest_path, cache_paths[scenario]),
            scenario=scenario,
            expected_cache_sha256=expected_caches[scenario],
            exporter_sha256=exporter_sha,
            build_spec_sha256=actual_build_spec_sha,
            expected_channel_config_sha256=expected_channel_configs[scenario],
            expected_physical_sample_ids_sha256=expected_physical_roots[scenario],
            expected_post_channel_iq_sha256_root=expected_iq_roots[scenario],
            expected_overlay_ids_sha256=expected_overlay_roots[scenario],
            expected_roles=expected_roles,
        )
        physical_ids_by_scenario[scenario] = physical_ids
        scenario_receipts[scenario] = cache_receipt
    observed: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        overlap = observed.intersection(physical_ids_by_scenario[scenario])
        if overlap:
            raise SomphLineageError(
                "physical samples are reused across LEO_weak scenarios"
            )
        observed.update(physical_ids_by_scenario[scenario])
    assignment_root = canonical_json_sha256(physical_ids_by_scenario)
    if (
        cache_set.get("physical_sample_ids_sha256_by_scenario")
        != expected_physical_roots
        or cache_set.get("physical_sample_scenario_assignment_sha256")
        != expected_assignment_root
        or assignment_root != expected_assignment_root
    ):
        raise SomphLineageError("cache-set physical scenario assignment root mismatch")

    receipt = {
        "schema": LINEAGE_RECEIPT_SCHEMA,
        "status": "BYTE_GROUNDED_SELF_CONSISTENCY_PASS",
        "cache_scope": expected_scope,
        "scenario_order": list(FORMAL_LEO_WEAK_SCENARIOS),
        "cache_set_manifest_sha256": set_sha,
        "cache_set_manifest_size_bytes": set_size,
        "exporter_sha256": exporter_sha,
        "exporter_size_bytes": exporter_size,
        "build_spec_sha256": actual_build_spec_sha,
        "build_spec_size_bytes": build_spec_size,
        "channel_code_closure_sha256": channel_closure_sha,
        "channel_code_members": channel_descriptors,
        "physical_sample_ids_sha256_by_scenario": expected_physical_roots,
        "physical_sample_scenario_assignment_sha256": expected_assignment_root,
        "scenario_receipts": scenario_receipts,
        "same_fd_nofollow_read": True,
        "npz_member_crc_size_ratio_audit": "PASS",
        "cross_scenario_physical_disjointness_audit": "PASS",
        "single_observation_contract_audit": "PASS",
        "sample_level_overlay_recompute": "PASS",
        "manifest_hex_self_declaration_sufficient": False,
        "external_authority_lock_verified": False,
        "contains_build_spec_or_dataset_paths": False,
        "formal_launch_authority": False,
    }
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    seal = {
        "schema": LINEAGE_SEAL_SCHEMA,
        "receipt_sha256": sha256_bytes(receipt_bytes),
        "receipt_size_bytes": len(receipt_bytes),
        "lineage_root_sha256": sha256_bytes(canonical_json_bytes(receipt)),
    }
    receipt_output = Path(receipt_path)
    seal_output = Path(detached_seal_path)
    if receipt_output.resolve(strict=False) == seal_output.resolve(strict=False):
        raise SomphLineageError("receipt and detached seal paths must be distinct")
    if receipt_output.exists() or seal_output.exists():
        raise FileExistsError("refusing to overwrite lineage receipt or detached seal")
    _atomic_write_new(receipt_output, receipt_bytes)
    try:
        _atomic_write_new(seal_output, canonical_json_bytes(seal) + b"\n")
    except Exception:
        receipt_output.unlink(missing_ok=True)
        raise
    os.chmod(receipt_output, 0o444)
    os.chmod(seal_output, 0o444)
    return receipt, seal


def verify_somph_leo_weak_lineage_seal(
    receipt_path: str | Path,
    detached_seal_path: str | Path,
    *,
    expected_detached_seal_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a detached seal before any consumer accepts a receipt.

    An orphan receipt is never sufficient.  This loader intentionally preserves
    ``formal_launch_authority=false`` and the explicit absence of an external
    authority lock.
    """

    expected_seal_sha = _require_sha256(
        expected_detached_seal_sha256,
        field="expected_detached_seal_sha256",
    )
    seal, actual_seal_sha, _seal_size = _json_external(
        detached_seal_path, context="SOMP-H LEO_weak detached lineage seal"
    )
    if actual_seal_sha != expected_seal_sha:
        raise SomphLineageError("external detached lineage seal SHA mismatch")
    if set(seal) != {
        "schema",
        "receipt_sha256",
        "receipt_size_bytes",
        "lineage_root_sha256",
    }:
        raise SomphLineageError("detached lineage seal exact schema drift")
    if seal.get("schema") != LINEAGE_SEAL_SCHEMA:
        raise SomphLineageError("detached lineage seal schema drift")
    receipt, receipt_sha, receipt_size = _json_external(
        receipt_path, context="SOMP-H LEO_weak lineage receipt"
    )
    if (
        receipt_sha != seal.get("receipt_sha256")
        or receipt_size != seal.get("receipt_size_bytes")
    ):
        raise SomphLineageError("detached lineage seal receipt binding mismatch")
    if receipt.get("schema") != LINEAGE_RECEIPT_SCHEMA:
        raise SomphLineageError("lineage receipt schema drift")
    if receipt.get("status") != "BYTE_GROUNDED_SELF_CONSISTENCY_PASS":
        raise SomphLineageError("lineage receipt status drift")
    if receipt.get("external_authority_lock_verified") is not False:
        raise SomphLineageError("structural receipt cannot claim an authority lock")
    if receipt.get("formal_launch_authority") is not False:
        raise SomphLineageError("lineage receipt cannot authorize launch")
    if seal.get("lineage_root_sha256") != sha256_bytes(
        canonical_json_bytes(receipt)
    ):
        raise SomphLineageError("detached lineage root digest mismatch")
    return receipt, seal
