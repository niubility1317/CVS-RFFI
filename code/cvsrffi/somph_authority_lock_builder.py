"""Build one unsigned SOMP-H authority lock from real offline bytes.

This module belongs to the Phase1/offline-controller side. It derives the
receiver, seed, TX registries and dataset paths from the sealed build spec,
recomputes every cache and sample-lineage root, and publishes an unsigned,
read-only lock package. It never grants Phase2 launch authority.
"""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cvsrffi import somph_leo_weak_lineage_seal as structural
from cvsrffi import somph_lineage_authority as authority
from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    canonical_json_sha256,
)
from cvsrffi.stage2_predictor_bundle import (
    _hash_handle,
    _zip_members_from_handle,
    canonical_json_bytes,
    sha256_bytes,
)
from training_controls import sat_channel_config_for_scenario


AUTHORITY_LOCK_BUILD_RECEIPT_SCHEMA = (
    "cvs.phase1.somph_authority_lock_build_receipt.v1"
)
AUTHORITY_LOCK_BUILD_STATUS = "UNSIGNED_OFFLINE_AUTHORITY_LOCK_BUILT"
AUTHORITY_LOCK_BUILD_RECEIPT_NAME = "authority_lock_build_receipt.json"
AUTHORITY_LOCK_BUILD_RECEIPT_KEYS = {
    "schema",
    "status",
    "cache_spec_manifest_sha256",
    "cache_spec_manifest_size_bytes",
    "cache_spec_cell_id",
    "required_samples_per_tx",
    "receiver",
    "seed",
    "cache_scope",
    "cache_set_manifest_sha256",
    "build_spec_file_sha256",
    "build_spec_canonical_sha256",
    "exporter_sha256",
    "channel_code_closure_sha256",
    "dataset_authority_root_sha256",
    "cache_role_inputs_root_sha256",
    "physical_sample_ids_sha256",
    "cache_sha256_by_scenario",
    "channel_config_sha256_by_scenario",
    "post_channel_iq_sha256_root_by_scenario",
    "overlay_ids_sha256_by_scenario",
    "cache_recompute_audits",
    "authority_lock_sha256",
    "authority_lock_canonical_sha256",
    "external_authority_lock_verified",
    "formal_launch_authority",
}
FORMAL_CACHE_SPEC_MANIFEST_SHA256 = (
    "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
)
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class SomphAuthorityLockBuildError(ValueError):
    """Raised when real inputs cannot produce a valid authority lock."""


def _file_descriptor(path: str | Path, *, field: str) -> dict[str, Any]:
    try:
        _raw, digest, size = authority._read_external_bytes(path, context=field)
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    return {
        "path": str(Path(path).absolute()),
        "sha256": digest,
        "size_bytes": size,
    }


def _build_spec_descriptor(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload, _raw, file_sha, size = authority._read_external_json(
            path, context="SOMP-H real authority build spec"
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    return (
        {
            "path": str(Path(path).absolute()),
            "file_sha256": file_sha,
            "canonical_sha256": canonical_json_sha256(payload),
            "size_bytes": size,
        },
        payload,
    )


def _derive_formal_identity(
    build_spec: Mapping[str, Any],
    *,
    build_spec_dir: Path,
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    role_specs = build_spec.get("role_specs")
    if not isinstance(role_specs, list) or not role_specs:
        raise SomphAuthorityLockBuildError("real build spec role_specs missing")
    by_role: dict[str, dict[str, Any]] = {}
    receivers: set[str] = set()
    for raw in role_specs:
        if not isinstance(raw, dict):
            raise SomphAuthorityLockBuildError(
                "real build spec role_specs must contain objects"
            )
        role = raw.get("role")
        receiver = raw.get("rxs")
        if (
            not isinstance(role, str)
            or not role
            or role in by_role
            or not isinstance(receiver, str)
            or not receiver
        ):
            raise SomphAuthorityLockBuildError(
                "real build spec role/receiver identity drift"
            )
        by_role[role] = dict(raw)
        receivers.add(receiver)
    if len(receivers) != 1:
        raise SomphAuthorityLockBuildError(
            "real build spec must use one target receiver"
        )
    scope = build_spec.get("cache_scope")
    roles = authority._expected_roles(str(scope))
    if tuple(by_role) != roles:
        raise SomphAuthorityLockBuildError(
            "real build spec role ordering does not match cache scope"
        )
    receiver = next(iter(receivers))
    seed = build_spec.get("dataset_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SomphAuthorityLockBuildError("real build spec dataset_seed invalid")
    old_tx = authority._split_tx_ids(
        by_role["target_old"].get("tx_ids"),
        field="build_spec.target_old.tx_ids",
    )
    new_tx = (
        authority._split_tx_ids(
            by_role["target_new"].get("tx_ids"),
            field="build_spec.target_new.tx_ids",
        )
        if "target_new" in by_role
        else []
    )
    identity = {
        "receiver": receiver,
        "seed": seed,
        "cache_scope": scope,
        "old_tx_ids": old_tx,
        "new_tx_ids": new_tx,
    }
    try:
        authority._validate_lock_formal_identity(identity)
        checked_roles = authority._validate_build_spec(
            dict(build_spec),
            lock=identity,
            receiver=receiver,
            seed=seed,
            roles=roles,
            build_spec_dir=build_spec_dir,
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    return identity, roles, checked_roles


def _resolve_relative_to(base: Path, raw: Any, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise SomphAuthorityLockBuildError(f"{field} path missing")
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    return path.absolute()


def _locked_cache_spec_cell(
    cache_spec_manifest_path: str | Path,
    *,
    cell_id: str,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, str, int]:
    manifest_path = Path(cache_spec_manifest_path).absolute()
    try:
        manifest, _raw, manifest_sha, manifest_size = (
            authority._read_external_json(
                manifest_path,
                context="SOMP-H locked cache-spec manifest",
            )
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    if manifest_sha != expected_manifest_sha256:
        raise SomphAuthorityLockBuildError(
            "locked cache-spec manifest SHA mismatch"
        )
    required_samples = manifest.get("required_samples_per_tx")
    support_pool_max_k = manifest.get("support_pool_max_k")
    query_samples = manifest.get("query_samples_per_tx")
    manifest_scope = manifest.get("cache_scope")
    production_manifest = (
        expected_manifest_sha256 == FORMAL_CACHE_SPEC_MANIFEST_SHA256
    )
    if (
        manifest.get("schema")
        != "cvs.phase2.somph_registered_cache_build_matrix.v1"
        or manifest_scope
        not in {"stage2_target_old", "stage2_registered"}
        or (production_manifest and manifest_scope != "stage2_registered")
        or manifest.get("phase2_sample_view_policy")
        != "leo_weak_only_no_clean_access"
        or manifest.get("formal_launch_authority") is not False
        or isinstance(required_samples, bool)
        or not isinstance(required_samples, int)
        or required_samples < 1
        or isinstance(support_pool_max_k, bool)
        or not isinstance(support_pool_max_k, int)
        or support_pool_max_k < 1
        or isinstance(query_samples, bool)
        or not isinstance(query_samples, int)
        or query_samples < 0
        or required_samples != support_pool_max_k + query_samples
        or (
            production_manifest
            and (
                required_samples != 40
                or support_pool_max_k != 20
                or query_samples != 20
            )
        )
    ):
        raise SomphAuthorityLockBuildError(
            "locked cache-spec manifest contract drift"
        )
    cells = manifest.get("cells")
    if not isinstance(cell_id, str) or not cell_id:
        raise SomphAuthorityLockBuildError("locked cache-spec cell_id missing")
    if not isinstance(cells, list):
        raise SomphAuthorityLockBuildError(
            "locked cache-spec manifest cells missing"
        )
    matches = [
        dict(item)
        for item in cells
        if isinstance(item, dict) and item.get("cell_id") == cell_id
    ]
    if len(matches) != 1:
        raise SomphAuthorityLockBuildError(
            "locked cache-spec cell_id is not unique"
        )
    cell = matches[0]
    required_cell_fields = {
        "cell_id",
        "receiver",
        "seed",
        "cache_scope",
        "cache_output_root",
        "spec_path",
        "spec_file_sha256",
        "spec_canonical_sha256",
        "required_samples_per_tx",
        "support_pool_max_k",
        "query_samples_per_tx",
    }
    if (
        not required_cell_fields.issubset(cell)
        or cell["cache_scope"] != manifest_scope
        or cell["required_samples_per_tx"] != required_samples
        or cell["support_pool_max_k"] != support_pool_max_k
        or cell["query_samples_per_tx"] != query_samples
    ):
        raise SomphAuthorityLockBuildError(
            "locked cache-spec cell contract drift"
        )
    spec_path = _resolve_relative_to(
        manifest_path.parent,
        cell["spec_path"],
        field="locked cache-spec cell spec",
    )
    return cell, manifest, spec_path, manifest_sha, manifest_size


def _embedded_manifest_and_sha(
    cache_path: Path,
    *,
    scenario: str,
) -> tuple[dict[str, Any], str, dict[str, np.ndarray]]:
    try:
        with structural._open_external_same_fd(cache_path) as handle:
            digest, _size = _hash_handle(handle)
            members = _zip_members_from_handle(
                handle, context=f"SOMP-H lock builder cache:{scenario}"
            )
            allowed = {
                structural._CACHE_MEMBERS,
                structural._CACHE_MEMBERS + structural._OPTIONAL_SPLIT_MEMBERS,
            }
            if members not in allowed:
                raise SomphAuthorityLockBuildError(
                    f"cache NPZ exact member allowlist drift for {scenario}"
                )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                manifest_json = np.array(archive["manifest_json"], copy=True)
                identity_arrays = {
                    "dataset_role": np.array(
                        archive["dataset_role"], copy=True
                    ),
                    "tx_ids": np.array(archive["tx_ids"], copy=True),
                    "rx_ids": np.array(archive["rx_ids"], copy=True),
                    "satellite_seeds": np.array(
                        archive["satellite_seeds"], copy=True
                    ),
                }
        manifest = structural._embedded_manifest(
            manifest_json, scenario=scenario
        )
    except (
        authority.PredictorPackageError,
        structural.SomphLineageError,
        KeyError,
        ValueError,
    ) as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    return manifest, digest, identity_arrays


def _expected_channel_config(
    build_spec: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, Any]:
    try:
        config = dict(sat_channel_config_for_scenario(scenario))
    except ValueError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    config.update(
        {
            "fs_hz": float(build_spec.get("sat_fs_hz", 25e6)),
            "fc_hz": float(build_spec.get("sat_fc_hz", 2.462e9)),
            "star_ground_channel_impl": "simplified_leo_residual",
        }
    )
    if config.get("channel_model") != "leo_residual":
        raise SomphAuthorityLockBuildError(
            "fixed channel code did not produce leo_residual"
        )
    return config


def _validate_identity_seed_and_coverage(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    scenario: str,
    receiver: str,
    roles: tuple[str, ...],
    tx_ids_by_role: Mapping[str, list[str]],
    base_satellite_seed: int,
    required_samples_per_tx: int,
) -> dict[str, int]:
    role_values = np.asarray(arrays["dataset_role"]).astype(str)
    tx_values = np.asarray(arrays["tx_ids"]).astype(str)
    rx_values = np.asarray(arrays["rx_ids"]).astype(str)
    seed_values = np.asarray(arrays["satellite_seeds"])
    if (
        role_values.shape != tx_values.shape
        or role_values.shape != rx_values.shape
        or role_values.shape != seed_values.shape
        or seed_values.dtype != np.int64
    ):
        raise SomphAuthorityLockBuildError(
            f"cache identity/seed shape drift for {scenario}"
        )
    expected_role_seeds = {
        role: int(base_satellite_seed) + index * 1_000_003
        for index, role in enumerate(roles)
    }
    if manifest.get("role_satellite_seeds") != expected_role_seeds:
        raise SomphAuthorityLockBuildError(
            f"cache role satellite seed/build-spec drift for {scenario}"
        )
    if not bool(np.all(rx_values == receiver)):
        raise SomphAuthorityLockBuildError(
            f"cache receiver drift for {scenario}"
        )
    expected_cells = {
        (role, tx_id, receiver)
        for role in roles
        for tx_id in tx_ids_by_role[role]
    }
    observed = Counter(
        zip(role_values.tolist(), tx_values.tolist(), rx_values.tolist())
    )
    if set(observed) != expected_cells or any(
        observed[cell] != required_samples_per_tx for cell in expected_cells
    ):
        raise SomphAuthorityLockBuildError(
            f"cache exact per-role/TX/receiver coverage drift for {scenario}"
        )
    for role in roles:
        observed_seeds = set(
            int(value) for value in seed_values[role_values == role].tolist()
        )
        if observed_seeds != {expected_role_seeds[role]}:
            raise SomphAuthorityLockBuildError(
                f"cache row satellite seed/build-spec drift for {scenario}:{role}"
            )
    return {
        role: required_samples_per_tx * len(tx_ids_by_role[role])
        for role in roles
    }


def _recompute_cache_roots(
    cache_set: dict[str, Any],
    *,
    cache_set_path: Path,
    exporter_sha256: str,
    build_spec_sha256: str,
    build_spec: Mapping[str, Any],
    receiver: str,
    roles: tuple[str, ...],
    tx_ids_by_role: Mapping[str, list[str]],
    required_samples_per_tx: int,
    expected_roles: set[str],
) -> tuple[
    dict[str, str],
    dict[str, str],
    str,
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, int],
]:
    declared_hashes = authority._scenario_sha_map(
        cache_set.get("cache_sha256_by_scenario"),
        field="cache_set.cache_sha256_by_scenario",
    )
    cache_paths = cache_set.get("cache_npz_by_scenario")
    cache_audits = cache_set.get("cache_audits")
    if (
        not isinstance(cache_paths, dict)
        or tuple(cache_paths) != FORMAL_LEO_WEAK_SCENARIOS
        or not isinstance(cache_audits, dict)
        or tuple(cache_audits) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise SomphAuthorityLockBuildError(
            "cache-set scenario path/audit map drift"
        )
    actual_hashes: dict[str, str] = {}
    channel_roots: dict[str, str] = {}
    iq_roots: dict[str, str] = {}
    overlay_roots: dict[str, str] = {}
    recomputed_audits: dict[str, dict[str, Any]] = {}
    role_row_counts: dict[str, int] | None = None
    physical_root: str | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        cache_path = structural._resolve_cache(
            cache_set_path, cache_paths[scenario]
        )
        manifest, actual_sha, identity_arrays = _embedded_manifest_and_sha(
            cache_path, scenario=scenario
        )
        if actual_sha != declared_hashes[scenario]:
            raise SomphAuthorityLockBuildError(
                f"cache-set self-declared cache SHA drift for {scenario}"
            )
        expected_channel_config = _expected_channel_config(
            build_spec, scenario=scenario
        )
        observed_channel_config = manifest.get("channel_config")
        if (
            not isinstance(observed_channel_config, dict)
            or canonical_json_sha256(observed_channel_config)
            != canonical_json_sha256(expected_channel_config)
        ):
            raise SomphAuthorityLockBuildError(
                f"cache channel_config/fixed-code drift for {scenario}"
            )
        channel_sha = canonical_json_sha256(expected_channel_config)
        scenario_counts = _validate_identity_seed_and_coverage(
            identity_arrays,
            manifest,
            scenario=scenario,
            receiver=receiver,
            roles=roles,
            tx_ids_by_role=tx_ids_by_role,
            base_satellite_seed=int(
                build_spec["satellite_seed_by_scenario"][scenario]
            ),
            required_samples_per_tx=required_samples_per_tx,
        )
        if role_row_counts is None:
            role_row_counts = scenario_counts
        elif scenario_counts != role_row_counts:
            raise SomphAuthorityLockBuildError(
                "cache role row counts drift across LEO_weak scenarios"
            )
        expected_physical = authority._require_sha256(
            manifest.get("physical_sample_ids_sha256"),
            field=f"cache manifest physical root:{scenario}",
        )
        expected_iq = authority._require_sha256(
            manifest.get("post_channel_iq_sha256_root"),
            field=f"cache manifest IQ root:{scenario}",
        )
        expected_overlay = authority._require_sha256(
            manifest.get("overlay_ids_sha256"),
            field=f"cache manifest overlay root:{scenario}",
        )
        try:
            _physical_ids, audit = structural._load_and_verify_cache(
                cache_path,
                scenario=scenario,
                expected_cache_sha256=actual_sha,
                exporter_sha256=exporter_sha256,
                build_spec_sha256=build_spec_sha256,
                expected_channel_config_sha256=channel_sha,
                expected_physical_sample_ids_sha256=expected_physical,
                expected_post_channel_iq_sha256_root=expected_iq,
                expected_overlay_ids_sha256=expected_overlay,
                expected_roles=expected_roles,
            )
        except structural.SomphLineageError as exc:
            raise SomphAuthorityLockBuildError(str(exc)) from exc
        declared_audit = cache_audits[scenario]
        if not isinstance(declared_audit, dict):
            raise SomphAuthorityLockBuildError(
                f"cache-set audit object drift for {scenario}"
            )
        expected_declared = {
            "sha256": audit["cache_sha256"],
            "scenario": scenario,
            "row_count": audit["row_count"],
            "physical_sample_ids_sha256": audit[
                "physical_sample_ids_sha256"
            ],
            "post_channel_iq_sha256_root": audit[
                "post_channel_iq_sha256_root"
            ],
            "overlay_ids_sha256": audit["overlay_ids_sha256"],
        }
        if any(
            declared_audit.get(key) != value
            for key, value in expected_declared.items()
        ):
            raise SomphAuthorityLockBuildError(
                f"cache-set audit/root drift for {scenario}"
            )
        if Path(str(declared_audit.get("path", ""))).resolve() != cache_path.resolve():
            raise SomphAuthorityLockBuildError(
                f"cache-set audit path drift for {scenario}"
            )
        current_physical = audit["physical_sample_ids_sha256"]
        if physical_root is None:
            physical_root = current_physical
        elif current_physical != physical_root:
            raise SomphAuthorityLockBuildError(
                "physical sample root drift across LEO_weak scenarios"
            )
        actual_hashes[scenario] = actual_sha
        channel_roots[scenario] = channel_sha
        iq_roots[scenario] = audit["post_channel_iq_sha256_root"]
        overlay_roots[scenario] = audit["overlay_ids_sha256"]
        recomputed_audits[scenario] = audit
    if physical_root is None:
        raise SomphAuthorityLockBuildError("cache set has no formal scenarios")
    if role_row_counts is None:
        raise SomphAuthorityLockBuildError("cache role row counts missing")
    if cache_set.get("physical_sample_ids_sha256") != physical_root:
        raise SomphAuthorityLockBuildError(
            "cache-set physical sample root drift"
        )
    return (
        actual_hashes,
        channel_roots,
        physical_root,
        iq_roots,
        overlay_roots,
        recomputed_audits,
        role_row_counts,
    )


def _channel_closure(
    members: Mapping[str, str | Path],
) -> dict[str, Any]:
    if tuple(members) != authority.CHANNEL_CODE_LOGICAL_MEMBERS:
        raise SomphAuthorityLockBuildError(
            "fixed channel code member order/allowlist drift"
        )
    try:
        closure_sha, descriptors = structural._channel_code_closure(members)
    except structural.SomphLineageError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    by_name = {item["logical_name"]: item for item in descriptors}
    closure = {
        "closure_sha256": closure_sha,
        "members": [
            {
                "logical_name": logical_name,
                "path": str(Path(members[logical_name]).resolve()),
                "sha256": by_name[logical_name]["sha256"],
                "size_bytes": by_name[logical_name]["size_bytes"],
            }
            for logical_name in authority.CHANNEL_CODE_LOGICAL_MEMBERS
        ],
    }
    try:
        authority._validate_channel_closure(closure)
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    return closure


def _dataset_descriptors(
    role_specs: list[dict[str, Any]],
    *,
    roles: tuple[str, ...],
    receiver: str,
    seed: int,
) -> tuple[list[dict[str, Any]], str]:
    datasets: list[dict[str, Any]] = []
    for role, role_spec in zip(roles, role_specs):
        descriptor = _file_descriptor(
            role_spec["pkl"], field=f"SOMP-H authority dataset:{role}"
        )
        datasets.append(
            {
                "role": role,
                **descriptor,
                "tx_ids": authority._split_tx_ids(
                    role_spec["tx_ids"],
                    field=f"build_spec.{role}.tx_ids",
                ),
            }
        )
    try:
        checked, root = authority._validate_datasets(
            datasets,
            roles=roles,
            role_specs=role_specs,
            receiver=receiver,
            seed=seed,
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    return checked, root


def _write_somph_authority_lock_package_impl(
    cache_set_manifest_path: str | Path,
    *,
    cache_spec_manifest_path: str | Path,
    cache_spec_cell_id: str,
    exporter_path: str | Path,
    channel_code_members: Mapping[str, str | Path],
    output_root: str | Path,
    expected_cache_spec_manifest_sha256: str,
) -> dict[str, Any]:
    """Recompute real roots and atomically publish one unsigned lock package."""

    (
        locked_cell,
        _locked_manifest,
        build_spec_path,
        locked_manifest_sha,
        locked_manifest_size,
    ) = _locked_cache_spec_cell(
        cache_spec_manifest_path,
        cell_id=cache_spec_cell_id,
        expected_manifest_sha256=expected_cache_spec_manifest_sha256,
    )
    cache_set_path = Path(cache_set_manifest_path).absolute()
    try:
        cache_set, _set_raw, set_sha, set_size = authority._read_external_json(
            cache_set_path, context="SOMP-H real cache-set manifest"
        )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    build_descriptor, build_spec = _build_spec_descriptor(build_spec_path)
    if (
        build_descriptor["file_sha256"]
        != locked_cell.get("spec_file_sha256")
        or build_descriptor["canonical_sha256"]
        != locked_cell.get("spec_canonical_sha256")
    ):
        raise SomphAuthorityLockBuildError(
            "locked cache-spec cell/build-spec SHA drift"
        )
    build_spec_dir = Path(str(build_descriptor["path"])).absolute().parent
    identity, roles, role_specs = _derive_formal_identity(
        build_spec,
        build_spec_dir=build_spec_dir,
    )
    expected_cell_id = (
        f"rx_{identity['receiver'].replace('-', '_')}_seed_{identity['seed']}"
    )
    if (
        expected_cell_id != cache_spec_cell_id
        or locked_cell.get("receiver") != identity["receiver"]
        or locked_cell.get("seed") != identity["seed"]
        or locked_cell.get("cache_scope") != identity["cache_scope"]
    ):
        raise SomphAuthorityLockBuildError(
            "locked cache-spec cell/formal identity drift"
        )
    required_samples_per_tx = locked_cell["required_samples_per_tx"]
    for role_spec in role_specs:
        if role_spec.get("max_samples_per_tx") != required_samples_per_tx:
            raise SomphAuthorityLockBuildError(
                "locked cache-spec exact sample count/build-spec drift"
            )
    try:
        expected_roles = structural._validate_cache_set(
            cache_set, expected_scope=identity["cache_scope"]
        )
    except structural.SomphLineageError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    build_out_manifest = _resolve_relative_to(
        build_spec_dir,
        build_spec["out_manifest"],
        field="build_spec.out_manifest",
    )
    locked_cache_root = Path(
        str(locked_cell.get("cache_output_root", ""))
    ).absolute()
    if (
        build_out_manifest.resolve() != cache_set_path.resolve()
        or build_out_manifest.parent.resolve() != locked_cache_root.resolve()
        or build_spec.get("cache_set_id") != cache_set.get("cache_set_id")
    ):
        raise SomphAuthorityLockBuildError(
            "real build spec/cache-set identity drift"
        )
    exporter = _file_descriptor(
        exporter_path, field="SOMP-H real LEO_weak cache exporter"
    )
    if cache_set.get("builder_sha256") != exporter["sha256"]:
        raise SomphAuthorityLockBuildError(
            "cache-set exporter SHA does not match real exporter bytes"
        )
    if cache_set.get("build_spec_sha256") != build_descriptor["canonical_sha256"]:
        raise SomphAuthorityLockBuildError(
            "cache-set build-spec canonical SHA drift"
        )
    cache_paths = cache_set.get("cache_npz_by_scenario")
    if not isinstance(cache_paths, dict):
        raise SomphAuthorityLockBuildError("cache-set cache path map missing")
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        build_out_npz = _resolve_relative_to(
            build_spec_dir,
            build_spec["out_npz_by_scenario"][scenario],
            field=f"build_spec.out_npz_by_scenario.{scenario}",
        )
        actual_path = structural._resolve_cache(
            cache_set_path, cache_paths.get(scenario)
        )
        if build_out_npz.resolve() != actual_path.resolve():
            raise SomphAuthorityLockBuildError(
                f"real build spec/cache output drift for {scenario}"
            )
    (
        cache_hashes,
        channel_roots,
        physical_root,
        iq_roots,
        overlay_roots,
        cache_audits,
        role_row_counts,
    ) = _recompute_cache_roots(
        cache_set,
        cache_set_path=cache_set_path,
        exporter_sha256=exporter["sha256"],
        build_spec_sha256=build_descriptor["canonical_sha256"],
        build_spec=build_spec,
        receiver=identity["receiver"],
        roles=roles,
        tx_ids_by_role={
            "target_old": identity["old_tx_ids"],
            **(
                {"target_new": identity["new_tx_ids"]}
                if "target_new" in roles
                else {}
            ),
        },
        required_samples_per_tx=required_samples_per_tx,
        expected_roles=expected_roles,
    )
    datasets, dataset_root = _dataset_descriptors(
        role_specs,
        roles=roles,
        receiver=identity["receiver"],
        seed=identity["seed"],
    )
    try:
        role_inputs = authority._cache_role_inputs(
            cache_set,
            manifest_path=cache_set_path,
            expected_cache_hashes=cache_hashes,
            expected_tx_by_role={
                "target_old": identity["old_tx_ids"],
                **(
                    {"target_new": identity["new_tx_ids"]}
                    if "target_new" in roles
                    else {}
                ),
            },
            receiver=identity["receiver"],
        )
        role_inputs_root = authority._verify_cache_role_inputs(
            role_inputs,
            datasets=datasets,
            role_specs=role_specs,
            seed=identity["seed"],
        )
        for row in role_inputs[FORMAL_LEO_WEAK_SCENARIOS[0]]:
            role = row.get("role")
            if (
                role not in role_row_counts
                or row.get("physical_sample_count") != role_row_counts[role]
            ):
                raise SomphAuthorityLockBuildError(
                    "cache role_inputs physical_sample_count drift"
                )
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    closure = _channel_closure(channel_code_members)
    lock = {
        "schema": authority.AUTHORITY_LOCK_SCHEMA,
        **identity,
        "cache_set_manifest": {
            "path": str(cache_set_path),
            "sha256": set_sha,
            "size_bytes": set_size,
        },
        "cache_sha256_by_scenario": cache_hashes,
        "exporter": exporter,
        "build_spec": build_descriptor,
        "channel_code_closure": closure,
        "channel_config_sha256_by_scenario": channel_roots,
        "physical_sample_ids_sha256": physical_root,
        "post_channel_iq_sha256_root_by_scenario": iq_roots,
        "overlay_ids_sha256_by_scenario": overlay_roots,
        "cache_role_inputs_root_sha256": role_inputs_root,
        "datasets": datasets,
    }
    try:
        authority._require_exact_dict(
            lock, authority._LOCK_KEYS, field="built authority lock"
        )
        authority._validate_lock_formal_identity(lock)
    except authority.SomphLineageAuthorityError as exc:
        raise SomphAuthorityLockBuildError(str(exc)) from exc
    lock_bytes = canonical_json_bytes(lock) + b"\n"
    lock_sha = sha256_bytes(lock_bytes)
    lock_canonical_sha = sha256_bytes(canonical_json_bytes(lock))
    destination = Path(output_root).absolute()
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent_stat = parent.lstat()
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise SomphAuthorityLockBuildError(
            "authority lock output parent must be a non-symlink directory"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite authority lock package: {destination}"
        )
    staging = parent / (
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(12)}.staging"
    )
    staging.mkdir(mode=0o700)
    published = False
    try:
        authority._write_new_readonly(
            staging / authority.AUTHORITY_LOCK_NAME, lock_bytes
        )
        receipt = {
            "schema": AUTHORITY_LOCK_BUILD_RECEIPT_SCHEMA,
            "status": AUTHORITY_LOCK_BUILD_STATUS,
            "cache_spec_manifest_sha256": locked_manifest_sha,
            "cache_spec_manifest_size_bytes": locked_manifest_size,
            "cache_spec_cell_id": cache_spec_cell_id,
            "required_samples_per_tx": required_samples_per_tx,
            "receiver": identity["receiver"],
            "seed": identity["seed"],
            "cache_scope": identity["cache_scope"],
            "cache_set_manifest_sha256": set_sha,
            "build_spec_file_sha256": build_descriptor["file_sha256"],
            "build_spec_canonical_sha256": build_descriptor[
                "canonical_sha256"
            ],
            "exporter_sha256": exporter["sha256"],
            "channel_code_closure_sha256": closure["closure_sha256"],
            "dataset_authority_root_sha256": dataset_root,
            "cache_role_inputs_root_sha256": role_inputs_root,
            "physical_sample_ids_sha256": physical_root,
            "cache_sha256_by_scenario": cache_hashes,
            "channel_config_sha256_by_scenario": channel_roots,
            "post_channel_iq_sha256_root_by_scenario": iq_roots,
            "overlay_ids_sha256_by_scenario": overlay_roots,
            "cache_recompute_audits": cache_audits,
            "authority_lock_sha256": lock_sha,
            "authority_lock_canonical_sha256": lock_canonical_sha,
            "external_authority_lock_verified": False,
            "formal_launch_authority": False,
        }
        if set(receipt) != AUTHORITY_LOCK_BUILD_RECEIPT_KEYS:
            raise AssertionError(
                "authority lock build receipt exact schema drift"
            )
        receipt_bytes = canonical_json_bytes(receipt) + b"\n"
        receipt_sha, _receipt_size = authority._write_new_readonly(
            staging / AUTHORITY_LOCK_BUILD_RECEIPT_NAME, receipt_bytes
        )
        authority._fsync_directory(staging)
        os.chmod(staging, 0o555)
        if stat.S_IMODE(staging.lstat().st_mode) & _WRITE_BITS:
            raise SomphAuthorityLockBuildError(
                "authority lock staging directory is not read-only"
            )
        authority._fsync_directory(parent)
        authority._publish_directory_noreplace(staging, destination)
        published = True
        authority._fsync_directory(parent)
    finally:
        if not published:
            authority._remove_tree(staging)
    return {
        "authority_lock_package_root": str(destination),
        "authority_lock_path": str(
            destination / authority.AUTHORITY_LOCK_NAME
        ),
        "authority_lock_sha256": lock_sha,
        "authority_lock_canonical_sha256": lock_canonical_sha,
        "authority_lock_build_receipt_sha256": receipt_sha,
        "cache_spec_manifest_sha256": locked_manifest_sha,
        "cache_spec_cell_id": cache_spec_cell_id,
        "external_authority_lock_verified": False,
        "formal_launch_authority": False,
    }


def write_somph_authority_lock_package(
    cache_set_manifest_path: str | Path,
    *,
    cache_spec_manifest_path: str | Path,
    cache_spec_cell_id: str,
    exporter_path: str | Path,
    channel_code_members: Mapping[str, str | Path],
    output_root: str | Path,
) -> dict[str, Any]:
    """Production wrapper pinned to the formal 30-cell cache-spec manifest."""

    return _write_somph_authority_lock_package_impl(
        cache_set_manifest_path,
        cache_spec_manifest_path=cache_spec_manifest_path,
        cache_spec_cell_id=cache_spec_cell_id,
        exporter_path=exporter_path,
        channel_code_members=channel_code_members,
        output_root=output_root,
        expected_cache_spec_manifest_sha256=(
            "0e1f09ba08afd52b43a1bc9188d319f389c6cb57c9c8e06eee087ac99b3666c5"
        ),
    )
