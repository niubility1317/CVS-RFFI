"""Plan the fixed SOMP-H Stage2 registered LEO_weak cache build matrix.

This module is an offline controller artifact.  It writes real
``cvs_leo_weak_iq_cache_build_spec_v1`` inputs for the existing cache builder,
but it never executes that builder and never grants formal launch authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SET_SCHEMA,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    physical_sample_id_from_values,
)
from cvsrffi.somph_formal_matrix import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEED,
    FORMAL_RECEIVERS,
    NEW_TX_IDS,
    OLD_TX_IDS,
    SUPPORT_POOL_MAX_K,
)


SCHEMA = "cvs.phase2.somph_registered_cache_build_matrix.v2"
BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v2"
ARTIFACT_BOUNDARY = "phase1_offline_cache_planner_never_mounted_in_phase2"
CONTROL_STATUS = "LOCAL_PROTOCOL_REPAIR_REQUIRED"
CACHE_SCOPE = "stage2_registered"
QUERY_SAMPLES_PER_TX = 20
REQUIRED_SAMPLES_PER_TX = SUPPORT_POOL_MAX_K + QUERY_SAMPLES_PER_TX
TOTAL_REQUIRED_SAMPLES_PER_TX = (
    REQUIRED_SAMPLES_PER_TX * len(FORMAL_LEO_WEAK_SCENARIOS)
)
SCENARIO_PARTITION_POLICY = "disjoint_preoverlay_tx_day_stratified_v1"
WISIG_OUT_LEN = 256
BATCH_SIZE = 256
FLOAT32_BYTES = 4
IQ_COMPONENTS = 2
MIN_DISK_BUDGET_BYTES_PER_CELL = 32 * 1024 * 1024
MANIFEST_NAME = "manifest.json"
SPEC_DIRECTORY = "specs"
SEEDS = (DEVELOPMENT_SEED, *CONFIRMATION_SEEDS)
FIXED_N607_CACHE_OUTPUT_ROOT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "somph_stage2bc_leo_weak_cache_20260716"
)
POST_BUILD_GATE_SCHEMA = "cvs.phase2.somph_registered_cache_coverage_gate.v1"
POST_BUILD_GATE_STATUS = "NOT_RUN_BLOCKS_FORMAL_LAUNCH"

_BUILD_SPEC_KEYS = {
    "schema",
    "cache_set_id",
    "cache_scope",
    "phase2_sample_view_policy",
    "clean_sample_access",
    "clean_derived_signal_access",
    "phase2_physical_sample_observation_policy",
    "phase2_cross_scenario_physical_sample_reuse",
    "phase2_additional_leo_channel_state_generation",
    "phase2_post_reception_equalization_augmentation_transform_allowed",
    "phase2_post_reception_view_from_fixed_received_iq_only",
    "phase2_post_reception_view_counts_as_additional_physical_sample",
    "phase2_physical_sample_root_id_policy",
    "phase2_query_post_reception_view_fit_access",
    "physical_sample_scenario_assignment_policy",
    "star_ground_channel_impl",
    "role_specs",
    "dataset_seed",
    "satellite_seed_by_scenario",
    "out_npz_by_scenario",
    "out_manifest",
    "batch_size",
    "wisig_out_len",
    "wisig_equalized",
    "wisig_domain",
}
_ROLE_SPEC_KEYS = {
    "role",
    "pkl",
    "tx_ids",
    "rxs",
    "days",
    "max_samples_per_tx",
}
_CELL_KEYS = {
    "cell_id",
    "receiver",
    "seed",
    "seed_role",
    "cache_scope",
    "support_pool_max_k",
    "query_samples_per_tx",
    "required_samples_per_tx",
    "development_selection_eligible",
    "development_selection_k_shot",
    "nondevelopment_selection_authority",
    "spec_path",
    "spec_file_sha256",
    "spec_canonical_sha256",
    "spec_size_bytes",
    "cache_output_root",
    "estimated_rows_per_scenario",
    "estimated_rows_all_scenarios",
    "estimated_iq_bytes_all_scenarios",
    "disk_budget_bytes",
    "post_build_coverage_status",
}
_DATASET_KEYS = {"logical_name", "role", "path", "tx_ids"}
_MANIFEST_KEYS = {
    "schema",
    "artifact_boundary",
    "control_status",
    "formal_launch_authority",
    "local_spec_generation_only",
    "ssh_performed",
    "cache_builder_executed",
    "build_spec_schema",
    "cache_scope",
    "cache_output_root",
    "phase2_sample_view_policy",
    "receivers",
    "development_seed",
    "confirmation_seeds",
    "seeds",
    "development_lock",
    "target_channel_scenarios",
    "old_tx_ids",
    "nested_new_tx_ids",
    "support_pool_max_k",
    "query_samples_per_tx",
    "required_samples_per_tx",
    "sample_count_requirement",
    "single_leo_weak_observation_per_physical_sample",
    "cross_scenario_physical_sample_reuse",
    "physical_sample_scenario_assignment_policy",
    "required_physical_samples_per_tx_all_scenarios",
    "datasets",
    "wisig_out_len",
    "estimated_rows_per_scenario",
    "estimated_rows_all_scenarios_per_cell",
    "estimated_iq_bytes_all_scenarios_per_cell",
    "disk_budget_bytes_per_cell",
    "disk_budget_bytes_total",
    "disk_budget_formula",
    "cell_count",
    "cells",
    "post_build_coverage_validator",
    "post_build_coverage_required",
    "post_build_coverage_status",
    "manifest_sha256",
}
_FORBIDDEN_PATH_TOKEN = re.compile(
    r"(^|[^a-z0-9])(clean|raw|phase2|predictor|scorer|package)([^a-z0-9]|$)"
)
_FORBIDDEN_NPZ_MEMBER_TOKEN = re.compile(r"clean|(^|_)raw_iq($|_)")


class SomphCacheBuildMatrixError(ValueError):
    """Raised when the fixed registered-cache plan drifts from its contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SomphCacheBuildMatrixError(
            f"{field} must be an integer >= {minimum}; booleans are forbidden"
        )
    return value


def _safe_receiver(receiver: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", receiver).strip("_")


def _runtime_path(root: str, *parts: str) -> str:
    if re.match(r"^[A-Za-z]:[\\/]", root):
        return str(PureWindowsPath(root, *parts))
    return str(PurePosixPath(root, *parts))


def _validate_path(
    value: Any,
    *,
    field: str,
    expected_dataset_name: str | None = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SomphCacheBuildMatrixError(f"{field} must be a nonempty string")
    normalized = value.replace("\\", "/")
    if _FORBIDDEN_PATH_TOKEN.search(normalized.casefold()):
        raise SomphCacheBuildMatrixError(f"{field} contains a forbidden path token")
    if expected_dataset_name is not None:
        actual_name = PurePosixPath(normalized).name.casefold()
        if actual_name != expected_dataset_name.casefold():
            raise SomphCacheBuildMatrixError(
                f"{field} must name the fixed {expected_dataset_name} dataset"
            )
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> bytes:
    raw = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    return raw


def _satellite_seeds_for_receiver(receiver: str, seed: int) -> dict[str, int]:
    try:
        receiver_index = FORMAL_RECEIVERS.index(receiver)
    except ValueError as exc:
        raise SomphCacheBuildMatrixError("receiver is outside the fixed formal grid") from exc
    return {
        scenario: seed * 100 + receiver_index * 10 + index
        for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
    }


def _estimated_rows_per_scenario() -> int:
    return (len(OLD_TX_IDS) + len(NEW_TX_IDS)) * REQUIRED_SAMPLES_PER_TX


def _estimated_iq_bytes_all_scenarios() -> int:
    return (
        _estimated_rows_per_scenario()
        * len(FORMAL_LEO_WEAK_SCENARIOS)
        * IQ_COMPONENTS
        * WISIG_OUT_LEN
        * FLOAT32_BYTES
    )


def _disk_budget_bytes_per_cell() -> int:
    estimated = _estimated_iq_bytes_all_scenarios()
    return max(MIN_DISK_BUDGET_BYTES_PER_CELL, estimated * 2)


def _build_cache_spec(
    *,
    receiver: str,
    seed: int,
    manysig_pkl: str,
    manytx_pkl: str,
    cache_output_root: str,
) -> dict[str, Any]:
    """Build one fixed receiver/seed ``stage2_registered`` cache spec."""

    if receiver not in FORMAL_RECEIVERS:
        raise SomphCacheBuildMatrixError("receiver is outside the fixed formal grid")
    if seed not in SEEDS or isinstance(seed, bool):
        raise SomphCacheBuildMatrixError("seed is outside the fixed 713101-713106 grid")
    manysig = _validate_path(
        manysig_pkl, field="manysig_pkl", expected_dataset_name="ManySig.pkl"
    )
    manytx = _validate_path(
        manytx_pkl, field="manytx_pkl", expected_dataset_name="ManyTx.pkl"
    )
    root = _validate_path(
        cache_output_root, field="cache_output_root"
    )
    cell_root = _runtime_path(root, f"rx_{_safe_receiver(receiver)}", f"seed_{seed}")
    spec = {
        "schema": BUILD_SPEC_SCHEMA,
        "cache_set_id": f"somph_stage2_registered_rx_{_safe_receiver(receiver)}_seed_{seed}",
        "cache_scope": CACHE_SCOPE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
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
        "physical_sample_scenario_assignment_policy": SCENARIO_PARTITION_POLICY,
        "star_ground_channel_impl": "simplified_leo_residual",
        "role_specs": [
            {
                "role": "target_old",
                "pkl": manysig,
                "tx_ids": ",".join(OLD_TX_IDS),
                "rxs": receiver,
                "days": "0,1,2",
                "max_samples_per_tx": TOTAL_REQUIRED_SAMPLES_PER_TX,
            },
            {
                "role": "target_new",
                "pkl": manytx,
                "tx_ids": ",".join(NEW_TX_IDS),
                "rxs": receiver,
                "days": "0,1,2",
                "max_samples_per_tx": TOTAL_REQUIRED_SAMPLES_PER_TX,
            },
        ],
        "dataset_seed": seed,
        "satellite_seed_by_scenario": _satellite_seeds_for_receiver(receiver, seed),
        "out_npz_by_scenario": {
            scenario: _runtime_path(cell_root, f"{scenario}.npz")
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "out_manifest": _runtime_path(cell_root, "cache_set.json"),
        "batch_size": BATCH_SIZE,
        "wisig_out_len": WISIG_OUT_LEN,
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
    }
    validate_cache_spec(spec, receiver=receiver, seed=seed)
    return spec


def validate_cache_spec(
    payload: Mapping[str, Any],
    *,
    receiver: str,
    seed: int,
) -> None:
    """Validate the exact real-builder schema and fixed formal data registry."""

    if not isinstance(payload, Mapping) or set(payload) != _BUILD_SPEC_KEYS:
        raise SomphCacheBuildMatrixError("cache build spec exact schema drift")
    _require_int(seed, field="seed", minimum=0)
    expected_scalars = {
        "schema": BUILD_SPEC_SCHEMA,
        "cache_scope": CACHE_SCOPE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
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
        "physical_sample_scenario_assignment_policy": SCENARIO_PARTITION_POLICY,
        "star_ground_channel_impl": "simplified_leo_residual",
        "dataset_seed": seed,
        "batch_size": BATCH_SIZE,
        "wisig_out_len": WISIG_OUT_LEN,
        "wisig_equalized": "1",
        "wisig_domain": "rx_day",
    }
    failed = [
        key for key, expected in expected_scalars.items() if payload.get(key) != expected
    ]
    if failed:
        raise SomphCacheBuildMatrixError(f"cache build spec contract drift: {failed}")
    if not isinstance(payload.get("cache_set_id"), str) or not payload["cache_set_id"]:
        raise SomphCacheBuildMatrixError("cache_set_id must be nonempty")
    _require_int(payload["dataset_seed"], field="dataset_seed", minimum=0)
    _require_int(payload["batch_size"], field="batch_size", minimum=1)
    _require_int(payload["wisig_out_len"], field="wisig_out_len", minimum=1)
    role_specs = payload.get("role_specs")
    if not isinstance(role_specs, list) or len(role_specs) != 2:
        raise SomphCacheBuildMatrixError("registered cache requires exactly two roles")
    expected_roles = (
        ("target_old", "ManySig.pkl", OLD_TX_IDS),
        ("target_new", "ManyTx.pkl", NEW_TX_IDS),
    )
    for index, (raw, expected_name, expected_tx_ids) in enumerate(expected_roles):
        role = role_specs[index]
        if not isinstance(role, Mapping) or set(role) != _ROLE_SPEC_KEYS:
            raise SomphCacheBuildMatrixError("role spec exact schema drift")
        if role.get("role") != raw:
            raise SomphCacheBuildMatrixError("registered cache role ordering drift")
        _validate_path(
            role.get("pkl"),
            field=f"role_specs[{index}].pkl",
            expected_dataset_name=expected_name,
        )
        if role.get("tx_ids") != ",".join(expected_tx_ids):
            raise SomphCacheBuildMatrixError("role TX registry drift")
        if role.get("rxs") != receiver or role.get("days") != "0,1,2":
            raise SomphCacheBuildMatrixError("role receiver/day binding drift")
        if (
            _require_int(
                role.get("max_samples_per_tx"),
                field=f"role_specs[{index}].max_samples_per_tx",
                minimum=1,
            )
            != TOTAL_REQUIRED_SAMPLES_PER_TX
        ):
            raise SomphCacheBuildMatrixError(
                "role sample cap must provide independent maxK20 plus Q20 "
                "for each of three scenarios"
            )
    seeds = payload.get("satellite_seed_by_scenario")
    outputs = payload.get("out_npz_by_scenario")
    if not isinstance(seeds, Mapping) or tuple(seeds) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphCacheBuildMatrixError("satellite scenario seed tuple drift")
    if not isinstance(outputs, Mapping) or tuple(outputs) != FORMAL_LEO_WEAK_SCENARIOS:
        raise SomphCacheBuildMatrixError("scenario output tuple drift")
    if dict(seeds) != _satellite_seeds_for_receiver(receiver, seed):
        raise SomphCacheBuildMatrixError("satellite scenario seed values drift")
    output_values: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        _require_int(seeds[scenario], field=f"satellite_seed.{scenario}", minimum=0)
        output = _validate_path(outputs[scenario], field=f"out_npz.{scenario}")
        if not output.endswith(f"{scenario}.npz") or output in output_values:
            raise SomphCacheBuildMatrixError("scenario outputs must be unique/exact")
        output_values.add(output)
    out_manifest = _validate_path(payload.get("out_manifest"), field="out_manifest")
    if not out_manifest.endswith("cache_set.json"):
        raise SomphCacheBuildMatrixError("out_manifest filename drift")
    output_parents = {
        value.replace("\\", "/").rsplit("/", 1)[0] for value in output_values
    }
    manifest_parent = out_manifest.replace("\\", "/").rsplit("/", 1)[0]
    if len(output_parents) != 1 or output_parents != {manifest_parent}:
        raise SomphCacheBuildMatrixError("cell outputs do not share one output root")


def _cell_descriptor(
    *,
    root: Path,
    receiver: str,
    seed: int,
    spec: Mapping[str, Any],
    spec_path: Path,
    raw: bytes,
) -> dict[str, Any]:
    development_eligible = receiver == "20-1" and seed == DEVELOPMENT_SEED
    cache_output_root = str(spec["out_manifest"]).replace("\\", "/").rsplit("/", 1)[0]
    descriptor = {
        "cell_id": f"rx_{_safe_receiver(receiver)}_seed_{seed}",
        "receiver": receiver,
        "seed": seed,
        "seed_role": (
            "development" if seed == DEVELOPMENT_SEED else "independent_confirmation"
        ),
        "cache_scope": CACHE_SCOPE,
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
        "query_samples_per_tx": QUERY_SAMPLES_PER_TX,
        "required_samples_per_tx": REQUIRED_SAMPLES_PER_TX,
        "development_selection_eligible": development_eligible,
        "development_selection_k_shot": 10 if development_eligible else None,
        "nondevelopment_selection_authority": False,
        "spec_path": spec_path.relative_to(root).as_posix(),
        "spec_file_sha256": hashlib.sha256(raw).hexdigest(),
        "spec_canonical_sha256": _canonical_sha256(spec),
        "spec_size_bytes": len(raw),
        "cache_output_root": cache_output_root,
        "estimated_rows_per_scenario": _estimated_rows_per_scenario(),
        "estimated_rows_all_scenarios": _estimated_rows_per_scenario()
        * len(FORMAL_LEO_WEAK_SCENARIOS),
        "estimated_iq_bytes_all_scenarios": _estimated_iq_bytes_all_scenarios(),
        "disk_budget_bytes": _disk_budget_bytes_per_cell(),
        "post_build_coverage_status": POST_BUILD_GATE_STATUS,
    }
    return descriptor


def _reject_symlink_components(path: Path, *, field: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.exists() and current.is_symlink():
            raise SomphCacheBuildMatrixError(f"{field} contains a symlink component")


def write_cache_build_matrix(
    *,
    output_root: Path,
    manysig_pkl: str,
    manytx_pkl: str,
    cache_output_root: str = FIXED_N607_CACHE_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Write all 30 fixed specs plus a hash-bound exact manifest.

    ``output_root`` must not already exist.  The function does not invoke the
    cache builder, SSH, SCP, or any experiment launcher.
    """

    if not output_root.is_absolute():
        raise SomphCacheBuildMatrixError("output_root must be absolute")
    _reject_symlink_components(output_root, field="output_root")
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite cache build matrix: {root}")
    _validate_path(str(root), field="output_root")
    manysig = _validate_path(
        manysig_pkl, field="manysig_pkl", expected_dataset_name="ManySig.pkl"
    )
    manytx = _validate_path(
        manytx_pkl, field="manytx_pkl", expected_dataset_name="ManyTx.pkl"
    )
    runtime_cache_root = _validate_path(
        cache_output_root, field="cache_output_root"
    )
    root.mkdir(parents=True, exist_ok=False)
    cells: list[dict[str, Any]] = []
    for receiver in FORMAL_RECEIVERS:
        for seed in SEEDS:
            spec = _build_cache_spec(
                receiver=receiver,
                seed=seed,
                manysig_pkl=manysig,
                manytx_pkl=manytx,
                cache_output_root=runtime_cache_root,
            )
            spec_path = (
                root
                / SPEC_DIRECTORY
                / f"rx_{_safe_receiver(receiver)}"
                / f"seed_{seed}.json"
            )
            raw = _write_new_json(spec_path, spec)
            cells.append(
                _cell_descriptor(
                    root=root,
                    receiver=receiver,
                    seed=seed,
                    spec=spec,
                    spec_path=spec_path,
                    raw=raw,
                )
            )
    rows = _estimated_rows_per_scenario()
    budget = _disk_budget_bytes_per_cell()
    payload = {
        "schema": SCHEMA,
        "artifact_boundary": ARTIFACT_BOUNDARY,
        "control_status": CONTROL_STATUS,
        "formal_launch_authority": False,
        "local_spec_generation_only": True,
        "ssh_performed": False,
        "cache_builder_executed": False,
        "build_spec_schema": BUILD_SPEC_SCHEMA,
        "cache_scope": CACHE_SCOPE,
        "cache_output_root": runtime_cache_root,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "receivers": list(FORMAL_RECEIVERS),
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "seeds": list(SEEDS),
        "development_lock": {
            "receiver": "20-1",
            "seed": DEVELOPMENT_SEED,
            "k_shot": 10,
        },
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "old_tx_ids": list(OLD_TX_IDS),
        "nested_new_tx_ids": list(NEW_TX_IDS),
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
        "query_samples_per_tx": QUERY_SAMPLES_PER_TX,
        "required_samples_per_tx": REQUIRED_SAMPLES_PER_TX,
        "sample_count_requirement": (
            "dataset_inventory_must_prove_at_least_120_distinct_rows_per_tx_"
            "before_single_scenario_assignment"
        ),
        "single_leo_weak_observation_per_physical_sample": True,
        "cross_scenario_physical_sample_reuse": False,
        "physical_sample_scenario_assignment_policy": SCENARIO_PARTITION_POLICY,
        "required_physical_samples_per_tx_all_scenarios": (
            TOTAL_REQUIRED_SAMPLES_PER_TX
        ),
        "datasets": [
            {
                "logical_name": "ManySig",
                "role": "target_old",
                "path": manysig,
                "tx_ids": list(OLD_TX_IDS),
            },
            {
                "logical_name": "ManyTx",
                "role": "target_new",
                "path": manytx,
                "tx_ids": list(NEW_TX_IDS),
            },
        ],
        "wisig_out_len": WISIG_OUT_LEN,
        "estimated_rows_per_scenario": rows,
        "estimated_rows_all_scenarios_per_cell": rows
        * len(FORMAL_LEO_WEAK_SCENARIOS),
        "estimated_iq_bytes_all_scenarios_per_cell": (
            _estimated_iq_bytes_all_scenarios()
        ),
        "disk_budget_bytes_per_cell": budget,
        "disk_budget_bytes_total": budget * len(cells),
        "disk_budget_formula": (
            "max(32MiB,2*(26TX*40independent_rows_per_scenario*"
            "3scenarios*2IQ*256*float32))"
        ),
        "cell_count": len(cells),
        "cells": cells,
        "post_build_coverage_validator": (
            "code/scripts/validate_cvs_somph_cache_coverage.py"
        ),
        "post_build_coverage_required": True,
        "post_build_coverage_status": POST_BUILD_GATE_STATUS,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    validate_cache_build_manifest(payload, manifest_root=root)
    _write_new_json(root / MANIFEST_NAME, payload)
    return payload


def validate_cache_build_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_root: Path | None = None,
) -> None:
    """Reject coverage, authority, digest, integer, or path drift."""

    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_KEYS:
        raise SomphCacheBuildMatrixError("cache matrix manifest exact schema drift")
    manifest_cache_output_root = _validate_path(
        payload.get("cache_output_root"), field="cache_output_root"
    )
    fixed = {
        "schema": SCHEMA,
        "artifact_boundary": ARTIFACT_BOUNDARY,
        "control_status": CONTROL_STATUS,
        "formal_launch_authority": False,
        "local_spec_generation_only": True,
        "ssh_performed": False,
        "cache_builder_executed": False,
        "build_spec_schema": BUILD_SPEC_SCHEMA,
        "cache_scope": CACHE_SCOPE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "receivers": list(FORMAL_RECEIVERS),
        "development_seed": DEVELOPMENT_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "seeds": list(SEEDS),
        "development_lock": {
            "receiver": "20-1",
            "seed": DEVELOPMENT_SEED,
            "k_shot": 10,
        },
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "old_tx_ids": list(OLD_TX_IDS),
        "nested_new_tx_ids": list(NEW_TX_IDS),
        "support_pool_max_k": SUPPORT_POOL_MAX_K,
        "query_samples_per_tx": QUERY_SAMPLES_PER_TX,
        "required_samples_per_tx": REQUIRED_SAMPLES_PER_TX,
        "sample_count_requirement": (
            "dataset_inventory_must_prove_at_least_120_distinct_rows_per_tx_"
            "before_single_scenario_assignment"
        ),
        "single_leo_weak_observation_per_physical_sample": True,
        "cross_scenario_physical_sample_reuse": False,
        "physical_sample_scenario_assignment_policy": SCENARIO_PARTITION_POLICY,
        "required_physical_samples_per_tx_all_scenarios": (
            TOTAL_REQUIRED_SAMPLES_PER_TX
        ),
        "wisig_out_len": WISIG_OUT_LEN,
        "estimated_rows_per_scenario": _estimated_rows_per_scenario(),
        "estimated_rows_all_scenarios_per_cell": _estimated_rows_per_scenario()
        * len(FORMAL_LEO_WEAK_SCENARIOS),
        "estimated_iq_bytes_all_scenarios_per_cell": (
            _estimated_iq_bytes_all_scenarios()
        ),
        "disk_budget_bytes_per_cell": _disk_budget_bytes_per_cell(),
        "disk_budget_bytes_total": _disk_budget_bytes_per_cell()
        * len(FORMAL_RECEIVERS)
        * len(SEEDS),
        "disk_budget_formula": (
            "max(32MiB,2*(26TX*40independent_rows_per_scenario*"
            "3scenarios*2IQ*256*float32))"
        ),
        "cell_count": len(FORMAL_RECEIVERS) * len(SEEDS),
        "post_build_coverage_validator": (
            "code/scripts/validate_cvs_somph_cache_coverage.py"
        ),
        "post_build_coverage_required": True,
        "post_build_coverage_status": POST_BUILD_GATE_STATUS,
    }
    failed = [key for key, expected in fixed.items() if payload.get(key) != expected]
    if failed:
        raise SomphCacheBuildMatrixError(f"cache matrix manifest drift: {failed}")
    for field in (
        "development_seed",
        "support_pool_max_k",
        "query_samples_per_tx",
        "required_samples_per_tx",
        "wisig_out_len",
        "estimated_rows_per_scenario",
        "estimated_rows_all_scenarios_per_cell",
        "estimated_iq_bytes_all_scenarios_per_cell",
        "disk_budget_bytes_per_cell",
        "disk_budget_bytes_total",
        "cell_count",
    ):
        _require_int(payload.get(field), field=field, minimum=1)
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise SomphCacheBuildMatrixError("dataset registry must contain ManySig/ManyTx")
    expected_datasets = (
        ("ManySig", "target_old", "ManySig.pkl", OLD_TX_IDS),
        ("ManyTx", "target_new", "ManyTx.pkl", NEW_TX_IDS),
    )
    for index, (logical, role, filename, tx_ids) in enumerate(expected_datasets):
        dataset = datasets[index]
        if not isinstance(dataset, Mapping) or set(dataset) != _DATASET_KEYS:
            raise SomphCacheBuildMatrixError("dataset descriptor exact schema drift")
        if dataset.get("logical_name") != logical or dataset.get("role") != role:
            raise SomphCacheBuildMatrixError("dataset descriptor role drift")
        _validate_path(
            dataset.get("path"),
            field=f"datasets[{index}].path",
            expected_dataset_name=filename,
        )
        if dataset.get("tx_ids") != list(tx_ids):
            raise SomphCacheBuildMatrixError("dataset TX registry drift")
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != len(FORMAL_RECEIVERS) * len(SEEDS):
        raise SomphCacheBuildMatrixError("cache matrix must contain exactly 30 cells")
    observed: set[tuple[str, int]] = set()
    output_roots: set[str] = set()
    eligible: list[tuple[str, int, int | None]] = []
    root = manifest_root.resolve() if manifest_root is not None else None
    for cell in cells:
        if not isinstance(cell, Mapping) or set(cell) != _CELL_KEYS:
            raise SomphCacheBuildMatrixError("cell exact schema drift")
        receiver = cell.get("receiver")
        seed = _require_int(cell.get("seed"), field="cell.seed", minimum=0)
        key = (receiver, seed)
        if receiver not in FORMAL_RECEIVERS or seed not in SEEDS or key in observed:
            raise SomphCacheBuildMatrixError("cell receiver/seed coverage drift")
        observed.add(key)
        expected_role = (
            "development" if seed == DEVELOPMENT_SEED else "independent_confirmation"
        )
        if cell.get("seed_role") != expected_role:
            raise SomphCacheBuildMatrixError("cell seed role drift")
        expected_eligible = receiver == "20-1" and seed == DEVELOPMENT_SEED
        if cell.get("development_selection_eligible") is not expected_eligible:
            raise SomphCacheBuildMatrixError("development selection eligibility drift")
        expected_k = 10 if expected_eligible else None
        if cell.get("development_selection_k_shot") != expected_k:
            raise SomphCacheBuildMatrixError("development selection K lock drift")
        if expected_eligible:
            eligible.append((receiver, seed, expected_k))
        if cell.get("nondevelopment_selection_authority") is not False:
            raise SomphCacheBuildMatrixError("nondevelopment cell gained selection authority")
        for field, expected in {
            "cache_scope": CACHE_SCOPE,
            "support_pool_max_k": SUPPORT_POOL_MAX_K,
            "query_samples_per_tx": QUERY_SAMPLES_PER_TX,
            "required_samples_per_tx": REQUIRED_SAMPLES_PER_TX,
            "estimated_rows_per_scenario": _estimated_rows_per_scenario(),
            "estimated_rows_all_scenarios": _estimated_rows_per_scenario()
            * len(FORMAL_LEO_WEAK_SCENARIOS),
            "estimated_iq_bytes_all_scenarios": (
                _estimated_iq_bytes_all_scenarios()
            ),
            "disk_budget_bytes": _disk_budget_bytes_per_cell(),
            "post_build_coverage_status": POST_BUILD_GATE_STATUS,
        }.items():
            if cell.get(field) != expected:
                raise SomphCacheBuildMatrixError(f"cell field drift: {field}")
            if field not in {"cache_scope", "post_build_coverage_status"}:
                _require_int(cell.get(field), field=f"cell.{field}", minimum=1)
        output_root = _validate_path(
            cell.get("cache_output_root"), field="cell.cache_output_root"
        )
        expected_output_root = _runtime_path(
            manifest_cache_output_root,
            f"rx_{_safe_receiver(str(receiver))}",
            f"seed_{seed}",
        )
        if output_root != expected_output_root:
            raise SomphCacheBuildMatrixError(
                "cell/manifest cache output root binding drift"
            )
        if output_root in output_roots:
            raise SomphCacheBuildMatrixError("cache output roots are not independent")
        output_roots.add(output_root)
        for digest_field in ("spec_file_sha256", "spec_canonical_sha256"):
            digest = cell.get(digest_field)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise SomphCacheBuildMatrixError(f"invalid {digest_field}")
        _require_int(cell.get("spec_size_bytes"), field="cell.spec_size_bytes", minimum=1)
        spec_rel = cell.get("spec_path")
        if (
            not isinstance(spec_rel, str)
            or not spec_rel.startswith(f"{SPEC_DIRECTORY}/")
            or Path(spec_rel).is_absolute()
            or ".." in PurePosixPath(spec_rel).parts
        ):
            raise SomphCacheBuildMatrixError("cell spec path is not a safe relative path")
        if root is not None:
            spec_path = (root / PurePosixPath(spec_rel)).resolve()
            try:
                spec_path.relative_to(root)
            except ValueError as exc:
                raise SomphCacheBuildMatrixError("spec path escapes manifest root") from exc
            if not spec_path.is_file():
                raise SomphCacheBuildMatrixError("referenced spec file is missing")
            raw = spec_path.read_bytes()
            if len(raw) != cell["spec_size_bytes"]:
                raise SomphCacheBuildMatrixError("spec size binding mismatch")
            if hashlib.sha256(raw).hexdigest() != cell["spec_file_sha256"]:
                raise SomphCacheBuildMatrixError("spec file SHA binding mismatch")
            try:
                spec = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SomphCacheBuildMatrixError("spec JSON is invalid") from exc
            if _canonical_sha256(spec) != cell["spec_canonical_sha256"]:
                raise SomphCacheBuildMatrixError("spec canonical SHA binding mismatch")
            validate_cache_spec(spec, receiver=receiver, seed=seed)
            actual_output_root = (
                str(spec["out_manifest"]).replace("\\", "/").rsplit("/", 1)[0]
            )
            if actual_output_root != output_root:
                raise SomphCacheBuildMatrixError("spec/cache output root binding drift")
    expected_coverage = {
        (receiver, seed) for receiver in FORMAL_RECEIVERS for seed in SEEDS
    }
    if observed != expected_coverage:
        raise SomphCacheBuildMatrixError("cache matrix receiver/seed coverage mismatch")
    if eligible != [("20-1", DEVELOPMENT_SEED, 10)]:
        raise SomphCacheBuildMatrixError("development selection lock is not unique")
    unsigned = dict(payload)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise SomphCacheBuildMatrixError("manifest canonical SHA mismatch")


def _read_same_fd_nofollow(path: Path, *, field: str) -> bytes:
    if not path.is_absolute():
        raise SomphCacheBuildMatrixError(f"{field} must be absolute")
    _reject_symlink_components(path, field=field)
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise SomphCacheBuildMatrixError(f"{field} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or before.st_size != after.st_size
            or (
                before.st_ino
                and after.st_ino
                and (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            )
        ):
            raise SomphCacheBuildMatrixError(f"{field} changed during safe open")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _load_npz_coverage_same_fd(
    path: Path,
    *,
    scenario: str,
    expected_sha256: str,
) -> dict[str, list[str]]:
    if not path.is_absolute():
        raise SomphCacheBuildMatrixError("cache NPZ path must be absolute")
    _reject_symlink_components(path, field=f"cache_npz.{scenario}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise SomphCacheBuildMatrixError("cache NPZ must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        after = os.fstat(descriptor)
        if (
            not stat.S_ISREG(after.st_mode)
            or before.st_size != after.st_size
            or (
                before.st_ino
                and after.st_ino
                and (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            )
        ):
            raise SomphCacheBuildMatrixError("cache NPZ changed during safe open")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
            if hashlib.sha256(raw).hexdigest() != expected_sha256:
                raise SomphCacheBuildMatrixError(
                    f"cache NPZ SHA mismatch for {scenario}"
                )
            handle.seek(0)
            with np.load(handle, allow_pickle=False) as archive:
                members = tuple(str(value) for value in archive.files)
                forbidden = sorted(
                    value
                    for value in members
                    if _FORBIDDEN_NPZ_MEMBER_TOKEN.search(value.casefold())
                )
                if forbidden:
                    raise SomphCacheBuildMatrixError(
                        f"cache NPZ exposes forbidden members: {forbidden}"
                    )
                required = {
                    "leo_weak_iq",
                    "dataset_role",
                    "tx_ids",
                    "rx_ids",
                    "day_ids",
                    "eq_ids",
                    "sig_ids",
                    "source_dataset_sha256",
                    "source_record_indices",
                    "sample_ids",
                    "sat_scenarios",
                }
                missing = sorted(required - set(members))
                if missing:
                    raise SomphCacheBuildMatrixError(
                        f"cache NPZ coverage members missing: {missing}"
                    )
                arrays = {
                    key: np.asarray(archive[key])
                    for key in required
                }
    finally:
        os.close(descriptor)
    row_count = int(arrays["leo_weak_iq"].shape[0])
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise SomphCacheBuildMatrixError(f"object array forbidden: {key}")
        if int(np.asarray(value).shape[0]) != row_count:
            raise SomphCacheBuildMatrixError(f"cache NPZ row drift: {key}")
    scenarios = np.asarray(arrays["sat_scenarios"]).astype(str)
    if not bool(np.all(scenarios == scenario)):
        raise SomphCacheBuildMatrixError("cache NPZ scenario row drift")
    stored_ids = np.asarray(arrays["sample_ids"]).astype(str).tolist()
    computed_ids = [
        physical_sample_id_from_values(
            dataset_sha256=str(arrays["source_dataset_sha256"][index]),
            source_record_index=int(arrays["source_record_indices"][index]),
            role=str(arrays["dataset_role"][index]),
            tx_id=str(arrays["tx_ids"][index]),
            rx_id=str(arrays["rx_ids"][index]),
            day_id=str(arrays["day_ids"][index]),
            eq_id=str(arrays["eq_ids"][index]),
            sig_id=str(arrays["sig_ids"][index]),
        )
        for index in range(row_count)
    ]
    if stored_ids != computed_ids:
        raise SomphCacheBuildMatrixError(
            "cache NPZ stable physical sample identity drift"
        )
    return {
        "roles": np.asarray(arrays["dataset_role"]).astype(str).tolist(),
        "tx_ids": np.asarray(arrays["tx_ids"]).astype(str).tolist(),
        "rx_ids": np.asarray(arrays["rx_ids"]).astype(str).tolist(),
        "day_ids": np.asarray(arrays["day_ids"]).astype(str).tolist(),
        "sample_ids": computed_ids,
    }


def validate_registered_cache_coverage(
    cache_set_manifest: str | Path,
    *,
    expected_receiver: str,
) -> dict[str, Any]:
    """Require exact 40-row coverage for every registered role/TX/scenario.

    The cache-set JSON and every NPZ are opened without following symlinks.
    NPZ hashes and arrays are read through the same file descriptor with
    ``allow_pickle=False``.
    """

    if expected_receiver not in FORMAL_RECEIVERS:
        raise SomphCacheBuildMatrixError("expected_receiver is outside formal grid")
    manifest_path = Path(cache_set_manifest)
    if not manifest_path.is_absolute():
        raise SomphCacheBuildMatrixError("cache_set_manifest must be absolute")
    if manifest_path.name != "cache_set.json":
        raise SomphCacheBuildMatrixError("cache_set manifest filename drift")
    raw = _read_same_fd_nofollow(manifest_path, field="cache_set_manifest")
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SomphCacheBuildMatrixError("cache_set manifest JSON invalid") from exc
    required = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "cache_scope": CACHE_SCOPE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
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
        "physical_sample_scenario_assignment_policy": SCENARIO_PARTITION_POLICY,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": ["target_old", "target_new"],
    }
    if not isinstance(payload, Mapping):
        raise SomphCacheBuildMatrixError("cache_set manifest root must be an object")
    failed = [key for key, expected in required.items() if payload.get(key) != expected]
    if failed:
        raise SomphCacheBuildMatrixError(f"cache_set manifest contract drift: {failed}")
    scenario_map = payload.get("cache_npz_by_scenario")
    hash_map = payload.get("cache_sha256_by_scenario")
    if (
        not isinstance(scenario_map, Mapping)
        or tuple(scenario_map) != FORMAL_LEO_WEAK_SCENARIOS
        or not isinstance(hash_map, Mapping)
        or tuple(hash_map) != FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise SomphCacheBuildMatrixError("cache_set scenario mapping drift")
    reference_counts: Counter[tuple[str, str, str]] | None = None
    sample_ids_by_scenario: dict[str, list[str]] = {}
    day_counts_by_scenario: dict[
        str, Counter[tuple[str, str, str]]
    ] = {}
    observed_all_ids: set[str] = set()
    scenario_audits: dict[str, Any] = {}
    expected_tx_by_role = {
        "target_old": set(OLD_TX_IDS),
        "target_new": set(NEW_TX_IDS),
    }
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        raw_rel = scenario_map[scenario]
        if raw_rel != f"{scenario}.npz":
            raise SomphCacheBuildMatrixError(
                "cache NPZ path must be the exact sibling scenario filename"
            )
        expected_hash = hash_map[scenario]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(value not in "0123456789abcdef" for value in expected_hash)
        ):
            raise SomphCacheBuildMatrixError("cache NPZ SHA declaration invalid")
        cache_path = (manifest_path.parent / raw_rel).absolute()
        rows = _load_npz_coverage_same_fd(
            cache_path,
            scenario=scenario,
            expected_sha256=expected_hash,
        )
        if len(rows["sample_ids"]) != _estimated_rows_per_scenario():
            raise SomphCacheBuildMatrixError(
                f"cache coverage total must be exactly 1040 for {scenario}"
            )
        if len(set(rows["sample_ids"])) != _estimated_rows_per_scenario():
            raise SomphCacheBuildMatrixError(
                f"cache coverage physical sample IDs must be unique for {scenario}"
            )
        counts: Counter[tuple[str, str, str]] = Counter()
        day_counts: Counter[tuple[str, str, str]] = Counter()
        for role, tx_id, receiver, day_id in zip(
            rows["roles"], rows["tx_ids"], rows["rx_ids"], rows["day_ids"]
        ):
            if role not in expected_tx_by_role:
                raise SomphCacheBuildMatrixError("cache coverage contains an extra role")
            if tx_id not in expected_tx_by_role[role]:
                raise SomphCacheBuildMatrixError(
                    f"cache coverage TX/role registry drift: {role}/{tx_id}"
                )
            if receiver != expected_receiver:
                raise SomphCacheBuildMatrixError(
                    "cache coverage contains another receiver"
                )
            counts[(role, tx_id, receiver)] += 1
            day_counts[(role, tx_id, day_id)] += 1
        expected_keys = {
            (role, tx_id, expected_receiver)
            for role, tx_ids in expected_tx_by_role.items()
            for tx_id in tx_ids
        }
        if set(counts) != expected_keys:
            raise SomphCacheBuildMatrixError(
                f"cache coverage role/TX registry is incomplete for {scenario}"
            )
        bad = {
            key: count for key, count in counts.items() if count != REQUIRED_SAMPLES_PER_TX
        }
        if bad:
            raise SomphCacheBuildMatrixError(
                f"cache coverage must be exactly 40 per role/TX/receiver: {bad}"
            )
        current_ids = rows["sample_ids"]
        current_set = set(current_ids)
        overlap = sorted(observed_all_ids.intersection(current_set))
        if overlap:
            raise SomphCacheBuildMatrixError(
                "PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION: "
                f"physical sample IDs overlap across LEO scenarios: {overlap[:3]}"
            )
        observed_all_ids.update(current_set)
        sample_ids_by_scenario[scenario] = current_ids
        day_counts_by_scenario[scenario] = day_counts
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise SomphCacheBuildMatrixError(
                "cache coverage role/TX counts drift across LEO_weak scenarios"
            )
        scenario_audits[scenario] = {
            "row_count": len(rows["sample_ids"]),
            "role_tx_receiver_cell_count": len(counts),
            "exact_rows_per_role_tx_receiver": REQUIRED_SAMPLES_PER_TX,
            "physical_sample_ids_sha256": ids_sha256(rows["sample_ids"]),
            "role_tx_day_counts": {
                "|".join(key): value for key, value in sorted(day_counts.items())
            },
        }
    for role, tx_ids in expected_tx_by_role.items():
        for tx_id in tx_ids:
            observed_days = sorted(
                {
                    day
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                    for current_role, current_tx, day in day_counts_by_scenario[
                        scenario
                    ]
                    if current_role == role and current_tx == tx_id
                }
            )
            if len(observed_days) != 3:
                raise SomphCacheBuildMatrixError(
                    f"cache coverage must span exactly three days: {role}/{tx_id}"
                )
            for day in observed_days:
                values = [
                    day_counts_by_scenario[scenario][(role, tx_id, day)]
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                ]
                if max(values) - min(values) > 1:
                    raise SomphCacheBuildMatrixError(
                        "cache scenario/day allocation is not stratified: "
                        f"{role}/{tx_id}/{day} counts={values}"
                    )
    physical_roots = {
        scenario: ids_sha256(sample_ids_by_scenario[scenario])
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    if payload.get("physical_sample_ids_sha256_by_scenario") != physical_roots:
        raise SomphCacheBuildMatrixError(
            "cache_set per-scenario physical sample ID roots do not match NPZ rows"
        )
    assignment_root = canonical_json_sha256(sample_ids_by_scenario)
    if payload.get("physical_sample_scenario_assignment_sha256") != assignment_root:
        raise SomphCacheBuildMatrixError(
            "cache_set physical sample scenario assignment root mismatch"
        )
    return {
        "schema": POST_BUILD_GATE_SCHEMA,
        "cache_set_manifest": str(manifest_path),
        "expected_receiver": expected_receiver,
        "scenario_count": len(FORMAL_LEO_WEAK_SCENARIOS),
        "row_count_per_scenario": _estimated_rows_per_scenario(),
        "exact_rows_per_role_tx_receiver": REQUIRED_SAMPLES_PER_TX,
        "physical_sample_ids_sha256_by_scenario": physical_roots,
        "physical_sample_scenario_assignment_sha256": assignment_root,
        "cross_scenario_physical_sample_overlap_count": 0,
        "tx_day_stratification_pass": True,
        "coverage_pass": True,
        "formal_launch_authority": False,
        "scenario_audits": scenario_audits,
    }
