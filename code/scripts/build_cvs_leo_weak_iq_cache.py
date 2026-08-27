#!/usr/bin/env python
"""Build sealed post-channel LEO_weak IQ caches outside the Phase2 boundary.

This is a Phase1/offline preprocessing tool.  It may read the source dataset,
but it writes only post-channel ``leo_weak_iq`` plus sample-level overlay
provenance.  Formal Phase2 consumers must use ``cvsrffi.leo_weak_cache`` and
must never receive this build spec or an input dataset path.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.eval import apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
    EXTERNAL_COMPARISON_SAMPLE_VIEW_POLICY,
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA,
    LEO_WEAK_CACHE_SET_SCHEMA,
    LEO_WEAK_CACHE_STAGE,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    load_verified_leo_weak_cache,
    load_verified_leo_weak_cache_set,
    overlay_id,
    physical_sample_id_from_values,
    post_channel_iq_sha256,
    sha256_file,
)
from cvsrffi.phase2_canonical_split import (  # noqa: E402
    K_VALUES as CANONICAL_K_VALUES,
    PROTOCOL_SCHEMA as CANONICAL_PROTOCOL_SCHEMA,
    QUERY_POLICIES as CANONICAL_QUERY_POLICIES,
    SPLIT_MANIFEST_SCHEMA as CANONICAL_SPLIT_MANIFEST_SCHEMA,
)
from cvsrffi.tensors import make_torch_generator  # noqa: E402
from cvsrffi.wisig_canonical_inventory import (  # noqa: E402
    canonical_physical_id,
)
from dataset_wisig import WiSigCompactDataset  # noqa: E402
from export_spaceborne_features import (  # noqa: E402
    _build_wisig_dataset,
    _meta_to_list,
)
from training_controls import sat_channel_config_for_scenario  # noqa: E402


BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v2"
LEGACY_BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v1"
CANONICAL_BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v3"
CANONICAL_CACHE_SCOPE = "stage2_canonical_registered"
CANONICAL_OLD_TX_COUNT = 6
PER_SCENARIO_SAMPLES_PER_TX = 40
SCENARIO_PARTITION_POLICY = "disjoint_preoverlay_tx_day_stratified_v1"
REFERENCE_EXCLUSION_POLICY = (
    "exclude_all_source_records_from_verified_reference_cache_set_v1"
)
SCOPE_ROLES = {
    "source_train": {"source"},
    "source_validation": {"source"},
    "stage2_target_old": {"target_old"},
    "stage2_registered": {"target_old", "target_new"},
    CANONICAL_CACHE_SCOPE: {"target_old", "target_new"},
    "external_comparison_registered": {"target_old", "target_new"},
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def validate_build_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    scope = str(spec.get("cache_scope", ""))
    if scope not in SCOPE_ROLES:
        raise ValueError(f"unsupported cache_scope={scope!r}")
    canonical_scope = scope == CANONICAL_CACHE_SCOPE
    expected_schema = (
        CANONICAL_BUILD_SPEC_SCHEMA
        if canonical_scope
        else (
            BUILD_SPEC_SCHEMA
            if scope in {"stage2_target_old", "stage2_registered"}
            else LEGACY_BUILD_SPEC_SCHEMA
        )
    )
    if spec.get("schema") != expected_schema:
        raise ValueError(f"build spec schema must be {expected_schema}")
    expected_view_policy = (
        EXTERNAL_COMPARISON_SAMPLE_VIEW_POLICY
        if scope == "external_comparison_registered"
        else PHASE2_SAMPLE_VIEW_POLICY
    )
    if spec.get("phase2_sample_view_policy") != expected_view_policy:
        raise ValueError("build spec sample view policy drift")
    expected_clean_access = scope == "external_comparison_registered"
    if spec.get("clean_sample_access") is not expected_clean_access:
        raise ValueError(
            "build spec clean_sample_access must match its cache scope"
        )
    if spec.get("clean_derived_signal_access") is not False:
        raise ValueError("build spec must declare clean_derived_signal_access=false")
    single_observation_contract = {
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
    }
    if scope in {
        "stage2_target_old",
        "stage2_registered",
        CANONICAL_CACHE_SCOPE,
    }:
        failed = [
            key
            for key, expected in single_observation_contract.items()
            if spec.get(key) != expected
        ]
        if failed:
            raise ValueError(f"build spec single-observation contract drift: {failed}")
    if spec.get("star_ground_channel_impl") != "simplified_leo_residual":
        raise ValueError("build spec requires simplified_leo_residual")
    if canonical_scope:
        if spec.get("protocol_schema") != CANONICAL_PROTOCOL_SCHEMA:
            raise ValueError(
                f"canonical build spec protocol_schema must be {CANONICAL_PROTOCOL_SCHEMA}"
            )
        if "role_specs" in spec:
            raise ValueError("canonical build spec forbids role_specs")
        for key in ("canonical_inventory", "split_manifest"):
            if not str(spec.get(key, "")).strip():
                raise ValueError(f"canonical build spec requires nonempty {key}")
        role_specs: list[Mapping[str, Any]] = []
        overlay_by_role = {"target_old": True, "target_new": True}
    else:
        role_specs = list(spec.get("role_specs", []))
        if not role_specs or any(not isinstance(item, Mapping) for item in role_specs):
            raise ValueError("build spec role_specs must be a nonempty object list")
        roles = [str(item.get("role", "")) for item in role_specs]
        if len(set(roles)) != len(roles) or set(roles) != SCOPE_ROLES[scope]:
            raise ValueError(
                f"cache_scope={scope} requires exact roles={sorted(SCOPE_ROLES[scope])}"
            )
        for item in role_specs:
            for key in ("role", "pkl", "tx_ids", "rxs"):
                if not str(item.get(key, "")).strip():
                    raise ValueError(f"role spec is missing {key}")
            apply_overlay = item.get("apply_leo_overlay", True)
            if not isinstance(apply_overlay, bool):
                raise ValueError("role apply_leo_overlay must be boolean")
            if scope in {"stage2_target_old", "stage2_registered"}:
                if str(item.get("days", "")) != "0,1,2":
                    raise ValueError(
                        "single-observation formal cache requires independent day pool "
                        "0,1,2"
                    )
                if int(item.get("max_samples_per_tx", 0)) != (
                    PER_SCENARIO_SAMPLES_PER_TX * len(FORMAL_LEO_WEAK_SCENARIOS)
                ):
                    raise ValueError(
                        "single-observation formal cache requires 120 physical samples "
                        "per TX before scenario partition"
                    )
        overlay_by_role = {
            str(item["role"]): bool(item.get("apply_leo_overlay", True))
            for item in role_specs
        }
    if scope == "external_comparison_registered":
        if overlay_by_role != {"target_old": False, "target_new": True}:
            raise ValueError(
                "external comparison requires unmodified target_old and "
                "LEO-overlaid target_new rows"
            )
    elif not all(overlay_by_role.values()):
        raise ValueError("formal cache scopes require LEO overlay for every role")
    seeds = dict(spec.get("satellite_seed_by_scenario", {}))
    outputs = dict(spec.get("out_npz_by_scenario", {}))
    if tuple(seeds) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("satellite_seed_by_scenario must use the formal ordered tuple")
    if tuple(outputs) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("out_npz_by_scenario must use the formal ordered tuple")
    if canonical_scope and any(
        not str(outputs[name]).strip() for name in FORMAL_LEO_WEAK_SCENARIOS
    ):
        raise ValueError("canonical build spec output paths must be nonempty")
    if any(int(seeds[name]) < 0 for name in FORMAL_LEO_WEAK_SCENARIOS):
        raise ValueError("satellite seeds must be nonnegative")
    if not str(spec.get("out_manifest", "")).strip():
        raise ValueError("build spec requires out_manifest")
    exclusion_policy = spec.get("physical_sample_exclusion_policy")
    exclusion_reference = spec.get(
        "physical_sample_exclusion_reference_cache_set"
    )
    if (exclusion_policy is None) != (exclusion_reference is None):
        raise ValueError(
            "physical sample exclusion policy and reference cache must be "
            "declared together"
        )
    if exclusion_policy is not None:
        if scope not in {"stage2_target_old", "stage2_registered"}:
            raise ValueError(
                "reference-cache physical sample exclusion is Stage2-only"
            )
        if str(exclusion_policy) != REFERENCE_EXCLUSION_POLICY:
            raise ValueError("unsupported physical sample exclusion policy")
        if not str(exclusion_reference).strip():
            raise ValueError(
                "physical sample exclusion reference cache must be nonempty"
            )
    if not 1 <= int(spec.get("batch_size", 256)) <= 4096:
        raise ValueError("batch_size must be in [1,4096]")
    if int(spec.get("wisig_out_len", 256)) <= 0:
        raise ValueError("wisig_out_len must be positive")
    return dict(spec)


def _load_reference_cache_exclusions(
    spec: Mapping[str, Any], *, spec_dir: Path
) -> tuple[dict[tuple[str, str], set[int]], dict[str, Any] | None]:
    raw_reference = spec.get(
        "physical_sample_exclusion_reference_cache_set"
    )
    if raw_reference is None:
        return {}, None
    reference_path = _resolve(spec_dir, str(raw_reference))
    if not reference_path.is_file() or reference_path.is_symlink():
        raise FileNotFoundError(
            f"reference exclusion cache-set is missing: {reference_path}"
        )
    arrays, manifest, audit = load_verified_leo_weak_cache_set(
        reference_path,
        expected_scope=str(spec["cache_scope"]),
        allowed_roles=SCOPE_ROLES[str(spec["cache_scope"])],
    )
    exclusions: dict[tuple[str, str], set[int]] = {}
    seen_rows: set[tuple[str, str, int]] = set()
    counts_by_role: dict[str, int] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        current = arrays[scenario]
        roles = np.asarray(current["dataset_role"]).reshape(-1)
        dataset_hashes = np.asarray(
            current["source_dataset_sha256"]
        ).reshape(-1)
        record_indices = np.asarray(
            current["source_record_indices"], dtype=np.int64
        ).reshape(-1)
        if not (
            len(roles) == len(dataset_hashes) == len(record_indices)
        ):
            raise ValueError(
                "reference cache exclusion provenance length drift"
            )
        for raw_role, raw_hash, raw_index in zip(
            roles.tolist(),
            dataset_hashes.tolist(),
            record_indices.tolist(),
        ):
            role = str(raw_role)
            dataset_sha = str(raw_hash)
            index = int(raw_index)
            if role not in SCOPE_ROLES[str(spec["cache_scope"])]:
                raise ValueError(
                    f"reference cache contains unexpected role={role!r}"
                )
            row_key = (role, dataset_sha, index)
            if row_key in seen_rows:
                raise ValueError(
                    "reference cache repeats a source record across scenarios"
                )
            seen_rows.add(row_key)
            exclusions.setdefault((role, dataset_sha), set()).add(index)
            counts_by_role[role] = counts_by_role.get(role, 0) + 1
    return exclusions, {
        "policy": REFERENCE_EXCLUSION_POLICY,
        "reference_cache_set_sha256": sha256_file(reference_path),
        "reference_cache_set_id": str(manifest["cache_set_id"]),
        "reference_physical_sample_count": int(
            audit["physical_sample_count"]
        ),
        "excluded_source_record_count": len(seen_rows),
        "excluded_source_record_count_by_role": counts_by_role,
        "reference_path_exposed_to_phase2": False,
    }


def _resolve(base: Path, raw: str) -> Path:
    value = Path(str(raw))
    return value if value.is_absolute() else (base / value).resolve()


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, start=base)
    except ValueError:
        return str(path)


def _build_role_datasets(spec: Mapping[str, Any], *, spec_dir: Path):
    datasets: list[tuple[dict[str, Any], Any, dict[str, Any], Path]] = []
    dataset_hash_cache: dict[Path, str] = {}
    dataset_object_cache: dict[str, dict[str, Any]] = {}
    reference_exclusions, exclusion_audit = (
        _load_reference_cache_exclusions(spec, spec_dir=spec_dir)
    )
    for role_index, raw_role_spec in enumerate(spec["role_specs"]):
        role_spec = dict(raw_role_spec)
        pkl_path = _resolve(spec_dir, str(role_spec["pkl"]))
        if not pkl_path.is_file():
            raise FileNotFoundError(f"input dataset is missing: {pkl_path}")
        if pkl_path not in dataset_hash_cache:
            dataset_hash_cache[pkl_path] = sha256_file(pkl_path)
        dataset_sha = dataset_hash_cache[pkl_path]
        role = str(role_spec["role"])
        excluded_records = reference_exclusions.get(
            (role, dataset_sha), set()
        )
        if reference_exclusions and not excluded_records:
            raise ValueError(
                "reference cache has no matching source-record lineage for "
                f"role={role} dataset_sha256={dataset_sha}"
            )
        dataset_seed = int(spec.get("dataset_seed", 4070391)) + role_index * 10_007
        dataset, info = _build_wisig_dataset(
            pkl_path=str(pkl_path),
            tx_spec=str(role_spec["tx_ids"]),
            role=str(role_spec["role"]),
            equalized=str(spec.get("wisig_equalized", "1")),
            out_len=int(spec.get("wisig_out_len", 256)),
            domain=str(spec.get("wisig_domain", "rx_day")),
            days=role_spec.get("days"),
            rxs=str(role_spec["rxs"]),
            max_samples_per_combo=int(role_spec.get("max_samples_per_combo", 0)),
            max_samples_per_tx=int(role_spec.get("max_samples_per_tx", 0)),
            seed=dataset_seed,
            dataset_cache=dataset_object_cache,
            exclude_source_record_indices=excluded_records,
        )
        if len(dataset) <= 0:
            raise ValueError(f"role={role_spec['role']} produced no physical samples")
        safe_info = {
            "role": role,
            "dataset_sha256": dataset_sha,
            "dataset_size_bytes": int(pkl_path.stat().st_size),
            "requested_tx_ids": str(role_spec["tx_ids"]),
            "requested_rxs": str(role_spec["rxs"]),
            "requested_days": role_spec.get("days"),
            "dataset_seed": dataset_seed,
            "resolved_info": _json_safe(info),
            "physical_sample_count": int(len(dataset)),
            "reference_excluded_source_record_count": len(
                excluded_records
            ),
        }
        datasets.append((role_spec, dataset, safe_info, pkl_path))
    return datasets, exclusion_audit


def _partition_role_datasets_by_scenario(role_datasets, *, batch_size: int):
    """Assign every physical sample to exactly one scenario before overlay."""

    result: dict[str, list[tuple[dict[str, Any], Any, dict[str, Any], Path]]] = {
        scenario: [] for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    for role_spec, dataset, safe_info, pkl_path in role_datasets:
        records_by_tx_day: dict[str, dict[str, list[tuple[str, int]]]] = {}
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        offset = 0
        for batch in loader:
            if len(batch) != 4:
                raise ValueError("WiSig partitioner expects (x,y,d,meta) batches")
            count = int(batch[0].shape[0])
            tx_values = _meta_to_list(batch[3], "tx", count)
            rx_values = _meta_to_list(batch[3], "rx", count)
            day_values = _meta_to_list(batch[3], "day", count)
            eq_values = _meta_to_list(batch[3], "equalized", count)
            sig_values = _meta_to_list(batch[3], "sig_i", count)
            for local_index, tx_id in enumerate(tx_values):
                logical_index = offset + local_index
                stable_key = hashlib.sha256(
                    (
                        f"{safe_info['dataset_seed']}|{safe_info['dataset_sha256']}|"
                        f"{tx_id}|{rx_values[local_index]}|{day_values[local_index]}|"
                        f"{eq_values[local_index]}|{sig_values[local_index]}"
                    ).encode("utf-8")
                ).hexdigest()
                records_by_tx_day.setdefault(str(tx_id), {}).setdefault(
                    str(day_values[local_index]), []
                ).append((stable_key, logical_index))
            offset += count
        if offset != len(dataset):
            raise RuntimeError("pre-overlay physical sample partition row drift")

        expected_txs = [value for value in str(role_spec["tx_ids"]).split(",") if value]
        selected_by_scenario = {
            scenario: [] for scenario in FORMAL_LEO_WEAK_SCENARIOS
        }
        for tx_id in expected_txs:
            by_day = records_by_tx_day.get(tx_id, {})
            required = PER_SCENARIO_SAMPLES_PER_TX * len(
                FORMAL_LEO_WEAK_SCENARIOS
            )
            observed = sum(len(values) for values in by_day.values())
            if observed != required:
                raise ValueError(
                    "single-observation scenario assignment requires the exact "
                    "independent physical sample pool: "
                    f"scenario assignment: role={role_spec['role']} tx={tx_id} "
                    f"observed={observed} required={required}"
                )
            if len(by_day) < 2:
                raise ValueError(
                    "single-observation scenario assignment requires multiple "
                    f"independent days: role={role_spec['role']} tx={tx_id}"
                )
            ordered_by_day = {
                day: sorted(values)
                for day, values in sorted(by_day.items())
            }
            stable_keys = [
                key
                for values in ordered_by_day.values()
                for key, _index in values
            ]
            if len(stable_keys) != len(set(stable_keys)):
                raise ValueError(
                    f"duplicate stable physical identity before overlay: {tx_id}"
                )
            chosen: dict[str, list[int]] | None = None
            days = tuple(ordered_by_day)
            for rotations in itertools.product(
                range(len(FORMAL_LEO_WEAK_SCENARIOS)), repeat=len(days)
            ):
                candidate = {
                    scenario: [] for scenario in FORMAL_LEO_WEAK_SCENARIOS
                }
                for day, rotation in zip(days, rotations):
                    for rank, (_key, logical_index) in enumerate(
                        ordered_by_day[day]
                    ):
                        scenario = FORMAL_LEO_WEAK_SCENARIOS[
                            (rank + rotation) % len(FORMAL_LEO_WEAK_SCENARIOS)
                        ]
                        candidate[scenario].append(logical_index)
                if all(
                    len(candidate[scenario]) == PER_SCENARIO_SAMPLES_PER_TX
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                ):
                    chosen = candidate
                    break
            if chosen is None:
                raise ValueError(
                    "cannot construct an exact TX/day-stratified disjoint scenario "
                    f"assignment: role={role_spec['role']} tx={tx_id}"
                )
            for scenario in FORMAL_LEO_WEAK_SCENARIOS:
                selected_by_scenario[scenario].extend(chosen[scenario])

        all_selected = [
            index
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
            for index in selected_by_scenario[scenario]
        ]
        if len(all_selected) != len(set(all_selected)):
            raise RuntimeError(
                "PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION: "
                "pre-overlay scenario partitions overlap"
            )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS:
            scenario_info = dict(safe_info)
            scenario_info.update(
                {
                    "physical_sample_scenario_assignment_policy": (
                        SCENARIO_PARTITION_POLICY
                    ),
                    "assigned_scenario": scenario,
                    "assigned_physical_sample_count": len(
                        selected_by_scenario[scenario]
                    ),
                }
            )
            result[scenario].append(
                (
                    role_spec,
                    Subset(dataset, selected_by_scenario[scenario]),
                    scenario_info,
                    pkl_path,
                )
            )
    return result


def _source_record_index(dataset: Any, logical_index: int) -> int:
    """Resolve nested torch/WiSig subsets to the underlying dataset record."""

    current = dataset
    resolved = int(logical_index)
    while True:
        if isinstance(current, Subset):
            resolved = int(current.indices[resolved])
            current = current.dataset
            continue
        if hasattr(current, "selected") and hasattr(current, "base"):
            selected = np.asarray(current.selected)
            resolved = int(selected[resolved])
            current = current.base
            continue
        break
    if resolved < 0:
        raise ValueError("source record index must be nonnegative")
    return resolved


def _build_one_scenario(
    *,
    scenario: str,
    base_seed: int,
    role_datasets,
    spec: Mapping[str, Any],
    out_path: Path,
    builder_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite LEO cache: {out_path}")
    channel_config = dict(sat_channel_config_for_scenario(str(scenario)))
    channel_config.update(
        {
            "fs_hz": float(spec.get("sat_fs_hz", 25e6)),
            "fc_hz": float(spec.get("sat_fc_hz", 2.462e9)),
            "star_ground_channel_impl": "simplified_leo_residual",
        }
    )
    if str(channel_config.get("channel_model", "")) != "leo_residual":
        raise ValueError("formal LEO_weak cache requires channel_model=leo_residual")
    channel_hash = canonical_json_sha256(channel_config)

    buffers: dict[str, list[Any]] = {
        "leo_weak_iq": [],
        "raw_labels": [],
        "domain_labels": [],
        "tx_ids": [],
        "rx_ids": [],
        "day_ids": [],
        "eq_ids": [],
        "sig_ids": [],
        "source_dataset_sha256": [],
        "source_record_indices": [],
        "dataset_role": [],
        "channel_views": [],
        "sat_scenarios": [],
        "satellite_seeds": [],
        "overlay_applied": [],
        "sample_ids": [],
        "post_channel_iq_sha256": [],
        "overlay_ids": [],
    }
    role_seed_map: dict[str, int] = {}
    channel_meta_keys: set[str] = set()
    role_inputs: list[dict[str, Any]] = []
    for role_index, (role_spec, dataset, safe_info, _pkl_path) in enumerate(
        role_datasets
    ):
        role = str(role_spec["role"])
        apply_overlay = bool(role_spec.get("apply_leo_overlay", True))
        role_seed = (
            int(base_seed) + role_index * 1_000_003
            if apply_overlay
            else -1
        )
        role_seed_map[role] = role_seed
        generator = make_torch_generator(device, role_seed)
        loader = DataLoader(
            dataset,
            batch_size=int(spec.get("batch_size", 256)),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        observed = 0
        for batch in loader:
            if len(batch) != 4:
                raise ValueError("WiSig cache builder expects (x,y,d,meta) batches")
            x, y, domain, meta = batch
            x = x.to(device, non_blocking=True)
            if apply_overlay:
                leo, channel_meta = apply_sat_channel_for_scenario(
                    x,
                    str(scenario),
                    argparse.Namespace(
                        sat_fs_hz=float(spec.get("sat_fs_hz", 25e6)),
                        sat_fc_hz=float(spec.get("sat_fc_hz", 2.462e9)),
                    ),
                    gen=generator,
                    return_meta=True,
                )
                if not isinstance(channel_meta, Mapping):
                    raise RuntimeError("LEO overlay did not return channel metadata")
                if str(channel_meta.get("channel_model", "")) != "leo_residual":
                    raise RuntimeError("LEO overlay metadata channel_model drift")
                channel_meta_keys.update(str(key) for key in channel_meta)
            else:
                leo = x
            leo_np = leo.detach().cpu().float().numpy().astype(np.float32)
            count = int(leo_np.shape[0])
            meta_tx = _meta_to_list(meta, "tx", count)
            meta_rx = _meta_to_list(meta, "rx", count)
            meta_day = _meta_to_list(meta, "day", count)
            meta_eq = _meta_to_list(meta, "equalized", count)
            meta_sig = _meta_to_list(meta, "sig_i", count)
            labels = [int(value) for value in y.detach().cpu().reshape(-1).tolist()]
            domains = [
                int(value) for value in domain.detach().cpu().reshape(-1).tolist()
            ]
            for index in range(count):
                source_record_index = _source_record_index(dataset, observed + index)
                sample_id = physical_sample_id_from_values(
                    dataset_sha256=str(safe_info["dataset_sha256"]),
                    source_record_index=source_record_index,
                    role=role,
                    tx_id=str(meta_tx[index]),
                    rx_id=str(meta_rx[index]),
                    day_id=str(meta_day[index]),
                    eq_id=str(meta_eq[index]),
                    sig_id=str(meta_sig[index]),
                )
                iq_hash = post_channel_iq_sha256(leo_np[index])
                evidence_id = overlay_id(
                    sample_id=sample_id,
                    scenario=str(scenario),
                    satellite_seed=role_seed,
                    channel_config_sha256=channel_hash,
                    iq_sha256=iq_hash,
                )
                buffers["sample_ids"].append(sample_id)
                buffers["source_dataset_sha256"].append(
                    str(safe_info["dataset_sha256"])
                )
                buffers["source_record_indices"].append(source_record_index)
                buffers["post_channel_iq_sha256"].append(iq_hash)
                buffers["overlay_ids"].append(evidence_id)
            buffers["leo_weak_iq"].append(leo_np)
            buffers["raw_labels"].extend(labels)
            buffers["domain_labels"].extend(domains)
            buffers["tx_ids"].extend(meta_tx)
            buffers["rx_ids"].extend(meta_rx)
            buffers["day_ids"].extend(meta_day)
            buffers["eq_ids"].extend(meta_eq)
            buffers["sig_ids"].extend(meta_sig)
            buffers["dataset_role"].extend([role] * count)
            buffers["channel_views"].extend(
                ["rx_base" if apply_overlay else "unmodified_received_iq"] * count
            )
            buffers["sat_scenarios"].extend([str(scenario)] * count)
            buffers["satellite_seeds"].extend([role_seed] * count)
            buffers["overlay_applied"].extend([apply_overlay] * count)
            observed += count
        if observed != int(len(dataset)):
            raise RuntimeError(
                f"cache builder row count drift for role={role}: {observed}!={len(dataset)}"
            )
        role_inputs.append(safe_info)

    sample_ids = [str(value) for value in buffers["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("cache builder encountered duplicate physical sample IDs")
    row_count = len(sample_ids)
    target_new_only_overlay = (
        str(spec["cache_scope"]) == "external_comparison_registered"
    )
    manifest = {
        "schema": LEO_WEAK_CACHE_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": (
            EXTERNAL_COMPARISON_SAMPLE_VIEW_POLICY
            if target_new_only_overlay
            else PHASE2_SAMPLE_VIEW_POLICY
        ),
        "clean_sample_access": target_new_only_overlay,
        "clean_derived_signal_access": False,
        "contains_post_channel_iq_only": not target_new_only_overlay,
        "contains_clean_rows": target_new_only_overlay,
        "target_channel_view": (
            "mixed_old_received_new_leo_weak"
            if target_new_only_overlay
            else "leo_weak_only"
        ),
        "target_channel_scenarios": [str(scenario)],
        "scenario": str(scenario),
        "iq_array_key": "leo_weak_iq",
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "overlay_role_policy": (
            "target_new_only" if target_new_only_overlay else "all_roles"
        ),
        "star_ground_channel_impl": "simplified_leo_residual",
        "channel_model": "leo_residual",
        "channel_config": _json_safe(channel_config),
        "channel_config_sha256": channel_hash,
        "builder_sha256": str(builder_sha256),
        "build_spec_sha256": canonical_json_sha256(spec),
        "output_roles": [str(item[0]["role"]) for item in role_datasets],
        "role_satellite_seeds": role_seed_map,
        "role_inputs": role_inputs,
        "row_count": row_count,
        "physical_sample_ids_sha256": ids_sha256(sample_ids),
        "post_channel_iq_sha256_root": ids_sha256(
            [str(value) for value in buffers["post_channel_iq_sha256"]]
        ),
        "overlay_ids_sha256": ids_sha256(
            [str(value) for value in buffers["overlay_ids"]]
        ),
        "channel_meta_keys": sorted(channel_meta_keys),
        "sample_overlay_provenance_fields": [
            "sample_ids",
            "source_dataset_sha256",
            "source_record_indices",
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
    }
    if str(spec["cache_scope"]) in {"stage2_target_old", "stage2_registered"}:
        manifest.update(
            {
                "phase2_physical_sample_observation_policy": (
                    "single_leo_weak_observation_per_physical_sample"
                ),
                "phase2_cross_scenario_physical_sample_reuse": False,
                "phase2_additional_leo_channel_state_generation": False,
                "phase2_post_reception_equalization_augmentation_transform_allowed": (
                    True
                ),
                "phase2_post_reception_view_from_fixed_received_iq_only": True,
                "phase2_post_reception_view_counts_as_additional_physical_sample": (
                    False
                ),
                "phase2_physical_sample_root_id_policy": (
                    "immutable_preoverlay_lineage_token"
                ),
                "phase2_query_post_reception_view_fit_access": False,
                "physical_sample_scenario_assignment_policy": (
                    SCENARIO_PARTITION_POLICY
                ),
            }
        )
    payload = {
        "leo_weak_iq": np.concatenate(buffers["leo_weak_iq"], axis=0).astype(
            np.float32
        ),
        "raw_labels": np.asarray(buffers["raw_labels"], dtype=np.int64),
        "domain_labels": np.asarray(buffers["domain_labels"], dtype=np.int64),
        "tx_ids": np.asarray(buffers["tx_ids"]),
        "rx_ids": np.asarray(buffers["rx_ids"]),
        "day_ids": np.asarray(buffers["day_ids"]),
        "eq_ids": np.asarray(buffers["eq_ids"]),
        "sig_ids": np.asarray(buffers["sig_ids"]),
        "source_dataset_sha256": np.asarray(
            buffers["source_dataset_sha256"]
        ),
        "source_record_indices": np.asarray(
            buffers["source_record_indices"], dtype=np.int64
        ),
        "dataset_role": np.asarray(buffers["dataset_role"]),
        "channel_views": np.asarray(buffers["channel_views"]),
        "sat_scenarios": np.asarray(buffers["sat_scenarios"]),
        "satellite_seeds": np.asarray(
            buffers["satellite_seeds"], dtype=np.int64
        ),
        "overlay_applied": np.asarray(buffers["overlay_applied"], dtype=bool),
        "sample_ids": np.asarray(sample_ids),
        "post_channel_iq_sha256": np.asarray(
            buffers["post_channel_iq_sha256"]
        ),
        "overlay_ids": np.asarray(buffers["overlay_ids"]),
        "manifest_json": np.asarray(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    _arrays, _loaded_manifest, audit = load_verified_leo_weak_cache(
        out_path,
        expected_scenario=str(scenario),
        allowed_roles=manifest["output_roles"],
        allow_target_new_only_overlay=target_new_only_overlay,
    )
    audit["physical_sample_ids"] = sample_ids
    return audit


def _exact_nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative exact integer")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _canonical_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _open_canonical_inventory_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"canonical inventory is missing: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _sequence_of_nonempty_strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(_nonempty_text(item, f"{name} member") for item in value)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _validate_canonical_split_manifest(
    split_path: Path,
    connection: sqlite3.Connection,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json_object(split_path, "canonical split manifest")
    if payload.get("schema") != CANONICAL_SPLIT_MANIFEST_SCHEMA:
        raise ValueError(
            f"canonical split schema must be {CANONICAL_SPLIT_MANIFEST_SCHEMA}"
        )
    if payload.get("protocol_schema") != CANONICAL_PROTOCOL_SCHEMA:
        raise ValueError(
            f"canonical split protocol_schema must be {CANONICAL_PROTOCOL_SCHEMA}"
        )
    query_policy = _nonempty_text(payload.get("query_policy"), "query_policy")
    if query_policy not in CANONICAL_QUERY_POLICIES:
        raise ValueError("canonical split declares an unsupported query_policy")
    k = _exact_nonnegative_integer(payload.get("k"), "canonical split K")
    if k not in CANONICAL_K_VALUES:
        raise ValueError(f"canonical split K must be one of {CANONICAL_K_VALUES}")
    registered_tx_ids = _sequence_of_nonempty_strings(
        payload.get("registered_tx_ids"), "registered_tx_ids"
    )
    if len(registered_tx_ids) <= CANONICAL_OLD_TX_COUNT:
        raise ValueError(
            "canonical split must contain six old TX IDs followed by new TX IDs"
        )
    eligible_receivers = _sequence_of_nonempty_strings(
        payload.get("eligible_receivers"), "eligible_receivers"
    )
    registered_set = set(registered_tx_ids)
    eligible_receiver_set = set(eligible_receivers)
    old_tx_set = set(registered_tx_ids[:CANONICAL_OLD_TX_COUNT])

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("canonical split rows must be a nonempty list")
    seen_roles: dict[str, str] = {}
    support_ids: set[str] = set()
    query_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    scenario_dataset_roles: dict[str, set[str]] = {
        scenario: set() for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    for row_index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"canonical split row {row_index} must be an object")
        physical_id = _nonempty_text(
            raw_row.get("physical_sample_id"),
            f"canonical split row {row_index} physical_sample_id",
        )
        role = _nonempty_text(raw_row.get("role"), f"row {row_index} role")
        if role not in {"support", "query"}:
            raise ValueError("canonical split role must be support or query")
        previous_role = seen_roles.get(physical_id)
        if previous_role is not None:
            if previous_role != role:
                raise ValueError(
                    "canonical split support/query overlap for duplicate physical sample ID"
                )
            raise ValueError("canonical split contains a duplicate physical sample ID")
        seen_roles[physical_id] = role
        (support_ids if role == "support" else query_ids).add(physical_id)
        rank = _exact_nonnegative_integer(
            raw_row.get("rank"), f"canonical split row {row_index} rank"
        )
        scene = _nonempty_text(raw_row.get("scene"), f"row {row_index} scene")
        if scene not in FORMAL_LEO_WEAK_SCENARIOS:
            raise ValueError("canonical split row uses an unsupported LEO scene")
        source_asset = _nonempty_text(
            raw_row.get("source_asset"), f"row {row_index} source_asset"
        )
        source_index = _exact_nonnegative_integer(
            raw_row.get("source_record_index"),
            f"canonical split row {row_index} source_record_index",
        )
        rx_id = _nonempty_text(raw_row.get("rx_id"), f"row {row_index} rx_id")
        day_id = _nonempty_text(raw_row.get("day_id"), f"row {row_index} day_id")

        canonical_row = connection.execute(
            """
            SELECT tx_id, rx_id, day_id, eq_id, sig_id, iq_sha256,
                   preferred_asset, preferred_source_record_index, eligible
            FROM canonical_records
            WHERE physical_sample_id = ?
            """,
            (physical_id,),
        ).fetchone()
        if canonical_row is None:
            raise ValueError(
                f"canonical split physical sample ID is absent from inventory: {physical_id}"
            )
        (
            tx_id,
            inventory_rx,
            inventory_day,
            eq_id,
            sig_id,
            iq_sha256,
            preferred_asset,
            preferred_source_index,
            eligible,
        ) = canonical_row
        tx_id = str(tx_id)
        inventory_rx = str(inventory_rx)
        inventory_day = str(inventory_day)
        eq_id = str(eq_id)
        sig_id = str(sig_id)
        iq_sha256 = str(iq_sha256)
        if int(eligible) != 1:
            raise ValueError("canonical split references an ineligible inventory row")
        expected_physical_id = canonical_physical_id(
            tx_id, inventory_rx, inventory_day, eq_id, sig_id
        )
        if expected_physical_id != physical_id:
            raise ValueError(
                "canonical inventory coordinate does not match physical sample ID"
            )
        if tx_id not in registered_set:
            raise ValueError("canonical split contains an unregistered TX")
        if rx_id != inventory_rx or day_id != inventory_day:
            raise ValueError("canonical split row identity disagrees with inventory")
        if rx_id not in eligible_receiver_set:
            raise ValueError("canonical split row uses an undeclared receiver")
        if source_asset != str(preferred_asset) or source_index != int(
            preferred_source_index
        ):
            raise ValueError(
                "canonical split preferred materialization reference disagrees with inventory"
            )
        if role == "support":
            if _nonempty_text(raw_row.get("tx_id"), "support tx_id") != tx_id:
                raise ValueError("canonical support TX label disagrees with inventory")
        elif "tx_id" in raw_row:
            raise ValueError("canonical query rows must omit tx_id")

        source_row = connection.execute(
            """
            SELECT dataset_path, iq_sha256
            FROM record_sources
            WHERE physical_sample_id = ? AND asset_name = ? AND source_record_index = ?
            """,
            (physical_id, source_asset, source_index),
        ).fetchone()
        if source_row is None:
            raise ValueError(
                "canonical preferred materialization reference is absent from record_sources"
            )
        dataset_path, source_iq_sha256 = source_row
        if str(source_iq_sha256) != iq_sha256:
            raise ValueError(
                "canonical inventory digest disagrees with preferred record source digest"
            )
        dataset_role = "target_old" if tx_id in old_tx_set else "target_new"
        scenario_dataset_roles[scene].add(dataset_role)
        rows.append(
            {
                "physical_sample_id": physical_id,
                "source_asset": source_asset,
                "source_record_index": source_index,
                "dataset_path": str(dataset_path),
                "tx_id": tx_id,
                "rx_id": inventory_rx,
                "day_id": inventory_day,
                "eq_id": eq_id,
                "sig_id": sig_id,
                "iq_sha256": iq_sha256,
                "scene": scene,
                "split_role": role,
                "split_rank": rank,
                "dataset_role": dataset_role,
            }
        )

    if support_ids.intersection(query_ids):
        raise ValueError("canonical split support/query physical IDs overlap")
    required_dataset_roles = SCOPE_ROLES[CANONICAL_CACHE_SCOPE]
    for scenario, observed_roles in scenario_dataset_roles.items():
        if observed_roles != required_dataset_roles:
            raise ValueError(
                f"canonical split scene {scenario} must contain exact old/new role coverage"
            )
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("canonical split counts must be an object")
    expected_counts = {
        "registered_tx_count": len(registered_tx_ids),
        "eligible_receiver_count": len(eligible_receivers),
        "eligible_count": len(rows),
        "support_count": len(support_ids),
        "query_count": len(query_ids),
        "row_count": len(rows),
    }
    for key, expected in expected_counts.items():
        if _exact_nonnegative_integer(counts.get(key), f"counts.{key}") != expected:
            raise ValueError(f"canonical split count inconsistency for {key}")
    return payload, rows


def _resolve_inventory_source_path(inventory_path: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path))
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (inventory_path.parent / candidate).resolve()
    )


def _load_pickle_payload(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical source PKL is missing: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("canonical source PKL must contain a mapping")
    required = {
        "data",
        "tx_list",
        "rx_list",
        "capture_date_list",
        "equalized_list",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"canonical source PKL is missing required fields: {missing}")
    return payload


def _prepare_canonical_sources(
    rows: list[dict[str, Any]],
    *,
    inventory_path: Path,
    spec: Mapping[str, Any],
) -> dict[Path, dict[str, Any]]:
    try:
        equalized = int(str(spec.get("wisig_equalized", "1")))
    except ValueError:
        raise ValueError("canonical wisig_equalized must be one exact integer label") from None
    sources: dict[Path, dict[str, Any]] = {}
    for row in rows:
        source_path = _resolve_inventory_source_path(
            inventory_path, str(row["dataset_path"])
        )
        row["source_path"] = source_path
        if source_path not in sources:
            payload = _load_pickle_payload(source_path)
            dataset = WiSigCompactDataset(
                dict(payload),
                out_len=int(spec.get("wisig_out_len", 256)),
                equalized=equalized,
                domain=str(spec.get("wisig_domain", "rx_day")),
                max_samples_per_combo=None,
                sample_strategy="front",
                seed=int(spec.get("dataset_seed", 4070391)),
                build_index=True,
            )
            sources[source_path] = {
                "payload": payload,
                "dataset": dataset,
                "dataset_sha256": sha256_file(source_path),
            }

    for row in rows:
        source = sources[row["source_path"]]
        payload = source["payload"]
        dataset: WiSigCompactDataset = source["dataset"]
        source_index = int(row["source_record_index"])
        if source_index >= len(dataset):
            raise ValueError(
                "canonical preferred source_record_index is outside Task 1 traversal"
            )
        item = dataset.index[source_index]
        coordinate = (
            _canonical_label(payload["tx_list"][item.tx_i]),
            _canonical_label(payload["rx_list"][item.rx_i]),
            _canonical_label(payload["capture_date_list"][item.day_i]),
            _canonical_label(payload["equalized_list"][item.eq_i]),
            str(item.sig_i),
        )
        expected_coordinate = (
            row["tx_id"],
            row["rx_id"],
            row["day_id"],
            row["eq_id"],
            row["sig_id"],
        )
        if coordinate != expected_coordinate:
            raise ValueError(
                "canonical coordinate from preferred source disagrees with inventory"
            )
        if canonical_physical_id(*coordinate) != row["physical_sample_id"]:
            raise ValueError(
                "preferred source canonical physical sample ID does not match inventory"
            )
        raw_iq = payload["data"][item.tx_i][item.rx_i][item.day_i][item.eq_i][
            item.sig_i
        ]
        raw_digest = hashlib.sha256(
            np.ascontiguousarray(np.asarray(raw_iq, dtype=np.float32)).tobytes(
                order="C"
            )
        ).hexdigest()
        if raw_digest != row["iq_sha256"]:
            raise ValueError(
                "preferred source raw pre-overlay IQ digest does not match inventory"
            )
        row["dataset_sha256"] = source["dataset_sha256"]
    return sources


def _canonical_scenario_payload(
    *,
    scenario: str,
    base_seed: int,
    rows: list[dict[str, Any]],
    sources: Mapping[Path, Mapping[str, Any]],
    spec: Mapping[str, Any],
    builder_sha256: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not rows:
        raise ValueError(f"canonical split has no rows for {scenario}")
    channel_config = dict(sat_channel_config_for_scenario(str(scenario)))
    channel_config.update(
        {
            "fs_hz": float(spec.get("sat_fs_hz", 25e6)),
            "fc_hz": float(spec.get("sat_fc_hz", 2.462e9)),
            "star_ground_channel_impl": "simplified_leo_residual",
        }
    )
    if str(channel_config.get("channel_model", "")) != "leo_residual":
        raise ValueError("formal LEO_weak cache requires channel_model=leo_residual")
    channel_hash = canonical_json_sha256(channel_config)
    generator = make_torch_generator(device, int(base_seed))
    buffers: dict[str, list[Any]] = {
        "leo_weak_iq": [],
        "raw_labels": [],
        "domain_labels": [],
        "tx_ids": [],
        "rx_ids": [],
        "day_ids": [],
        "eq_ids": [],
        "sig_ids": [],
        "source_dataset_sha256": [],
        "source_record_indices": [],
        "dataset_role": [],
        "channel_views": [],
        "sat_scenarios": [],
        "satellite_seeds": [],
        "overlay_applied": [],
        "sample_ids": [],
        "post_channel_iq_sha256": [],
        "overlay_ids": [],
        "canonical_physical_sample_ids": [],
        "split_roles": [],
        "split_ranks": [],
    }
    channel_meta_keys: set[str] = set()
    batch_size = int(spec.get("batch_size", 256))
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        samples = [
            sources[row["source_path"]]["dataset"][row["source_record_index"]]
            for row in batch_rows
        ]
        x = torch.stack([sample[0] for sample in samples], dim=0).to(
            device, non_blocking=True
        )
        leo, channel_meta = apply_sat_channel_for_scenario(
            x,
            str(scenario),
            argparse.Namespace(
                sat_fs_hz=float(spec.get("sat_fs_hz", 25e6)),
                sat_fc_hz=float(spec.get("sat_fc_hz", 2.462e9)),
            ),
            gen=generator,
            return_meta=True,
        )
        if not isinstance(channel_meta, Mapping):
            raise RuntimeError("LEO overlay did not return channel metadata")
        if str(channel_meta.get("channel_model", "")) != "leo_residual":
            raise RuntimeError("LEO overlay metadata channel_model drift")
        channel_meta_keys.update(str(key) for key in channel_meta)
        leo_np = leo.detach().cpu().float().numpy().astype(np.float32)
        if int(leo_np.shape[0]) != len(batch_rows):
            raise RuntimeError("canonical overlay row count drift")
        buffers["leo_weak_iq"].append(leo_np)
        for local_index, (row, sample) in enumerate(zip(batch_rows, samples)):
            sample_id = str(row["physical_sample_id"])
            iq_hash = post_channel_iq_sha256(leo_np[local_index])
            evidence_id = overlay_id(
                sample_id=sample_id,
                scenario=str(scenario),
                satellite_seed=int(base_seed),
                channel_config_sha256=channel_hash,
                iq_sha256=iq_hash,
            )
            buffers["raw_labels"].append(int(sample[1]))
            buffers["domain_labels"].append(int(sample[2]))
            buffers["tx_ids"].append(str(row["tx_id"]))
            buffers["rx_ids"].append(str(row["rx_id"]))
            buffers["day_ids"].append(str(row["day_id"]))
            buffers["eq_ids"].append(str(row["eq_id"]))
            buffers["sig_ids"].append(str(row["sig_id"]))
            buffers["source_dataset_sha256"].append(str(row["dataset_sha256"]))
            buffers["source_record_indices"].append(int(row["source_record_index"]))
            buffers["dataset_role"].append(str(row["dataset_role"]))
            buffers["channel_views"].append("rx_base")
            buffers["sat_scenarios"].append(str(scenario))
            buffers["satellite_seeds"].append(int(base_seed))
            buffers["overlay_applied"].append(True)
            buffers["sample_ids"].append(sample_id)
            buffers["post_channel_iq_sha256"].append(iq_hash)
            buffers["overlay_ids"].append(evidence_id)
            buffers["canonical_physical_sample_ids"].append(sample_id)
            buffers["split_roles"].append(str(row["split_role"]))
            buffers["split_ranks"].append(int(row["split_rank"]))

    sample_ids = [str(value) for value in buffers["sample_ids"]]
    output_roles = ["target_old", "target_new"]
    manifest = {
        "schema": LEO_WEAK_CACHE_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "contains_post_channel_iq_only": True,
        "contains_clean_rows": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": [str(scenario)],
        "scenario": str(scenario),
        "iq_array_key": "leo_weak_iq",
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "overlay_role_policy": "all_roles",
        "star_ground_channel_impl": "simplified_leo_residual",
        "channel_model": "leo_residual",
        "channel_config": _json_safe(channel_config),
        "channel_config_sha256": channel_hash,
        "builder_sha256": str(builder_sha256),
        "build_spec_sha256": canonical_json_sha256(spec),
        "output_roles": output_roles,
        "role_satellite_seeds": {
            role: int(base_seed) for role in output_roles
        },
        "role_inputs": [
            {
                "source_asset": asset,
                "physical_sample_count": sum(
                    str(row["source_asset"]) == asset for row in rows
                ),
            }
            for asset in sorted({str(row["source_asset"]) for row in rows})
        ],
        "row_count": len(sample_ids),
        "physical_sample_ids_sha256": ids_sha256(sample_ids),
        "post_channel_iq_sha256_root": ids_sha256(
            [str(value) for value in buffers["post_channel_iq_sha256"]]
        ),
        "overlay_ids_sha256": ids_sha256(
            [str(value) for value in buffers["overlay_ids"]]
        ),
        "channel_meta_keys": sorted(channel_meta_keys),
        "sample_overlay_provenance_fields": [
            "sample_ids",
            "source_dataset_sha256",
            "source_record_indices",
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
        **{
            key: value
            for key, value in _single_observation_manifest_contract().items()
        },
    }
    payload = {
        "leo_weak_iq": np.concatenate(buffers["leo_weak_iq"], axis=0).astype(
            np.float32
        ),
        "raw_labels": np.asarray(buffers["raw_labels"], dtype=np.int64),
        "domain_labels": np.asarray(buffers["domain_labels"], dtype=np.int64),
        "tx_ids": np.asarray(buffers["tx_ids"]),
        "rx_ids": np.asarray(buffers["rx_ids"]),
        "day_ids": np.asarray(buffers["day_ids"]),
        "eq_ids": np.asarray(buffers["eq_ids"]),
        "sig_ids": np.asarray(buffers["sig_ids"]),
        "source_dataset_sha256": np.asarray(buffers["source_dataset_sha256"]),
        "source_record_indices": np.asarray(
            buffers["source_record_indices"], dtype=np.int64
        ),
        "dataset_role": np.asarray(buffers["dataset_role"]),
        "channel_views": np.asarray(buffers["channel_views"]),
        "sat_scenarios": np.asarray(buffers["sat_scenarios"]),
        "satellite_seeds": np.asarray(buffers["satellite_seeds"], dtype=np.int64),
        "overlay_applied": np.asarray(buffers["overlay_applied"], dtype=bool),
        "sample_ids": np.asarray(sample_ids),
        "post_channel_iq_sha256": np.asarray(buffers["post_channel_iq_sha256"]),
        "overlay_ids": np.asarray(buffers["overlay_ids"]),
        "canonical_physical_sample_ids": np.asarray(
            buffers["canonical_physical_sample_ids"]
        ),
        "split_roles": np.asarray(buffers["split_roles"]),
        "split_ranks": np.asarray(buffers["split_ranks"], dtype=np.int64),
        "manifest_json": np.asarray(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        ),
    }
    return payload, manifest


def _single_observation_manifest_contract() -> dict[str, Any]:
    return {
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
    }


def _build_canonical_cache_set(
    path: Path,
    spec: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    spec_dir = path.parent
    inventory_path = _resolve(spec_dir, str(spec["canonical_inventory"]))
    split_path = _resolve(spec_dir, str(spec["split_manifest"]))
    out_manifest = _resolve(spec_dir, str(spec["out_manifest"]))
    cache_paths = {
        scenario: _resolve(
            spec_dir, str(dict(spec["out_npz_by_scenario"])[scenario])
        )
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    all_outputs = (out_manifest, *cache_paths.values())
    if len({candidate.resolve() for candidate in all_outputs}) != len(all_outputs):
        raise ValueError("canonical cache outputs must use distinct paths")
    for candidate in all_outputs:
        if candidate.exists():
            raise FileExistsError(
                f"refusing to overwrite canonical LEO cache output: {candidate}"
            )

    connection = _open_canonical_inventory_read_only(inventory_path)
    try:
        split_manifest, rows = _validate_canonical_split_manifest(
            split_path, connection
        )
    finally:
        connection.close()
    sources = _prepare_canonical_sources(
        rows, inventory_path=inventory_path, spec=spec
    )

    builder_hash = sha256_file(Path(__file__))
    cache_audits: dict[str, Any] = {}
    physical_roots: dict[str, str] = {}
    physical_ids_by_scenario: dict[str, list[str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        scenario_rows = [row for row in rows if row["scene"] == scenario]
        payload, _manifest = _canonical_scenario_payload(
            scenario=scenario,
            base_seed=int(dict(spec["satellite_seed_by_scenario"])[scenario]),
            rows=scenario_rows,
            sources=sources,
            spec=spec,
            builder_sha256=builder_hash,
            device=device,
        )
        out_path = cache_paths[scenario]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("xb") as handle:
            np.savez(handle, **payload)
        _arrays, _loaded_manifest, audit = load_verified_leo_weak_cache(
            out_path,
            expected_scenario=scenario,
            allowed_roles=SCOPE_ROLES[CANONICAL_CACHE_SCOPE],
        )
        current_ids = [
            str(value)
            for value in np.asarray(
                _arrays["canonical_physical_sample_ids"]
            ).tolist()
        ]
        physical_ids_by_scenario[scenario] = current_ids
        physical_roots[scenario] = str(audit["physical_sample_ids_sha256"])
        cache_audits[scenario] = audit

    assignment_root = canonical_json_sha256(physical_ids_by_scenario)
    output_roles = ["target_old", "target_new"]
    set_manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_set_id": str(spec.get("cache_set_id", path.stem)),
        "cache_scope": CANONICAL_CACHE_SCOPE,
        "protocol_schema": CANONICAL_PROTOCOL_SCHEMA,
        "profile_id": str(split_manifest["profile_id"]),
        "query_policy": str(split_manifest["query_policy"]),
        "k": int(split_manifest["k"]),
        "capsule_id": str(split_manifest["capsule_id"]),
        "split_id": str(split_manifest["split_id"]),
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "overlay_role_policy": "all_roles",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": output_roles,
        "cache_npz_by_scenario": {
            scenario: _relative_or_absolute(
                cache_paths[scenario], out_manifest.parent
            )
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_sha256_by_scenario": {
            scenario: sha256_file(cache_paths[scenario])
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_audits": cache_audits,
        "builder_sha256": builder_hash,
        "build_spec_sha256": canonical_json_sha256(spec),
        "build_spec_path_exposed_to_phase2": False,
        **_single_observation_manifest_contract(),
        "physical_sample_ids_sha256_by_scenario": physical_roots,
        "physical_sample_scenario_assignment_sha256": assignment_root,
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(set_manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _verified_arrays, _verified_manifest, verified_audit = (
        load_verified_leo_weak_cache_set(
            out_manifest,
            expected_scope=CANONICAL_CACHE_SCOPE,
            allowed_roles=SCOPE_ROLES[CANONICAL_CACHE_SCOPE],
        )
    )
    return {
        "cache_set_manifest": str(out_manifest),
        "cache_set_manifest_sha256": sha256_file(out_manifest),
        "cache_scope": CANONICAL_CACHE_SCOPE,
        "output_roles": output_roles,
        "physical_sample_ids_sha256_by_scenario": physical_roots,
        "physical_sample_scenario_assignment_sha256": assignment_root,
        "cache_audits": cache_audits,
        "canonical_cache_set_audit": verified_audit,
        "physical_sample_exclusion_audit": None,
    }


def build_cache_set(spec_path: str | Path, *, device: torch.device) -> dict[str, Any]:
    path = Path(spec_path).resolve()
    spec = validate_build_spec(
        json.loads(path.read_text(encoding="utf-8-sig"))
    )
    if str(spec["cache_scope"]) == CANONICAL_CACHE_SCOPE:
        return _build_canonical_cache_set(path, spec, device=device)
    out_manifest = _resolve(path.parent, str(spec["out_manifest"]))
    if out_manifest.exists():
        raise FileExistsError(
            f"refusing to overwrite LEO cache-set manifest: {out_manifest}"
        )
    role_datasets, exclusion_audit = _build_role_datasets(
        spec, spec_dir=path.parent
    )
    single_observation_scope = str(spec["cache_scope"]) in {
        "stage2_target_old",
        "stage2_registered",
    }
    role_datasets_by_scenario = (
        _partition_role_datasets_by_scenario(
            role_datasets,
            batch_size=int(spec.get("batch_size", 256)),
        )
        if single_observation_scope
        else {
            scenario: role_datasets for scenario in FORMAL_LEO_WEAK_SCENARIOS
        }
    )
    builder_hash = sha256_file(Path(__file__))
    cache_paths: dict[str, Path] = {}
    cache_audits: dict[str, Any] = {}
    physical_roots: dict[str, str] = {}
    physical_ids_by_scenario: dict[str, list[str]] = {}
    observed_physical_ids: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        out_path = _resolve(
            path.parent, str(dict(spec["out_npz_by_scenario"])[scenario])
        )
        audit = _build_one_scenario(
            scenario=scenario,
            base_seed=int(dict(spec["satellite_seed_by_scenario"])[scenario]),
            role_datasets=role_datasets_by_scenario[scenario],
            spec=spec,
            out_path=out_path,
            builder_sha256=builder_hash,
            device=device,
        )
        current_ids = [str(value) for value in audit["physical_sample_ids"]]
        overlap = sorted(observed_physical_ids.intersection(current_ids))
        if single_observation_scope and overlap:
            raise RuntimeError(
                "PROTOCOL_INVALID_FOR_PHASE2_SINGLE_OBSERVATION: "
                f"generated scenario physical IDs overlap: {overlap[:3]}"
            )
        observed_physical_ids.update(current_ids)
        physical_ids_by_scenario[scenario] = current_ids
        physical_roots[scenario] = str(audit["physical_sample_ids_sha256"])
        cache_paths[scenario] = out_path
        cache_audit = dict(audit)
        cache_audit.pop("physical_sample_ids", None)
        cache_audits[scenario] = cache_audit

    output_roles = [str(item[0]["role"]) for item in role_datasets]
    assignment_root = canonical_json_sha256(physical_ids_by_scenario)
    set_manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_set_id": str(spec.get("cache_set_id", path.stem)),
        "cache_scope": str(spec["cache_scope"]),
        "phase2_sample_view_policy": (
            EXTERNAL_COMPARISON_SAMPLE_VIEW_POLICY
            if str(spec["cache_scope"]) == "external_comparison_registered"
            else PHASE2_SAMPLE_VIEW_POLICY
        ),
        "clean_sample_access": (
            str(spec["cache_scope"]) == "external_comparison_registered"
        ),
        "clean_derived_signal_access": False,
        "target_channel_view": (
            "mixed_old_received_new_leo_weak"
            if str(spec["cache_scope"]) == "external_comparison_registered"
            else "leo_weak_only"
        ),
        "overlay_role_policy": (
            "target_new_only"
            if str(spec["cache_scope"]) == "external_comparison_registered"
            else "all_roles"
        ),
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": output_roles,
        "cache_npz_by_scenario": {
            scenario: _relative_or_absolute(cache_paths[scenario], out_manifest.parent)
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_sha256_by_scenario": {
            scenario: sha256_file(cache_paths[scenario])
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_audits": cache_audits,
        "builder_sha256": builder_hash,
        "build_spec_sha256": canonical_json_sha256(spec),
        "build_spec_path_exposed_to_phase2": False,
    }
    if single_observation_scope:
        set_manifest.update(
            {
                "phase2_physical_sample_observation_policy": (
                    "single_leo_weak_observation_per_physical_sample"
                ),
                "phase2_cross_scenario_physical_sample_reuse": False,
                "phase2_additional_leo_channel_state_generation": False,
                "phase2_post_reception_equalization_augmentation_transform_allowed": (
                    True
                ),
                "phase2_post_reception_view_from_fixed_received_iq_only": True,
                "phase2_post_reception_view_counts_as_additional_physical_sample": (
                    False
                ),
                "phase2_physical_sample_root_id_policy": (
                    "immutable_preoverlay_lineage_token"
                ),
                "phase2_query_post_reception_view_fit_access": False,
                "physical_sample_scenario_assignment_policy": (
                    SCENARIO_PARTITION_POLICY
                ),
                "physical_sample_ids_sha256_by_scenario": physical_roots,
                "physical_sample_scenario_assignment_sha256": assignment_root,
            }
        )
        if exclusion_audit is not None:
            set_manifest["physical_sample_exclusion_audit"] = exclusion_audit
    else:
        set_manifest["physical_sample_ids_sha256"] = physical_roots[
            FORMAL_LEO_WEAK_SCENARIOS[0]
        ]
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(set_manifest, ensure_ascii=False, indent=2, sort_keys=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        "cache_set_manifest": str(out_manifest),
        "cache_set_manifest_sha256": sha256_file(out_manifest),
        "cache_scope": str(spec["cache_scope"]),
        "output_roles": output_roles,
        "physical_sample_ids_sha256_by_scenario": (
            physical_roots if single_observation_scope else None
        ),
        "physical_sample_scenario_assignment_sha256": (
            assignment_root if single_observation_scope else None
        ),
        "cache_audits": cache_audits,
        "physical_sample_exclusion_audit": exclusion_audit,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    result = build_cache_set(args.spec, device=device)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
