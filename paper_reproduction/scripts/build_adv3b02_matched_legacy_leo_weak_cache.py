"""Build post-channel caches with the historical K10 split/View schedule.

This is a Phase1-only exporter.  It loads clean ManySig rows, reconstructs the
historical nested support pool/query split, applies the historical support and
query overlay seeds separately, then persists only LEO_weak IQ and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from training_controls import sat_channel_config_for_scenario  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
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
from paper_reproduction.common.wisig_runtime import load_wisig_compact_pkl  # noqa: E402
from paper_reproduction.cvs_aligned.evaluate import (  # noqa: E402
    _apply_scenario,
    _build_stage2_tensors,
)


LEGACY_RUNNER_SHA256 = "1270dbdb40285393519796a65a4f9bce3a0a89debdfce0e9a3ca1521a930a9db"
LEGACY_RUNNER_GIT_COMMIT = "d7f2f549ceb4903c1ab8b219b44f581379deacf3"
APPLY_SCENARIO_SOURCE_SHA256 = "0441168c391db173db25501165098e0b7236d475003cfdb31b56f5a1f139a22d"
LEGACY_CALL_AST_SHA256 = "1d6f306184fdee90b1c3333714fc187e3c25a0f6836a88c93bf43aa401ecfdf4"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve(base: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _numpy_from_torch(value: torch.Tensor, *, dtype: Any) -> np.ndarray:
    """Copy through Python values for the N607 NumPy2/PyTorch2.1 runtime."""

    return np.asarray(value.detach().cpu().tolist(), dtype=dtype)


def _meta_arrays(
    support_meta: list[dict[str, Any]], query_meta: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    rows = [*support_meta, *query_meta]
    partitions = ["support_pool"] * len(support_meta) + ["query"] * len(query_meta)
    counters: dict[tuple[str, str], int] = {}
    ranks: list[int] = []
    for partition, row in zip(partitions, rows):
        key = (partition, str(row["tx_label"]))
        ranks.append(counters.get(key, 0))
        counters[key] = counters.get(key, 0) + 1
    tx = [str(row["tx_label"]) for row in rows]
    rx = [str(row["rx_label"]) for row in rows]
    day = [f"day{int(row['day_i'])}" for row in rows]
    eq = ["eq1"] * len(rows)
    sig = [f"sig{int(row['sig_i'])}" for row in rows]
    sample_ids = [
        physical_sample_id_from_values(
            role="target_old",
            tx_id=tx[index],
            rx_id=rx[index],
            day_id=day[index],
            eq_id=eq[index],
            sig_id=sig[index],
        )
        for index in range(len(rows))
    ]
    return {
        "tx_ids": np.asarray(tx),
        "rx_ids": np.asarray(rx),
        "day_ids": np.asarray(day),
        "eq_ids": np.asarray(eq),
        "sig_ids": np.asarray(sig),
        "dataset_role": np.asarray(["target_old"] * len(rows)),
        "sample_ids": np.asarray(sample_ids),
        "split_partition": np.asarray(partitions),
        "split_rank": np.asarray(ranks, dtype=np.int64),
    }


def build(spec_path: Path, *, device: torch.device) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if spec.get("offline_split_partition_policy") != "legacy_seeded_nested_exact":
        raise ValueError("matched cache requires the exact historical split policy")
    if spec.get("legacy_runner_sha256") != LEGACY_RUNNER_SHA256:
        raise ValueError("historical View runner binding drift")
    if spec.get("legacy_runner_git_commit") != LEGACY_RUNNER_GIT_COMMIT:
        raise ValueError("historical View runner Git binding drift")
    if spec.get("apply_scenario_source_sha256") != APPLY_SCENARIO_SOURCE_SHA256:
        raise ValueError("historical View function binding drift")
    if spec.get("legacy_support_query_call_ast_sha256") != LEGACY_CALL_AST_SHA256:
        raise ValueError("historical View call-site binding drift")
    observed_apply_hash = hashlib.sha256(
        textwrap.dedent(inspect.getsource(_apply_scenario)).encode("utf-8")
    ).hexdigest()
    if observed_apply_hash != APPLY_SCENARIO_SOURCE_SHA256:
        raise ValueError("current _apply_scenario implementation differs from historical code")
    support_seeds = dict(spec["support_satellite_seed_by_scenario"])
    query_seeds = dict(spec["query_satellite_seed_by_scenario"])
    if tuple(support_seeds) != FORMAL_LEO_WEAK_SCENARIOS or tuple(
        query_seeds
    ) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("historical View seed scenario order drift")
    role = dict(spec["role_specs"][0])
    receiver = str(role["rxs"])
    old_labels = [value for value in str(role["tx_ids"]).split(",") if value]
    support_pool_max_k = int(spec["support_pool_max_k"])
    query_per_tx = int(spec["query_per_tx"])
    seed = int(spec["dataset_seed"])
    if support_pool_max_k != 20 or query_per_tx != 20:
        raise ValueError("matched cache requires a 20-row nested pool and 20 queries/class")
    expected_support = {
        scenario: seed + 1000 + index
        for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
    }
    expected_query = {
        scenario: seed + 2000 + index
        for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
    }
    if support_seeds != expected_support or query_seeds != expected_query:
        raise ValueError("historical support/query View seed formula drift")

    dataset_path = _resolve(spec_path.parent, str(role["pkl"]))
    manysig = load_wisig_compact_pkl(str(dataset_path))
    tensors = _build_stage2_tensors(
        {
            "target_receiver_labels": [receiver],
            "target_old_tx_labels": old_labels,
            "target_new_tx_labels": [],
            "target_unknown_tx_labels": [],
            "target_day": 0,
            "equalized": 1,
            "k_shot": support_pool_max_k,
            "support_pool_max_k": support_pool_max_k,
            "query_per_tx": query_per_tx,
            "target_sample_strategy": "seeded_nested",
            "split_seed": seed,
            "seed": seed,
        },
        manysig,
        manysig,
    )
    support_meta = list(tensors["support_meta"])
    query_meta = list(tensors["query_meta"])
    metadata = _meta_arrays(support_meta, query_meta)
    if len(set(metadata["sample_ids"].astype(str).tolist())) != len(metadata["sample_ids"]):
        raise ValueError("matched cache physical IDs are not unique")
    labels = _numpy_from_torch(
        torch.cat([tensors["support_y"], tensors["query_y"]]), dtype=np.int64
    )
    partition = metadata["split_partition"].astype(str)
    support_count = len(support_meta)
    query_count = len(query_meta)
    if support_count != len(old_labels) * 20 or query_count != len(old_labels) * 20:
        raise ValueError("matched cache support/query row count drift")

    builder_sha = sha256_file(Path(__file__))
    output_paths: dict[str, Path] = {}
    class_count = len(old_labels)
    historical_k10_positions = torch.as_tensor(
        [
            class_index * support_pool_max_k + rank
            for class_index in range(class_count)
            for rank in range(10)
        ],
        dtype=torch.long,
    )
    support_extra_positions = torch.as_tensor(
        [
            class_index * support_pool_max_k + rank
            for class_index in range(class_count)
            for rank in range(10, support_pool_max_k)
        ],
        dtype=torch.long,
    )
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        out_path = _resolve(spec_path.parent, str(spec["out_npz_by_scenario"][scenario]))
        if out_path.exists():
            raise FileExistsError(f"refusing to overwrite matched cache: {out_path}")
        historical_support_iq = _apply_scenario(
            tensors["support_x"][historical_k10_positions].to(device),
            scenario,
            seed=int(support_seeds[scenario]),
        )
        historical_support_iq = _numpy_from_torch(
            historical_support_iq, dtype=np.float32
        )
        extra_seed = int(support_seeds[scenario]) + 3000
        extra_support_iq = _apply_scenario(
            tensors["support_x"][support_extra_positions].to(device),
            scenario,
            seed=extra_seed,
        )
        extra_support_iq = _numpy_from_torch(extra_support_iq, dtype=np.float32)
        support_iq = np.empty(
            (support_count, *historical_support_iq.shape[1:]), dtype=np.float32
        )
        support_iq[historical_k10_positions.numpy()] = historical_support_iq
        support_iq[support_extra_positions.numpy()] = extra_support_iq
        query_iq = _apply_scenario(
            tensors["query_x"].to(device),
            scenario,
            seed=int(query_seeds[scenario]),
        )
        query_iq = _numpy_from_torch(query_iq, dtype=np.float32)
        iq = np.empty(
            (support_count + query_count, *support_iq.shape[1:]), dtype=np.float32
        )
        iq[:support_count] = support_iq
        iq[support_count:] = query_iq
        support_seed_rows = np.full(support_count, extra_seed, dtype=np.int64)
        support_seed_rows[historical_k10_positions.numpy()] = int(
            support_seeds[scenario]
        )
        satellite_seeds = np.empty(support_count + query_count, dtype=np.int64)
        satellite_seeds[:support_count] = support_seed_rows
        satellite_seeds[support_count:] = int(query_seeds[scenario])
        channel_config = dict(sat_channel_config_for_scenario(scenario))
        channel_config.update(
            {
                "fs_hz": 25e6,
                "fc_hz": 2.462e9,
                "star_ground_channel_impl": "simplified_leo_residual",
            }
        )
        channel_hash = canonical_json_sha256(channel_config)
        ids = metadata["sample_ids"].astype(str).tolist()
        iq_hashes = [post_channel_iq_sha256(row) for row in iq]
        overlay_ids = [
            overlay_id(
                sample_id=ids[index],
                scenario=scenario,
                satellite_seed=int(satellite_seeds[index]),
                channel_config_sha256=channel_hash,
                iq_sha256=iq_hashes[index],
            )
            for index in range(len(ids))
        ]
        manifest = {
            "schema": LEO_WEAK_CACHE_SCHEMA,
            "artifact_stage": LEO_WEAK_CACHE_STAGE,
            "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "contains_post_channel_iq_only": True,
            "contains_clean_rows": False,
            "target_channel_view": "leo_weak_only",
            "target_channel_scenarios": [scenario],
            "scenario": scenario,
            "iq_array_key": "leo_weak_iq",
            "raw_or_clean_iq_key_present": False,
            "overlay_applied_before_phase2": True,
            "star_ground_channel_impl": "simplified_leo_residual",
            "channel_model": "leo_residual",
            "channel_config": channel_config,
            "channel_config_sha256": channel_hash,
            "builder_sha256": builder_sha,
            "build_spec_sha256": canonical_json_sha256(spec),
            "output_roles": ["target_old"],
            "row_count": len(ids),
            "physical_sample_ids_sha256": ids_sha256(ids),
            "post_channel_iq_sha256_root": ids_sha256(iq_hashes),
            "overlay_ids_sha256": ids_sha256(overlay_ids),
            "sample_overlay_provenance_fields": [
                "sample_ids",
                "sat_scenarios",
                "satellite_seeds",
                "post_channel_iq_sha256",
                "overlay_ids",
            ],
            "offline_split_partition_policy": "legacy_seeded_nested_exact",
            "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
            "legacy_runner_git_commit": LEGACY_RUNNER_GIT_COMMIT,
            "apply_scenario_source_sha256": observed_apply_hash,
            "legacy_support_query_call_ast_sha256": LEGACY_CALL_AST_SHA256,
            "support_satellite_seed": int(support_seeds[scenario]),
            "unused_support_pool_extra_satellite_seed": extra_seed,
            "query_satellite_seed": int(query_seeds[scenario]),
            "query_view_count": 1,
        }
        payload = {
            "leo_weak_iq": iq,
            "raw_labels": labels.astype(np.int64),
            "domain_labels": np.zeros(len(ids), dtype=np.int64),
            **metadata,
            "channel_views": np.asarray(["rx_base"] * len(ids)),
            "sat_scenarios": np.asarray([scenario] * len(ids)),
            "satellite_seeds": satellite_seeds,
            "overlay_applied": np.ones(len(ids), dtype=bool),
            "post_channel_iq_sha256": np.asarray(iq_hashes),
            "overlay_ids": np.asarray(overlay_ids),
            "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **payload)
        load_verified_leo_weak_cache(
            out_path, expected_scenario=scenario, allowed_roles={"target_old"}
        )
        output_paths[scenario] = out_path

    out_manifest = _resolve(spec_path.parent, str(spec["out_manifest"]))
    if out_manifest.exists():
        raise FileExistsError(f"refusing to overwrite matched cache set: {out_manifest}")
    set_manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_set_id": str(spec["cache_set_id"]),
        "cache_scope": "stage2_target_old",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": ["target_old"],
        "cache_npz_by_scenario": {
            scenario: os.path.relpath(output_paths[scenario], out_manifest.parent)
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "cache_sha256_by_scenario": {
            scenario: sha256_file(output_paths[scenario])
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "physical_sample_ids_sha256": ids_sha256(
            metadata["sample_ids"].astype(str).tolist()
        ),
        "offline_split_partition_policy": "legacy_seeded_nested_exact",
        "legacy_runner_sha256": LEGACY_RUNNER_SHA256,
        "legacy_runner_git_commit": LEGACY_RUNNER_GIT_COMMIT,
        "apply_scenario_source_sha256": APPLY_SCENARIO_SOURCE_SHA256,
        "legacy_support_query_call_ast_sha256": LEGACY_CALL_AST_SHA256,
        "support_satellite_seed_by_scenario": support_seeds,
        "query_satellite_seed_by_scenario": query_seeds,
        "query_view_count": 1,
    }
    _write_json(out_manifest, set_manifest)
    _arrays, _manifest, audit = load_verified_leo_weak_cache_set(
        out_manifest,
        expected_scope="stage2_target_old",
        allowed_roles={"target_old"},
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = build(args.spec, device=torch.device(args.device))
    print(json.dumps({"status": "PASS", "audit": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
