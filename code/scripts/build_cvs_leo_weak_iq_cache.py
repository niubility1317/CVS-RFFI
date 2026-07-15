#!/usr/bin/env python
"""Build sealed post-channel LEO_weak IQ caches outside the Phase2 boundary.

This is a Phase1/offline preprocessing tool.  It may read the source dataset,
but it writes only post-channel ``leo_weak_iq`` plus sample-level overlay
provenance.  Formal Phase2 consumers must use ``cvsrffi.leo_weak_cache`` and
must never receive this build spec or an input dataset path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    while candidate in sys.path:
        sys.path.remove(candidate)
for candidate in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, candidate)

from cvsrffi.eval import apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.leo_weak_cache import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA,
    LEO_WEAK_CACHE_SET_SCHEMA,
    LEO_WEAK_CACHE_STAGE,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    load_verified_leo_weak_cache,
    overlay_id,
    physical_sample_id_from_values,
    post_channel_iq_sha256,
    sha256_file,
)
from cvsrffi.tensors import make_torch_generator  # noqa: E402
from export_spaceborne_features import (  # noqa: E402
    _build_wisig_dataset,
    _meta_to_list,
)
from training_controls import sat_channel_config_for_scenario  # noqa: E402


BUILD_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v1"
SCOPE_ROLES = {
    "source_train": {"source"},
    "source_validation": {"source"},
    "stage2_target_old": {"target_old"},
    "stage2_registered": {"target_old", "target_new"},
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
    if spec.get("schema") != BUILD_SPEC_SCHEMA:
        raise ValueError(f"build spec schema must be {BUILD_SPEC_SCHEMA}")
    if spec.get("phase2_sample_view_policy") != PHASE2_SAMPLE_VIEW_POLICY:
        raise ValueError("build spec phase2_sample_view_policy drift")
    if spec.get("clean_sample_access") is not False:
        raise ValueError("build spec must declare clean_sample_access=false")
    if spec.get("clean_derived_signal_access") is not False:
        raise ValueError("build spec must declare clean_derived_signal_access=false")
    if spec.get("star_ground_channel_impl") != "simplified_leo_residual":
        raise ValueError("build spec requires simplified_leo_residual")
    scope = str(spec.get("cache_scope", ""))
    if scope not in SCOPE_ROLES:
        raise ValueError(f"unsupported cache_scope={scope!r}")
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
    seeds = dict(spec.get("satellite_seed_by_scenario", {}))
    outputs = dict(spec.get("out_npz_by_scenario", {}))
    if tuple(seeds) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("satellite_seed_by_scenario must use the formal ordered tuple")
    if tuple(outputs) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("out_npz_by_scenario must use the formal ordered tuple")
    if any(int(seeds[name]) < 0 for name in FORMAL_LEO_WEAK_SCENARIOS):
        raise ValueError("satellite seeds must be nonnegative")
    if not str(spec.get("out_manifest", "")).strip():
        raise ValueError("build spec requires out_manifest")
    if not 1 <= int(spec.get("batch_size", 256)) <= 4096:
        raise ValueError("batch_size must be in [1,4096]")
    if int(spec.get("wisig_out_len", 256)) <= 0:
        raise ValueError("wisig_out_len must be positive")
    return dict(spec)


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
    for role_index, raw_role_spec in enumerate(spec["role_specs"]):
        role_spec = dict(raw_role_spec)
        pkl_path = _resolve(spec_dir, str(role_spec["pkl"]))
        if not pkl_path.is_file():
            raise FileNotFoundError(f"input dataset is missing: {pkl_path}")
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
        )
        if len(dataset) <= 0:
            raise ValueError(f"role={role_spec['role']} produced no physical samples")
        if pkl_path not in dataset_hash_cache:
            dataset_hash_cache[pkl_path] = sha256_file(pkl_path)
        safe_info = {
            "role": str(role_spec["role"]),
            "dataset_sha256": dataset_hash_cache[pkl_path],
            "dataset_size_bytes": int(pkl_path.stat().st_size),
            "requested_tx_ids": str(role_spec["tx_ids"]),
            "requested_rxs": str(role_spec["rxs"]),
            "requested_days": role_spec.get("days"),
            "dataset_seed": dataset_seed,
            "resolved_info": _json_safe(info),
            "physical_sample_count": int(len(dataset)),
        }
        datasets.append((role_spec, dataset, safe_info, pkl_path))
    return datasets


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
        role_seed = int(base_seed) + role_index * 1_000_003
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
                sample_id = physical_sample_id_from_values(
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
            buffers["channel_views"].extend(["rx_base"] * count)
            buffers["sat_scenarios"].extend([str(scenario)] * count)
            buffers["satellite_seeds"].extend([role_seed] * count)
            buffers["overlay_applied"].extend([True] * count)
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
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
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
    )
    return audit


def build_cache_set(spec_path: str | Path, *, device: torch.device) -> dict[str, Any]:
    path = Path(spec_path).resolve()
    spec = validate_build_spec(
        json.loads(path.read_text(encoding="utf-8-sig"))
    )
    out_manifest = _resolve(path.parent, str(spec["out_manifest"]))
    if out_manifest.exists():
        raise FileExistsError(
            f"refusing to overwrite LEO cache-set manifest: {out_manifest}"
        )
    role_datasets = _build_role_datasets(spec, spec_dir=path.parent)
    builder_hash = sha256_file(Path(__file__))
    cache_paths: dict[str, Path] = {}
    cache_audits: dict[str, Any] = {}
    reference_ids_hash: str | None = None
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        out_path = _resolve(
            path.parent, str(dict(spec["out_npz_by_scenario"])[scenario])
        )
        audit = _build_one_scenario(
            scenario=scenario,
            base_seed=int(dict(spec["satellite_seed_by_scenario"])[scenario]),
            role_datasets=role_datasets,
            spec=spec,
            out_path=out_path,
            builder_sha256=builder_hash,
            device=device,
        )
        current_ids_hash = str(audit["physical_sample_ids_sha256"])
        if reference_ids_hash is None:
            reference_ids_hash = current_ids_hash
        elif current_ids_hash != reference_ids_hash:
            raise RuntimeError("physical sample IDs drift across generated scenarios")
        cache_paths[scenario] = out_path
        cache_audits[scenario] = audit

    output_roles = [str(item[0]["role"]) for item in role_datasets]
    set_manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_set_id": str(spec.get("cache_set_id", path.stem)),
        "cache_scope": str(spec["cache_scope"]),
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
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
        "physical_sample_ids_sha256": str(reference_ids_hash),
        "builder_sha256": builder_hash,
        "build_spec_sha256": canonical_json_sha256(spec),
        "build_spec_path_exposed_to_phase2": False,
    }
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
        "physical_sample_ids_sha256": str(reference_ids_hash),
        "cache_audits": cache_audits,
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
