#!/usr/bin/env python3
"""Build the sealed MRIOR-preadapted Stage2-C CI matrix from frozen v7 input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    FORMAL_LEO_WEAK_SCENARIOS,
    _materialize_npz,
    _validate_support_arrays,
    preflight_stage2_predictor_package,
)
from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (  # noqa: E402
    MRIORPreadaptInputBinding,
    expected_mrior_preadapt_method_lock,
    preadapt_key,
)


SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_ci_plan.v1"
SOURCE_PLAN_SCHEMA = "cvs.phase2.adv3b02_paper_full_ci_plan.v1"
SOURCE_EXPERIMENT_ID = "adv3b02_unfrozen_paperfull_ci_20260723_v7"
SOURCE_PLAN_SHA256 = "1c5fb08231bc9d150d625e5360162c3ee287bdf778ac1c16a069ac187b96d65b"
SOURCE_CACHE_MANIFEST_PATH = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "adv3b02_three_da_leoweakonly_20260715_v1/phase1_caches/source/cache_set.json"
)
SOURCE_CACHE_SCOPE = "source_train"
SOURCE_METHODS = ("csil_paper_full", "mopc_hr_paper_full")
METHOD_MAP = {
    "csil_paper_full": "mrior_sda_then_csil_paper_full",
    "mopc_hr_paper_full": "mrior_sda_then_mopc_hr_paper_full",
}
FORMAL_RECEIVERS = ("20-1", "3-19", "7-14", "7-7", "8-8")
FORMAL_SEEDS = (713101, 713102, 713103, 713104, 713105)
FORMAL_K_VALUES = (1, 5, 10, 20)
FORMAL_NEW_COUNTS = (2, 5, 10, 20)
PREADAPT_ANCHOR_NEW_CLASS_COUNT = 2
FORMAL_EXPERIMENT_ID = "adv3b02_mrior_preadapt_ci_20260817_v1"


@dataclass(frozen=True)
class _MatrixContract:
    receivers: tuple[str, ...]
    seeds: tuple[int, ...]
    k_values: tuple[int, ...]
    new_counts: tuple[int, ...]
    expected_source_cache_path: str


@dataclass(frozen=True)
class _PackageEvidence:
    package_id: str
    receiver: str
    seed: int
    new_class_count: int
    old_class_labels: tuple[str, ...]
    predictor_package_root: Path
    detached_seal_path: Path
    detached_seal_sha256: str
    manifest_sha256: str
    checkpoint_sha256: str
    old_support_token_sha256_by_k_scene: dict[tuple[int, str], str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return lowered


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _expected_counts(contract: _MatrixContract) -> dict[str, int]:
    package_count = (
        len(contract.receivers) * len(contract.seeds) * len(contract.new_counts)
    )
    cell_count = package_count * len(SOURCE_METHODS) * len(contract.k_values)
    return {
        "packages": package_count,
        "cells": cell_count,
        "scenario_rows": cell_count * len(FORMAL_LEO_WEAK_SCENARIOS),
    }


def _normal_tuple(values: Any, *, field: str, cast: type[str] | type[int]) -> tuple:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    try:
        normalized = tuple(cast(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} contains an invalid value") from exc
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} contains duplicates")
    return normalized


def _validate_source_plan(
    payload: Mapping[str, Any], *, contract: _MatrixContract
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], list[dict[str, Any]], str]:
    if payload.get("schema") != SOURCE_PLAN_SCHEMA:
        raise ValueError("source plan must use the authorized v7 schema")
    if payload.get("experiment_id") != SOURCE_EXPERIMENT_ID:
        raise ValueError("source plan is not the authorized v7 experiment")
    if _normal_tuple(payload.get("methods"), field="source methods", cast=str) != SOURCE_METHODS:
        raise ValueError("source plan methods drift")
    if _normal_tuple(payload.get("receivers"), field="source receivers", cast=str) != contract.receivers:
        raise ValueError("source plan receiver matrix drift")
    if _normal_tuple(payload.get("seeds"), field="source seeds", cast=int) != contract.seeds:
        raise ValueError("source plan seed matrix drift")
    if _normal_tuple(payload.get("k_values"), field="source K values", cast=int) != contract.k_values:
        raise ValueError("source plan K matrix drift")
    if _normal_tuple(
        payload.get("new_class_counts"), field="source new counts", cast=int
    ) != contract.new_counts:
        raise ValueError("source plan new-count matrix drift")
    if tuple(payload.get("scenarios", [])) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("source plan LEO scenario matrix drift")
    if payload.get("counts") != _expected_counts(contract):
        raise ValueError("source plan matrix counts drift")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("source plan artifact surface missing")
    checkpoint = artifacts.get("base_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("source plan checkpoint descriptor missing")
    checkpoint_sha256 = _require_sha256(
        checkpoint.get("sha256"), field="source plan checkpoint SHA"
    )

    raw_packages = payload.get("packages")
    raw_cells = payload.get("cells")
    if not isinstance(raw_packages, list) or not isinstance(raw_cells, list):
        raise ValueError("source plan package/cell surface missing")
    packages: dict[tuple[str, int, int], dict[str, Any]] = {}
    package_ids: set[str] = set()
    for raw in raw_packages:
        if not isinstance(raw, Mapping):
            raise ValueError("source plan package entry drift")
        package = dict(raw)
        receiver = str(package.get("receiver", ""))
        seed = package.get("seed")
        new_count = package.get("new_class_count")
        package_id = package.get("package_id")
        if (
            receiver not in contract.receivers
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in contract.seeds
            or isinstance(new_count, bool)
            or not isinstance(new_count, int)
            or new_count not in contract.new_counts
            or not isinstance(package_id, str)
            or not package_id
        ):
            raise ValueError("source plan package identity drift")
        key = (receiver, int(seed), int(new_count))
        if key in packages or package_id in package_ids:
            raise ValueError("source plan duplicate package identity")
        old_labels = package.get("old_class_labels")
        new_labels = package.get("new_class_labels")
        if (
            not isinstance(old_labels, list)
            or len(old_labels) != 6
            or len(set(map(str, old_labels))) != 6
            or not isinstance(new_labels, list)
            or len(new_labels) != new_count
            or len(set(map(str, new_labels))) != new_count
            or set(map(str, old_labels)) & set(map(str, new_labels))
        ):
            raise ValueError("source plan package class surface drift")
        for field in ("predictor_package_root", "detached_seal"):
            if not isinstance(package.get(field), str) or not str(package[field]).strip():
                raise ValueError(f"source plan package {field} missing")
        packages[key] = package
        package_ids.add(package_id)
    if len(packages) != _expected_counts(contract)["packages"]:
        raise ValueError("source plan package coverage drift")

    expected_cell_keys = {
        (receiver, seed, new_count, method, k_shot)
        for receiver in contract.receivers
        for seed in contract.seeds
        for new_count in contract.new_counts
        for method in SOURCE_METHODS
        for k_shot in contract.k_values
    }
    observed_cell_keys: set[tuple[str, int, int, str, int]] = set()
    cell_ids: set[str] = set()
    normalized_cells: list[dict[str, Any]] = []
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise ValueError("source plan cell entry drift")
        cell = dict(raw)
        cell_id = cell.get("cell_id")
        receiver = str(cell.get("receiver", ""))
        seed = cell.get("seed")
        new_count = cell.get("new_class_count")
        method = cell.get("method")
        k_shot = cell.get("k_shot")
        package_id = cell.get("package_id")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or cell_id in cell_ids
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or isinstance(new_count, bool)
            or not isinstance(new_count, int)
            or isinstance(k_shot, bool)
            or not isinstance(k_shot, int)
            or method not in SOURCE_METHODS
        ):
            raise ValueError("source plan cell identity drift")
        key = (receiver, seed, new_count, str(method), k_shot)
        if key not in expected_cell_keys or key in observed_cell_keys:
            raise ValueError("source plan cell matrix drift")
        package = packages.get((receiver, seed, new_count))
        if package is None or package_id != package["package_id"]:
            raise ValueError("source plan cell/package binding drift")
        observed_cell_keys.add(key)
        cell_ids.add(cell_id)
        normalized_cells.append(cell)
    if observed_cell_keys != expected_cell_keys:
        raise ValueError("source plan cell coverage drift")
    return packages, normalized_cells, checkpoint_sha256


def _validate_source_cache(
    path: Path,
    *,
    expected_sha256: str,
    expected_path: str,
) -> dict[str, Any]:
    if str(path) != str(expected_path):
        raise ValueError("source cache manifest path drift")
    expected_digest = _require_sha256(
        expected_sha256, field="expected source-cache manifest SHA"
    )
    if _sha256(path) != expected_digest:
        raise ValueError("source cache manifest SHA drift")
    payload = _read_json(path, context="source cache manifest")
    if payload.get("cache_scope") != SOURCE_CACHE_SCOPE:
        raise ValueError("source cache scope drift")
    if payload.get("target_channel_view") != "leo_weak_only":
        raise ValueError("source cache channel view drift")
    scenario_map = payload.get("cache_npz_by_scenario")
    if not isinstance(scenario_map, Mapping) or tuple(scenario_map) != FORMAL_LEO_WEAK_SCENARIOS:
        raise ValueError("source cache LEO scenario mapping drift")
    return payload


def _old_support_token_sha256(
    *,
    package_root: Path,
    manifest: Mapping[str, Any],
    old_class_count: int,
    k_shot: int,
    scenario: str,
) -> str:
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ValueError("package manifest members missing")
    descriptor = next(
        (
            value
            for value in members
            if isinstance(value, Mapping)
            and value.get("artifact_role") == f"support:{scenario}"
        ),
        None,
    )
    if descriptor is None:
        raise ValueError("package manifest support member missing")
    arrays, support_manifest = _materialize_npz(package_root, descriptor)
    _validate_support_arrays(
        arrays,
        support_manifest,
        scenario=scenario,
        class_count=int(manifest["registered_class_count"]),
        max_k=int(manifest["support_pool_max_k"]),
    )
    labels = np.asarray(arrays["support_pool_class_indices"], dtype=np.int64)
    ranks = np.asarray(arrays["support_pool_rank_within_class"], dtype=np.int64)
    tokens = np.asarray(arrays["support_pool_tokens"]).astype(str)
    selected = sorted(
        (
            (int(label), int(rank), str(token))
            for label, rank, token in zip(labels.tolist(), ranks.tolist(), tokens.tolist())
            if int(label) < old_class_count and int(rank) < k_shot
        ),
        key=lambda value: (value[0], value[1]),
    )
    expected_pairs = [
        (class_index, rank)
        for class_index in range(old_class_count)
        for rank in range(k_shot)
    ]
    if [(label, rank) for label, rank, _token in selected] != expected_pairs:
        raise ValueError("target-old support K-shot identity drift")
    ordered_tokens = [token for _label, _rank, token in selected]
    if len(set(ordered_tokens)) != len(ordered_tokens):
        raise ValueError("target-old support token collision")
    return _canonical_sha256(
        {
            "old_class_count": old_class_count,
            "k_shot": k_shot,
            "scenario": scenario,
            "ordered_support_tokens": ordered_tokens,
        }
    )


def _package_evidence(
    package: Mapping[str, Any],
    *,
    contract: _MatrixContract,
    source_checkpoint_sha256: str,
) -> _PackageEvidence:
    receiver = str(package["receiver"])
    seed = int(package["seed"])
    new_count = int(package["new_class_count"])
    old_labels = tuple(str(value) for value in package["old_class_labels"])
    package_root = Path(str(package["predictor_package_root"])).resolve(strict=True)
    seal_path = Path(str(package["detached_seal"])).resolve(strict=True)
    seal_sha256 = _sha256(seal_path)
    manifest, seal, _audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=seal_path,
        expected_seal_sha256=seal_sha256,
    )
    if (
        manifest.get("receiver") != receiver
        or int(manifest.get("seed", -1)) != seed
        or int(manifest.get("new_class_count", -1)) != new_count
        or int(manifest.get("registered_class_count", -1)) != len(old_labels) + new_count
        or int(manifest.get("support_pool_max_k", 0)) < max(contract.k_values)
    ):
        raise ValueError("source package manifest identity drift")
    checkpoint_members = [
        value
        for value in manifest.get("members", [])
        if isinstance(value, Mapping) and value.get("artifact_role") == "checkpoint"
    ]
    if len(checkpoint_members) != 1:
        raise ValueError("source package checkpoint member drift")
    checkpoint_sha256 = _require_sha256(
        checkpoint_members[0].get("sha256"), field="source package checkpoint SHA"
    )
    if checkpoint_sha256 != source_checkpoint_sha256:
        raise ValueError("source package checkpoint SHA drift")
    support_digests = {
        (k_shot, scenario): _old_support_token_sha256(
            package_root=package_root,
            manifest=manifest,
            old_class_count=len(old_labels),
            k_shot=k_shot,
            scenario=scenario,
        )
        for k_shot in contract.k_values
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    return _PackageEvidence(
        package_id=str(package["package_id"]),
        receiver=receiver,
        seed=seed,
        new_class_count=new_count,
        old_class_labels=old_labels,
        predictor_package_root=package_root,
        detached_seal_path=seal_path,
        detached_seal_sha256=seal_sha256,
        manifest_sha256=_require_sha256(
            seal.get("manifest_sha256"), field="source package manifest SHA"
        ),
        checkpoint_sha256=checkpoint_sha256,
        old_support_token_sha256_by_k_scene=support_digests,
    )


def _formal_contract() -> _MatrixContract:
    return _MatrixContract(
        receivers=FORMAL_RECEIVERS,
        seeds=FORMAL_SEEDS,
        k_values=FORMAL_K_VALUES,
        new_counts=FORMAL_NEW_COUNTS,
        expected_source_cache_path=SOURCE_CACHE_MANIFEST_PATH,
    )


def _build(
    *,
    source_plan: Path,
    expected_source_plan_sha256: str,
    source_cache_manifest: Path,
    expected_source_cache_manifest_sha256: str,
    run_root: Path,
    output: Path,
    contract: _MatrixContract,
    experiment_id: str,
) -> dict[str, Any]:
    source_plan = source_plan.resolve(strict=True)
    expected_source_plan_digest = _require_sha256(
        expected_source_plan_sha256, field="expected source-plan SHA"
    )
    if _sha256(source_plan) != expected_source_plan_digest:
        raise ValueError("source plan SHA drift")
    source = _read_json(source_plan, context="source plan")
    packages, source_cells, checkpoint_sha256 = _validate_source_plan(
        source, contract=contract
    )
    source_cache_manifest = source_cache_manifest.resolve(strict=True)
    _validate_source_cache(
        source_cache_manifest,
        expected_sha256=expected_source_cache_manifest_sha256,
        expected_path=contract.expected_source_cache_path,
    )
    source_cache_sha256 = _require_sha256(
        expected_source_cache_manifest_sha256,
        field="expected source-cache manifest SHA",
    )
    if PREADAPT_ANCHOR_NEW_CLASS_COUNT not in contract.new_counts:
        raise ValueError("preadapt anchor new-count is unavailable")

    evidence = {
        key: _package_evidence(
            package,
            contract=contract,
            source_checkpoint_sha256=checkpoint_sha256,
        )
        for key, package in packages.items()
    }
    anchor_by_receiver_seed: dict[tuple[str, int], _PackageEvidence] = {}
    for receiver in contract.receivers:
        for seed in contract.seeds:
            anchor = evidence.get((receiver, seed, PREADAPT_ANCHOR_NEW_CLASS_COUNT))
            if anchor is None:
                raise ValueError("preadapt anchor package missing")
            anchor_by_receiver_seed[(receiver, seed)] = anchor
            for new_count in contract.new_counts:
                candidate = evidence[(receiver, seed, new_count)]
                if candidate.old_class_labels != anchor.old_class_labels:
                    raise ValueError("target-old class identity drift across new-count packages")
                if candidate.checkpoint_sha256 != anchor.checkpoint_sha256:
                    raise ValueError("target package checkpoint drift across new-count packages")
                for k_shot in contract.k_values:
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
                        if (
                            candidate.old_support_token_sha256_by_k_scene[(k_shot, scenario)]
                            != anchor.old_support_token_sha256_by_k_scene[(k_shot, scenario)]
                        ):
                            raise ValueError(
                                "target-old support identity drift across new-count packages"
                            )

    resolved_run_root = run_root.resolve()
    jobs: list[dict[str, Any]] = []
    job_by_identity: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    method_lock = expected_mrior_preadapt_method_lock()
    method_lock_sha256 = _canonical_sha256(method_lock)
    for receiver in contract.receivers:
        for seed in contract.seeds:
            anchor = anchor_by_receiver_seed[(receiver, seed)]
            for k_shot in contract.k_values:
                for scenario in FORMAL_LEO_WEAK_SCENARIOS:
                    identity = (receiver, seed, k_shot, scenario)
                    if identity in job_by_identity:
                        raise ValueError("duplicate preadapt job key")
                    job_id = preadapt_key(receiver, seed, k_shot, scenario)
                    support_token_sha256 = anchor.old_support_token_sha256_by_k_scene[
                        (k_shot, scenario)
                    ]
                    binding = MRIORPreadaptInputBinding.from_verified_values(
                        checkpoint_sha256=anchor.checkpoint_sha256,
                        source_cache_sha256=source_cache_sha256,
                        support_token_sha256=support_token_sha256,
                        target_package_seal_sha256=anchor.detached_seal_sha256,
                        receiver=receiver,
                        seed=seed,
                        k_shot=k_shot,
                        scene=scenario,
                    )
                    artifact_root = resolved_run_root / "preadapt_jobs" / job_id
                    job = {
                        "job_id": job_id,
                        "receiver": receiver,
                        "seed": seed,
                        "k_shot": k_shot,
                        "scenario": scenario,
                        "preadapt_anchor_new_class_count": PREADAPT_ANCHOR_NEW_CLASS_COUNT,
                        "target_package_id": anchor.package_id,
                        "target_package_root": str(anchor.predictor_package_root),
                        "target_package_seal_path": str(anchor.detached_seal_path),
                        "target_package_seal_sha256": anchor.detached_seal_sha256,
                        "target_package_manifest_sha256": anchor.manifest_sha256,
                        "checkpoint_sha256": anchor.checkpoint_sha256,
                        "source_cache_manifest": str(source_cache_manifest),
                        "source_cache_sha256": source_cache_sha256,
                        "old_support_token_sha256": support_token_sha256,
                        "input_binding": binding.canonical_payload(),
                        "input_binding_sha256": binding.canonical_sha256,
                        "method_lock": method_lock,
                        "method_lock_sha256": method_lock_sha256,
                        "query_opened_before_model_lock": False,
                        "artifact_root": str(artifact_root),
                        "artifact_state_path": str(artifact_root / "mrior_preadapt_state.pt"),
                        "artifact_manifest_path": str(artifact_root / "manifest.json"),
                    }
                    jobs.append(job)
                    job_by_identity[identity] = job
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate preadapt job ID")

    source_cell_by_key = {
        (
            str(cell["receiver"]),
            int(cell["seed"]),
            int(cell["new_class_count"]),
            str(cell["method"]),
            int(cell["k_shot"]),
        ): cell
        for cell in source_cells
    }
    cells: list[dict[str, Any]] = []
    for receiver in contract.receivers:
        for seed in contract.seeds:
            anchor = anchor_by_receiver_seed[(receiver, seed)]
            for k_shot in contract.k_values:
                job_ids_by_scenario = {
                    scenario: job_by_identity[(receiver, seed, k_shot, scenario)][
                        "job_id"
                    ]
                    for scenario in FORMAL_LEO_WEAK_SCENARIOS
                }
                for new_count in contract.new_counts:
                    package = evidence[(receiver, seed, new_count)]
                    reuse_proof_by_scenario = {
                        scenario: {
                            "anchor_package_id": anchor.package_id,
                            "anchor_new_class_count": PREADAPT_ANCHOR_NEW_CLASS_COUNT,
                            "anchor_old_support_token_sha256": anchor.old_support_token_sha256_by_k_scene[
                                (k_shot, scenario)
                            ],
                            "cell_package_id": package.package_id,
                            "cell_package_old_support_token_sha256": package.old_support_token_sha256_by_k_scene[
                                (k_shot, scenario)
                            ],
                            "matched": True,
                        }
                        for scenario in FORMAL_LEO_WEAK_SCENARIOS
                    }
                    for source_method in SOURCE_METHODS:
                        source_cell = source_cell_by_key[
                            (receiver, seed, new_count, source_method, k_shot)
                        ]
                        method = METHOD_MAP[source_method]
                        cell_id = (
                            f"rx_{receiver.replace('-', '_')}__seed_{seed}__new_{new_count}"
                            f"__{method}__k_{k_shot}"
                        )
                        cells.append(
                            {
                                "cell_id": cell_id,
                                "baseline_v7_cell_id": source_cell["cell_id"],
                                "source_v7_method": source_method,
                                "method": method,
                                "receiver": receiver,
                                "seed": seed,
                                "new_class_count": new_count,
                                "k_shot": k_shot,
                                "target_package_id": package.package_id,
                                "target_package_root": str(package.predictor_package_root),
                                "target_package_seal_path": str(package.detached_seal_path),
                                "target_package_seal_sha256": package.detached_seal_sha256,
                                "preadapt_anchor_new_class_count": PREADAPT_ANCHOR_NEW_CLASS_COUNT,
                                "preadapt_anchor_target_package_seal_sha256": anchor.detached_seal_sha256,
                                "preadapt_job_ids_by_scenario": job_ids_by_scenario,
                                "preadapt_reuse_proof_by_scenario": reuse_proof_by_scenario,
                                "output_root": str(resolved_run_root / "cells" / cell_id),
                            }
                        )
    if len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("duplicate MRIOR CI cell ID")
    if len({cell["output_root"] for cell in cells}) != len(cells):
        raise ValueError("duplicate MRIOR CI output path")

    expected_job_count = (
        len(contract.receivers)
        * len(contract.seeds)
        * len(contract.k_values)
        * len(FORMAL_LEO_WEAK_SCENARIOS)
    )
    expected_cell_count = _expected_counts(contract)["cells"]
    if len(jobs) != expected_job_count or len(cells) != expected_cell_count:
        raise ValueError("MRIOR preadapt CI matrix count drift")
    first_receiver = contract.receivers[0]
    first_seed = contract.seeds[0]
    smoke_job_ids = [
        job_by_identity[(first_receiver, first_seed, contract.k_values[0], scenario)][
            "job_id"
        ]
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    ]
    smoke_cell_ids = [
        cell["cell_id"]
        for cell in cells
        if cell["receiver"] == first_receiver
        and cell["seed"] == first_seed
        and (
            (
                cell["new_class_count"] == contract.new_counts[0]
                and cell["k_shot"] == contract.k_values[0]
            )
            or (
                cell["new_class_count"] == contract.new_counts[-1]
                and cell["k_shot"] == contract.k_values[-1]
            )
        )
    ]
    plan = {
        "schema": SCHEMA,
        "experiment_id": str(experiment_id),
        "protocol_schema": "p2_min_v1",
        "claim_boundary": "formal_paper_method_comparison_mrior_preadapt",
        "comparison_method_protocol_scope": (
            "stage2_main_method_protocol_exempt_new_class_leo_required"
        ),
        "source_v7_plan": {
            "path": str(source_plan),
            "sha256": expected_source_plan_digest,
            "schema": SOURCE_PLAN_SCHEMA,
            "experiment_id": SOURCE_EXPERIMENT_ID,
        },
        "source_cache": {
            "path": str(source_cache_manifest),
            "sha256": source_cache_sha256,
            "expected_scope": SOURCE_CACHE_SCOPE,
        },
        "run_root": str(resolved_run_root),
        "methods": list(METHOD_MAP.values()),
        "source_methods": list(SOURCE_METHODS),
        "receivers": list(contract.receivers),
        "seeds": list(contract.seeds),
        "k_values": list(contract.k_values),
        "new_class_counts": list(contract.new_counts),
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "preadapt_anchor_new_class_count": PREADAPT_ANCHOR_NEW_CLASS_COUNT,
        "canonical_ordering": "receiver,seed,k_shot,scenario,new_class_count,method",
        "preadapt_jobs": jobs,
        "cells": cells,
        "counts": {
            "preadapt_jobs": len(jobs),
            "cells": len(cells),
            "scenario_rows": len(cells) * len(FORMAL_LEO_WEAK_SCENARIOS),
        },
        "smoke_preadapt_job_ids": smoke_job_ids,
        "smoke_cell_ids": smoke_cell_ids,
        "launch_authority": False,
        "authority_state": "N607_MRIOR_PREADAPT_CI_SMOKE_REQUIRED",
    }
    contract_payload = {
        key: value
        for key, value in plan.items()
        if key not in {"launch_authority", "authority_state", "plan_contract_sha256"}
    }
    plan["plan_contract_sha256"] = _canonical_sha256(contract_payload)
    _write_new(output.resolve(), plan)
    return plan


def _build_for_test(
    *,
    source_plan: Path,
    expected_source_plan_sha256: str,
    source_cache_manifest: Path,
    expected_source_cache_manifest_sha256: str,
    run_root: Path,
    output: Path,
    expected_receivers: Sequence[str],
    expected_seeds: Sequence[int],
    expected_k_values: Sequence[int],
    expected_new_counts: Sequence[int],
    expected_source_cache_path: str,
) -> dict[str, Any]:
    """Internal miniature-matrix hook; production CLI always uses `_formal_contract`."""

    return _build(
        source_plan=Path(source_plan),
        expected_source_plan_sha256=expected_source_plan_sha256,
        source_cache_manifest=Path(source_cache_manifest),
        expected_source_cache_manifest_sha256=expected_source_cache_manifest_sha256,
        run_root=Path(run_root),
        output=Path(output),
        contract=_MatrixContract(
            receivers=tuple(str(value) for value in expected_receivers),
            seeds=tuple(int(value) for value in expected_seeds),
            k_values=tuple(int(value) for value in expected_k_values),
            new_counts=tuple(int(value) for value in expected_new_counts),
            expected_source_cache_path=str(expected_source_cache_path),
        ),
        experiment_id="test_mrior_preadapt_ci_plan",
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    expected_source_plan_sha256 = _require_sha256(
        getattr(args, "expected_source_plan_sha256", None),
        field="expected source-plan SHA",
    )
    if expected_source_plan_sha256 != SOURCE_PLAN_SHA256:
        raise ValueError("expected source-plan SHA must equal the frozen v7 SHA")
    return _build(
        source_plan=Path(args.source_plan),
        expected_source_plan_sha256=expected_source_plan_sha256,
        source_cache_manifest=Path(args.source_cache_manifest),
        expected_source_cache_manifest_sha256=str(
            args.expected_source_cache_manifest_sha256
        ),
        run_root=Path(args.run_root),
        output=Path(args.output),
        contract=_formal_contract(),
        experiment_id=str(getattr(args, "experiment_id", FORMAL_EXPERIMENT_ID)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--expected-source-plan-sha256", required=True)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-cache-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default=FORMAL_EXPERIMENT_ID)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=True, sort_keys=True))
