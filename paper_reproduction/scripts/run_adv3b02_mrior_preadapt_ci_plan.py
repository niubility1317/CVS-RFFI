#!/usr/bin/env python3
"""Run the sealed MRIOR-preadapted Stage2-C CI comparison matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for _import_root in (REPO_ROOT, CODE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


RUN_ROOT_IDENTITY_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_ci_run_root.v1"
PLAN_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_ci_plan.v1"
HEALTH_STATE_SCHEMA = "cvs.phase2.adv3b02_mrior_preadapt_ci_health_state.v1"
FORMAL_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
FORMAL_METHOD_MAP = {
    "csil_paper_full": "mrior_sda_then_csil_paper_full",
    "mopc_hr_paper_full": "mrior_sda_then_mopc_hr_paper_full",
}
FORMAL_COUNTS = {"preadapt_jobs": 1200, "cells": 800, "scenario_rows": 2400}
FORMAL_SHARD_COUNT = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _normal_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _validate_plan_payload(raw_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on any matrix drift before a runner creates an output root."""

    if not isinstance(raw_plan, Mapping):
        raise ValueError("MRIOR preadapt CI plan must be a JSON object")
    plan = dict(raw_plan)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("MRIOR preadapt CI plan schema drift")
    if plan.get("counts") != FORMAL_COUNTS:
        raise ValueError("MRIOR preadapt CI plan must close 1200/800/2400")
    scenarios = tuple(str(value) for value in _normal_list(plan.get("scenarios"), field="scenarios"))
    if scenarios != FORMAL_SCENARIOS:
        raise ValueError("MRIOR preadapt CI plan scenario matrix drift")
    methods = tuple(str(value) for value in _normal_list(plan.get("methods"), field="methods"))
    if methods != tuple(FORMAL_METHOD_MAP.values()):
        raise ValueError("MRIOR preadapt CI plan method matrix drift")
    source_methods = tuple(
        str(value) for value in _normal_list(plan.get("source_methods"), field="source methods")
    )
    if source_methods != tuple(FORMAL_METHOD_MAP):
        raise ValueError("MRIOR preadapt CI source method matrix drift")
    receivers = tuple(str(value) for value in _normal_list(plan.get("receivers"), field="receivers"))
    seeds = tuple(int(value) for value in _normal_list(plan.get("seeds"), field="seeds"))
    k_values = tuple(int(value) for value in _normal_list(plan.get("k_values"), field="K values"))
    new_counts = tuple(
        int(value) for value in _normal_list(plan.get("new_class_counts"), field="new counts")
    )
    if (
        len(receivers) != 5
        or len(set(receivers)) != 5
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or len(k_values) != 4
        or len(set(k_values)) != 4
        or len(new_counts) != 4
        or len(set(new_counts)) != 4
    ):
        raise ValueError("MRIOR preadapt CI formal matrix dimensions drift")
    if plan.get("preadapt_scope") != "receiver_seed_newcount_k_scene":
        raise ValueError("MRIOR preadapt CI preadapt scope drift")
    expected_job_keys = {
        (receiver, seed, new_count, k_shot, scenario)
        for receiver in receivers
        for seed in seeds
        for new_count in new_counts
        for k_shot in k_values
        for scenario in scenarios
    }
    jobs = _normal_list(plan.get("preadapt_jobs"), field="preadapt jobs")
    if len(jobs) != FORMAL_COUNTS["preadapt_jobs"]:
        raise ValueError("MRIOR preadapt CI plan requires 1200 preadapt jobs")
    job_ids: set[str] = set()
    job_keys: set[tuple[str, int, int, int, str]] = set()
    jobs_by_id: dict[str, dict[str, Any]] = {}
    for raw_job in jobs:
        if not isinstance(raw_job, Mapping):
            raise ValueError("MRIOR preadapt job schema drift")
        job = dict(raw_job)
        job_id = job.get("job_id")
        key = (
            str(job.get("receiver", "")),
            job.get("seed"),
            job.get("new_class_count"),
            job.get("k_shot"),
            str(job.get("scenario", "")),
        )
        if (
            not isinstance(job_id, str)
            or not job_id
            or job_id in job_ids
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or isinstance(key[2], bool)
            or not isinstance(key[2], int)
            or isinstance(key[3], bool)
            or not isinstance(key[3], int)
            or key not in expected_job_keys
            or key in job_keys
            or not isinstance(job.get("artifact_root"), str)
        ):
            raise ValueError("MRIOR preadapt job identity drift")
        _require_sha256(job.get("input_binding_sha256"), field="preadapt input binding SHA")
        _require_sha256(job.get("method_lock_sha256"), field="preadapt method lock SHA")
        job_ids.add(job_id)
        job_keys.add(key)
        jobs_by_id[job_id] = job
    if job_keys != expected_job_keys:
        raise ValueError("MRIOR preadapt job coverage drift")
    if len({str(job["artifact_root"]) for job in jobs_by_id.values()}) != len(jobs_by_id):
        raise ValueError("MRIOR preadapt artifact root collision")

    expected_cell_keys = {
        (receiver, seed, new_count, method, k_shot)
        for receiver in receivers
        for seed in seeds
        for new_count in new_counts
        for method in methods
        for k_shot in k_values
    }
    cells = _normal_list(plan.get("cells"), field="cells")
    if len(cells) != FORMAL_COUNTS["cells"]:
        raise ValueError("MRIOR preadapt CI plan requires 800 cells")
    cell_ids: set[str] = set()
    cell_keys: set[tuple[str, int, int, str, int]] = set()
    output_roots: set[str] = set()
    for raw_cell in cells:
        if not isinstance(raw_cell, Mapping):
            raise ValueError("MRIOR CI cell schema drift")
        cell = dict(raw_cell)
        cell_id = cell.get("cell_id")
        method = str(cell.get("method", ""))
        source_method = str(cell.get("source_v7_method", ""))
        key = (
            str(cell.get("receiver", "")),
            cell.get("seed"),
            cell.get("new_class_count"),
            method,
            cell.get("k_shot"),
        )
        output_root = cell.get("output_root")
        bindings = cell.get("preadapt_job_ids_by_scenario")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or cell_id in cell_ids
            or isinstance(key[1], bool)
            or not isinstance(key[1], int)
            or isinstance(key[2], bool)
            or not isinstance(key[2], int)
            or isinstance(key[4], bool)
            or not isinstance(key[4], int)
            or key not in expected_cell_keys
            or key in cell_keys
            or FORMAL_METHOD_MAP.get(source_method) != method
            or not isinstance(output_root, str)
            or output_root in output_roots
            or not isinstance(bindings, Mapping)
            or set(bindings) != set(scenarios)
        ):
            raise ValueError("MRIOR CI cell identity drift")
        for scenario in scenarios:
            job_id = bindings[scenario]
            job = jobs_by_id.get(str(job_id))
            if (
                job is None
                or job["receiver"] != key[0]
                or int(job["seed"]) != key[1]
                or int(job["new_class_count"]) != key[2]
                or int(job["k_shot"]) != key[4]
                or job["scenario"] != scenario
            ):
                raise ValueError("MRIOR CI cell preadapt binding drift")
        cell_ids.add(cell_id)
        cell_keys.add(key)
        output_roots.add(output_root)
    if cell_keys != expected_cell_keys:
        raise ValueError("MRIOR CI cell coverage drift")
    for field in ("smoke_preadapt_job_ids", "smoke_cell_ids"):
        values = _normal_list(plan.get(field), field=field)
        known = job_ids if field == "smoke_preadapt_job_ids" else cell_ids
        if not values or len(values) != len(set(values)) or not set(values) <= known:
            raise ValueError(f"MRIOR preadapt CI {field} drift")
    if plan.get("launch_authority") is not False or plan.get("authority_state") != (
        "N607_MRIOR_PREADAPT_CI_SMOKE_REQUIRED"
    ):
        raise ValueError("MRIOR preadapt CI plan authority surface drift")
    contract_payload = {
        key: value
        for key, value in plan.items()
        if key not in {"launch_authority", "authority_state", "plan_contract_sha256"}
    }
    if _canonical_sha256(contract_payload) != _require_sha256(
        plan.get("plan_contract_sha256"), field="plan contract SHA"
    ):
        raise ValueError("MRIOR preadapt CI plan contract hash drift")
    return plan


def _select_shard(
    entries: list[Mapping[str, Any]], *, shard_index: int, shard_count: int
) -> list[Mapping[str, Any]]:
    if int(shard_count) != FORMAL_SHARD_COUNT or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("MRIOR preadapt CI requires exactly eight valid shards")
    identity_field = "job_id" if all("job_id" in item for item in entries) else "cell_id"
    ordered = sorted(entries, key=lambda item: str(item[identity_field]))
    return [
        entry
        for index, entry in enumerate(ordered)
        if index % FORMAL_SHARD_COUNT == int(shard_index)
    ]


def _normalized_exception_fingerprint(exc: BaseException) -> str:
    lines = [line for line in str(exc).splitlines() if line.strip()]
    message = lines[-1] if lines else ""
    message = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
    message = re.sub(r"\b\d+\b", "<n>", message)
    message = re.sub(r"[/\\][^\s:]+", "<path>", message)
    return hashlib.sha256(f"{type(exc).__name__}:{message}".encode("utf-8")).hexdigest()


def _read_health_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema": HEALTH_STATE_SCHEMA,
            "stop_dispatch": False,
            "result_state": "RUNNING",
            "failures": [],
        }
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_health_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(state), handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _is_p0_protocol_safety_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "protocol violation",
            "query truth",
            "query role",
            "query quota",
            "global reassignment",
            "overwrite",
            "plan contract hash drift",
        )
    )


def _update_health_state(
    plan: Mapping[str, Any],
    *,
    row_id: str,
    exc: BaseException,
    prediction_produced: bool,
) -> dict[str, Any]:
    """Persist an execution-only stop gate; no metric can trigger this path."""

    state_path = Path(str(plan["run_root"])) / "run_health_state.json"
    state = _read_health_state(state_path)
    fingerprint = _normalized_exception_fingerprint(exc)
    state["failures"].append(
        {
            "row_id": str(row_id),
            "fingerprint": fingerprint,
            "exception_type": type(exc).__name__,
            "prediction_produced": bool(prediction_produced),
        }
    )
    matching_rows = {
        item["row_id"]
        for item in state["failures"]
        if item["fingerprint"] == fingerprint and item["prediction_produced"] is False
    }
    if _is_p0_protocol_safety_failure(exc):
        state["stop_dispatch"] = True
        state["stop_reason"] = "P0_PROTOCOL_OR_SAFETY_VIOLATION"
    elif len(matching_rows) >= 2:
        state["stop_dispatch"] = True
        state["stop_reason"] = (
            "TWO_DISTINCT_ROWS_SAME_NORMALIZED_EXCEPTION_BEFORE_PREDICTION"
        )
    if state.get("stop_dispatch") is True:
        state["result_state"] = "NO_PERFORMANCE_RESULT"
        state["trigger_fingerprint"] = fingerprint
    _write_health_state(state_path, state)
    return state


def _verify_matrix_closure(
    plan: Mapping[str, Any],
    *,
    artifacts: list[Mapping[str, Any]],
    cells: list[Mapping[str, Any]],
) -> dict[str, int]:
    """Require one verified artifact, prediction, score, and row set per matrix unit."""

    expected_jobs = {str(job["job_id"]) for job in plan["preadapt_jobs"]}
    observed_jobs = [str(item.get("job_id", "")) for item in artifacts]
    if len(observed_jobs) != FORMAL_COUNTS["preadapt_jobs"] or set(observed_jobs) != expected_jobs:
        raise ValueError("matrix closure requires exactly 1200 preadapt artifacts")
    expected_cells = {str(cell["cell_id"]) for cell in plan["cells"]}
    observed_cells = [str(item.get("cell_id", "")) for item in cells]
    if len(observed_cells) != FORMAL_COUNTS["cells"] or set(observed_cells) != expected_cells:
        raise ValueError("matrix closure requires exactly 800 prediction/score cells")
    if any(item.get("prediction") is not True or item.get("score") is not True for item in cells):
        raise ValueError("matrix closure requires one prediction and score per cell")
    expected_rows = {(cell_id, scenario) for cell_id in expected_cells for scenario in FORMAL_SCENARIOS}
    observed_rows = [
        (str(item["cell_id"]), str(scenario))
        for item in cells
        for scenario in item.get("scenarios", [])
    ]
    if len(observed_rows) != FORMAL_COUNTS["scenario_rows"] or set(observed_rows) != expected_rows:
        raise ValueError("matrix closure requires exactly 2400 unique scene rows")
    return dict(FORMAL_COUNTS)


def _claim_run_root(plan: Mapping[str, Any]) -> Path:
    """Create or verify the only root that this sealed plan is allowed to use."""

    run_root = Path(str(plan["run_root"])).resolve()
    plan_contract_sha256 = str(plan["plan_contract_sha256"])
    identity_path = run_root / "run_root_identity.json"
    expected_identity = {
        "schema": RUN_ROOT_IDENTITY_SCHEMA,
        "plan_contract_sha256": plan_contract_sha256,
    }
    if not run_root.exists():
        run_root.mkdir(parents=True, exist_ok=False)
        _write_new(identity_path, expected_identity)
        return run_root
    if not run_root.is_dir() or not identity_path.is_file():
        raise RuntimeError("unowned existing run root; refusing overwrite")
    try:
        observed = json.loads(identity_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("unowned existing run root; refusing overwrite") from exc
    if observed != expected_identity:
        raise RuntimeError("run root belongs to a different plan; refusing overwrite")
    return run_root


def _verify_smoke_authority(
    plan: Mapping[str, Any], *, project_root: Path
) -> None:
    """Accept only a runner-written smoke receipt for this immutable plan."""

    del project_root
    if plan.get("launch_authority") is not False or plan.get("authority_state") != (
        "N607_MRIOR_PREADAPT_CI_SMOKE_REQUIRED"
    ):
        raise ValueError("MRIOR preadapt CI matrix lacks smoke authority")
    receipt_path = Path(str(plan.get("run_root", ""))) / "smoke_receipt.json"
    try:
        receipt = _read_json(receipt_path, context="MRIOR preadapt CI smoke receipt")
    except ValueError as exc:
        raise ValueError("MRIOR preadapt CI matrix lacks smoke authority") from exc
    if (
        receipt.get("schema") != "cvs.phase2.adv3b02_mrior_preadapt_ci_smoke_receipt.v1"
        or receipt.get("status") != "PASS"
        or receipt.get("plan_contract_sha256") != plan["plan_contract_sha256"]
        or receipt.get("completed_preadapt_job_ids") != plan["smoke_preadapt_job_ids"]
        or receipt.get("completed_cell_ids") != plan["smoke_cell_ids"]
    ):
        raise ValueError("MRIOR preadapt CI smoke authority receipt drift")


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _load_plan(path: Path) -> dict[str, Any]:
    return _validate_plan_payload(_read_json(path.resolve(strict=True), context="plan"))


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(map(str, command)).encode("utf-8")).hexdigest()


def _run_json_command(
    command: Sequence[str], *, cwd: Path, receipt_path: Path
) -> dict[str, Any]:
    """Run one immutable child command and preserve its exact receipt."""

    process = subprocess.run(
        list(command), cwd=cwd, text=True, capture_output=True, check=False
    )
    receipt = {
        "schema": "cvs.phase2.adv3b02_mrior_preadapt_ci_command_receipt.v1",
        "command": [str(value) for value in command],
        "command_sha256": _command_sha256(command),
        "cwd": str(cwd.resolve()),
        "returncode": int(process.returncode),
        "stdout_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(process.stderr.encode("utf-8")).hexdigest(),
    }
    _write_new(receipt_path, receipt)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(map(str, command))}\n"
            f"stdout={process.stdout}\nstderr={process.stderr}"
        )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("command returned no JSON")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("command did not return final JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("command JSON result must be an object")
    return result


def _source_v7_context(plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    descriptor = plan.get("source_v7_plan")
    if not isinstance(descriptor, Mapping):
        raise ValueError("MRIOR preadapt CI source v7 plan descriptor is missing")
    path = Path(str(descriptor.get("path", ""))).resolve(strict=True)
    expected_sha = _require_sha256(descriptor.get("sha256"), field="source v7 plan SHA")
    if _sha256(path) != expected_sha:
        raise ValueError("MRIOR preadapt CI source v7 plan hash drift")
    source = _read_json(path, context="source v7 plan")
    packages = source.get("packages")
    if not isinstance(packages, list):
        raise ValueError("MRIOR preadapt CI source v7 packages are missing")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in packages:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("package_id"), str):
            raise ValueError("MRIOR preadapt CI source v7 package schema drift")
        package = dict(raw)
        if package["package_id"] in by_id:
            raise ValueError("MRIOR preadapt CI source v7 package duplicate")
        by_id[package["package_id"]] = package
    return source, by_id


def _torch_from_array(value: Any, *, dtype):
    import numpy as np
    import torch

    array = np.ascontiguousarray(value)
    if dtype == torch.float32:
        array = array.astype(np.float32, copy=False)
    elif dtype == torch.int64:
        array = array.astype(np.int64, copy=False)
    else:
        raise TypeError(dtype)
    return torch.frombuffer(memoryview(array), dtype=dtype).reshape(array.shape).clone()


def _load_preadapt_source_cache(path: Path):
    """Load the frozen source cache under its exact v1 or current v2 contract."""

    payload = _read_json(path.resolve(strict=True), context="preadapt source cache")
    schema = str(payload.get("schema", ""))
    if str(payload.get("cache_scope", "")) != "source_train":
        raise ValueError("preadapt source cache scope drift")
    if schema == "cvs_leo_weak_iq_cache_set_v2":
        from cvsrffi.leo_weak_cache import load_verified_leo_weak_cache_set

        return load_verified_leo_weak_cache_set(
            path, expected_scope="source_train", allowed_roles={"source"}
        )
    if schema == "cvs_leo_weak_iq_cache_set_v1":
        from paper_reproduction.scripts.build_adv3b02_paper_full_ci_bundle import (
            load_comparison_leo_cache_set,
        )

        arrays, manifest, audit = load_comparison_leo_cache_set(
            path, expected_scope="source_train", allowed_roles={"source"}
        )
        if (
            manifest.get("schema") != "cvs_leo_weak_iq_cache_set_v1"
            or manifest.get("cache_scope") != "source_train"
            or audit.get("status") != "PASS_COMPARISON_SCOPE"
        ):
            raise ValueError("legacy preadapt source cache verification drift")
        return arrays, manifest, {
            **audit,
            "legacy_source_cache_compatibility": "STRICT_V1",
        }
    raise ValueError(f"unsupported source cache-set schema: {schema!r}")


def _required_total_capacity(
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    package: Mapping[str, Any],
) -> int:
    """Recover the v7 capacity from its frozen class registry when omitted."""

    old_labels = package.get("old_class_labels")
    new_counts = plan.get("new_class_counts")
    if (
        not isinstance(old_labels, list)
        or not old_labels
        or not isinstance(new_counts, list)
        or not new_counts
        or any(isinstance(value, bool) or not isinstance(value, int) for value in new_counts)
    ):
        raise ValueError("source v7 class registry is incomplete")
    inferred = len(old_labels) + max(new_counts)
    declared = source.get("required_total_capacity", inferred)
    if (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or declared != inferred
    ):
        raise ValueError("source v7 total capacity drift")
    return inferred


def _read_existing_preadapt_receipt(job: Mapping[str, Any]) -> dict[str, Any]:
    from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (
        load_verified_mrior_preadapt_artifact,
    )

    root = Path(str(job["artifact_root"])).resolve(strict=True)
    result = load_verified_mrior_preadapt_artifact(
        root,
        expected_input_binding_sha256=str(job["input_binding_sha256"]),
        expected_method_lock_sha256=str(job["method_lock_sha256"]),
    )
    receipt = _read_json(root / "job_receipt.json", context="preadapt job receipt")
    if (
        receipt.get("status") != "PASS"
        or receipt.get("job_id") != job["job_id"]
        or receipt.get("query_unopened_receipt") != result.query_unopened_receipt
    ):
        raise ValueError("existing MRIOR preadapt artifact receipt drift")
    return receipt


def _run_preadapt_job(
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    project_root: Path,
    device_name: str,
    invocation: Sequence[str],
) -> dict[str, Any]:
    """Create one query-unopened MRIOR artifact, or verify its immutable receipt."""

    del project_root
    artifact_root = Path(str(job["artifact_root"])).resolve()
    if artifact_root.exists():
        return _read_existing_preadapt_receipt(job)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from cvsrffi.stage2_predictor_bundle import (
        _materialize_npz,
        _validate_support_arrays,
        preflight_stage2_predictor_package,
    )
    from paper_reproduction.cvs_aligned.adv3b02_mrior_preadapt_ci import (
        MRIORPreadaptInputBinding,
        fit_mrior_preadapted_backbone,
        write_mrior_preadapt_artifact,
    )
    from paper_reproduction.scripts.run_adv3b02_paper_full_ci_truth_free_predictor import (
        _load_exact_backbone,
    )

    source, packages = _source_v7_context(plan)
    package = packages.get(str(job["target_package_id"]))
    if package is None:
        raise ValueError("preadapt job target package is absent from source v7 plan")
    old_labels = package.get("old_class_labels")
    if not isinstance(old_labels, list) or not old_labels:
        raise ValueError("preadapt job source v7 old-class labels are missing")
    cache_path = Path(str(job["source_cache_manifest"])).resolve(strict=True)
    if _sha256(cache_path) != _require_sha256(
        job.get("source_cache_sha256"), field="preadapt source cache SHA"
    ):
        raise ValueError("preadapt source cache hash drift")
    source_arrays, _cache_manifest, _cache_audit = _load_preadapt_source_cache(
        cache_path
    )
    package_root = Path(str(job["target_package_root"])).resolve(strict=True)
    seal_path = Path(str(job["target_package_seal_path"])).resolve(strict=True)
    manifest, _seal, _audit = preflight_stage2_predictor_package(
        package_root,
        detached_seal_path=seal_path,
        expected_seal_sha256=str(job["target_package_seal_sha256"]),
    )
    roles = {item["artifact_role"]: item for item in manifest["members"]}
    scenario = str(job["scenario"])
    support, support_manifest = _materialize_npz(package_root, roles[f"support:{scenario}"])
    _validate_support_arrays(
        support,
        support_manifest,
        scenario=scenario,
        class_count=int(manifest["registered_class_count"]),
        max_k=int(manifest["support_pool_max_k"]),
    )
    support_labels = np.asarray(support["support_pool_class_indices"], dtype=np.int64)
    support_ranks = np.asarray(support["support_pool_rank_within_class"], dtype=np.int64)
    selected = np.flatnonzero(
        (support_labels < len(old_labels)) & (support_ranks < int(job["k_shot"]))
    )
    if len(selected) != len(old_labels) * int(job["k_shot"]):
        raise ValueError("preadapt target-old K-shot support coverage drift")
    target_x = _torch_from_array(
        np.asarray(support["support_pool_leo_weak_iq"])[selected], dtype=torch.float32
    )
    target_y = _torch_from_array(support_labels[selected], dtype=torch.int64).long()
    label_map = {str(label): index for index, label in enumerate(old_labels)}
    source_tx = np.asarray(source_arrays[scenario]["tx_ids"]).astype(str)
    if not set(source_tx.tolist()) <= set(label_map):
        raise ValueError("preadapt source cache contains a non-old transmitter")
    source_y = _torch_from_array(
        np.asarray([label_map[str(value)] for value in source_tx], dtype=np.int64),
        dtype=torch.int64,
    ).long()
    source_x = _torch_from_array(source_arrays[scenario]["leo_weak_iq"], dtype=torch.float32)
    source_loader = DataLoader(
        TensorDataset(source_x, source_y),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(job["seed"])),
    )
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    backbone, _feature_fn, _backbone_audit = _load_exact_backbone(
        package_root, manifest, device=device, verify_checkpoint_member=True
    )
    binding = MRIORPreadaptInputBinding.from_payload(job["input_binding"])
    result = fit_mrior_preadapted_backbone(
        backbone,
        source_loader,
        target_x.to(device),
        target_y.to(device),
        binding=binding,
        seed=int(job["seed"]),
    )
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    artifact_manifest = write_mrior_preadapt_artifact(artifact_root, result)
    receipt = {
        "schema": "cvs.phase2.adv3b02_mrior_preadapt_ci_job_receipt.v1",
        "status": "PASS",
        "job_id": job["job_id"],
        "artifact_manifest_sha256": _sha256(artifact_root / "manifest.json"),
        "artifact_state_sha256": _sha256(artifact_root / "mrior_preadapt_state.pt"),
        "input_binding_sha256": artifact_manifest["input_binding_sha256"],
        "method_lock_sha256": artifact_manifest["method_lock_sha256"],
        "query_unopened_receipt": artifact_manifest["query_unopened_receipt"],
        "runner_command": [str(value) for value in invocation],
        "runner_command_sha256": _command_sha256(invocation),
    }
    _write_new(artifact_root / "job_receipt.json", receipt)
    return receipt


def _preadapt_bindings_for_cell(
    plan: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any]:
    jobs = {str(job["job_id"]): job for job in plan["preadapt_jobs"]}
    bindings: dict[str, Any] = {}
    for scenario in FORMAL_SCENARIOS:
        job = jobs[str(cell["preadapt_job_ids_by_scenario"][scenario])]
        _read_existing_preadapt_receipt(job)
        bindings[scenario] = {
            "artifact_root": str(Path(str(job["artifact_root"])).resolve(strict=True)),
            "expected_input_binding_sha256": job["input_binding_sha256"],
            "expected_method_lock_sha256": job["method_lock_sha256"],
        }
    return {
        "schema": "cvs.phase2.adv3b02_mrior_preadapt_predictor_bindings.v1",
        "bindings": bindings,
    }


def _run_smoke_cell(
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    project_root: Path,
    device_name: str,
) -> dict[str, Any]:
    """Run one immutable MRIOR-prepared Task3 predictor and independent scorer cell."""

    output_root = Path(str(cell["output_root"])).resolve()
    receipt_path = output_root / "cell_receipt.json"
    if output_root.exists():
        if not receipt_path.is_file():
            raise RuntimeError("partial MRIOR CI cell output exists; refusing overwrite")
        receipt = _read_json(receipt_path, context="MRIOR CI cell receipt")
        if receipt.get("status") != "PASS" or receipt.get("cell_id") != cell["cell_id"]:
            raise ValueError("existing MRIOR CI cell receipt drift")
        return receipt
    source, packages = _source_v7_context(plan)
    package = packages.get(str(cell["target_package_id"]))
    if package is None:
        raise ValueError("MRIOR CI cell target package is absent from source v7 plan")
    old_labels = package.get("old_class_labels")
    if not isinstance(old_labels, list) or len(old_labels) != 6:
        raise ValueError("MRIOR CI cell old-class labels drift")
    required_total_capacity = _required_total_capacity(source, plan, package)
    build_receipt = _read_json(
        Path(str(package["build_receipt"])).resolve(strict=True),
        context="source v7 package build receipt",
    )
    if build_receipt.get("status") != "PASS":
        raise ValueError("source v7 package build receipt is not PASS")
    scoring_manifest_sha256 = _require_sha256(
        build_receipt.get("scoring_manifest_sha256"), field="source v7 scoring manifest SHA"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    bindings_path = output_root / "mrior_preadapt_bindings.json"
    _write_new(bindings_path, _preadapt_bindings_for_cell(plan, cell))
    predictor_output = output_root / "predictor"
    predictor_command = [
        sys.executable,
        str(project_root / "paper_reproduction/scripts/run_adv3b02_paper_full_ci_truth_free_predictor.py"),
        "--package-root",
        str(cell["target_package_root"]),
        "--detached-seal",
        str(cell["target_package_seal_path"]),
        "--expected-seal-sha256",
        str(cell["target_package_seal_sha256"]),
        "--method",
        str(cell["method"]),
        "--old-class-count",
        str(len(old_labels)),
        "--expected-total-capacity",
        str(required_total_capacity),
        "--k-shot",
        str(cell["k_shot"]),
        "--seed",
        str(cell["seed"]),
        "--row-id",
        str(cell["cell_id"]),
        "--output-dir",
        str(predictor_output),
        "--device",
        device_name,
        "--mrior-preadapt-bindings",
        str(bindings_path),
    ]
    predictor = _run_json_command(
        predictor_command,
        cwd=project_root,
        receipt_path=output_root / "predictor_command_receipt.json",
    )
    scoring_root = output_root / "scoring"
    scoring_command = [
        sys.executable,
        str(project_root / "code/scripts/score_cvs_stage2_sealed_prediction.py"),
        "--prediction-artifact",
        str(predictor["prediction_artifact"]),
        "--expected-prediction-artifact-sha256",
        str(predictor["prediction_artifact_sha256"]),
        "--expected-prediction-seal-sha256",
        str(predictor["prediction_seal_sha256"]),
        "--scoring-manifest",
        str(Path(str(package["scorer_root"])) / "scoring_manifest.json"),
        "--expected-scoring-manifest-sha256",
        scoring_manifest_sha256,
        "--formal-rows",
        str(scoring_root / "formal_rows.json"),
        "--formal-predictions",
        str(scoring_root / "formal_predictions.json"),
        "--scoring-receipt",
        str(scoring_root / "scoring_receipt.json"),
    ]
    scoring = _run_json_command(
        scoring_command,
        cwd=project_root,
        receipt_path=output_root / "scoring_command_receipt.json",
    )
    rows_payload = _read_json(scoring_root / "formal_rows.json", context="formal rows")
    rows = rows_payload.get("rows")
    if (
        rows_payload.get("schema") != "cvs.phase2.formal_metric_rows.v1"
        or not isinstance(rows, list)
        or len(rows) != len(FORMAL_SCENARIOS)
        or {row.get("scenario") for row in rows if isinstance(row, Mapping)}
        != set(FORMAL_SCENARIOS)
    ):
        raise ValueError("MRIOR CI scorer did not produce exactly three scenario rows")
    predictor_receipt = _read_json(
        predictor_output / "predictor_receipt.json", context="predictor receipt"
    )
    enrollment_receipt = _read_json(
        predictor_output / "enrollment_receipt.json", context="enrollment receipt"
    )
    if (
        predictor.get("status") != "FORMAL_COMPARISON_MRIOR_PREADAPT"
        or scoring.get("status") != "PASS"
        or predictor_receipt.get("query_opened_after_model_lock") is not True
        or enrollment_receipt.get("query_members_opened_before_model_lock") is not False
        or enrollment_receipt.get("query_rows_used_for_training") != 0
    ):
        raise ValueError("MRIOR CI query-boundary receipt drift")
    receipt = {
        "schema": "cvs.phase2.adv3b02_mrior_preadapt_ci_cell_receipt.v1",
        "status": "PASS",
        "cell_id": cell["cell_id"],
        "receiver": cell["receiver"],
        "seed": cell["seed"],
        "new_class_count": cell["new_class_count"],
        "k_shot": cell["k_shot"],
        "method": cell["method"],
        "prediction_artifact_sha256": predictor["prediction_artifact_sha256"],
        "prediction_seal_sha256": predictor["prediction_seal_sha256"],
        "predictor_receipt_sha256": _sha256(predictor_output / "predictor_receipt.json"),
        "scoring_receipt_sha256": _sha256(scoring_root / "scoring_receipt.json"),
        "formal_rows_sha256": _sha256(scoring_root / "formal_rows.json"),
        "query_boundary_receipt": {
            "query_opened_after_model_lock": True,
            "query_members_opened_before_model_lock": False,
            "query_rows_used_for_training": 0,
        },
    }
    _write_new(receipt_path, receipt)
    return receipt


def _all_preadapt_receipts(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_read_existing_preadapt_receipt(job) for job in plan["preadapt_jobs"]]


def _collect_cell_closure_entries(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read every completed cell into the strict matrix-closure shape."""

    entries: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        cell_id = str(cell["cell_id"])
        output_root = Path(str(cell["output_root"])).resolve()
        receipt = _read_json(
            output_root / "cell_receipt.json", context=f"MRIOR CI cell receipt {cell_id}"
        )
        if receipt.get("status") != "PASS" or receipt.get("cell_id") != cell_id:
            raise ValueError(f"MRIOR CI cell receipt drift: {cell_id}")
        if not (output_root / "predictor" / "prediction_artifact.cvspred").is_file():
            raise ValueError(f"MRIOR CI prediction artifact is missing: {cell_id}")
        scoring_receipt = _read_json(
            output_root / "scoring" / "scoring_receipt.json",
            context=f"MRIOR CI scoring receipt {cell_id}",
        )
        if scoring_receipt.get("status") != "PASS":
            raise ValueError(f"MRIOR CI scoring receipt is not PASS: {cell_id}")
        rows_payload = _read_json(
            output_root / "scoring" / "formal_rows.json",
            context=f"MRIOR CI formal rows {cell_id}",
        )
        rows = rows_payload.get("rows")
        if (
            rows_payload.get("schema") != "cvs.phase2.formal_metric_rows.v1"
            or not isinstance(rows, list)
            or len(rows) != len(FORMAL_SCENARIOS)
            or any(not isinstance(row, Mapping) for row in rows)
            or {str(row.get("scenario")) for row in rows} != set(FORMAL_SCENARIOS)
        ):
            raise ValueError(f"MRIOR CI formal rows are incomplete: {cell_id}")
        entries.append(
            {
                "cell_id": cell_id,
                "prediction": True,
                "score": True,
                "scenarios": [str(row["scenario"]) for row in rows],
            }
        )
    return entries


def _assert_dispatch_allowed(plan: Mapping[str, Any]) -> None:
    state = _read_health_state(Path(str(plan["run_root"])) / "run_health_state.json")
    if state.get("stop_dispatch") is True:
        raise RuntimeError("systemic technical failure stop gate is active; refusing dispatch")


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(Path(args.plan))
    project_root = Path(args.project_root).resolve(strict=True)
    stage = str(args.stage)
    if stage in {"preadapt_shard", "ci_shard"}:
        _verify_smoke_authority(plan, project_root=project_root)
    _claim_run_root(plan)
    invocation = [str(value) for value in getattr(args, "invocation", sys.argv)]
    if stage == "prepare":
        return {"status": "PASS", "stage": stage, "run_root": plan["run_root"]}
    if stage == "preadapt_smoke":
        if int(args.shard_index) != 0:
            raise ValueError("MRIOR preadapt CI preadapt smoke runs only on shard 0")
        jobs = {str(job["job_id"]): job for job in plan["preadapt_jobs"]}
        completed = []
        for job_id in plan["smoke_preadapt_job_ids"]:
            _assert_dispatch_allowed(plan)
            job = jobs[str(job_id)]
            try:
                receipt = _run_preadapt_job(
                    plan,
                    job,
                    project_root=project_root,
                    device_name=str(args.device),
                    invocation=invocation,
                )
            except Exception as exc:
                _update_health_state(
                    plan,
                    row_id=str(job["job_id"]),
                    exc=exc,
                    prediction_produced=False,
                )
                raise
            completed.append(receipt["job_id"])
        return {"status": "PASS", "stage": stage, "completed": completed}
    if stage == "preadapt_shard":
        selected = _select_shard(
            plan["preadapt_jobs"],
            shard_index=int(args.shard_index),
            shard_count=int(args.shard_count),
        )
        completed = []
        for job in selected:
            _assert_dispatch_allowed(plan)
            try:
                receipt = _run_preadapt_job(
                    plan,
                    job,
                    project_root=project_root,
                    device_name=str(args.device),
                    invocation=invocation,
                )
            except Exception as exc:
                _update_health_state(
                    plan,
                    row_id=str(job["job_id"]),
                    exc=exc,
                    prediction_produced=False,
                )
                raise
            completed.append(receipt["job_id"])
        return {"status": "PASS", "stage": stage, "completed": completed}
    if stage == "ci_shard":
        preadapt_receipts = _all_preadapt_receipts(plan)
        if len(preadapt_receipts) != FORMAL_COUNTS["preadapt_jobs"]:
            raise ValueError("ci_shard requires exactly 1200 verified preadapt artifacts")
        selected = _select_shard(
            plan["cells"],
            shard_index=int(args.shard_index),
            shard_count=int(args.shard_count),
        )
        completed = []
        for cell in selected:
            _assert_dispatch_allowed(plan)
            try:
                receipt = _run_smoke_cell(
                    plan, cell, project_root=project_root, device_name=str(args.device)
                )
            except Exception as exc:
                prediction_path = (
                    Path(str(cell["output_root"]))
                    / "predictor"
                    / "prediction_artifact.cvspred"
                )
                _update_health_state(
                    plan,
                    row_id=str(cell["cell_id"]),
                    exc=exc,
                    prediction_produced=prediction_path.is_file(),
                )
                raise
            completed.append(receipt["cell_id"])
        return {
            "status": "PASS",
            "stage": stage,
            "completed": completed,
            "verified_preadapt_artifacts": len(preadapt_receipts),
        }
    if stage == "smoke":
        if int(args.shard_index) != 0:
            raise ValueError("MRIOR preadapt CI smoke runs only on shard 0")
        jobs = {str(job["job_id"]): job for job in plan["preadapt_jobs"]}
        preadapt_receipts = [
            _read_existing_preadapt_receipt(jobs[str(job_id)])
            for job_id in plan["smoke_preadapt_job_ids"]
        ]
        cells = {str(cell["cell_id"]): cell for cell in plan["cells"]}
        completed_cells = []
        for cell_id in plan["smoke_cell_ids"]:
            _assert_dispatch_allowed(plan)
            cell = cells[str(cell_id)]
            try:
                receipt = _run_smoke_cell(
                    plan, cell, project_root=project_root, device_name=str(args.device)
                )
            except Exception as exc:
                prediction_path = Path(str(cell["output_root"])) / "predictor" / "prediction_artifact.cvspred"
                _update_health_state(
                    plan,
                    row_id=str(cell_id),
                    exc=exc,
                    prediction_produced=prediction_path.is_file(),
                )
                raise
            completed_cells.append(receipt["cell_id"])
        smoke_path = Path(str(plan["run_root"])) / "smoke_receipt.json"
        _write_new(
            smoke_path,
            {
                "schema": "cvs.phase2.adv3b02_mrior_preadapt_ci_smoke_receipt.v1",
                "status": "PASS",
                "plan_contract_sha256": plan["plan_contract_sha256"],
                "completed_preadapt_job_ids": plan["smoke_preadapt_job_ids"],
                "preadapt_receipt_sha256": {
                    receipt["job_id"]: _sha256(
                        Path(str(jobs[str(receipt["job_id"])]["artifact_root"]))
                        / "job_receipt.json"
                    )
                    for receipt in preadapt_receipts
                },
                "completed_cell_ids": completed_cells,
                "cell_receipt_sha256": {
                    cell_id: _sha256(Path(str(cells[cell_id]["output_root"])) / "cell_receipt.json")
                    for cell_id in completed_cells
                },
            },
        )
        return {"status": "PASS", "stage": stage, "completed": completed_cells}
    if stage == "finalize":
        artifacts = _all_preadapt_receipts(plan)
        if len(artifacts) != FORMAL_COUNTS["preadapt_jobs"]:
            raise ValueError("finalize requires exactly 1200 verified preadapt artifacts")
        cells = _collect_cell_closure_entries(plan)
        counts = _verify_matrix_closure(plan, artifacts=artifacts, cells=cells)
        final_path = Path(str(plan["run_root"])) / "final_receipt.json"
        _write_new(
            final_path,
            {
                "schema": "cvs.phase2.adv3b02_mrior_preadapt_ci_final_receipt.v1",
                "status": "PASS",
                "plan_contract_sha256": plan["plan_contract_sha256"],
                "counts": counts,
            },
        )
        return {
            "status": "PASS",
            "stage": stage,
            "final_receipt": str(final_path),
            "counts": counts,
        }
    raise ValueError(
        "stage must be prepare, preadapt_smoke, preadapt_shard, smoke, ci_shard, or finalize"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "preadapt_smoke", "preadapt_shard", "smoke", "ci_shard", "finalize"),
        required=True,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=FORMAL_SHARD_COUNT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=True, sort_keys=True))
