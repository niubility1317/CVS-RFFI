#!/usr/bin/env python3
"""Build packages and run the sharded paper-mechanism ADV3B02 CI plan."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_reproduction.scripts.build_adv3b02_paper_full_ci_plan import (
    validate_adapter_release_matrix,
)

METHODS = ("csil_paper_full", "mopc_hr_paper_full")
OFFICIAL_METHODS = (
    "csil_official_repo",
    "mopc_hr_official_repo",
    "csil_official_repo_corefix_cvs_adapter",
    "mopc_hr_official_repo_cvs_adapter",
    "mopc_hr_official_repo_sequential5_cvs_adapter",
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_json(
    command: Sequence[str],
    *,
    cwd: Path,
    health_plan: dict[str, Any] | None = None,
    row_id: str | None = None,
) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name != "nt"),
    )
    if health_plan is not None and row_id is not None:
        _register_active_child(
            health_plan,
            row_id=row_id,
            pid=int(process.pid),
            command=command,
            cwd=cwd,
        )
    try:
        stdout, stderr = process.communicate()
    finally:
        if health_plan is not None and row_id is not None:
            _unregister_active_child(health_plan, pid=int(process.pid))
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n"
            f"stdout={stdout}\nstderr={stderr}"
        )
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("command returned no JSON")
    return json.loads(lines[-1])


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "cvs.phase2.adv3b02_paper_full_ci_plan.v1":
        raise ValueError("paper-full plan schema drift")
    methods = tuple(str(value) for value in plan.get("methods", []))
    if methods != METHODS and not (
        methods
        and len(set(methods)) == len(methods)
        and all(value in OFFICIAL_METHODS for value in methods)
    ):
        raise ValueError("paper-full methods drift")
    new_counts = tuple(int(value) for value in plan.get("new_class_counts", []))
    validate_adapter_release_matrix(methods, new_counts)
    packages = plan.get("packages")
    cells = plan.get("cells")
    if not isinstance(packages, list) or not isinstance(cells, list):
        raise ValueError("paper-full package/cell surface drift")
    expected_counts = {
        "packages": len(packages),
        "cells": len(cells),
        "scenario_rows": len(cells) * 3,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError("paper-full matrix counts drift")
    if len({item.get("package_id") for item in packages}) != len(packages):
        raise ValueError("paper-full duplicate package id")
    if len({item.get("cell_id") for item in cells}) != len(cells):
        raise ValueError("paper-full duplicate cell id")
    required_capacity = int(plan.get("required_total_capacity", 0))
    if required_capacity <= 0:
        raise ValueError("paper-full required total capacity missing")
    if plan.get("expected_cache_scope") not in {
        "stage2_registered",
        "external_comparison_registered",
    }:
        raise ValueError("paper-full cache scope drift")
    import hashlib

    contract_payload = {
        key: value
        for key, value in plan.items()
        if key
        not in {
            "smoke_receipt_sha256",
            "smoke_receipt_path",
            "launch_authority",
            "authority_state",
            "plan_contract_sha256",
        }
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(
            contract_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if plan.get("plan_contract_sha256") != contract_sha256:
        raise ValueError("paper-full plan contract hash drift")
    for name, descriptor in plan.get("artifacts", {}).items():
        artifact_path = Path(str(descriptor.get("path", ""))).resolve(strict=True)
        if _sha256(artifact_path) != str(descriptor.get("sha256", "")):
            raise ValueError(f"paper-full artifact hash drift: {name}")
    return plan


def _normalized_exception_fingerprint(exc: BaseException) -> str:
    message = str(exc).splitlines()[-1] if str(exc).splitlines() else ""
    message = re.sub(r"0x[0-9a-fA-F]+", "<hex>", message)
    message = re.sub(r"\b\d+\b", "<n>", message)
    message = re.sub(r"[/\\][^\s:]+", "<path>", message)
    import hashlib

    normalized = f"{type(exc).__name__}:{message}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@contextmanager
def _health_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_health_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema": "cvs.adv3b02.paper_full_ci_health_state.v1",
            "stop_dispatch": False,
            "failures": [],
            "active_children": [],
            "termination_events": [],
        }
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_health_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _command_sha256(command: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256(
        "\0".join(str(value) for value in command).encode("utf-8")
    ).hexdigest()


def _register_active_child(
    plan: dict[str, Any],
    *,
    row_id: str,
    pid: int,
    command: Sequence[str],
    cwd: Path,
) -> None:
    root = Path(plan["run_root"])
    state_path = root / "run_health_state.json"
    with _health_lock(root / ".run_health_state.lock"):
        state = _read_health_state(state_path)
        state.setdefault("active_children", []).append(
            {
                "row_id": str(row_id),
                "pid": int(pid),
                "process_group_id": int(pid) if os.name != "nt" else None,
                "cwd": str(Path(cwd).resolve()),
                "command_sha256": _command_sha256(command),
            }
        )
        _write_health_state(state_path, state)


def _unregister_active_child(plan: dict[str, Any], *, pid: int) -> None:
    root = Path(plan["run_root"])
    state_path = root / "run_health_state.json"
    with _health_lock(root / ".run_health_state.lock"):
        state = _read_health_state(state_path)
        state["active_children"] = [
            item
            for item in state.get("active_children", [])
            if int(item.get("pid", -1)) != int(pid)
        ]
        _write_health_state(state_path, state)


def _is_p0_protocol_safety_failure(exc: BaseException) -> bool:
    lines = [line.strip().lower() for line in str(exc).splitlines() if line.strip()]
    message = lines[-1] if lines else ""
    markers = (
        "protocol",
        "query",
        "truth",
        "overwrite",
        "seal",
        "hash drift",
        "hash mismatch",
        "scope drift",
        "capacity drift",
        "parity",
        "class sets overlap",
    )
    return any(marker in message for marker in markers)


def _linux_child_identity_matches(child: dict[str, Any]) -> bool:
    if os.name == "nt":
        return True
    pid = int(child["pid"])
    proc_root = Path("/proc") / str(pid)
    try:
        cwd = str((proc_root / "cwd").resolve(strict=True))
        command = (proc_root / "cmdline").read_bytes().rstrip(b"\0")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    import hashlib

    command_sha = hashlib.sha256(command).hexdigest()
    return cwd == child["cwd"] and command_sha == child["command_sha256"]


def _process_group_exists(process_group_id: int) -> bool:
    if os.name == "nt":
        try:
            os.kill(int(process_group_id), 0)
            return True
        except ProcessLookupError:
            return False
    try:
        os.killpg(int(process_group_id), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_registered_children(children: list[dict[str, Any]]) -> list[dict]:
    events = []
    for child in children:
        pid = int(child["pid"])
        if not _linux_child_identity_matches(child):
            events.append(
                {"pid": pid, "row_id": child["row_id"], "status": "IDENTITY_REJECTED"}
            )
            continue
        try:
            if os.name == "nt":
                os.kill(pid, 15)
            else:
                import signal

                os.killpg(int(child["process_group_id"]), signal.SIGTERM)
            status = "SIGTERM_SENT"
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if not _process_group_exists(
                    int(child["process_group_id"] or pid)
                ):
                    status = "TERMINATED_GRACEFULLY"
                    break
                time.sleep(0.1)
            else:
                if os.name != "nt" and _process_group_exists(
                    int(child["process_group_id"])
                ):
                    import signal

                    os.killpg(int(child["process_group_id"]), signal.SIGKILL)
                    status = "SIGKILL_SENT_AFTER_GRACE"
            events.append({"pid": pid, "row_id": child["row_id"], "status": status})
        except (ProcessLookupError, PermissionError) as exc:
            events.append(
                {
                    "pid": pid,
                    "row_id": child["row_id"],
                    "status": type(exc).__name__,
                }
            )
    return events


def _update_health_state(
    plan: dict[str, Any],
    *,
    row_id: str | None = None,
    exc: BaseException | None = None,
    prediction_produced: bool = False,
) -> dict[str, Any]:
    root = Path(plan["run_root"])
    state_path = root / "run_health_state.json"
    lock_path = root / ".run_health_state.lock"
    children_to_terminate = []
    with _health_lock(lock_path):
        state = _read_health_state(state_path)
        if row_id is not None and exc is not None:
            fingerprint = _normalized_exception_fingerprint(exc)
            state["failures"].append(
                {
                    "row_id": str(row_id),
                    "fingerprint": fingerprint,
                    "exception_type": type(exc).__name__,
                    "prediction_produced": bool(prediction_produced),
                }
            )
            distinct_rows = {
                item["row_id"]
                for item in state["failures"]
                if item["fingerprint"] == fingerprint
                and item["prediction_produced"] is False
            }
            if _is_p0_protocol_safety_failure(exc):
                state["stop_dispatch"] = True
                state["stop_reason"] = "P0_PROTOCOL_OR_SAFETY_VIOLATION"
                state["trigger_fingerprint"] = fingerprint
            elif len(distinct_rows) >= 2:
                state["stop_dispatch"] = True
                state["stop_reason"] = (
                    "TWO_DISTINCT_ROWS_SAME_DETERMINISTIC_EXCEPTION_"
                    "BEFORE_PREDICTION"
                )
                state["trigger_fingerprint"] = fingerprint
            if state.get("stop_dispatch") is True:
                children_to_terminate = list(state.get("active_children", []))
        _write_health_state(state_path, state)
    if children_to_terminate:
        events = _terminate_registered_children(children_to_terminate)
        with _health_lock(lock_path):
            state = _read_health_state(state_path)
            state.setdefault("termination_events", []).extend(events)
            _write_health_state(state_path, state)
    return state


def _assert_dispatch_allowed(plan: dict[str, Any]) -> None:
    state = _update_health_state(plan)
    if state.get("stop_dispatch") is True:
        raise RuntimeError(
            "systemic technical failure stop gate is active; refusing dispatch"
        )


def _selected(index: int, shard_index: int, shard_count: int) -> bool:
    return int(index) % int(shard_count) == int(shard_index)


def _load_formal_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema") != "cvs.phase2.formal_metric_rows.v1":
        raise ValueError("formal metric row schema drift")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("formal metric rows payload drift")
    return rows


def _verify_cache_parity_receipt(
    plan: dict[str, Any], package: dict[str, Any]
) -> dict[str, Any] | None:
    if plan["expected_cache_scope"] != "external_comparison_registered":
        return None
    verification_mode = str(
        plan.get("cache_verification_mode", "historical_reference")
    )
    if verification_mode not in {
        "historical_reference",
        "same_cache_new20_integrity",
    }:
        raise ValueError("unsupported cache verification mode")
    raw = package.get("cache_parity_receipt")
    if not raw:
        raise ValueError("external comparison package misses cache parity receipt")
    path = Path(raw).resolve(strict=True)
    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    target_cache = Path(package["target_cache_set"]).resolve(strict=True)
    reference_cache = Path(
        package["cache_parity_reference_cache_set"]
    ).resolve(strict=True)
    scenario_receipts = receipt.get("scenario_receipts")
    expected_rows_per_scenario = (
        len(plan["parity_preserved_class_labels"]) * 50
    )
    expected_schema = (
        "cvs.adv3b02.same_cache_new20_integrity_receipt.v1"
        if verification_mode == "same_cache_new20_integrity"
        else "cvs.adv3b02.official_scale_cache_parity_receipt.v1"
    )
    same_cache_paths_valid = (
        (
            verification_mode == "same_cache_new20_integrity"
            and reference_cache == target_cache
        )
        or (
            verification_mode == "historical_reference"
            and reference_cache != target_cache
        )
    )
    valid_scenario_receipts = (
        isinstance(scenario_receipts, dict)
        and tuple(scenario_receipts)
        == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
        and all(
            int(value.get("row_count", -1)) == expected_rows_per_scenario
            and len(str(value.get("sample_ids_sha256", ""))) == 64
            and len(str(value.get("post_channel_iq_sha256_root", ""))) == 64
            for value in scenario_receipts.values()
        )
    )
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("verification_mode", "historical_reference")
        != verification_mode
        or receipt.get("status") != "PASS"
        or Path(str(receipt.get("expanded_cache_set", ""))).resolve()
        != target_cache
        or receipt.get("expanded_cache_set_sha256") != _sha256(target_cache)
        or Path(str(receipt.get("reference_cache_set", ""))).resolve()
        != reference_cache
        or receipt.get("reference_cache_set_sha256")
        != _sha256(reference_cache)
        or receipt.get("preserved_class_labels")
        != plan["parity_preserved_class_labels"]
        or receipt.get("verified_fields")
        != ["tx_ids", "sample_ids", "post_channel_iq_sha256"]
        or not same_cache_paths_valid
        or not valid_scenario_receipts
    ):
        raise ValueError("cache parity receipt does not authorize package build")
    return receipt


def _verify_smoke_authority(
    plan: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    if (
        plan.get("launch_authority") is not True
        or plan.get("authority_state") != "N607_PAPER_FULL_CI_SMOKE_PASS"
    ):
        raise ValueError("formal paper-full matrix lacks smoke authority")
    raw_path = plan.get("smoke_receipt_path")
    if not raw_path:
        raise ValueError("formal paper-full matrix misses smoke receipt path")
    path = Path(str(raw_path)).resolve(strict=True)
    receipt = json.loads(path.read_text(encoding="utf-8-sig"))
    executed_plan = Path(
        str(receipt.get("executed_plan_path", ""))
    ).resolve(strict=True)
    expected_artifacts = {
        key: value["sha256"] for key, value in plan["artifacts"].items()
    }
    cell_receipt_hashes = receipt.get("cell_receipt_sha256")
    cells = {item["cell_id"]: item for item in plan["cells"]}
    smoke_artifacts_valid = isinstance(cell_receipt_hashes, dict)
    if smoke_artifacts_valid:
        for cell_id in plan["smoke_cell_ids"]:
            output_root = Path(cells[cell_id]["output_root"])
            cell_receipt_path = output_root / "cell_receipt.json"
            try:
                cell_receipt = json.loads(
                    cell_receipt_path.read_text(encoding="utf-8-sig")
                )
                smoke_artifacts_valid = (
                    cell_receipt_hashes.get(cell_id)
                    == _sha256(cell_receipt_path)
                    and cell_receipt.get("status")
                    == "FORMAL_COMPARISON_BASELINE"
                    and cell_receipt.get("cell_id") == cell_id
                    and cell_receipt.get("prediction_artifact_sha256")
                    == _sha256(
                        output_root
                        / "predictor"
                        / "prediction_artifact.cvspred"
                    )
                    and cell_receipt.get("scoring_receipt_sha256")
                    == _sha256(
                        output_root / "scoring" / "scoring_receipt.json"
                    )
                    and cell_receipt.get("formal_rows_sha256")
                    == _sha256(
                        output_root / "scoring" / "formal_rows.json"
                    )
                )
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                smoke_artifacts_valid = False
            if not smoke_artifacts_valid:
                break
    if (
        _sha256(path) != plan.get("smoke_receipt_sha256")
        or receipt.get("schema")
        != "cvs.phase2.adv3b02_paper_full_ci_smoke_receipt.v1"
        or receipt.get("status") != "PASS"
        or receipt.get("completed_cell_ids") != plan["smoke_cell_ids"]
        or receipt.get("plan_contract_sha256")
        != plan["plan_contract_sha256"]
        or receipt.get("executed_plan_sha256") != _sha256(executed_plan)
        or receipt.get("artifact_sha256") != expected_artifacts
        or receipt.get("predictor_script_sha256")
        != _sha256(project_root / plan["predictor_script"])
        or not smoke_artifacts_valid
    ):
        raise ValueError("formal smoke receipt authority verification failed")
    return receipt


def _build_package(
    plan: dict[str, Any],
    package: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    parity_receipt = _verify_cache_parity_receipt(plan, package)
    receipt_path = Path(package["build_receipt"])
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        expected_parity_sha = (
            _sha256(Path(package["cache_parity_receipt"]))
            if parity_receipt is not None
            else None
        )
        if (
            receipt.get("status") != "PASS"
            or receipt.get("cache_parity_receipt_sha256")
            != expected_parity_sha
        ):
            raise ValueError("existing package receipt is not PASS")
        return receipt
    if receipt_path.parent.exists():
        raise RuntimeError("partial package parent exists; refusing destructive resume")
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    artifacts = plan["artifacts"]
    result = _run_json(
        [
            sys.executable,
            str(
                project_root
                / "paper_reproduction/scripts/build_adv3b02_paper_full_ci_bundle.py"
            ),
            "--target-cache-set",
            package["target_cache_set"],
            "--expected-cache-scope",
            str(plan["expected_cache_scope"]),
            "--predictor-out-root",
            package["predictor_package_root"],
            "--scorer-out-root",
            package["scorer_root"],
            "--detached-seal-path",
            package["detached_seal"],
            "--stage",
            "stage2c",
            "--receiver",
            package["receiver"],
            "--seed",
            str(package["seed"]),
            "--old-class-labels",
            ",".join(package["old_class_labels"]),
            "--new-class-labels",
            ",".join(package["new_class_labels"]),
            "--new-class-count",
            str(package["new_class_count"]),
            "--support-pool-max-k",
            "20",
            "--query-per-tx",
            "20",
            "--candidate-lock",
            artifacts["candidate_lock"]["path"],
            "--checkpoint",
            artifacts["base_checkpoint"]["path"],
            "--adapter",
            artifacts["adapter"]["path"],
            "--head-artifact",
            artifacts["head_artifact"]["path"],
            "--tta-policy-json",
            artifacts["tta_policy"]["path"],
        ],
        cwd=project_root,
        health_plan=plan,
        row_id=package["package_id"],
    )
    receipt = {
        "schema": "cvs.phase2.adv3b02_paper_full_ci_package_build_receipt.v1",
        "status": "PASS",
        "package_id": package["package_id"],
        "predictor_package_root_sha256": result["predictor_package_root_sha256"],
        "predictor_package_seal_sha256": result["predictor_package_seal_sha256"],
        "scoring_manifest_sha256": result["scoring_manifest_sha256"],
        "cache_parity_receipt_sha256": (
            _sha256(Path(package["cache_parity_receipt"]))
            if parity_receipt is not None
            else None
        ),
        "builder_result": result,
    }
    _write_new(receipt_path, receipt)
    return receipt


def _run_cell(
    plan: dict[str, Any],
    cell: dict[str, Any],
    package: dict[str, Any],
    *,
    project_root: Path,
    device: str,
) -> dict[str, Any]:
    output_root = Path(cell["output_root"])
    cell_receipt = output_root / "cell_receipt.json"
    if cell_receipt.is_file():
        value = json.loads(cell_receipt.read_text(encoding="utf-8-sig"))
        if value.get("status") != "FORMAL_COMPARISON_BASELINE":
            raise ValueError("existing paper-full cell receipt status drift")
        return value
    if output_root.exists():
        raise RuntimeError("partial cell output exists; refusing destructive resume")
    build_receipt = _build_package(plan, package, project_root=project_root)
    predictor_root = output_root / "predictor"
    predictor = _run_json(
        [
            sys.executable,
            str(project_root / plan["predictor_script"]),
            "--package-root",
            package["predictor_package_root"],
            "--detached-seal",
            package["detached_seal"],
            "--expected-seal-sha256",
            build_receipt["predictor_package_seal_sha256"],
            "--method",
            cell["method"],
            "--old-class-count",
            "6",
            "--expected-total-capacity",
            str(plan["required_total_capacity"]),
            "--k-shot",
            str(cell["k_shot"]),
            "--seed",
            str(cell["seed"]),
            "--row-id",
            cell["cell_id"],
            "--output-dir",
            str(predictor_root),
            "--device",
            device,
        ],
        cwd=project_root,
        health_plan=plan,
        row_id=cell["cell_id"],
    )
    if predictor.get("status") != "FORMAL_COMPARISON_BASELINE":
        raise ValueError("paper-full predictor status drift")
    scoring_root = output_root / "scoring"
    scoring_root.mkdir(parents=True, exist_ok=False)
    scoring_manifest = Path(package["scorer_root"]) / "scoring_manifest.json"
    scoring = _run_json(
        [
            sys.executable,
            str(project_root / "code/scripts/score_cvs_stage2_sealed_prediction.py"),
            "--prediction-artifact",
            predictor["prediction_artifact"],
            "--expected-prediction-artifact-sha256",
            predictor["prediction_artifact_sha256"],
            "--expected-prediction-seal-sha256",
            predictor["prediction_seal_sha256"],
            "--scoring-manifest",
            str(scoring_manifest),
            "--expected-scoring-manifest-sha256",
            build_receipt["scoring_manifest_sha256"],
            "--formal-rows",
            str(scoring_root / "formal_rows.json"),
            "--formal-predictions",
            str(scoring_root / "formal_predictions.json"),
            "--scoring-receipt",
            str(scoring_root / "scoring_receipt.json"),
        ],
        cwd=project_root,
        health_plan=plan,
        row_id=cell["cell_id"],
    )
    if len(_load_formal_rows(scoring_root / "formal_rows.json")) != 3:
        raise ValueError("cell scorer did not produce three scenario rows")
    receipt = {
        "schema": "cvs.phase2.adv3b02_paper_full_ci_cell_receipt.v1",
        "status": "FORMAL_COMPARISON_BASELINE",
        **{
            key: cell[key]
            for key in (
                "cell_id",
                "package_id",
                "receiver",
                "seed",
                "new_class_count",
                "method",
                "k_shot",
            )
        },
        "predictor_receipt_sha256": predictor["predictor_receipt_sha256"],
        "prediction_artifact_sha256": predictor["prediction_artifact_sha256"],
        "prediction_seal_sha256": predictor["prediction_seal_sha256"],
        "scoring_receipt_sha256": _sha256(scoring_root / "scoring_receipt.json"),
        "formal_rows_sha256": _sha256(scoring_root / "formal_rows.json"),
        "scoring_status": scoring.get("status"),
    }
    _write_new(cell_receipt, receipt)
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(Path(args.plan).resolve(strict=True))
    project_root = Path(args.project_root).resolve(strict=True)
    if not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index/count")
    packages = {item["package_id"]: item for item in plan["packages"]}
    cells = {item["cell_id"]: item for item in plan["cells"]}
    completed = []
    if args.stage == "build_shard":
        for index, package in enumerate(plan["packages"]):
            if _selected(index, args.shard_index, args.shard_count):
                _assert_dispatch_allowed(plan)
                try:
                    receipt = _build_package(
                        plan, package, project_root=project_root
                    )
                except Exception as exc:
                    _update_health_state(
                        plan,
                        row_id=package["package_id"],
                        exc=exc,
                        prediction_produced=False,
                    )
                    raise
                completed.append(receipt["package_id"])
    elif args.stage == "smoke":
        if int(args.shard_index) != 0:
            raise ValueError("smoke runs only on shard 0")
        for cell_id in plan["smoke_cell_ids"]:
            _assert_dispatch_allowed(plan)
            cell = cells[cell_id]
            try:
                _run_cell(
                    plan,
                    cell,
                    packages[cell["package_id"]],
                    project_root=project_root,
                    device=args.device,
                )
            except Exception as exc:
                prediction_path = (
                    Path(cell["output_root"])
                    / "predictor"
                    / "prediction_artifact.cvspred"
                )
                _update_health_state(
                    plan,
                    row_id=cell_id,
                    exc=exc,
                    prediction_produced=prediction_path.is_file(),
                )
                raise
            completed.append(cell_id)
        receipt = {
            "schema": "cvs.phase2.adv3b02_paper_full_ci_smoke_receipt.v1",
            "status": "PASS",
            "completed_cell_ids": completed,
            "executed_plan_path": str(Path(args.plan).resolve(strict=True)),
            "executed_plan_sha256": _sha256(Path(args.plan).resolve(strict=True)),
            "plan_contract_sha256": plan["plan_contract_sha256"],
            "artifact_sha256": {
                key: value["sha256"] for key, value in plan["artifacts"].items()
            },
            "predictor_script_sha256": _sha256(
                project_root / plan["predictor_script"]
            ),
            "cell_receipt_sha256": {
                cell_id: _sha256(Path(cells[cell_id]["output_root"]) / "cell_receipt.json")
                for cell_id in completed
            },
        }
        _write_new(Path(plan["run_root"]) / "smoke_receipt.json", receipt)
    else:
        _verify_smoke_authority(plan, project_root=project_root)
        cells_by_package = {}
        for cell in plan["cells"]:
            cells_by_package.setdefault(cell["package_id"], []).append(cell)
        for index, package in enumerate(plan["packages"]):
            if not _selected(index, args.shard_index, args.shard_count):
                continue
            for cell in cells_by_package[package["package_id"]]:
                _assert_dispatch_allowed(plan)
                try:
                    receipt = _run_cell(
                        plan,
                        cell,
                        package,
                        project_root=project_root,
                        device=args.device,
                    )
                except Exception as exc:
                    prediction_path = (
                        Path(cell["output_root"])
                        / "predictor"
                        / "prediction_artifact.cvspred"
                    )
                    _update_health_state(
                        plan,
                        row_id=cell["cell_id"],
                        exc=exc,
                        prediction_produced=prediction_path.is_file(),
                    )
                    raise
                completed.append(receipt["cell_id"])
    return {
        "status": "PASS",
        "stage": args.stage,
        "shard_index": args.shard_index,
        "completed": completed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("build_shard", "smoke", "matrix_shard"), required=True
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
