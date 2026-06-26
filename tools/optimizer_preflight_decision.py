#!/usr/bin/env python
"""Build a read-only local preflight decision for CV-SincNet optimizer runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from optimizer_state_current_view import current_view
from optimizer_validate_matrix import (
    expected_count_for_matrix,
    load_stage2_sample_protocol,
    validate,
)
from optimizer_workflow_lib import item_list, load_json_compat, read_text_compat, write_json


PROJECT_DOC_NAME = "\u9879\u76ee.md"
DEFAULT_ACTIVE_PROMPT = (
    Path("automation_reports")
    / "CV-SincNet"
    / "automation_prompt_backups"
    / "20260615_001820_stage2_closed_loop_v4"
    / "stage2_prompt.md"
)
DEFAULT_STATE = Path("automation_reports") / "CV-SincNet" / "stage2_optimizer_state.json"
DEFAULT_REQUIRED_FILES = (
    Path("AGENTS.md"),
    Path(PROJECT_DOC_NAME),
    Path("tools") / "optimizer_control_manifest.md",
    DEFAULT_ACTIVE_PROMPT,
    Path("tools") / "optimizer_workflow_contract.md",
    DEFAULT_STATE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def file_record(project_root: Path, relative_path: Path) -> Dict[str, Any]:
    path = project_root / relative_path
    record: Dict[str, Any] = {
        "relative_path": str(relative_path).replace("\\", "/"),
        "path": str(path),
        "status": "PASS",
    }
    try:
        record.update(
            {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "mtime_utc": utc_mtime(path),
            }
        )
        if path.suffix.lower() in {".md", ".json", ".toml", ".txt", ".py", ".sh"}:
            read_text_compat(path)
    except Exception as exc:  # pragma: no cover - exception type varies by platform
        record.update({"status": "UNREADABLE", "error": str(exc)})
    return record


def control_readiness(project_root: Path, state_path: Path) -> Dict[str, Any]:
    required = [file_record(project_root, path) for path in DEFAULT_REQUIRED_FILES]
    status = "PASS" if all(item["status"] == "PASS" for item in required) else "BLOCKED"
    payload: Dict[str, Any] = {
        "status": status,
        "hard_blocker": None if status == "PASS" else "USER_REQUIRED_SAFETY_STOP",
        "required_files": required,
    }
    if status != "PASS":
        return payload

    state_abs = project_root / state_path
    try:
        state = load_json_compat(state_abs)
        if not isinstance(state, Mapping):
            raise ValueError("stage2 optimizer state root must be an object")
        view = current_view(state, state_path)
        payload["state_view"] = {
            "schema": view["schema"],
            "source_path": view["source_path"],
            "source_state_sha256": sha256_file(state_abs),
            "current_decision_keys": view["current_decision_keys"],
            "audit_only_keys_present": view["audit_only_keys_present"],
            "state_size_bytes": view["state_size_bytes"],
            "current_view_size_bytes": view["current_view_size_bytes"],
        }
    except Exception as exc:  # pragma: no cover - exception type varies by platform
        payload["status"] = "BLOCKED"
        payload["hard_blocker"] = "USER_REQUIRED_SAFETY_STOP"
        payload["state_view_error"] = str(exc)
    return payload


def load_optional_text(path: Optional[Path]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if path is None:
        return None, None
    try:
        return read_text_compat(path), None
    except Exception as exc:  # pragma: no cover - exception type varies by platform
        return "", {"scope": "launcher", "issue": "launcher_unreadable", "launcher_path": str(path), "error": str(exc)}


def matrix_readiness(
    project_root: Path,
    matrix_path: Optional[Path],
    launcher_path: Optional[Path],
    expected_count: Optional[int],
    state_path: Path,
) -> Dict[str, Any]:
    if matrix_path is None:
        return {"status": "NOT_PROVIDED", "runner_readiness": "NO_MATRIX_PROVIDED", "issues": []}

    root = load_json_compat(matrix_path)
    items = item_list(root)
    resolved_expected = expected_count_for_matrix(root, expected_count)
    sample_protocol: Optional[Mapping[str, Any]] = None
    if isinstance(root, Mapping) and isinstance(root.get("stage2_sample_protocol") or root.get("sample_protocol"), Mapping):
        sample_protocol = load_stage2_sample_protocol(root)
    else:
        state_abs = project_root / state_path
        if state_abs.exists():
            state = load_json_compat(state_abs)
            if isinstance(state, Mapping):
                sample_protocol = load_stage2_sample_protocol(state)

    launcher_text, launcher_issue = load_optional_text(launcher_path)
    result = validate(
        items,
        resolved_expected,
        sample_protocol=sample_protocol,
        matrix_root=root if isinstance(root, Mapping) else None,
        launcher_text=launcher_text,
        launcher_path=str(launcher_path) if launcher_path else None,
    )
    if launcher_issue:
        result["issues"].insert(0, launcher_issue)
        result["verdict"] = "FAIL"
    total = result.get("launchability_summary", {}).get("total", {})
    return {
        "status": "PASS" if result["verdict"] == "PASS" else "BLOCKED",
        "matrix_path": str(matrix_path),
        "verdict": result["verdict"],
        "expected_count": result["expected_count"],
        "candidate_count": result["candidate_count"],
        "runner_readiness": total.get("runner_readiness", "UNKNOWN"),
        "launchable_count": total.get("launchable", 0),
        "deferred_count": total.get("deferred", 0),
        "non_launchable_count": total.get("non_launchable", 0),
        "issue_count": len(result["issues"]),
        "issues": result["issues"],
        "launchability_summary": result.get("launchability_summary", {}),
    }


def launcher_readiness(matrix_payload: Mapping[str, Any], launcher_path: Optional[Path]) -> Dict[str, Any]:
    if launcher_path is None:
        return {"status": "NOT_PROVIDED", "issues": []}
    launcher_issues = [
        issue
        for issue in matrix_payload.get("issues", [])
        if issue.get("scope") == "launcher" or str(issue.get("issue", "")).startswith("launcher_")
    ]
    return {
        "status": "PASS" if not launcher_issues else "BLOCKED",
        "launcher_path": str(launcher_path),
        "issues": launcher_issues,
    }


def duplicate_readiness(matrix_path: Optional[Path]) -> Dict[str, Any]:
    if matrix_path is None:
        return {"status": "NOT_PROVIDED", "issues": []}
    root = load_json_compat(matrix_path)
    items = item_list(root)
    issues: List[Dict[str, Any]] = []
    for field in ("registry_key", "command_hash"):
        values = [str(item.get(field) or "").strip() for item in items]
        missing = [
            str(item.get("candidate_id") or "UNKNOWN")
            for item, value in zip(items, values)
            if not value
        ]
        duplicates = sorted({value for value in values if value and values.count(value) > 1})
        if missing:
            issues.append({"scope": "matrix", "issue": f"missing_{field}", "candidate_ids": missing})
        if duplicates:
            issues.append({"scope": "matrix", "issue": f"duplicate_{field}", "values": duplicates})
    return {"status": "PASS" if not issues else "BLOCKED", "issues": issues}


def remote_readiness() -> Dict[str, Any]:
    return {
        "status": "PENDING_REMOTE_MONITOR",
        "remote_actions_performed": False,
        "required_next_step": "Run AGENTS-approved N607 preflight and live process/CWD/cmdline/GPU monitor before remote gates.",
    }


def derive_overall_status(parts: Mapping[str, Mapping[str, Any]]) -> tuple[str, Optional[str]]:
    if parts["control_readiness"]["status"] == "BLOCKED":
        return "BLOCKED", parts["control_readiness"].get("hard_blocker") or "USER_REQUIRED_SAFETY_STOP"
    for key in ("matrix_readiness", "launcher_readiness", "duplicate_readiness"):
        status = parts[key]["status"]
        if status == "BLOCKED":
            return "BLOCKED", f"{key.upper()}_BLOCKED"
    if parts["matrix_readiness"]["status"] == "NOT_PROVIDED":
        return "PENDING_LOCAL_ARTIFACTS", None
    return "PENDING_REMOTE_MONITOR", None


def preflight_decision(
    project_root: Path = Path("."),
    matrix_path: Optional[Path] = None,
    launcher_path: Optional[Path] = None,
    expected_count: Optional[int] = None,
    state_path: Path = DEFAULT_STATE,
) -> Dict[str, Any]:
    project_root = project_root.resolve()
    matrix_path = matrix_path.resolve() if matrix_path else None
    launcher_path = launcher_path.resolve() if launcher_path else None
    control = control_readiness(project_root, state_path)
    matrix = (
        {"status": "SKIPPED_CONTROL_BLOCKED", "runner_readiness": "CONTROL_BLOCKED", "issues": []}
        if control["status"] == "BLOCKED"
        else matrix_readiness(project_root, matrix_path, launcher_path, expected_count, state_path)
    )
    launcher = (
        {"status": "SKIPPED_CONTROL_BLOCKED", "issues": []}
        if control["status"] == "BLOCKED"
        else launcher_readiness(matrix, launcher_path)
    )
    duplicates = (
        {"status": "SKIPPED_CONTROL_BLOCKED", "issues": []}
        if control["status"] == "BLOCKED"
        else duplicate_readiness(matrix_path)
    )
    remote = remote_readiness()
    parts = {
        "control_readiness": control,
        "matrix_readiness": matrix,
        "launcher_readiness": launcher,
        "duplicate_readiness": duplicates,
        "remote_readiness": remote,
    }
    overall_status, blocker_code = derive_overall_status(parts)
    return {
        "schema": "optimizer_preflight_decision_v1",
        "project_root": str(project_root),
        "overall_status": overall_status,
        "blocker_code": blocker_code,
        **parts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = preflight_decision(
        project_root=args.project_root,
        matrix_path=args.matrix,
        launcher_path=args.launcher,
        expected_count=args.expected_count,
        state_path=args.state,
    )
    if args.output:
        write_json(args.output, payload)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "overall_status": payload["overall_status"],
                "blocker_code": payload["blocker_code"],
                "control_status": payload["control_readiness"]["status"],
                "matrix_status": payload["matrix_readiness"]["status"],
                "launcher_status": payload["launcher_readiness"]["status"],
                "duplicate_status": payload["duplicate_readiness"]["status"],
                "remote_status": payload["remote_readiness"]["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if payload["overall_status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
