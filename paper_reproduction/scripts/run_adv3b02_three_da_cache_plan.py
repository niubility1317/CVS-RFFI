"""Execute and verify the offline LEO_weak cache-preparation phase only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for value in (str(CODE_ROOT), str(REPO_ROOT)):
    while value in sys.path:
        sys.path.remove(value)
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    sys.path.insert(0, value)

from cvsrffi.leo_weak_cache import load_verified_leo_weak_cache_set  # noqa: E402
from cvsrffi.phase2_runtime_contract import validate_phase2_contract  # noqa: E402
from cvsrffi.stage2_predictor_bundle import (  # noqa: E402
    load_verified_stage2_predictor_bundle,
)
from cvsrffi.stage2_scoring_sidecar import load_verified_scoring_sidecar  # noqa: E402


def _resolve_manifest_from_spec(spec_path: Path) -> tuple[Path, dict[str, Any]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    path = Path(str(spec["out_manifest"]))
    return (path if path.is_absolute() else spec_path.parent / path), spec


def _verify(spec_path: Path) -> dict[str, Any]:
    manifest_path, spec = _resolve_manifest_from_spec(spec_path)
    roles = {str(value["role"]) for value in spec["role_specs"]}
    _arrays, _manifest, audit = load_verified_leo_weak_cache_set(
        manifest_path,
        expected_scope=str(spec["cache_scope"]),
        allowed_roles=roles,
    )
    return audit


def _argument(command: list[str], flag: str) -> Path:
    index = command.index(flag)
    return Path(str(command[index + 1]))


def _verify_bundle(command: list[str]) -> dict[str, Any]:
    root = _argument(command, "--out-root")
    seal = root / "predictor_package_seal.json"
    seal_sha = (root / "predictor_package_seal.sha256").read_text(encoding="ascii").strip()
    _support, _query, _manifest, predictor_audit = load_verified_stage2_predictor_bundle(
        root / "predictor_package",
        detached_seal_path=seal,
        expected_seal_sha256=seal_sha,
    )
    _truth, _scoring, scorer_audit = load_verified_scoring_sidecar(
        root / "scoring_manifest.json"
    )
    return {"predictor": predictor_audit, "scoring": scorer_audit}


def _verify_seal(command: list[str]) -> dict[str, Any]:
    root = _argument(command, "--out-root")
    index = json.loads(
        (root / "runtime_isolation_evidence_index.json").read_text(encoding="utf-8-sig")
    )
    if index.get("count") != 25 or len(index.get("entries", [])) != 25:
        raise ValueError("runtime isolation evidence index must contain 25 receiver-seed entries")
    config = json.loads(_argument(command, "--config").read_text(encoding="utf-8-sig"))
    for entry in index["entries"]:
        evidence = json.loads(
            Path(str(entry["runtime_isolation_evidence"])).read_text(encoding="utf-8-sig")
        )
        record = dict(config)
        record["phase2_runtime_isolation_evidence"] = evidence
        validate_phase2_contract(record, evidence_phase="pre_run")
    return {"count": 25, "index": str(root / "runtime_isolation_evidence_index.json")}


def _run(command: list[str]) -> None:
    values = [str(value) for value in command]
    if values and values[0] == "python":
        values[0] = sys.executable
    subprocess.run(values, check=True, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan_manifest.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "adv3b02_three_da_leo_weak_only_plan_v1":
        raise ValueError("unexpected plan schema")
    if plan.get("phase2_config_exposes_dataset_path") is not False:
        raise ValueError("Phase2 config exposure guard failed")
    commands = list(plan["commands"]["phase1_offline_cache_build"])
    spec_entries = list(plan["cache_specs"])
    if len(commands) != len(spec_entries):
        raise ValueError("cache command/spec count drift")
    completed = skipped = 0
    audits: list[dict[str, Any]] = []
    plan_root = Path(str(plan["runtime_plan_dir"]))
    for command, entry in zip(commands, spec_entries):
        spec_path = plan_root / str(entry["spec"])
        try:
            audit = _verify(spec_path)
            skipped += 1
        except (FileNotFoundError, OSError):
            if not args.execute:
                continue
            _run(command)
            audit = _verify(spec_path)
            completed += 1
        audits.append({"scope": entry["scope"], "spec": str(spec_path), "audit": audit})
    bundle_commands = list(plan["commands"].get("phase1_offline_predictor_bundle_build", []))
    bundle_audits: list[dict[str, Any]] = []
    bundle_completed = bundle_skipped = 0
    for command in bundle_commands:
        try:
            audit = _verify_bundle(command)
            bundle_skipped += 1
        except (FileNotFoundError, OSError):
            if not args.execute:
                continue
            _run(command)
            audit = _verify_bundle(command)
            bundle_completed += 1
        bundle_audits.append(audit)
    seal_command = list(plan["commands"].get("phase2_runtime_seal", []))
    seal_audit: dict[str, Any] | None = None
    seal_completed = seal_skipped = 0
    if seal_command:
        try:
            seal_audit = _verify_seal(seal_command)
            seal_skipped = 1
        except (FileNotFoundError, OSError):
            if args.execute:
                _run(seal_command)
                seal_audit = _verify_seal(seal_command)
                seal_completed = 1
    expected_total = len(commands) + len(bundle_commands) + (1 if seal_command else 0)
    verified_total = len(audits) + len(bundle_audits) + (1 if seal_audit else 0)
    summary = {
        "schema": "adv3b02_three_da_cache_prep_summary_v1",
        "execute": bool(args.execute),
        "expected": expected_total,
        "completed": completed + bundle_completed + seal_completed,
        "skipped_verified": skipped + bundle_skipped + seal_skipped,
        "verified": verified_total,
        "cache_verified": len(audits),
        "predictor_bundle_verified": len(bundle_audits),
        "runtime_seal_verified": bool(seal_audit),
        "phase2_started": False,
        "audits": audits,
        "bundle_audits": bundle_audits,
        "runtime_seal_audit": seal_audit,
    }
    out_path = args.plan_manifest.parent / "cache_prep_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "audits"}, ensure_ascii=False))
    if args.execute and verified_total != expected_total:
        raise RuntimeError("not all offline caches, predictor bundles, and runtime seal were verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
