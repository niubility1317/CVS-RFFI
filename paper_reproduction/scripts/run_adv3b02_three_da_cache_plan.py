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
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi.leo_weak_cache import load_verified_leo_weak_cache_set  # noqa: E402


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
            subprocess.run([str(value) for value in command], check=True, cwd=REPO_ROOT)
            audit = _verify(spec_path)
            completed += 1
        audits.append({"scope": entry["scope"], "spec": str(spec_path), "audit": audit})
    summary = {
        "schema": "adv3b02_three_da_cache_prep_summary_v1",
        "execute": bool(args.execute),
        "expected": len(commands),
        "completed": completed,
        "skipped_verified": skipped,
        "verified": len(audits),
        "phase2_started": False,
        "audits": audits,
    }
    out_path = args.plan_manifest.parent / "cache_prep_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "audits"}, ensure_ascii=False))
    if args.execute and len(audits) != len(commands):
        raise RuntimeError("not all cache artifacts were verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
