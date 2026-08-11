#!/usr/bin/env python3
"""Analyze one complete D92-E0OCF Hard12-v3 artifact set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_e0ocf_analysis import analyze_d92_e0ocf_hard12v3  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix-manifest", required=True, help="frozen Hard12-v3 matrix_manifest.json")
    result.add_argument("--run-root", default=None, help="retrieved artifact root overriding manifest output_root")
    result.add_argument("--method-lock", default=None, help="local method-lock override")
    result.add_argument("--output-root", default=None, help="directory for summary.json/gates.json/paired_rows.csv/analysis.md")
    result.add_argument("--dry-run", action="store_true", help="validate CLI inputs without reading performance artifacts")
    return result


def _write_outputs(result: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "gates.json").write_text(json.dumps(result["gates"], ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(result.get("paired_rows", []))
    if rows:
        fields = sorted({str(key) for row in rows for key in row})
        with (output_root / "paired_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        (output_root / "paired_rows.csv").write_text("outer_key\n", encoding="utf-8")
    lines = [
        "# D92-E0OCF Hard12-v3分析摘要",
        "",
        f"- 状态：`{result.get('status')}`",
        f"- claim scope：`{result.get('claim_scope')}`",
        f"- promotion candidate：`{result.get('promotion_candidate')}`",
        f"- verdict：`{result.get('verdict')}`",
        "",
        "Hard12-v3是development-only stress screen，不等同于正式Target125，也不构成无偏估计。E0_OCF50仅作diagnostic-only，不得改变verdict。",
    ]
    (output_root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parser().parse_args()
    manifest = Path(args.matrix_manifest).resolve(strict=True)
    if args.dry_run:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        print(json.dumps({"status": "DRY_RUN", "matrix_manifest": str(manifest), "job_count": payload.get("job_count"), "shard_count": payload.get("shard_count")}, ensure_ascii=True, sort_keys=True))
        return 0
    result = analyze_d92_e0ocf_hard12v3(manifest, run_root=args.run_root, method_lock_path=args.method_lock)
    output_root = Path(args.output_root) if args.output_root else (Path(args.run_root) if args.run_root else Path(str(result["matrix_manifest"])).parent)
    _write_outputs(result, output_root)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
