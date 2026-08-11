#!/usr/bin/env python3
"""Write immutable strict D92-E0D Hard12-v2 analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_e0d_analysis import analyze_d92_e0d_hard12v2  # noqa: E402


def _write_json_new(path: Path, payload: object) -> None:
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run(
    *,
    matrix_manifest: str | Path,
    output_dir: str | Path,
    run_root: str | Path | None = None,
    method_lock: str | Path | None = None,
) -> dict:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    result = analyze_d92_e0d_hard12v2(
        matrix_manifest,
        run_root=run_root,
        method_lock_path=method_lock,
    )
    output.mkdir(parents=True)
    _write_json_new(output / "summary.json", result)
    _write_json_new(output / "gates.json", result["gates"])
    paired_path = output / "paired_rows.csv"
    with paired_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["paired_rows"][0]))
        writer.writeheader()
        writer.writerows(result["paired_rows"])
    os.chmod(paired_path, 0o444)
    return {
        "status": "D92_E0D_HARD12V2_STRICT_GEOMETRY_ANALYSIS_COMPLETE",
        "verdict": result["verdict"],
        "output_dir": str(output.resolve()),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--matrix-manifest", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--run-root")
    result.add_argument("--method-lock")
    return result


def main() -> int:
    args = parser().parse_args()
    result = run(
        matrix_manifest=args.matrix_manifest,
        output_dir=args.output_dir,
        run_root=args.run_root,
        method_lock=args.method_lock,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
