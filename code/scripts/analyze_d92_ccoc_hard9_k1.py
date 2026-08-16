#!/usr/bin/env python3
"""CLI for the truth-last D92 CCOC Hard9+K1 analyzer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.stage2_d92_ccoc_hard9_k1_analysis import (  # noqa: E402
    analyze_d92_ccoc_hard9_k1,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Analyze one immutable D92 CCOC Hard9+K1 run after truth-sidecar scoring."
    )
    result.add_argument("--matrix-manifest", "--manifest", dest="matrix_manifest", required=True)
    result.add_argument("--method-lock", required=True)
    result.add_argument("--run-root", required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--baseline-paired-rows")
    result.add_argument("--per-old-class-rows")
    result.add_argument("--truth-sidecar-root")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = analyze_d92_ccoc_hard9_k1(
        args.matrix_manifest,
        run_root=args.run_root,
        method_lock_path=args.method_lock,
        baseline_paired_rows_path=args.baseline_paired_rows,
        per_old_class_rows_path=args.per_old_class_rows,
        truth_sidecar_root=args.truth_sidecar_root,
        output_root=args.output_root,
    )
    outputs = result.get("output_paths", {})
    print(json.dumps({"verdict": result["verdict"], "outputs": outputs}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
