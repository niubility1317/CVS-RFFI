#!/usr/bin/env python3
"""Minimal fail-closed preflight for an M2.4 prediction suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


def _json_object(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if json.loads(canonical) != value:
        raise ValueError(f"{path} cannot be represented canonically")
    return dict(value)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_preflight(
    *,
    base_manifest: str | Path,
    overlay_manifest: str | Path,
    scoring_manifest: str | Path,
    output_root: str | Path,
    expected_commit: str,
    repository: str | Path,
) -> dict[str, Any]:
    base = _json_object(base_manifest)
    overlay = _json_object(overlay_manifest)
    scoring = _json_object(scoring_manifest)
    if base.get("protocol_schema") != "p2_min_v1" or base.get("phase2_data_status") != "VALIDATED_ONCE":
        raise ValueError("base manifest protocol status drift")
    for field in ("capsule_id", "split_id", "receiver", "k_shot", "method_seed"):
        if base.get(field) != overlay.get(field):
            raise ValueError(f"base/overlay {field} drift")
    if not any("truth" in str(key).lower() or "scoring" in str(key).lower() for key in scoring):
        raise ValueError("scoring manifest schema is not recognizable")
    if Path(output_root).exists():
        raise FileExistsError("M2.4 output root already exists")
    for module in ("numpy", "cvsrffi.stage2_m24_row_executor", "cvsrffi.stage2_m24_safe_residual"):
        importlib.import_module(module)
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if head != str(expected_commit):
        raise ValueError("run commit does not match repository HEAD")
    return {
        "status": "PASS",
        "python": sys.executable,
        "commit": head,
        "base_manifest_sha256": _sha256(base_manifest),
        "overlay_manifest_sha256": _sha256(overlay_manifest),
        "scoring_manifest_sha256": _sha256(scoring_manifest),
        "output_root_absent": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--overlay-manifest", required=True)
    parser.add_argument("--scoring-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repository", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_preflight(
        base_manifest=args.base_manifest,
        overlay_manifest=args.overlay_manifest,
        scoring_manifest=args.scoring_manifest,
        output_root=args.output_root,
        expected_commit=args.expected_commit,
        repository=args.repository,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
