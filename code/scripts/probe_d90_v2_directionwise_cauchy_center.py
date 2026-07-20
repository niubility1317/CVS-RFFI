#!/usr/bin/env python3
"""D90 v2 directionwise Cauchy-center diagnostic probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
D89_PROBE_PATH = SCRIPT_DIR / "probe_d89_v2_radius_cauchy_center.py"
D89_CORE_PATH = CODE_ROOT / "cvsrffi" / "stage2_d89_v2_radius_cauchy_center.py"
CORE_PATH = CODE_ROOT / "cvsrffi" / "stage2_d90_v2_directionwise_cauchy_center.py"

ARM = "v2_directionwise_cauchy_center"
STRUCTURE = "d62_with_v2_directionwise_cauchy_support_center"
FORMULA = (
    "reuse the fixed D89 v2 cellwise SNR-radius ground spectrum; inside every "
    "target class preserve the D81 radial robust-center component orthogonal to "
    "the retained ground subspace, but replace its subspace component by one "
    "independent Cauchy robust center per retained ground direction using only "
    "that class support; translate the z160 class-common center once, preserve "
    "within-class residuals and FFT96/RF32, then reuse the locked D62 OOF closure "
    "and compile one INT8 affine head"
)


class D90ProbeError(RuntimeError):
    """Raised when D90 integration or evidence closure drifts."""


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise D90ProbeError(f"D90 could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


d89 = _load("d90_d89_probe_scaffold", D89_PROBE_PATH)
core = _load("d90_directionwise_core", CORE_PATH)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--d90-arm", required=True, choices=(ARM,))
    return parser


def main(argv: list[str] | None = None) -> int:
    _known, remaining = build_parser().parse_known_args(argv)
    output = d89.base.d43._runner_output(remaining)
    d89.ARM = ARM
    d89.STRUCTURE = STRUCTURE
    d89.FORMULA = FORMULA
    d89.CORE_PATH = CORE_PATH
    d89.core = core
    d89.__file__ = str(Path(__file__).resolve())
    d89.OUTPUT_METADATA_NAME = "D90_PROBE_METADATA.json"
    d89.OUTPUT_METADATA_SCHEMA = (
        "cvs.phase2.d90.v2_directionwise_cauchy_center_probe.v1"
    )
    d89.EXTRA_SOURCE_CLOSURE = {
        "d90_d89_probe_scaffold_sha256": _sha256(D89_PROBE_PATH),
        "d90_d89_core_sha256": _sha256(D89_CORE_PATH),
    }
    exit_code = int(d89.main(["--d89-arm", ARM, *remaining]))
    if exit_code != 0:
        return exit_code
    metadata_path = output / d89.OUTPUT_METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "schema": d89.OUTPUT_METADATA_SCHEMA,
        "arm": ARM,
        "formula": FORMULA,
        "d90_core_sha256": _sha256(CORE_PATH),
        "d90_probe_sha256": _sha256(Path(__file__).resolve()),
        "d90_d89_probe_scaffold_sha256": _sha256(D89_PROBE_PATH),
        "d90_d89_core_sha256": _sha256(D89_CORE_PATH),
        "directionwise_subspace_center": True,
        "d81_orthogonal_center_preserved": True,
        "forced_nonpromotable": True,
    })
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
