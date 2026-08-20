#!/usr/bin/env python3
"""Reuse the validated Stage2-C controllers to add seeds 7282104/7282105."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


SUPPLEMENTAL_METHOD_SEEDS = (7282104, 7282105)
MODE_CONSTANTS = {
    "package": {
        "METHOD_SEEDS": SUPPLEMENTAL_METHOD_SEEDS,
        "EXPECTED_TASKS": 30,
    },
    "feature": {
        "METHOD_SEEDS": SUPPLEMENTAL_METHOD_SEEDS,
        "EXPECTED_TASKS": 50,
        "EXPECTED_SCOPE_CACHES": 150,
    },
}


def patched_constants(mode: str) -> dict[str, object]:
    if mode not in MODE_CONSTANTS:
        raise ValueError(f"unsupported extension mode: {mode}")
    return dict(MODE_CONSTANTS[mode])


def _load_controller(path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"controller is not a regular file: {path}")
    spec = importlib.util.spec_from_file_location("m24_full125_controller", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(MODE_CONSTANTS))
    parser.add_argument("--controller", type=Path, required=True)
    return parser


def main() -> int:
    args, forwarded = _parser().parse_known_args()
    module = _load_controller(args.controller.absolute())
    for name, value in patched_constants(args.mode).items():
        if not hasattr(module, name):
            raise AttributeError(f"controller lacks expected constant: {name}")
        setattr(module, name, value)
    if not callable(getattr(module, "main", None)):
        raise AttributeError("controller lacks callable main()")
    forwarded = list(forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    sys.argv = [str(args.controller), *forwarded]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
