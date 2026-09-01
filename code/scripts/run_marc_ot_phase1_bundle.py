#!/usr/bin/env python3
"""Build one immutable source-only MARC-OT Phase1 weight bundle."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.marc_ot_phase1_entry import (  # noqa: E402
    run_marc_ot_phase1_bundle,
    validate_marc_ot_phase1_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", required=True)
    return parser


def _resolve_project_input(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    payload = validate_marc_ot_phase1_config(
        json.loads(config_path.read_text(encoding="utf-8-sig"))
    )
    wisig_path = _resolve_project_input(str(payload["wisig_pkl"]))
    if not wisig_path.is_file() or wisig_path.is_symlink():
        raise FileNotFoundError(f"ManySig pickle is not a regular file: {wisig_path}")
    with wisig_path.open("rb") as handle:
        ds_w = pickle.load(handle)
    result = run_marc_ot_phase1_bundle(args, ds_w)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
