#!/usr/bin/env python
"""Build an ADV3B02 CI bundle with structural NPZ truth-surface auditing."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "code" / "scripts"
for value in (str(REPO_ROOT), str(SCRIPT_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_cvs_stage2_predictor_bundle as base_builder  # noqa: E402


def reject_predictor_truth_leaks_structurally(
    root: Path, forbidden_values: Iterable[str]
) -> None:
    """Reject text-bearing truth surfaces without scanning compressed numeric bytes."""
    needles = {
        value.encode("utf-8")
        for value in forbidden_values
        if isinstance(value, str) and value
    }
    pre_registered_binary_artifacts = {"checkpoint.bin", "adapter.bin", "head.bin"}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"predictor package contains a non-regular member: {path}")
        if path.name in pre_registered_binary_artifacts:
            continue
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for member_name in archive.files:
                    if any(needle in member_name.encode("utf-8") for needle in needles):
                        raise ValueError(
                            "predictor package contains forbidden truth/role token "
                            f"in {path.name}:{member_name}"
                        )
                    array = np.asarray(archive[member_name])
                    if array.dtype.kind not in {"S", "U"}:
                        continue
                    for value in array.reshape(-1).tolist():
                        payload = (
                            bytes(value)
                            if isinstance(value, (bytes, bytearray))
                            else str(value).encode("utf-8")
                        )
                        if any(needle in payload for needle in needles):
                            raise ValueError(
                                "predictor package contains forbidden truth/role token "
                                f"in {path.name}:{member_name}"
                            )
            continue
        payload = path.read_bytes()
        if any(needle in payload for needle in needles):
            raise ValueError(
                f"predictor package contains forbidden truth/role token in {path.name}"
            )


def main() -> int:
    base_builder._reject_predictor_truth_leaks = reject_predictor_truth_leaks_structurally
    return base_builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
