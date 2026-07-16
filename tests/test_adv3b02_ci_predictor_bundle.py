from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_reproduction.scripts.build_adv3b02_ci_predictor_bundle import (
    reject_predictor_truth_leaks_structurally,
)


def test_numeric_npz_bytes_do_not_create_false_truth_match(tmp_path: Path) -> None:
    root = tmp_path / "predictor"
    root.mkdir()
    with (root / "query.npz").open("xb") as handle:
        np.savez(handle, iq=np.frombuffer(b"prefix-14-10-suffix", dtype=np.uint8))
    reject_predictor_truth_leaks_structurally(root, ["14-10"])


@pytest.mark.parametrize("surface", ["member", "value", "json"])
def test_text_surfaces_still_fail_closed(tmp_path: Path, surface: str) -> None:
    root = tmp_path / "predictor"
    root.mkdir()
    if surface == "member":
        with (root / "query.npz").open("xb") as handle:
            np.savez(handle, **{"14-10": np.asarray([1], dtype=np.int64)})
    elif surface == "value":
        with (root / "query.npz").open("xb") as handle:
            np.savez(handle, token=np.asarray(["qid_14-10"]))
    else:
        (root / "tta_policy.json").write_text(
            json.dumps({"tx": "14-10"}), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="forbidden truth/role token"):
        reject_predictor_truth_leaks_structurally(root, ["14-10"])
