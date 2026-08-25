from __future__ import annotations

from pathlib import Path

import pytest

from cvsrffi.slow_fast_no_query_smoke import run_slow_fast_no_query_smoke


def test_no_query_smoke_has_no_query_path_capability(tmp_path: Path) -> None:
    config = {
        "candidate_id": "COMMON_SHIFT_R4",
        "bundle_id": "bundle",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule",
        "split_id": "split",
        "base_checkpoint_path": "base.pth",
        "bundle_path": "bundle.pt",
        "support_path": "support.npz",
        "prototype_path": "prototype.npz",
        "receiver": "20-1",
        "scenario": "leo_clear_weak",
        "operating_point": "K10/new10",
        "seed": 392002,
        "k_shot": 10,
        "steps": 3,
        "query_path": "forbidden.npz",
    }
    with pytest.raises(ValueError, match="allowlist.*query_path"):
        run_slow_fast_no_query_smoke(config, tmp_path / "smoke.json", device="cpu")
    assert not (tmp_path / "smoke.json").exists()
