from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "code/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_cvs_phase2_effective8_lock_artifacts import build  # noqa: E402
from build_cvs_phase2_effective8_candidate_capsule import (  # noqa: E402
    _audit_head_and_tta_policy,
)


def test_v14_three_threshold_lock_exports_six_slots_with_noop_defaults(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "candidate_lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": "cvs_stage2c_source_candidate_lock_v2",
                "locked_candidate": {
                    "head": {
                        "mode": "symmetric_locked",
                        "selected": {
                            "prototype_rule": "mean",
                            "ridge": None,
                            "use_alignment": False,
                        },
                    },
                    "adaptive_tta": {
                        "thresholds": {
                            "base_stop_margin": 0.05,
                            "shift3_stop_margin": 0.0,
                            "shift3_max_disagreement": 2.0 / 3.0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    result = build(argparse.Namespace(candidate_lock=lock, out_dir=tmp_path / "out"))
    tta = json.loads(Path(result["tta_policy"]).read_text(encoding="utf-8"))
    assert tta["base_stop_min_score"] == -1.0e9
    assert tta["shift3_stop_min_score"] == -1.0e9
    assert tta["fusion_std_penalty"] == 0.0
    assert result["compatibility_defaults"] == {
        "base_stop_min_score": -1.0e9,
        "shift3_stop_min_score": -1.0e9,
        "fusion_std_penalty": 0.0,
    }
    _audit_head_and_tta_policy(
        json.loads(lock.read_text(encoding="utf-8")),
        head_lock=Path(result["head_lock"]),
        tta_policy=Path(result["tta_policy"]),
    )
