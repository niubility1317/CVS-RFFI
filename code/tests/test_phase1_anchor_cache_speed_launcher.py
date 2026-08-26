import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "phase1_adv3b02_fasttrust_qb3_anchor_cache_speed_e6_20260826.json"
LAUNCHER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_fasttrust_qb3_anchor_cache_speed_e6_20260826.sh"
MATRIX_LAUNCHER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_fasttrust_qb3_matrix_20260826.sh"


def test_anchor_cache_speed_matrix_is_a_paired_single_factor_e6_probe():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert data["epochs"] == 6
    assert data["unlabeled_batch_size"] == 256
    assert data["training_schedule"] == "LEO_WEAK"
    rows = data["rows"]
    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {392002}
    assert {row["gpu"] for row in rows} == {6, 7}
    assert {row["cache_anchor_logits"] for row in rows} == {False, True}
    for row in rows:
        assert row["hard"] and row["partial"] and row["partial_set"]
        assert row["partial_conditional"] is False
        assert row["feature_anchor"] == 0.0
        assert row["eval_batch_size"] == 512
        assert row["recovery_checkpoint_interval"] == 1


def test_anchor_cache_speed_launcher_and_matrix_runner_propagate_the_flag():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    runner = MATRIX_LAUNCHER.read_text(encoding="utf-8")

    assert "phase1_adv3b02_fasttrust_qb3_anchor_cache_speed_e6_20260826.json" in launcher
    assert 'SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"' in launcher
    assert 'CODE_ROOT="${CODE_ROOT:-${SCRIPT_ROOT}}"' in launcher
    assert 'RC4_CACHE_ANCHOR_LOGITS="${cache_anchor_logits}"' in runner
