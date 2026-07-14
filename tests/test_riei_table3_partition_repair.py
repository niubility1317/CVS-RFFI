from pathlib import Path
import re


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "launch_riei_table3_partition_repair_20260714.sh"


def test_partition_repair_covers_table3_and_paper_metric() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    rows = re.findall(r'^\s+"(\d+)\|([^|]+)\|([^|]+)\|([^|]+)\|([0-9.]+)\|([0-9.]+)"$', text, re.MULTILINE)
    assert len(rows) == 12
    assert [int(row[0]) for row in rows] == list(range(1, 13))
    assert '"RIEI_PAPER_EVAL_LAST_N=10"' in text
    assert "stable_group_seed_shared_train_test_holdout" in text
    assert '"RIEI_OPTIMIZER=sgd"' in text
    assert '"RIEI_CE_REDUCTION=mean"' in text


def test_partition_repair_uses_capacity_gated_sequential_queues() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '[[ "${#GPU_IDS[@]}" -eq 8 ]]' in text
    assert "planned_peak=1" in text
    assert "total <= MAX_TRAIN_PER_GPU" in text
    assert 'nohup bash "${queue}"' in text
    assert "unique run/log root already exists" in text
