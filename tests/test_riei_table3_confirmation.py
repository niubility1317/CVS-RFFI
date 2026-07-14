from pathlib import Path
import re


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "scripts" / "launch_riei_table3_confirm_sgd_mean_20260714.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_table3_confirmation_has_all_twelve_paper_rows() -> None:
    text = _text()
    rows = re.findall(
        r'^\s+"(\d+)\|([^|]+)\|([^|]+)\|([^|]+)\|([0-9.]+)\|([0-9.]+)"$',
        text,
        flags=re.MULTILINE,
    )
    assert len(rows) == 12
    assert [int(row[0]) for row in rows] == list(range(1, 13))
    assert len({row[1] for row in rows}) == 12
    assert [float(row[4]) for row in rows] == [
        77.88, 79.43, 66.09, 70.51, 77.35, 75.48,
        71.91, 68.33, 73.54, 73.52, 72.05, 73.46,
    ]
    assert [float(row[5]) for row in rows] == [
        2.23, 1.66, 0.67, 3.53, 1.53, 1.21,
        2.08, 2.37, 1.27, 3.15, 2.71, 2.00,
    ]


def test_table3_confirmation_fixes_p02_training_semantics() -> None:
    text = _text()
    for setting in (
        '"BASELINE_EPOCHS=200"',
        '"RIEI_PAPER_EVAL_LAST_N=5"',
        '"RIEI_OPTIMIZER=sgd"',
        '"RIEI_SGD_MOMENTUM=0"',
        '"RIEI_CE_REDUCTION=mean"',
        '"RIEI_MI_REDUCTION=mean"',
        '"RIEI_IE_REDUCTION=mean"',
        '"RIEI_WISIG_RMS_NORMALIZE=0"',
        '"RIEI_LAMBDA_FEATURE_NORM=0"',
    ):
        assert setting in text
    assert "DRIFT_" not in text
    assert "target-oracle" not in text


def test_table3_confirmation_uses_eight_capacity_gated_queues() -> None:
    text = _text()
    assert 'if [[ "${#GPU_IDS[@]}" -ne 8 ]]' in text
    assert "planned_peak=1" in text
    assert "total > MAX_TRAIN_PER_GPU" in text
    assert 'queues/gpu_${gpu}.sh' in text
    assert 'nohup bash "${queue_file}"' in text
    assert "unique run/log root already exists" in text
