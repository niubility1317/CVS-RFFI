from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh"
LAUNCHER = PROJECT_ROOT / SCRIPT
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
EXPECTED_C0_CHECKPOINT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/"
    "ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth"
)
EXPECTED_ROWS = [
    "C1",
    "C2",
    "C3",
    "S0",
    "S1",
    "S2",
    "S3",
    "S4",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
]


def _parse_keyvals(line: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        payload[key] = value
    return payload


def _dry_run() -> SimpleNamespace:
    result = subprocess.run(
        [str(GIT_BASH), "--noprofile", "--norc", SCRIPT, "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    rows = []
    c0 = None
    prepare = None
    predicts = []
    scores = []
    for line in result.stdout.splitlines():
        if line.startswith("[FCRV2-C0]"):
            c0 = _parse_keyvals(line)
        elif line.startswith("[FCRV2-ROW]"):
            rows.append(_parse_keyvals(line))
        elif line.startswith("[FCRV2-PREPARE]"):
            prepare = _parse_keyvals(line)
        elif line.startswith("[FCRV2-PREDICT]"):
            predicts.append(_parse_keyvals(line))
        elif line.startswith("[FCRV2-SCORE]"):
            scores.append(_parse_keyvals(line))
    return SimpleNamespace(
        stdout=result.stdout,
        c0=c0,
        rows=[row["row"] for row in rows],
        rowspecs=rows,
        prepare=prepare,
        predicts=predicts,
        scores=scores,
    )


def test_complete_matrix_has_all_rows_once_and_final_only() -> None:
    dry = _dry_run()

    assert dry.c0 is not None
    assert dry.c0["checkpoint"] == EXPECTED_C0_CHECKPOINT
    assert dry.c0["train"] == "0"
    assert dry.rows == EXPECTED_ROWS
    assert len({row["output"] for row in dry.rowspecs}) == len(EXPECTED_ROWS)
    for row in dry.rowspecs:
        assert row["epochs"] == "200"
        assert row["checkpoint_selection"] == "final_only"
        assert row["init_checkpoint"] == EXPECTED_C0_CHECKPOINT
        assert row["final_checkpoint"].endswith("/final.pth")
        assert row["diagnostics"].endswith("/fcr_diagnostics.json")


def test_complete_matrix_uses_two_waves_and_shared_truth_last_prepare() -> None:
    dry = _dry_run()

    wave1 = [row for row in dry.rowspecs if row["wave"] == "1"]
    wave2 = [row for row in dry.rowspecs if row["wave"] == "2"]
    assert [row["row"] for row in wave1] == EXPECTED_ROWS[:8]
    assert [row["gpu"] for row in wave1] == [str(gpu) for gpu in range(8)]
    assert [row["row"] for row in wave2] == EXPECTED_ROWS[8:]
    assert [row["gpu"] for row in wave2] == [str(gpu) for gpu in range(6)]

    assert dry.prepare is not None
    assert dry.prepare["rows_ready"] == "14"
    assert dry.prepare["truth_sidecar"].endswith("/target_truth/truth_sidecar.json")
    assert dry.prepare["input_package"].endswith("/target_inputs")

    assert [item["row"] for item in dry.predicts] == ["C0", *EXPECTED_ROWS]
    assert [item["row"] for item in dry.scores] == ["C0", *EXPECTED_ROWS]
    assert dry.predicts[0]["checkpoint"] == EXPECTED_C0_CHECKPOINT
    assert all(item["checkpoint"].endswith("/final.pth") for item in dry.predicts[1:])
    assert all(item["predictions"].endswith("/predictions.json") for item in dry.scores)


def test_launcher_text_orders_target_eval_after_training_waits_and_refuses_overwrite() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '[[ "${DRY_RUN}" != "1" && -e "${OUTPUT_ROOT}" ]]' in text
    assert "wait_training_rows" in text
    assert text.index("wait_training_rows") < text.index("--mode prepare")
    assert text.index("--mode prepare") < text.index("--mode predict")
    assert text.index("--mode predict") < text.index("--predictions")
    assert "C0_SCORE_JSON" in text
    assert "rows_ready=14" in text
