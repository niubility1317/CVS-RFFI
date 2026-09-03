from __future__ import annotations

import subprocess
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import train  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


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


def _formal_train_argv(row: str) -> list[str]:
    text = LAUNCHER.read_text(encoding="utf-8")
    common_block = text.split("COMMON_ARGS=(", 1)[1].split("\n)", 1)[0]
    replacements = {
        "${WISIG_PKL}": "placeholder_manysig.pkl",
        "${SEED}": "392005",
        "${SOURCE_DAYS}": "1,2,3",
        "${SOURCE_RXS}": "1,3,4,6,8",
        "${TARGET_DAYS}": "0,1,2,3",
        "${TARGET_RXS}": "0,2,5,7,9,10,11",
        "${C0_CHECKPOINT}": "placeholder_c0.pth",
    }
    common = [replacements.get(token, token) for token in shlex.split(common_block)]
    return [
        *common,
        "--phase1_method", "adv3b02_fcr",
        "--use_fcr",
        "--fcr_ablation_row", row,
        "--run_name", f"formal_{row}",
        "--final_save_path", f"out/{row}/final.pth",
        "--log_dir", f"out/{row}/logs",
        "--fcr_diagnostics_path", f"out/{row}/fcr_diagnostics.json",
        "--fcr_predictions_path", f"out/{row}/fcr_predictions.json",
        "--fcr_config_dry_run",
    ]


def _argv_value(argv: list[str], flag: str, default: str) -> str:
    if flag not in argv:
        return default
    return argv[argv.index(flag) + 1]


def test_formal_launcher_reconstructs_c0_identity_before_fcr_v2_initialization(
    tmp_path: Path,
) -> None:
    argv = _formal_train_argv("M6")
    mature = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        branch_ablation="no_dac",
        domain_branch_ablation="no_stats",
        fast_infer_when_no_aux=False,
        use_fcr=False,
    )
    checkpoint = tmp_path / "c0.pth"
    torch.save(
        {
            "model": mature.state_dict(),
            "epoch": 200,
            "candidate_id": "ADV3B02_CORE90_SOFT_E200",
            "args": {"seed": 392005},
        },
        checkpoint,
    )
    candidate = build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        branch_ablation=_argv_value(argv, "--branch_ablation", "none"),
        domain_branch_ablation=_argv_value(argv, "--domain_branch_ablation", "same"),
        fast_infer_when_no_aux=False,
        use_fcr=True,
        fcr_version="v2",
    )
    new_v2_initial = {
        name: value.detach().clone()
        for name, value in candidate.state_dict().items()
        if name.startswith(("fcr.", "fcr_identity_projection."))
    }

    report = train.load_init_checkpoint_weights(
        candidate,
        str(checkpoint),
        torch.device("cpu"),
        expected_seed=392005,
        expected_epoch=200,
        expected_candidate_id="ADV3B02_CORE90_SOFT_E200",
        require_mature_identity_complete=True,
    )

    assert report["mature_identity_complete"] is True
    for name, value in mature.state_dict().items():
        if name.startswith("id_backbone."):
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0.0, atol=0.0)
    for name, value in new_v2_initial.items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0.0, atol=0.0)
    assert candidate.fcr_identity_head_matches_legacy()


def test_launcher_c1_and_m6_formal_argv_reach_real_train_parser() -> None:
    for row in ("C1", "M6"):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "code" / "train.py"), *_formal_train_argv(row)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        assert f'"fcr_matrix_row": "{row}"' in result.stdout

    text = LAUNCHER.read_text(encoding="utf-8")
    probe = text.split("probe_training_contract()", 1)[1].split("\n}", 1)[0]
    assert "build_train_command" in probe
    assert "C1 M6" in probe
