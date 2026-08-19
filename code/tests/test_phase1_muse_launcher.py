from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/{drive}/{resolved.as_posix().split(':', 1)[1].lstrip('/')}"


def _base_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ROOT": _bash_path(ROOT),
            "RUNS_ROOT": _bash_path(tmp_path / "runs" / "muse_task7"),
            "GPU": "3",
            "PYTHON": "/opt/conda/envs/cvs/bin/python",
        }
    )
    return env


def _dry_run(tmp_path: Path, only: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [GIT_BASH.as_posix(), LAUNCHER.relative_to(ROOT).as_posix(), "--dry-run", f"--only={only}"],
        cwd=ROOT,
        env=_base_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )


def _prepare_fake_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    fake_root = tmp_path / "fake_project"
    trainer = fake_root / "code/SSDG/train_ssdg.py"
    evaluator = fake_root / "code/scripts/eval_ssdg_sat_per_rx.py"
    trainer.parent.mkdir(parents=True)
    evaluator.parent.mkdir(parents=True)
    call_log = tmp_path / "calls.log"
    trainer.write_text(
        """from __future__ import annotations
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--muse_external_final_eval", default="false")
args, _ = parser.parse_known_args()
with Path(os.environ["FAKE_CALL_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write("train\\n")
print("fake trainer invoked", flush=True)
if os.environ.get("FAKE_TRAIN") == "fail":
    raise SystemExit(19)
Path(args.output_dir, "final_ssdg.pth").write_bytes(b"fake-checkpoint")
if str(args.muse_external_final_eval).lower() not in {"1", "true", "yes"}:
    with Path(os.environ["FAKE_CALL_LOG"]).open("a", encoding="utf-8") as handle:
        handle.write("internal_eval\\n")
    Path(args.output_dir, "frozen_phase1_heldout_eval.json").write_text(
        "{}", encoding="utf-8"
    )
""",
        encoding="utf-8",
        newline="\n",
    )
    evaluator.write_text(
        """from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--output_json", required=True)
parser.add_argument("--scenarios", required=True)
parser.add_argument("--strict_reconstruction", action="store_true")
args, _ = parser.parse_known_args()
scenarios = [item for item in args.scenarios.split(",") if item]
with Path(os.environ["FAKE_CALL_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write("eval:" + ",".join(scenarios) + "\\n")
print("fake joint evaluator invoked", flush=True)
mode = os.environ.get("FAKE_EVAL", "success")
if mode == "exit":
    raise SystemExit(23)
sat_correct = {"leo_clear_weak": (7, 8), "leo_low_elev_weak": (6, 7), "leo_rain_weak": (5, 6)}
rows = []
for scenario in scenarios:
    if mode == "missing_" + scenario:
        continue
    for rx_idx in (0, 1):
        row = {
            "name": f"target_rx_{rx_idx}",
            "rx_idx": rx_idx,
            "rx_label": f"rx-{rx_idx}",
            "days_label": "2,3",
            "scenario": scenario,
            "sat_acc": 10.0 * sat_correct[scenario][rx_idx],
            "sat_correct": sat_correct[scenario][rx_idx],
            "sat_total": 10,
        }
        if mode != "missing_clean":
            row.update({"clean_acc": 80.0 + 10.0 * rx_idx, "clean_correct": 8 + rx_idx, "clean_total": 10})
        rows.append(row)
aggregates = []
for scenario in scenarios:
    selected = [row for row in rows if row["scenario"] == scenario]
    if selected:
        correct = sum(row["sat_correct"] for row in selected)
        total = sum(row["sat_total"] for row in selected)
        aggregates.append({"scenario": scenario, "tx_acc": 100.0 * correct / total, "tx_correct": correct, "tx_total": total})
payload = {
    "schema": "ssdg_sat_per_rx_eval_v1",
    "checkpoint": str(Path(args.ckpt).resolve()),
    "checkpoint_epoch": 200,
    "run_name": "fake",
    "reconstruction": "fixture",
    "reconstruction_audit": {
        "strict_requested": bool(args.strict_reconstruction),
        "checkpoint_load_strict": bool(args.strict_reconstruction),
        "fallback_used": mode == "fallback_metadata",
        "missing_keys": 0,
        "unexpected_keys": 0,
        "shape_mismatches": 0,
    },
    "eval_on": "unseen_rx",
    "group_loader": "test_unseen_day_unseen_rx",
    "group_key": "rx_i",
    "selected_names": ["test_unseen_day_unseen_rx"],
    "scenarios": scenarios,
    "max_batches": -1,
    "sat_seed": 392002,
    "split": {"fixture": True},
    "rows": rows,
    "aggregates": aggregates,
}
Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
""",
        encoding="utf-8",
        newline="\n",
    )
    return fake_root, call_log, tmp_path / "runs" / "muse_task7"


def _fake_run(
    tmp_path: Path,
    *,
    fake_train: str = "success",
    fake_eval: str = "success",
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    fake_root, call_log, runs_root = _prepare_fake_project(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "ROOT": _bash_path(fake_root),
            "RUNS_ROOT": _bash_path(runs_root),
            "GPU": "0",
            "PYTHON": sys.executable,
            "CONTROL_PYTHON": sys.executable,
            "FAKE_CALL_LOG": call_log.as_posix(),
            "FAKE_TRAIN": fake_train,
            "FAKE_EVAL": fake_eval,
        }
    )
    result = subprocess.run(
        [GIT_BASH.as_posix(), LAUNCHER.relative_to(ROOT).as_posix(), "--only=M0"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, runs_root / "M0", call_log


def test_launcher_freezes_protocol_paic_base_and_all_required_evaluations():
    text = LAUNCHER.read_text(encoding="utf-8")
    for token in (
        "--labeled_ratio 0.07",
        "--unlabeled_ratio 0.63",
        "--source_cal_ratio 0.15",
        "--source_select_ratio 0.15",
        "--checkpoint_selection final_only",
        "--epochs 200",
        "--paic_guard_enabled true",
        "--paic_guard_sat_ce_delta 0.12",
        "--paic_guard_grad_delta 3.0",
        "--paic_guard_reliable_drop 0.01",
        "--paic_guard_cooldown_epochs 1",
        "--paic_guard_sat_scale 0.75",
    ):
        assert token in text
    for scenario in SCENARIOS:
        assert scenario in text
    assert "ARTIFACTS_COMPLETE" in text
    assert "EVAL_FAILED_" in text


def test_m3_dry_run_prints_one_training_and_one_joint_real_evaluation_without_outputs(tmp_path):
    result = _dry_run(tmp_path, "M3")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[MUSE-TRAIN-CMD]") == 1
    assert result.stdout.count("[MUSE-EVAL-CMD]") == 1
    assert result.stdout.count("[MUSE-EVAL-OUTPUT]") == 4
    assert "--muse_level M3" in result.stdout
    assert "--base_candidate ADV3B02_CORE90_SOFT_E200" in result.stdout
    assert "--muse_external_final_eval true" in result.stdout
    assert "--strict_reconstruction" in result.stdout
    assert "eval_ssdg_sat_per_rx.py" in result.stdout
    assert "--scenarios leo_clear_weak\\,leo_low_elev_weak\\,leo_rain_weak" in result.stdout
    for scenario in SCENARIOS:
        assert f"scenario={scenario}" in result.stdout
        assert f"metrics_{scenario}.json" in result.stdout
        assert f"eval_{scenario}.log" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_dry_run_maps_levels_and_shares_paic_base_across_all_four_arms(tmp_path):
    result = _dry_run(tmp_path, "M0,M1,M2,M3")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[MUSE-TRAIN-CMD]") == 4
    assert result.stdout.count("[MUSE-EVAL-CMD]") == 4
    for level in ("M0", "M1", "M2", "M3"):
        assert f"candidate={level}" in result.stdout
        assert f"--muse_level {level}" in result.stdout
    for option in (
        "--paic_guard_enabled true",
        "--paic_guard_sat_ce_delta 0.12",
        "--paic_guard_grad_delta 3.0",
        "--paic_guard_reliable_drop 0.01",
        "--paic_guard_cooldown_epochs 1",
        "--paic_guard_sat_scale 0.75",
    ):
        assert result.stdout.count(option) == 4
    assert "candidate=M0 capabilities=ADV3B02_CONTROL" in result.stdout
    assert "candidate=M1 capabilities=BASE" in result.stdout
    assert "candidate=M2 capabilities=BASE_FUSION_HML" in result.stdout
    assert "candidate=M3 capabilities=BASE_FUSION_HML_SATELLITE_CROSSRX_PROTO" in result.stdout


def test_only_rejects_unknown_candidate_without_creating_outputs(tmp_path):
    result = _dry_run(tmp_path, "M4")

    assert result.returncode == 2
    assert "unknown candidate" in result.stderr.lower()
    assert not (tmp_path / "runs").exists()


def test_existing_candidate_root_is_rejected_before_fake_training(tmp_path):
    fake_root, call_log, runs_root = _prepare_fake_project(tmp_path)
    candidate_root = runs_root / "M0"
    candidate_root.mkdir(parents=True)
    sentinel = candidate_root / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "ROOT": _bash_path(fake_root),
            "RUNS_ROOT": _bash_path(runs_root),
            "PYTHON": sys.executable,
            "CONTROL_PYTHON": sys.executable,
            "FAKE_CALL_LOG": call_log.as_posix(),
        }
    )
    result = subprocess.run(
        [GIT_BASH.as_posix(), LAUNCHER.relative_to(ROOT).as_posix(), "--only=M0"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 3
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not call_log.exists()


def test_fake_training_failure_is_preserved_and_evaluation_never_runs(tmp_path):
    result, candidate_root, call_log = _fake_run(tmp_path, fake_train="fail")

    assert result.returncode == 4
    assert (candidate_root / "status.txt").read_text(encoding="utf-8").strip() == "TRAIN_FAILED"
    assert (candidate_root / "config.json").is_file()
    assert (candidate_root / "train.log").is_file()
    assert call_log.read_text(encoding="utf-8").splitlines() == ["train"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_missing_joint_metric_records_the_specific_evaluation_failure(tmp_path, scenario):
    result, candidate_root, call_log = _fake_run(tmp_path, fake_eval=f"missing_{scenario}")

    assert result.returncode != 0
    assert (candidate_root / "status.txt").read_text(encoding="utf-8").strip() == f"EVAL_FAILED_{scenario.upper()}"
    assert (candidate_root / "final_ssdg.pth").is_file()
    assert "eval:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in call_log.read_text(encoding="utf-8")
    assert "ARTIFACTS_COMPLETE" not in (candidate_root / "status.txt").read_text(encoding="utf-8")


def test_fake_joint_evaluator_runs_once_and_writes_four_semantic_metrics_before_complete(tmp_path):
    result, candidate_root, call_log = _fake_run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "train",
        "eval:leo_clear_weak,leo_low_elev_weak,leo_rain_weak",
    ]
    assert not (candidate_root / "frozen_phase1_heldout_eval.json").exists()
    expected_acc = {
        "clean": 85.0,
        "leo_clear_weak": 75.0,
        "leo_low_elev_weak": 65.0,
        "leo_rain_weak": 55.0,
    }
    for scenario in SCENARIOS:
        metrics_path = candidate_root / f"metrics_{scenario}.json"
        log_path = candidate_root / f"eval_{scenario}.log"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert payload["scenario"] == scenario
        assert payload["aggregate"]["scenario"] == scenario
        assert payload["aggregate"]["tx_acc"] == expected_acc[scenario]
        assert {row["scenario"] for row in payload["rows"]} == {scenario}
        assert log_path.stat().st_size > 0
        assert f"scenario={scenario}" in log_path.read_text(encoding="utf-8")
    assert (candidate_root / "status.txt").read_text(encoding="utf-8").strip() == "ARTIFACTS_COMPLETE"


def test_launcher_rejects_non_strict_or_fallback_reconstruction_metadata(tmp_path):
    result, candidate_root, _call_log = _fake_run(
        tmp_path,
        fake_eval="fallback_metadata",
    )

    assert result.returncode != 0
    assert (candidate_root / "status.txt").read_text(encoding="utf-8").strip() == "EVAL_FAILED_JOINT"
    assert not (candidate_root / "metrics_clean.json").exists()
    assert "ARTIFACTS_COMPLETE" not in (candidate_root / "status.txt").read_text(encoding="utf-8")
