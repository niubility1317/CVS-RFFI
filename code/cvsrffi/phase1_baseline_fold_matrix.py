from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


SOURCE_RECEIVERS = (1, 3, 4, 6, 8)
TARGET_RECEIVERS = (0, 2, 5, 7, 9, 10, 11)
FOLDS = (1, 8)
SEED = 392002
SAT_SCENARIOS = "leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
SAT_SCHEDULE = (
    "1@0.30:leo_clear_weak;"
    "41@0.60:leo_low_elev_weak,leo_rain_weak;"
    "91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
)


@dataclass(frozen=True)
class MatrixRow:
    row_id: str
    method: str
    method_version: str
    fold: int
    gpu: int
    train_receivers: tuple[int, ...]
    target_receivers: tuple[int, ...]
    output_dir: str
    log_file: str
    command: tuple[str, ...]


def _common_command(
    *,
    python_bin: str,
    module: str,
    wisig_pkl: Path,
    output_dir: Path,
    fold: int,
) -> list[str]:
    return [
        str(python_bin),
        "-u",
        "-m",
        module,
        "--wisig_pkl",
        str(wisig_pkl),
        "--wisig_protocol",
        "cvs_day_rx",
        "--wisig_equalized",
        "1",
        "--wisig_domain",
        "rx_day",
        "--wisig_out_len",
        "256",
        "--use_source_ssl_split",
        "--wisig_labeled_ratio",
        "0.07",
        "--wisig_unlabeled_ratio",
        "0.63",
        "--wisig_source_val_ratio",
        "0.3",
        "--wisig_train_days",
        "0,1,2",
        "--wisig_test_days",
        "3",
        "--wisig_train_rxs",
        ",".join(map(str, SOURCE_RECEIVERS)),
        "--wisig_source_holdout_rxs",
        str(fold),
        "--wisig_test_rxs",
        ",".join(map(str, TARGET_RECEIVERS)),
        "--wisig_split_strategy",
        "random",
        "--wisig_cap_strategy",
        "random",
        "--wisig_split_seed",
        str(SEED),
        "--seed",
        str(SEED),
        "--use_pseudo_labels",
        "--epochs",
        "200",
        "--use_concat_sat_channel_aug",
        "--concat_sat_ce_only",
        "--concat_sat_start_epoch",
        "80",
        "--lambda_sat_cls",
        "0.68",
        "--lambda_sat_cons",
        "0",
        "--sat_train_scenarios",
        SAT_SCENARIOS,
        "--sat_view_schedule",
        SAT_SCHEDULE,
        "--sat_view_seed",
        str(SEED),
        "--eval_sat_channel",
        "--eval_sat_on",
        "test_seen_day_unseen_rx,test_unseen_day_unseen_rx",
        "--eval_sat_scenarios",
        SAT_SCENARIOS,
        "--no_test_on_val_improve",
        "--test_eval_interval",
        "0",
        "--paper_eval_last_n",
        "0",
        "--final_test_best_by_val",
        "--final_test_target_only",
        "--output_dir",
        str(output_dir),
    ]


def build_rows(
    *,
    run_id: str,
    project_root: Path,
    wisig_pkl: Path,
    run_root: Path,
    log_root: Path,
    python_bin: str,
    gpu_ids: Sequence[int],
) -> list[MatrixRow]:
    if len(gpu_ids) != 4:
        raise ValueError("exactly four GPU ids are required")
    rows: list[MatrixRow] = []
    specs = (("RIEI", 1), ("RIEI", 8), ("DRIFT", 1), ("DRIFT", 8))
    for (method, fold), gpu in zip(specs, gpu_ids):
        train_receivers = tuple(rx for rx in SOURCE_RECEIVERS if rx != fold)
        row_id = f"{method}_FOLD{fold}_S{SEED}"
        output_dir = run_root / row_id
        log_file = log_root / f"{row_id}.log"
        module = "baselines.riei_fd.train" if method == "RIEI" else "baselines.drift.train"
        command = _common_command(
            python_bin=python_bin,
            module=module,
            wisig_pkl=wisig_pkl,
            output_dir=output_dir,
            fold=fold,
        )
        command.extend(["--batch_size", "64", "--eval_batch_size", "256"])
        if method == "RIEI":
            command.extend(
                [
                    "--lr_all",
                    "0.0001",
                    "--lr_fed",
                    "0.0001",
                    "--lambda_mi",
                    "1.2",
                    "--lambda_ie",
                    "1.2",
                    "--ce_reduction",
                    "sum",
                    "--mi_reduction",
                    "sum",
                    "--ie_reduction",
                    "sum",
                    "--lambda_feature_norm",
                    "0.0001",
                ]
            )
        else:
            command.extend(
                [
                    "--lr",
                    "0.0001",
                    "--lambda_grl",
                    "1.0",
                    "--grl_coeff",
                    "1.0",
                    "--lambda_center",
                    "0.01",
                    "--center_mode",
                    "ema",
                    "--center_momentum",
                    "0.95",
                    "--lambda_mse",
                    "0.02",
                    "--no-normalize_features_for_mse",
                    "--mse_reduction",
                    "sum",
                    "--mse_cap",
                    "4000",
                    "--domain_discriminator_layers",
                    "2",
                    "--grl_schedule",
                    "constant",
                ]
            )
        rows.append(
            MatrixRow(
                row_id=row_id,
                method=method,
                method_version=(
                    "RIEI_C06_sum_featnorm1e4"
                    if method == "RIEI"
                    else "DRIFT_N02_raw_cap4000"
                ),
                fold=fold,
                gpu=int(gpu),
                train_receivers=train_receivers,
                target_receivers=TARGET_RECEIVERS,
                output_dir=str(output_dir),
                log_file=str(log_file),
                command=tuple(command),
            )
        )
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_gpu_ids(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("--gpu-ids must contain four distinct GPU ids")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--wisig-pkl", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu-ids", default="4,5,6,7")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    run_root = (args.run_root or project_root / "runs" / args.run_id).resolve()
    log_root = (args.log_root or project_root / "logs" / args.run_id).resolve()
    rows = build_rows(
        run_id=args.run_id,
        project_root=project_root,
        wisig_pkl=args.wisig_pkl,
        run_root=run_root,
        log_root=log_root,
        python_bin=args.python_bin,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
    )
    payload = {
        "run_id": args.run_id,
        "seed": SEED,
        "roles": {"L_s": 0.07, "U_s": 0.63, "V": 0.30},
        "source_receivers": list(SOURCE_RECEIVERS),
        "target_receivers": list(TARGET_RECEIVERS),
        "train_days": [0, 1, 2],
        "target_test_days": [0, 1, 2, 3],
        "checkpoint_selection": "source_V_only",
        "target_feedback": False,
        "rows": [asdict(row) for row in rows],
    }
    if not args.execute:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not project_root.is_dir():
        raise FileNotFoundError(project_root)
    if not args.wisig_pkl.is_file():
        raise FileNotFoundError(args.wisig_pkl)
    if run_root.exists() or log_root.exists():
        raise FileExistsError("run/log root already exists; choose a fresh immutable run id")
    run_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    _write_json(run_root / "matrix_manifest.json", payload)

    launched = []
    for row in rows:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(row.gpu)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root / "code"), str(project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        log_path = Path(row.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_handle:
            proc = subprocess.Popen(
                list(row.command),
                cwd=str(project_root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched.append(
            {
                "row_id": row.row_id,
                "pid": int(proc.pid),
                "gpu": row.gpu,
                "cwd": str(project_root),
                "output_dir": row.output_dir,
                "log_file": row.log_file,
                "command": list(row.command),
            }
        )
    receipt = {"run_id": args.run_id, "status": "RUNNING", "launched": launched}
    _write_json(run_root / "launch_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
