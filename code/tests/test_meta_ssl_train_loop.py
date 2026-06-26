import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_wisig(path: Path, samples_per_combo=8, length=256):
    data = []
    for tx in range(3):
        tx_rows = []
        for rx in range(3):
            rx_rows = []
            for day in range(4):
                arr = np.zeros((samples_per_combo, length, 2), dtype=np.float32)
                arr[:, :, 0] = tx + rx * 0.1 + day * 0.01
                arr[:, :, 1] = np.linspace(-0.2, 0.2, length, dtype=np.float32)
                rx_rows.append([arr])
            tx_rows.append(rx_rows)
        data.append(tx_rows)
    payload = {
        "data": data,
        "tx_list": [f"tx{i}" for i in range(3)],
        "rx_list": [f"rx{i}" for i in range(3)],
        "capture_date_list": [f"day{i}" for i in range(4)],
        "equalized_list": [1],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_meta_ssl_training_loop_writes_weighted_losses_and_epoch_logs(tmp_path):
    wisig_pkl = tmp_path / "fake_wisig.pkl"
    log_dir = tmp_path / "logs"
    run_dir = tmp_path / "runs"
    _write_fake_wisig(wisig_pkl)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "code")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "train.py"),
        "--dataset",
        "wisig",
        "--wisig_pkl",
        str(wisig_pkl),
        "--wisig_train_days",
        "0,1",
        "--wisig_test_days",
        "2,3",
        "--wisig_train_rxs",
        "0,1",
        "--wisig_test_rxs",
        "2",
        "--wisig_max_day123_per_combo",
        "8",
        "--wisig_max_test_per_combo",
        "2",
        "--wisig_out_len",
        "128",
        "--num_classes",
        "3",
        "--model_size",
        "S",
        "--batch_size",
        "4",
        "--eval_batch_size",
        "8",
        "--train_steps_per_epoch",
        "1",
        "--eval_max_batches",
        "1",
        "--test_eval_policy",
        "interval_final",
        "--test_eval_interval",
        "1",
        "--no_eval_sat_channel",
        "--use_meta_ssl_cvs",
        "--use_meta_rxday_episodes",
        "--lambda_ssl_tx",
        "0.5",
        "--lambda_ssl_proto",
        "0.1",
        "--lambda_meta_ssl",
        "0.05",
        "--ssl_min_conf",
        "0.0",
        "--ssl_min_margin",
        "0.0",
        "--ssl_max_uncertainty",
        "10.0",
        "--device",
        "cpu",
        "--num_workers",
        "0",
        "--epochs",
        "1",
        "--log_dir",
        str(log_dir),
        "--best_save_path",
        str(run_dir / "best.pth"),
        "--latest_save_path",
        str(run_dir / "latest.pth"),
        "--run_name",
        "meta_ssl_train_loop_smoke",
        "--seed",
        "7",
    ]
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    assert "[META-SSL-CVS-TRAIN]" in proc.stdout
    assert "[LOSS-META-SSL]" in proc.stdout

    logs_path = log_dir / "logs.jsonl"
    metrics_path = log_dir / "metrics.csv"
    assert logs_path.exists()
    assert metrics_path.exists()
    rows = [json.loads(line) for line in logs_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["meta_ssl_enabled"] is True
    assert rows[-1]["train_meta_ssl_coverage"] > 0.0
    assert rows[-1]["train_meta_ssl_tx_loss"] >= 0.0
    assert rows[-1]["train_meta_ssl_proto_loss"] >= 0.0
    metrics_text = metrics_path.read_text(encoding="utf-8")
    assert "train_meta_ssl_coverage" in metrics_text
