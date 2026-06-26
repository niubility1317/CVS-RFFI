import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_fake_wisig(path: Path, samples_per_combo=8):
    data = []
    for tx in range(3):
        tx_rows = []
        for rx in range(3):
            rx_rows = []
            for day in range(4):
                arr = np.zeros((samples_per_combo, 16, 2), dtype=np.float32)
                arr[:, :, 0] = tx + rx * 0.1 + day * 0.01
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


def test_train_meta_ssl_protocol_check_is_default_off_and_exits_before_training(tmp_path):
    wisig_pkl = tmp_path / "fake_wisig.pkl"
    report = tmp_path / "meta_ssl_report.json"
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
        "4",
        "--no_eval_sat_channel",
        "--use_meta_ssl_cvs",
        "--meta_ssl_protocol_check_only",
        "--meta_ssl_protocol_report",
        str(report),
        "--device",
        "cpu",
        "--num_workers",
        "0",
        "--epochs",
        "1",
    ]
    proc = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    assert "[META-SSL-CVS]" in proc.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS"
    assert payload["split"]["overlap_count"] == 0
    assert payload["split"]["tx_label_policy"]["unlabeled_source"] == "masked_y_minus_1_true_tx_in_meta_only"
