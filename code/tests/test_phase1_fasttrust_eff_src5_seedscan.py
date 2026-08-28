from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from dataset_wisig import make_wisig_trainval_test_by_day_rx
from SSDG import train_ssdg


LAUNCHER = ROOT / "code/scripts/launch_phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828.sh"
MATRIX = ROOT / "configs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828.json"
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")


def _bash_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/{drive}/{resolved.as_posix().split(':', 1)[1].lstrip('/')}"


def _fake_wisig(samples_per_combo: int = 8):
    data = []
    for tx in range(2):
        tx_rows = []
        for rx in range(4):
            rx_rows = []
            for day in range(4):
                arr = np.zeros((samples_per_combo, 16, 2), dtype=np.float32)
                arr[:, :, 0] = tx + rx * 0.1 + day * 0.01
                rx_rows.append([arr])
            tx_rows.append(rx_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": ["tx0", "tx1"],
        "rx_list": ["src0", "src1", "tgt0", "tgt1"],
        "capture_date_list": [f"day{i}" for i in range(4)],
        "equalized_list": [1],
    }


def test_shared_four_days_are_allowed_only_for_receiver_disjoint_domains():
    _train, _val, test, named, _meta, info = make_wisig_trainval_test_by_day_rx(
        _fake_wisig(),
        train_days=[0, 1, 2, 3],
        test_days=[0, 1, 2, 3],
        train_rxs=[0, 1],
        test_rxs=[2, 3],
        train_ratio=0.5,
        allow_shared_days_if_receivers_disjoint=True,
        seed=713101,
    )

    assert info["train_days_idx"] == [0, 1, 2, 3]
    assert info["test_days_idx"] == [0, 1, 2, 3]
    assert info["shared_days_idx"] == [0, 1, 2, 3]
    assert info["train_rxs_idx"] == [0, 1]
    assert info["test_rxs_idx"] == [2, 3]
    assert info["receiver_disjoint"] is True
    assert len(test) == 2 * 2 * 4 * 8
    assert "test_seen_day_unseen_rx" in named
    assert "test_unseen_day_unseen_rx" not in named

    with pytest.raises(ValueError, match="shared train/test days require disjoint receiver sets"):
        make_wisig_trainval_test_by_day_rx(
            _fake_wisig(),
            train_days=[0, 1, 2, 3],
            test_days=[0, 1, 2, 3],
            train_rxs=[0, 1],
            test_rxs=[1, 2],
            train_ratio=0.5,
            allow_shared_days_if_receivers_disjoint=True,
            seed=713101,
        )


def test_matrix_freezes_eight_paired_seeds_and_exactly_two_rows_per_gpu():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert data["schema"] == "cvs.phase1.fasttrust_eff_seedscan.v1"
    assert data["run_id"] == "phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1"
    assert data["epochs"] == 200
    assert data["unlabeled_batch_size"] == 256
    assert data["eval_batch_size"] == 512
    assert data["source_profile"]["profile_id"] == "SRC5_MAXP2"
    assert data["source_profile"]["train_days"] == [0, 1, 2, 3]
    assert data["source_profile"]["test_days"] == [0, 1, 2, 3]
    assert data["source_profile"]["train_receiver_indices"] == [1, 3, 4, 6, 8]
    assert data["source_profile"]["test_receiver_indices"] == [0, 2, 5, 7, 9, 10, 11]

    rows = data["rows"]
    assert len(rows) == 16
    assert Counter(row["gpu"] for row in rows) == Counter({gpu: 2 for gpu in range(8)})
    by_gpu = defaultdict(list)
    for row in rows:
        by_gpu[row["gpu"]].append(row)
        assert row["init"] == "scratch"
    for gpu, pair in by_gpu.items():
        assert {row["variant"] for row in pair} == {"CONTROL", "FASTTRUST_EFF"}
        assert {row["seed"] for row in pair} == {713101 + gpu}


def test_seed_scan_data_context_exposes_only_source_v_select_before_freeze(tmp_path):
    dataset_path = tmp_path / "ManySig.pkl"
    with dataset_path.open("wb") as handle:
        pickle.dump(_fake_wisig(samples_per_combo=100), handle)
    args = train_ssdg.build_arg_parser().parse_args(
        [
            "--output_dir",
            str(tmp_path / "out"),
            "--wisig_pkl",
            str(dataset_path),
            "--wisig_train_days",
            "0,1,2,3",
            "--wisig_train_rxs",
            "0,1",
            "--wisig_test_days",
            "",
            "--wisig_test_rxs",
            "",
            "--phase1_source_only_eval",
            "true",
            "--split_mode",
            "tx_rx_day_1_7_2",
            "--phase1_source_role_protocol",
            "l_s_u_s_v_cal_v_select",
            "--labeled_ratio",
            "0.07",
            "--unlabeled_ratio",
            "0.63",
            "--source_val_ratio",
            "0.30",
            "--source_cal_ratio",
            "0.15",
            "--source_select_ratio",
            "0.15",
            "--num_workers",
            "0",
        ]
    )

    data_ctx = train_ssdg._build_ssdg_wisig_data(args, torch.device("cpu"))

    assert set(data_ctx["named_test_loaders"]) == {"source_v_select"}
    assert data_ctx["split_info"]["target_access"] is False
    assert data_ctx["split_info"]["source_split_receipt"]["target_days"] == []
    assert data_ctx["split_info"]["source_split_receipt"]["target_receivers"] == []
    assert data_ctx["split_info"]["named_test_meta"]["source_v_select"]["rxs_idx"] == [0, 1]


def test_worker_config_records_scratch_without_unused_checkpoint_and_source_profile():
    worker = (
        ROOT / "code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh"
    ).read_text(encoding="utf-8")

    assert 'local effective_base_ckpt=""' in worker
    assert 'effective_base_ckpt="${BASE_CKPT}"' in worker
    assert '"base_checkpoint": "%s"' in worker
    assert '"phase1_source_only_eval": %s' in worker
    assert '"wisig_train_days": "%s"' in worker
    assert '"wisig_train_receivers": "%s"' in worker
    assert '"wisig_test_days": "%s"' in worker
    assert '"wisig_test_receivers": "%s"' in worker


def test_control_batch_initializes_optional_rc4_route_before_muse_branch():
    source = (ROOT / "code/SSDG/train_ssdg.py").read_text(encoding="utf-8")
    loop_start = source.index(
        "        for batch_idx, (labeled_batch, muse_unlabeled_batch) "
        "in enumerate(epoch_pairs, start=1):"
    )
    first_optional_muse_branch = source.index(
        "            if muse_state is not None:", loop_start
    )
    route_default = source.index("            rc4_route = None", loop_start)

    assert loop_start < route_default < first_optional_muse_branch


def test_dry_run_expands_scratch_src5_pair_and_effective_fasttrust_only(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "ROOT": _bash_path(ROOT),
            "CODE_ROOT": _bash_path(ROOT),
            "MATRIX": _bash_path(MATRIX),
            "RUNS_ROOT": _bash_path(tmp_path / "runs"),
            "PYTHON": "/opt/conda/envs/cvs/bin/python",
            "CONTROL_PYTHON": os.sys.executable,
        }
    )
    result = subprocess.run(
        [GIT_BASH.as_posix(), LAUNCHER.relative_to(ROOT).as_posix(), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[SEEDSCAN-ROW]") == 16
    assert result.stdout.count("[MUSE-TRAIN-CMD]") == 16
    assert result.stdout.count("[MUSE-EVAL-CMD]") == 16
    assert result.stdout.count("--from_scratch true") == 16
    assert "--baseline_ckpt" not in result.stdout
    for token in (
        "--wisig_train_days 0,1,2,3",
        "--wisig_train_rxs 1,3,4,6,8",
        "--phase1_source_only_eval true",
        "--eval_on source_v_select",
        "--group_loader source_v_select",
        "--eval_batch_size 512",
        "--muse_unlabeled_batch_size 256",
        "--source_val_heavy_eval_interval 10",
        "--source_val_heavy_eval_final_window 20",
    ):
        assert token in result.stdout
    assert "--wisig_test_rxs 0,2,5,7,9,10,11" not in result.stdout
    for token in (
        "--muse_fasttrust_hard_only_no_fill true",
        "--muse_class_balanced_cap true",
        "--muse_prior_alignment_gamma 0.35",
        "--muse_require_temporal_stability false",
        "--muse_use_prototype_evidence false",
        "--muse_enable_u_prototype_update false",
        "--muse_lambda_cross_receiver 0",
        "--muse_lambda_nuisance 0",
    ):
        assert result.stdout.count(token) == 8
    assert not (tmp_path / "runs").exists()
