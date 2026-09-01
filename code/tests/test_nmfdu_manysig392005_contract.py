from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from SSDG import train_ssdg as train_ssdg_module
from SSDG.train_ssdg import (
    build_arg_parser,
    resolve_phase1_source_target_scope,
    split_tx_rx_day_1_7_2,
)


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh"
)
CONFIG = ROOT / "code/configs/phase1_adv3b02_nmfdu_gate8_manysig392005.json"


@dataclass(slots=True)
class _Record:
    tx_i: int
    rx_i: int
    day_i: int
    eq_i: int
    sig_i: int


class _IndexOnlyDataset:
    def __init__(self) -> None:
        self.index = [
            _Record(tx_i=tx, rx_i=rx, day_i=day, eq_i=1, sig_i=sig)
            for tx in range(6)
            for rx in (1, 3, 4, 6, 8)
            for day in (1, 2, 3)
            for sig in range(1000)
        ]


def test_manysig392005_scope_preserves_shared_days_only_for_disjoint_receivers() -> None:
    source_days, target_days, source_rxs, target_rxs = (
        resolve_phase1_source_target_scope(
            [1, 2, 3],
            [0, 1, 2, 3],
            [1, 3, 4, 6, 8],
            [0, 2, 5, 7, 9, 10, 11],
            allow_day_overlap_by_disjoint_rx=True,
        )
    )
    assert source_days == [1, 2, 3]
    assert target_days == [0, 1, 2, 3]
    assert source_rxs == [1, 3, 4, 6, 8]
    assert target_rxs == [0, 2, 5, 7, 9, 10, 11]


def test_day_overlap_mode_fails_closed_when_receivers_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint receiver sets"):
        resolve_phase1_source_target_scope(
            [1, 2, 3],
            [0, 1, 2, 3],
            [1, 3, 4, 6, 8],
            [0, 2, 3, 5, 7, 9, 10, 11],
            allow_day_overlap_by_disjoint_rx=True,
        )


def test_exact_target_builder_uses_only_declared_cartesian_scope(monkeypatch) -> None:
    calls = []

    class _FakeTarget:
        def __init__(self, dataset, **kwargs) -> None:
            calls.append((dataset, kwargs))

        def __len__(self) -> int:
            return 168000

    monkeypatch.setattr(train_ssdg_module, "WiSigCompactDataset", _FakeTarget)
    dataset = {
        "capture_date_list": [
            "2021_03_01",
            "2021_03_08",
            "2021_03_15",
            "2021_03_23",
        ],
        "rx_list": [str(index) for index in range(12)],
    }
    tests, metadata, info = train_ssdg_module._build_exact_target_test_scope(
        dataset,
        equalized=1,
        out_len=256,
        domain="rx_day",
        target_days=[0, 1, 2, 3],
        target_receivers=[0, 2, 5, 7, 9, 10, 11],
        max_samples_per_combo=None,
        seed=392005,
    )
    assert tuple(tests) == ("test_unseen_day_unseen_rx",)
    assert metadata["test_unseen_day_unseen_rx"]["size"] == 168000
    assert info["test_size"] == 168000
    assert calls[0][1]["day_keep"] == [0, 1, 2, 3]
    assert calls[0][1]["rx_keep"] == [0, 2, 5, 7, 9, 10, 11]
    assert calls[0][1]["max_samples_per_combo"] is None
    assert calls[0][1]["seed"] == 392005


def test_manysig_source_pool_and_single_validation_counts_are_exact() -> None:
    labeled, unlabeled, validation = split_tx_rx_day_1_7_2(
        _IndexOnlyDataset(),
        labeled_ratio=0.07,
        unlabeled_ratio=0.63,
        source_val_ratio=0.30,
    )
    assert len(labeled) == 6300
    assert len(unlabeled) == 56700
    assert len(validation) == 27000
    assert len(labeled) + len(unlabeled) + len(validation) == 90000
    assert not (set(labeled) & set(unlabeled))
    assert not (set(labeled) & set(validation))
    assert not (set(unlabeled) & set(validation))


def test_cli_exposes_fixed_split_seed_and_single_validation_protocol() -> None:
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "unused",
            "--wisig_split_seed",
            "392005",
            "--allow_source_target_day_overlap_by_disjoint_rx",
            "true",
            "--phase1_source_role_protocol",
            "l_s_u_s_v",
        ]
    )
    assert args.wisig_split_seed == 392005
    assert args.allow_source_target_day_overlap_by_disjoint_rx is True
    assert args.phase1_source_role_protocol == "l_s_u_s_v"


def test_gate8_matrix_is_complete_and_causally_ordered() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = payload["matrix"]
    assert [row["row"] for row in rows] == [f"E{index}" for index in range(1, 9)]
    assert [row["nmfdu_ablation_mode"] for row in rows] == [
        "equal",
        "i_only",
        "i_d",
        "i_d_s",
        "physical_fixed",
        "physical_full",
        "full_no_null",
        "full",
    ]
    assert {row["physical_gate_variant"] for row in rows} == {"nmfdu_v1"}
    assert payload["dataset"]["source"]["roles"] == {
        "L_s": 6300,
        "U_s": 56700,
        "V": 27000,
    }
    assert payload["dataset"]["target_test"]["samples_per_scenario"] == 168000
    assert payload["phase2_access"] is False
    assert payload["query_access_before_independent_scoring"] is False


def test_gate8_launcher_freezes_exact_manysig_scope_and_refuses_overwrite() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    required = (
        "--wisig_equalized 1",
        "--wisig_train_rxs 1,3,4,6,8",
        "--wisig_test_rxs 0,2,5,7,9,10,11",
        "--wisig_train_days 1,2,3",
        "--wisig_test_days 0,1,2,3",
        '--wisig_split_seed "${SPLIT_SEED}"',
        "--allow_source_target_day_overlap_by_disjoint_rx true",
        "--wisig_max_day123_per_combo 0",
        "--wisig_max_test_per_combo 0",
        "--phase1_source_role_protocol l_s_u_s_v",
        "--labeled_ratio 0.07",
        "--unlabeled_ratio 0.63",
        "--source_val_ratio 0.30",
        "--epochs 200",
        "--best_metric source_val_sat_hmean",
        "--enable_joint_safe_guard false",
        "--checkpoint_selection final_only",
    )
    for token in required:
        assert token in text
    assert "ROWS=(E1 E2 E3 E4 E5 E6 E7 E8)" in text
    assert "MODES=(equal i_only i_d i_d_s physical_fixed physical_full full_no_null full)" in text
    assert "refusing existing run/log root" in text
    assert "refusing to overwrite row=" in text
