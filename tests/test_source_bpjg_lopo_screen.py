from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS
from paper_reproduction.scripts.screen_cvs_p4_bpjg_lopo_source import (
    build_view_major_support,
    select_role_symmetric_source_split,
)


def _toy_source(class_count: int = 3, per_class: int = 25):
    labels = np.repeat(np.arange(class_count, dtype=np.int64), per_class)
    sample_ids = np.asarray(
        [f"source|tx-{label}|rx|day|eq|sig-{index}" for index, label in enumerate(labels)]
    )
    arrays = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        rows = np.zeros((len(labels), 2, 8), dtype=np.float32)
        rows[:, 0, :] = scenario_index + np.arange(len(labels))[:, None]
        arrays[scenario] = {
            "sample_ids": sample_ids.copy(),
            "raw_labels": labels.copy(),
            "leo_weak_iq": rows,
        }
    return labels, sample_ids, arrays


def test_nested_source_split_has_fixed_query_outside_k20_pool() -> None:
    labels, sample_ids, _ = _toy_source()
    candidates = np.arange(len(labels), dtype=np.int64)
    rows = {
        k_shot: select_role_symmetric_source_split(
            labels,
            sample_ids,
            candidates,
            class_count=3,
            k_shot=k_shot,
            support_pool_max_k=20,
            seed=71,
        )
        for k_shot in (1, 5, 10, 20)
    }
    k1, query1, split1 = rows[1]
    k5, query5, split5 = rows[5]
    k10, query10, split10 = rows[10]
    k20, query20, split20 = rows[20]
    assert np.array_equal(query1, query5)
    assert np.array_equal(query1, query10)
    assert np.array_equal(query1, query20)
    assert set(k1.tolist()).issubset(set(k5.tolist()))
    assert set(k5.tolist()).issubset(set(k10.tolist()))
    assert set(k10.tolist()).issubset(set(k20.tolist()))
    assert {
        split1["query_ids_sha256"],
        split5["query_ids_sha256"],
        split10["query_ids_sha256"],
        split20["query_ids_sha256"],
    } == {split1["query_ids_sha256"]}
    assert split10["physical_support_count"] == 30
    assert split10["physical_support_pool_count"] == 60
    assert split10["physical_query_count"] == 15
    assert split10["support_query_overlap_count"] == 0


def test_view_major_support_preserves_physical_order() -> None:
    labels, sample_ids, arrays = _toy_source()
    support = np.asarray([0, 1, 25, 26, 50, 51], dtype=np.int64)
    rows, repeated_labels, physical_ids, row_ids = build_view_major_support(
        arrays, support
    )
    assert rows.shape == (18, 2, 8)
    assert repeated_labels.tolist() == labels[support].tolist() * 3
    assert physical_ids == sample_ids[support].tolist()
    assert row_ids == physical_ids * 3


def test_view_major_support_rejects_cross_scenario_id_drift() -> None:
    _, _, arrays = _toy_source()
    arrays[FORMAL_LEO_WEAK_SCENARIOS[1]]["sample_ids"][0] = "drift"
    with pytest.raises(ValueError, match="alignment drift"):
        build_view_major_support(arrays, np.asarray([0, 25, 50], dtype=np.int64))


def test_v20_launcher_binds_exact_lopo_trainer_dependency() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "paper_reproduction"
        / "scripts"
        / "launch_cvs_p4_bpjg_lopo_source_v20.sh"
    ).read_text(encoding="utf-8")
    assert "qknnv42_p4_bpjg_lopo_source_k10_20260715_v20" in launcher
    assert (
        "TRAINER_SHA256="
        "f985f5e5f718f1c60ab75e6b41684bf4962edce454c1612a7d2f7c0e14406f7e"
        in launcher
    )
    assert 'sha256sum "$TRAINER"' in launcher


def test_source_screen_uses_numpy2_torch21_compat_bridge() -> None:
    root = Path(__file__).resolve().parents[1]
    screen = (
        root
        / "paper_reproduction"
        / "scripts"
        / "screen_cvs_p4_bpjg_lopo_source.py"
    ).read_text(encoding="utf-8")
    assert "torch.from_numpy" not in screen
    assert screen.count("numpy_to_tensor_compat(") >= 4


def test_v21_launcher_locks_screen_and_supports_staged_arms() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "paper_reproduction"
        / "scripts"
        / "launch_cvs_p4_bpjg_lopo_source_v21.sh"
    ).read_text(encoding="utf-8")
    assert "qknnv42_p4_bpjg_lopo_source_k10_20260715_v21" in launcher
    assert (
        "SCREEN_SHA256="
        "3c4ad69ee148831f0d401a9f5fb73287400bc3a8cc994c6c85dd166c058b194f"
        in launcher
    )
    assert 'ARM_INDEXES_RAW="${ARM_INDEXES:-0 1 2 3}"' in launcher
    assert 'refusing to overwrite existing v21 arm' in launcher


def test_v22_launcher_confirms_locked_joint_gate_across_k_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "paper_reproduction"
        / "scripts"
        / "launch_cvs_p4_bpjg_lopo_source_v22.sh"
    ).read_text(encoding="utf-8")
    screen = (
        root
        / "paper_reproduction"
        / "scripts"
        / "screen_cvs_p4_bpjg_lopo_source.py"
    ).read_text(encoding="utf-8")
    assert "qknnv42_p4_bpjg_lopo_source_kgrid_20260715_v22" in launcher
    assert "labels=(JG020_K1 JG020_K5 JG020_K20)" in launcher
    assert "ks=(1 5 20)" in launcher
    assert "--scope joint_gate" in launcher
    assert "--learning_rate 0.02" in launcher
    assert (
        "SCREEN_SHA256="
        "ff061f84ea279bdee50299c1f2a7da83e7dd6abb9f75410771e7c420b07a25bc"
        in launcher
    )
    assert 'ARM_INDEXES_RAW="${ARM_INDEXES:-0 1 2}"' in launcher
    assert 'refusing to overwrite existing v22 arm' in launcher
    assert 'f"source LEO_weak K{int(args.k_shot)} method screen only"' in screen


def test_v23_launcher_tests_the_minimal_k1_joint_projection_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root
        / "paper_reproduction"
        / "scripts"
        / "launch_cvs_p4_bpjg_lopo_source_v23.sh"
    ).read_text(encoding="utf-8")
    screen = (
        root
        / "paper_reproduction"
        / "scripts"
        / "screen_cvs_p4_bpjg_lopo_source.py"
    ).read_text(encoding="utf-8")
    assert "qknnv42_p4_bpjg_lopo_source_k1_layer_20260715_v23" in launcher
    assert "labels=(JP8_LR005 JP8_LR010 JP8_LR020 JG8_LR005)" in launcher
    assert "scopes=(joint_projection joint_projection joint_projection joint_gate)" in launcher
    assert "lrs=(0.005 0.01 0.02 0.005)" in launcher
    assert "--k_shot 1" in launcher
    assert (
        "SCREEN_SHA256="
        "f19be0b4c3745c4c950161199faa82008b001f4904640fc8a6129e00a8fd1834"
        in launcher
    )
    assert 'refusing to overwrite existing v23 arm' in launcher
    assert '"joint_projection", "joint_gate"' in screen
