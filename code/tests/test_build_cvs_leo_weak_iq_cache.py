from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    PHASE2_SAMPLE_VIEW_POLICY,
    load_verified_leo_weak_cache_set,
)
from scripts import build_cvs_leo_weak_iq_cache as builder


class _TinyWiSig(Dataset):
    def __init__(self, role: str) -> None:
        self.role = str(role)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        x = torch.full((2, 8), float(index + 1), dtype=torch.float32)
        y = torch.tensor(index, dtype=torch.long)
        domain = torch.tensor(0, dtype=torch.long)
        meta = {
            "tx": f"tx{index}",
            "rx": "rx0",
            "day": "d0",
            "equalized": "1",
            "sig_i": str(index),
        }
        return x, y, domain, meta


def _spec(tmp_path: Path) -> dict:
    return {
        "schema": builder.BUILD_SPEC_SCHEMA,
        "cache_set_id": "tiny",
        "cache_scope": "source_train",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "star_ground_channel_impl": "simplified_leo_residual",
        "role_specs": [
            {
                "role": "source",
                "pkl": "fake.pkl",
                "tx_ids": "0,1",
                "rxs": "0",
                "max_samples_per_tx": 2,
            }
        ],
        "satellite_seed_by_scenario": {
            scenario: 100 + index
            for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
        },
        "out_npz_by_scenario": {
            scenario: f"{scenario}.npz" for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "out_manifest": "cache_set.json",
        "batch_size": 2,
        "wisig_out_len": 8,
    }


def test_builder_writes_only_verified_post_channel_iq(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "fake.pkl").write_bytes(b"not-a-real-pkl")

    def fake_dataset(**kwargs):
        role = str(kwargs["role"])
        return _TinyWiSig(role), {
            "tx_labels": ["tx0", "tx1"],
            "rx_labels": ["rx0"],
        }

    def fake_overlay(x, scenario, args, *, gen, return_meta):
        offset = 0.1 * (FORMAL_LEO_WEAK_SCENARIOS.index(str(scenario)) + 1)
        return x + offset, {"channel_model": "leo_residual", "snr_db": torch.ones(len(x))}

    monkeypatch.setattr(builder, "_build_wisig_dataset", fake_dataset)
    monkeypatch.setattr(builder, "apply_sat_channel_for_scenario", fake_overlay)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec(tmp_path)), encoding="utf-8")
    result = builder.build_cache_set(spec_path, device=torch.device("cpu"))
    assert result["cache_scope"] == "source_train"

    arrays, manifest, audit = load_verified_leo_weak_cache_set(
        tmp_path / "cache_set.json",
        expected_scope="source_train",
        allowed_roles={"source"},
    )
    assert manifest["clean_sample_access"] is False
    assert audit["physical_sample_count"] == 2
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        assert "leo_weak_iq" in arrays[scenario]
        with np.load(tmp_path / f"{scenario}.npz", allow_pickle=False) as archive:
            assert "raw_iq" not in archive.files
            assert all(not key.startswith("clean") for key in archive.files)


def test_build_spec_rejects_target_cache_without_both_registered_roles() -> None:
    spec = _spec(Path("."))
    spec["cache_scope"] = "stage2_registered"
    try:
        builder.validate_build_spec(spec)
    except ValueError as exc:
        assert "exact roles" in str(exc)
    else:
        raise AssertionError("invalid target cache spec was accepted")


def test_build_spec_accepts_stage2b_target_old_only_scope() -> None:
    spec = _spec(Path("."))
    spec["cache_scope"] = "stage2_target_old"
    spec["role_specs"][0]["role"] = "target_old"
    checked = builder.validate_build_spec(spec)
    assert checked["cache_scope"] == "stage2_target_old"
    assert checked["role_specs"][0]["role"] == "target_old"
