from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
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


class _PartitionWiSig(Dataset):
    def __init__(self) -> None:
        self.rows = [
            (tx_id, sig_i)
            for tx_id in ("tx0", "tx1")
            for sig_i in range(120)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        tx_id, sig_i = self.rows[index]
        return (
            torch.zeros((2, 8), dtype=torch.float32),
            torch.tensor(0, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
            {
                "tx": tx_id,
                "rx": "rx0",
                "day": str(sig_i // 40),
                "equalized": "1",
                "sig_i": str(sig_i),
            },
        )


def _spec(tmp_path: Path) -> dict:
    return {
        "schema": builder.LEGACY_BUILD_SPEC_SCHEMA,
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


def _enable_single_observation_target(spec: dict) -> None:
    spec.update(
        {
            "schema": builder.BUILD_SPEC_SCHEMA,
            "phase2_physical_sample_observation_policy": (
                "single_leo_weak_observation_per_physical_sample"
            ),
            "phase2_cross_scenario_physical_sample_reuse": False,
            "phase2_additional_leo_channel_state_generation": False,
            "phase2_post_reception_equalization_augmentation_transform_allowed": True,
            "phase2_post_reception_view_from_fixed_received_iq_only": True,
            "phase2_post_reception_view_counts_as_additional_physical_sample": False,
            "phase2_physical_sample_root_id_policy": (
                "immutable_preoverlay_lineage_token"
            ),
            "phase2_query_post_reception_view_fit_access": False,
            "physical_sample_scenario_assignment_policy": (
                builder.SCENARIO_PARTITION_POLICY
            ),
        }
    )


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


def test_external_comparison_overlays_new_rows_only(
    tmp_path: Path, monkeypatch
) -> None:
    overlay_batches = []

    class _RoleTiny(_TinyWiSig):
        def __getitem__(self, index: int):
            x, y, domain, meta = super().__getitem__(index)
            meta["tx"] = f"{self.role}_tx{index}"
            return x, y, domain, meta

    def fake_overlay(x, scenario, args, *, gen, return_meta):
        overlay_batches.append(x.detach().cpu().clone())
        return x + 0.5, {"channel_model": "leo_residual"}

    monkeypatch.setattr(builder, "apply_sat_channel_for_scenario", fake_overlay)
    safe_info = {"dataset_sha256": "a" * 64}
    role_datasets = [
        (
            {"role": "target_old", "apply_leo_overlay": False},
            _RoleTiny("target_old"),
            safe_info,
            tmp_path / "fake.pkl",
        ),
        (
            {"role": "target_new", "apply_leo_overlay": True},
            _RoleTiny("target_new"),
            safe_info,
            tmp_path / "fake.pkl",
        ),
    ]
    out = tmp_path / "mixed.npz"
    audit = builder._build_one_scenario(
        scenario="leo_clear_weak",
        base_seed=123,
        role_datasets=role_datasets,
        spec={"cache_scope": "external_comparison_registered"},
        out_path=out,
        builder_sha256="b" * 64,
        device=torch.device("cpu"),
    )
    assert audit["row_count"] == 4
    from paper_reproduction.scripts.build_adv3b02_paper_full_ci_bundle import (
        load_comparison_inner_leo_cache,
    )

    comparison_arrays, _, comparison_audit = load_comparison_inner_leo_cache(
        out,
        expected_scenario="leo_clear_weak",
        allowed_roles={"target_old", "target_new"},
    )
    assert comparison_audit["old_class_unmodified_received_iq_verified"] is True
    assert comparison_audit["new_class_leo_iq_verified"] is True
    assert comparison_arrays["leo_weak_iq"].shape[0] == 4
    with np.load(out, allow_pickle=False) as archive:
        tampered_payload = {
            key: np.asarray(archive[key]) for key in archive.files
        }
    tampered_manifest = json.loads(
        str(np.asarray(tampered_payload["manifest_json"]).reshape(-1)[0])
    )
    tampered_manifest["phase2_sample_view_policy"] = PHASE2_SAMPLE_VIEW_POLICY
    tampered_payload["manifest_json"] = np.asarray(
        json.dumps(tampered_manifest, sort_keys=True)
    )
    tampered = tmp_path / "mixed_policy_tampered.npz"
    np.savez(tampered, **tampered_payload)
    with pytest.raises(ValueError, match="manifest contract"):
        load_comparison_inner_leo_cache(
            tampered,
            expected_scenario="leo_clear_weak",
            allowed_roles={"target_old", "target_new"},
        )
    assert len(overlay_batches) == 1
    with np.load(out, allow_pickle=False) as archive:
        roles = np.asarray(archive["dataset_role"]).astype(str)
        applied = np.asarray(archive["overlay_applied"]).astype(bool)
        views = np.asarray(archive["channel_views"]).astype(str)
        iq = np.asarray(archive["leo_weak_iq"])
        assert np.all(~applied[roles == "target_old"])
        assert np.all(applied[roles == "target_new"])
        assert np.all(views[roles == "target_old"] == "unmodified_received_iq")
        assert np.all(views[roles == "target_new"] == "rx_base")
        assert np.allclose(iq[roles == "target_old", 0, 0], [1.0, 2.0])
        assert np.allclose(iq[roles == "target_new", 0, 0], [1.5, 2.5])


def test_build_spec_rejects_target_cache_without_both_registered_roles() -> None:
    spec = _spec(Path("."))
    _enable_single_observation_target(spec)
    spec["cache_scope"] = "stage2_registered"
    try:
        builder.validate_build_spec(spec)
    except ValueError as exc:
        assert "exact roles" in str(exc)
    else:
        raise AssertionError("invalid target cache spec was accepted")


def test_build_spec_accepts_stage2b_target_old_only_scope() -> None:
    spec = _spec(Path("."))
    _enable_single_observation_target(spec)
    spec["cache_scope"] = "stage2_target_old"
    spec["role_specs"][0]["role"] = "target_old"
    spec["role_specs"][0]["days"] = "0,1,2"
    spec["role_specs"][0]["max_samples_per_tx"] = 120
    checked = builder.validate_build_spec(spec)
    assert checked["cache_scope"] == "stage2_target_old"
    assert checked["role_specs"][0]["role"] == "target_old"


def test_build_spec_requires_reference_exclusion_pair() -> None:
    spec = _spec(Path("."))
    _enable_single_observation_target(spec)
    spec["cache_scope"] = "stage2_target_old"
    spec["role_specs"][0]["role"] = "target_old"
    spec["role_specs"][0]["days"] = "0,1,2"
    spec["role_specs"][0]["max_samples_per_tx"] = 120
    spec["physical_sample_exclusion_policy"] = (
        builder.REFERENCE_EXCLUSION_POLICY
    )
    try:
        builder.validate_build_spec(spec)
    except ValueError as exc:
        assert "declared together" in str(exc)
    else:
        raise AssertionError("unpaired reference exclusion was accepted")


def test_reference_cache_exclusions_are_forwarded_by_role_and_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_sha = "a" * 64
    reference = tmp_path / "reference.json"
    reference.write_text("{}", encoding="utf-8")
    pkl_path = tmp_path / "ManySig.pkl"
    pkl_path.write_bytes(b"dataset")
    arrays = {}
    for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        arrays[scenario] = {
            "dataset_role": np.asarray(["target_old"]),
            "source_dataset_sha256": np.asarray([dataset_sha]),
            "source_record_indices": np.asarray(
                [scenario_index], dtype=np.int64
            ),
        }
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *args, **kwargs: (
            arrays,
            {"cache_set_id": "reference"},
            {"physical_sample_count": 3},
        ),
    )
    monkeypatch.setattr(
        builder,
        "sha256_file",
        lambda path: dataset_sha
        if Path(path).name == "ManySig.pkl"
        else "b" * 64,
    )
    observed = {}

    def fake_dataset(**kwargs):
        observed.update(kwargs)
        return _TinyWiSig("target_old"), {"tx_labels": ["tx0"]}

    monkeypatch.setattr(builder, "_build_wisig_dataset", fake_dataset)
    spec = _spec(tmp_path)
    _enable_single_observation_target(spec)
    spec["cache_scope"] = "stage2_target_old"
    spec["role_specs"] = [
        {
            "role": "target_old",
            "pkl": str(pkl_path),
            "tx_ids": "tx0",
            "rxs": "rx0",
            "days": "0,1,2",
            "max_samples_per_tx": 120,
        }
    ]
    spec["physical_sample_exclusion_policy"] = (
        builder.REFERENCE_EXCLUSION_POLICY
    )
    spec["physical_sample_exclusion_reference_cache_set"] = str(reference)
    datasets, audit = builder._build_role_datasets(
        builder.validate_build_spec(spec), spec_dir=tmp_path
    )
    assert len(datasets) == 1
    assert observed["exclude_source_record_indices"] == {0, 1, 2}
    assert audit["excluded_source_record_count"] == 3


def test_preoverlay_partition_assigns_each_physical_row_to_one_scenario() -> None:
    dataset = _PartitionWiSig()
    role_spec = {"role": "target_old", "tx_ids": "tx0,tx1"}
    partitions = builder._partition_role_datasets_by_scenario(
        [
            (
                role_spec,
                dataset,
                {"dataset_sha256": "a" * 64, "dataset_seed": 713101},
                Path("ManySig.pkl"),
            )
        ],
        batch_size=32,
    )
    observed: set[int] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        subset = partitions[scenario][0][1]
        assert len(subset) == 80
        current = set(int(value) for value in subset.indices)
        assert observed.isdisjoint(current)
        observed.update(current)
        for tx_id in ("tx0", "tx1"):
            day_counts = [
                sum(
                    1
                    for index in current
                    if dataset.rows[index][0] == tx_id
                    and dataset.rows[index][1] // 40 == day
                )
                for day in range(3)
            ]
            assert sorted(day_counts) == [13, 13, 14]
    assert observed == set(range(240))
