from __future__ import annotations

import json
import shutil
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


class _ConfirmationWiSig(Dataset):
    """Small deterministic source pool for the target-confirmation scope."""

    def __init__(self, tx_ids: tuple[str, ...]) -> None:
        self.rows = [
            (str(tx_id), sig_i)
            for tx_id in tx_ids
            for sig_i in range(120)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        tx_id, sig_i = self.rows[index]
        return (
            torch.full((2, 8), float(index + 1), dtype=torch.float32),
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


def _target_confirmation_spec(
    tmp_path: Path,
    *,
    registered_tx_ids: tuple[str, ...] = ("known-a", "known-b"),
    unknown_tx_ids: tuple[str, ...] = ("unknown-a", "unknown-b"),
) -> dict:
    spec = _spec(tmp_path)
    _enable_single_observation_target(spec)
    spec["cache_scope"] = "phase1_clic_target_confirmation"
    spec["role_specs"] = [
        {
            "role": "target_registered_known",
            "pkl": "fake.pkl",
            "tx_ids": ",".join(registered_tx_ids),
            "rxs": "rx0",
            "days": "0,1,2",
            "max_samples_per_tx": 120,
        },
        {
            "role": "target_unknown",
            "pkl": "fake.pkl",
            "tx_ids": ",".join(unknown_tx_ids),
            "rxs": "rx0",
            "days": "0,1,2",
            "max_samples_per_tx": 120,
        },
    ]
    return spec


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


def test_phase1_clic_target_confirmation_builder_and_loader_use_exact_roles_and_disjoint_physical_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirmation cache is a new, explicit two-role single-LEO scope."""

    (tmp_path / "fake.pkl").write_bytes(b"synthetic-target-confirmation-dataset")
    spec = _target_confirmation_spec(tmp_path)

    def fake_dataset(**kwargs):
        tx_ids = tuple(
            value for value in str(kwargs["tx_spec"]).split(",") if value
        )
        return _ConfirmationWiSig(tx_ids), {
            "tx_labels": list(tx_ids),
            "rx_labels": ["rx0"],
        }

    def fake_overlay(x, scenario, args, *, gen, return_meta):
        del args, gen
        offset = 0.01 * (FORMAL_LEO_WEAK_SCENARIOS.index(str(scenario)) + 1)
        return x + offset, {
            "channel_model": "leo_residual",
            "snr_db": torch.ones(len(x)),
        }

    monkeypatch.setattr(builder, "_build_wisig_dataset", fake_dataset)
    monkeypatch.setattr(builder, "apply_sat_channel_for_scenario", fake_overlay)
    spec_path = tmp_path / "target_confirmation.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")

    result = builder.build_cache_set(spec_path, device=torch.device("cpu"))
    assert result["cache_scope"] == "phase1_clic_target_confirmation"
    assert result["output_roles"] == ["target_registered_known", "target_unknown"]
    manifest_path = Path(result["cache_set_manifest"])
    arrays, manifest, audit = load_verified_leo_weak_cache_set(
        manifest_path,
        expected_scope="phase1_clic_target_confirmation",
        allowed_roles={"target_registered_known", "target_unknown"},
    )
    assert manifest["output_roles"] == [
        "target_registered_known",
        "target_unknown",
    ]
    assert tuple(arrays) == FORMAL_LEO_WEAK_SCENARIOS
    assert audit["physical_sample_count"] == 4 * 3 * 40
    observed_physical: set[str] = set()
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        current = arrays[scene]
        cache_path = manifest_path.parent / str(
            manifest["cache_npz_by_scenario"][scene]
        )
        with np.load(cache_path, allow_pickle=False) as archive:
            current_manifest = json.loads(
                str(np.asarray(archive["manifest_json"]).item())
            )
        assert current_manifest["phase2_physical_sample_observation_policy"] == (
            "single_leo_weak_observation_per_physical_sample"
        )
        assert current_manifest["phase2_cross_scenario_physical_sample_reuse"] is False
        assert current_manifest["phase2_additional_leo_channel_state_generation"] is False
        assert current_manifest["phase2_query_post_reception_view_fit_access"] is False
        roles = np.asarray(current["dataset_role"], dtype=str)
        assert set(roles) == {"target_registered_known", "target_unknown"}
        physical = np.asarray(current["sample_ids"], dtype=str).tolist()
        assert len(physical) == len(set(physical))
        assert observed_physical.isdisjoint(physical)
        observed_physical.update(physical)
    assert len(observed_physical) == audit["physical_sample_count"]


def test_target_confirmation_builder_torch_numpy2_compatibility_without_numpy_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real channel-to-NPZ path must not depend on legacy NumPy bridges."""

    (tmp_path / "fake.pkl").write_bytes(b"torch-numpy2-compat-fixture")
    spec = _target_confirmation_spec(tmp_path)

    def fake_dataset(**kwargs):
        tx_ids = tuple(value for value in str(kwargs["tx_spec"]).split(",") if value)
        return _ConfirmationWiSig(tx_ids), {
            "tx_labels": list(tx_ids),
            "rx_labels": ["rx0"],
        }

    def fake_overlay(x, scenario, args, *, gen, return_meta):
        del args, gen
        offset = 0.01 * (FORMAL_LEO_WEAK_SCENARIOS.index(str(scenario)) + 1)
        return x + offset, {
            "channel_model": "leo_residual",
            "snr_db": torch.ones(len(x)),
        }

    def forbidden_tensor_numpy(*_args, **_kwargs):
        raise AssertionError("builder must not call Tensor.numpy()")

    def forbidden_torch_from_numpy(*_args, **_kwargs):
        raise AssertionError("builder must not call torch.from_numpy()")

    monkeypatch.setattr(builder, "_build_wisig_dataset", fake_dataset)
    monkeypatch.setattr(builder, "apply_sat_channel_for_scenario", fake_overlay)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden_tensor_numpy)
    monkeypatch.setattr(torch, "from_numpy", forbidden_torch_from_numpy)
    spec_path = tmp_path / "torch_numpy2_compat.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")

    result = builder.build_cache_set(spec_path, device=torch.device("cpu"))
    assert result["cache_scope"] == "phase1_clic_target_confirmation"
    assert result["output_roles"] == ["target_registered_known", "target_unknown"]
    assert sum(
        int(audit["row_count"])
        for audit in result["cache_audits"].values()
    ) == 4 * 3 * 40

    manifest_path = Path(result["cache_set_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_rows = 0
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        cache_path = (manifest_path.parent / str(manifest["cache_npz_by_scenario"][scene])).resolve()
        with np.load(cache_path, allow_pickle=False) as archive:
            iq = np.asarray(archive["leo_weak_iq"])
            assert iq.shape == (4 * 40, 2, 8)
            assert np.isfinite(iq).all()
            total_rows += int(iq.shape[0])
    assert total_rows == 4 * 3 * 40


@pytest.mark.parametrize(
    "mutation",
    ("legacy_roles", "overlapping_tx_ids", "role_scene_seed_dependency"),
)
def test_phase1_clic_target_confirmation_rejects_alias_overlap_or_scene_seed_role_contract(
    tmp_path: Path, mutation: str
) -> None:
    spec = _target_confirmation_spec(tmp_path)
    if mutation == "legacy_roles":
        spec["role_specs"][0]["role"] = "target_old"
        spec["role_specs"][1]["role"] = "target_new"
    elif mutation == "overlapping_tx_ids":
        spec["role_specs"][1]["tx_ids"] = spec["role_specs"][0]["tx_ids"]
    else:
        spec["role_specs"][0]["scene"] = "leo_clear_weak"
        spec["role_specs"][1]["satellite_seed"] = 17
    with pytest.raises(ValueError, match="scope|role|exact|TX|tx|overlap|scene|seed|role"):
        builder.validate_build_spec(spec)


def test_cache_set_loader_rejects_npz_toctou_after_manifest_hash_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache replaced after its manifest hash check must never be accepted."""

    # Reuse the existing tiny builder output so this test targets only the
    # loader's verify-to-open window, not a second data-construction path.
    (tmp_path / "fake.pkl").write_bytes(b"race-fixture")
    spec = _spec(tmp_path)

    def fake_dataset(**kwargs):
        return _TinyWiSig(str(kwargs["role"])), {
            "tx_labels": ["tx0", "tx1"],
            "rx_labels": ["rx0"],
        }

    monkeypatch.setattr(builder, "_build_wisig_dataset", fake_dataset)
    monkeypatch.setattr(
        builder,
        "apply_sat_channel_for_scenario",
        lambda x, scenario, args, *, gen, return_meta: (
            x,
            {"channel_model": "leo_residual"},
        ),
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    result = builder.build_cache_set(spec_path, device=torch.device("cpu"))
    manifest_path = Path(result["cache_set_manifest"])
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_npz = (manifest_path.parent / cache_manifest["cache_npz_by_scenario"][FORMAL_LEO_WEAK_SCENARIOS[0]]).resolve()
    original_sha = __import__("hashlib").sha256(target_npz.read_bytes()).hexdigest()
    tampered_npz = target_npz.with_suffix(".race.npz")
    shutil.copy2(target_npz, tampered_npz)
    tampered_npz.write_bytes(tampered_npz.read_bytes() + b"post-hash-race")

    import cvsrffi.leo_weak_cache as cache_module

    original_hash = cache_module.sha256_file
    raced = False

    def race_hash(path: str | Path) -> str:
        nonlocal raced
        value = original_hash(path)
        if Path(path).resolve() == target_npz and not raced:
            raced = True
            target_npz.write_bytes(tampered_npz.read_bytes())
            return original_sha
        return value

    monkeypatch.setattr(cache_module, "sha256_file", race_hash)
    with pytest.raises(Exception, match="TOCTOU|changed|race|hash|SHA|mismatch|drift"):
        load_verified_leo_weak_cache_set(
            manifest_path,
            expected_scope="source_train",
            allowed_roles={"source"},
        )


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
