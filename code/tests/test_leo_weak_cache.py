from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA,
    LEO_WEAK_CACHE_SET_SCHEMA,
    LEO_WEAK_CACHE_STAGE,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    load_verified_leo_weak_cache,
    load_verified_leo_weak_cache_set,
    overlay_id,
    physical_sample_id_from_values,
    post_channel_iq_sha256,
    sha256_file,
)


def _write_cache(
    path: Path,
    *,
    scenario: str,
    seed: int = 17,
    role: str = "target_old",
    forbidden_member: str | None = None,
    manifest_patch: dict | None = None,
    iq_offset: float = 0.0,
) -> None:
    iq = (np.arange(32, dtype=np.float32).reshape(2, 2, 8) + iq_offset) / 100.0
    tx = np.asarray(["14-10", "14-7"])
    rx = np.asarray(["20-1", "20-1"])
    day = np.asarray(["d0", "d0"])
    eq = np.asarray(["1", "1"])
    sig = np.asarray(["0", "1"])
    roles = np.asarray([role, role])
    sample_ids = np.asarray(
        [
            physical_sample_id_from_values(
                role=role,
                tx_id=str(tx[index]),
                rx_id=str(rx[index]),
                day_id=str(day[index]),
                eq_id=str(eq[index]),
                sig_id=str(sig[index]),
            )
            for index in range(2)
        ]
    )
    channel_config = {"scenario": scenario, "fs_hz": 25e6, "fc_hz": 2.462e9}
    channel_hash = canonical_json_sha256(channel_config)
    iq_hashes = np.asarray([post_channel_iq_sha256(row) for row in iq])
    overlay_ids = np.asarray(
        [
            overlay_id(
                sample_id=str(sample_ids[index]),
                scenario=scenario,
                satellite_seed=seed,
                channel_config_sha256=channel_hash,
                iq_sha256=str(iq_hashes[index]),
            )
            for index in range(2)
        ]
    )
    manifest = {
        "schema": LEO_WEAK_CACHE_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "contains_post_channel_iq_only": True,
        "contains_clean_rows": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": [scenario],
        "scenario": scenario,
        "iq_array_key": "leo_weak_iq",
        "raw_or_clean_iq_key_present": False,
        "overlay_applied_before_phase2": True,
        "star_ground_channel_impl": "simplified_leo_residual",
        "channel_model": "leo_residual",
        "builder_sha256": hashlib.sha256(b"builder").hexdigest(),
        "output_roles": [role],
        "sample_overlay_provenance_fields": [
            "sample_ids",
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
        "channel_config_sha256": channel_hash,
        "row_count": 2,
    }
    manifest.update(manifest_patch or {})
    payload = {
        "leo_weak_iq": iq,
        "raw_labels": np.asarray([0, 1], dtype=np.int64),
        "domain_labels": np.asarray([0, 0], dtype=np.int64),
        "tx_ids": tx,
        "rx_ids": rx,
        "day_ids": day,
        "eq_ids": eq,
        "sig_ids": sig,
        "dataset_role": roles,
        "channel_views": np.asarray(["rx_base", "rx_base"]),
        "sat_scenarios": np.asarray([scenario, scenario]),
        "satellite_seeds": np.asarray([seed, seed], dtype=np.int64),
        "overlay_applied": np.asarray([True, True]),
        "sample_ids": sample_ids,
        "post_channel_iq_sha256": iq_hashes,
        "overlay_ids": overlay_ids,
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    if forbidden_member:
        payload[forbidden_member] = iq.copy()
    np.savez(path, **payload)


def test_verified_cache_accepts_sample_level_overlay_provenance(tmp_path: Path) -> None:
    path = tmp_path / "leo.npz"
    _write_cache(path, scenario=FORMAL_LEO_WEAK_SCENARIOS[0])
    arrays, manifest, audit = load_verified_leo_weak_cache(
        path,
        expected_scenario=FORMAL_LEO_WEAK_SCENARIOS[0],
        allowed_roles={"target_old"},
    )
    assert arrays["leo_weak_iq"].shape == (2, 2, 8)
    assert manifest["clean_sample_access"] is False
    assert audit["forbidden_members_checked_before_iq_read"] is True
    assert audit["row_count"] == 2


@pytest.mark.parametrize("member", ["raw_iq", "clean_iq", "features", "tx_logits"])
def test_verified_cache_rejects_forbidden_member_before_iq_read(
    tmp_path: Path, member: str
) -> None:
    path = tmp_path / "bad.npz"
    _write_cache(
        path,
        scenario=FORMAL_LEO_WEAK_SCENARIOS[0],
        forbidden_member=member,
    )
    with pytest.raises(ValueError, match="forbidden raw/clean/derived members"):
        load_verified_leo_weak_cache(
            path,
            expected_scenario=FORMAL_LEO_WEAK_SCENARIOS[0],
            allowed_roles={"target_old"},
        )


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"phase2_sample_view_policy": "legacy"}, "manifest contract"),
        ({"clean_sample_access": True}, "manifest contract"),
        ({"channel_config_sha256": ""}, "channel_config_sha256"),
    ],
)
def test_verified_cache_rejects_manifest_protocol_drift(
    tmp_path: Path, patch: dict, match: str
) -> None:
    path = tmp_path / "bad_manifest.npz"
    _write_cache(
        path,
        scenario=FORMAL_LEO_WEAK_SCENARIOS[0],
        manifest_patch=patch,
    )
    with pytest.raises(ValueError, match=match):
        load_verified_leo_weak_cache(
            path,
            expected_scenario=FORMAL_LEO_WEAK_SCENARIOS[0],
            allowed_roles={"target_old"},
        )


def test_verified_cache_rejects_waveform_or_overlay_tamper(tmp_path: Path) -> None:
    path = tmp_path / "tampered.npz"
    _write_cache(path, scenario=FORMAL_LEO_WEAK_SCENARIOS[0])
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["leo_weak_iq"] = payload["leo_weak_iq"].copy()
    payload["leo_weak_iq"][0, 0, 0] += 1.0
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="IQ digest mismatch"):
        load_verified_leo_weak_cache(
            path,
            expected_scenario=FORMAL_LEO_WEAK_SCENARIOS[0],
            allowed_roles={"target_old"},
        )


def test_cache_set_requires_aligned_physical_ids_and_hashes(tmp_path: Path) -> None:
    mapping: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
        path = tmp_path / f"{scenario}.npz"
        _write_cache(path, scenario=scenario, seed=100 + index, iq_offset=float(index))
        mapping[scenario] = path.name
        hashes[scenario] = sha256_file(path)
    physical_ids = [
        "target_old|14-10|20-1|d0|1|0",
        "target_old|14-7|20-1|d0|1|1",
    ]
    manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "cache_scope": "stage2_registered",
        "output_roles": ["target_old"],
        "cache_npz_by_scenario": mapping,
        "cache_sha256_by_scenario": hashes,
        "physical_sample_ids_sha256": ids_sha256(physical_ids),
    }
    set_path = tmp_path / "cache_set.json"
    set_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    arrays, loaded, audit = load_verified_leo_weak_cache_set(
        set_path,
        expected_scope="stage2_registered",
        allowed_roles={"target_old"},
    )
    assert tuple(arrays) == FORMAL_LEO_WEAK_SCENARIOS
    assert loaded["clean_sample_access"] is False
    assert audit["physical_sample_count"] == 2


def test_cache_set_rejects_same_ids_with_unregistered_file_hash(tmp_path: Path) -> None:
    mapping: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        path = tmp_path / f"{scenario}.npz"
        _write_cache(path, scenario=scenario)
        mapping[scenario] = path.name
        hashes[scenario] = sha256_file(path)
    hashes[FORMAL_LEO_WEAK_SCENARIOS[-1]] = hashlib.sha256(b"wrong").hexdigest()
    manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "cache_scope": "stage2_registered",
        "output_roles": ["target_old"],
        "cache_npz_by_scenario": mapping,
        "cache_sha256_by_scenario": hashes,
        "physical_sample_ids_sha256": ids_sha256(
            [
                "target_old|14-10|20-1|d0|1|0",
                "target_old|14-7|20-1|d0|1|1",
            ]
        ),
    }
    set_path = tmp_path / "bad_set.json"
    set_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_verified_leo_weak_cache_set(
            set_path,
            expected_scope="stage2_registered",
            allowed_roles={"target_old"},
        )
