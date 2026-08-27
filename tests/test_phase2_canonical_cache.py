from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

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
    physical_sample_id,
    physical_sample_id_from_values,
    post_channel_iq_sha256,
    sha256_file,
)
from cvsrffi.wisig_canonical_inventory import build_inventory


CODE_SCRIPTS = Path(__file__).resolve().parents[1] / "code" / "scripts"
if str(CODE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CODE_SCRIPTS))
import build_cvs_leo_weak_iq_cache as cache_builder  # noqa: E402


CANONICAL_SCOPE = "stage2_canonical_registered"
CANONICAL_SPEC_SCHEMA = "cvs_leo_weak_iq_cache_build_spec_v3"
SPLIT_SCHEMA = "cvs.phase2.canonical_split_manifest.v1"
PROTOCOL_SCHEMA = "p2_min_v1"
OLD_TX_IDS = tuple(f"old-{index}" for index in range(6))
NEW_TX_ID = "new-0"
REGISTERED_TX_IDS = OLD_TX_IDS + (NEW_TX_ID,)


def _single_observation_contract() -> dict[str, object]:
    return {
        "phase2_physical_sample_observation_policy": (
            "single_leo_weak_observation_per_physical_sample"
        ),
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_equalization_augmentation_transform_allowed": True,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_physical_sample_root_id_policy": "immutable_preoverlay_lineage_token",
        "phase2_query_post_reception_view_fit_access": False,
        "physical_sample_scenario_assignment_policy": (
            "disjoint_preoverlay_tx_day_stratified_v1"
        ),
    }


def _canonical_spec() -> dict[str, object]:
    return {
        "schema": CANONICAL_SPEC_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "cache_scope": CANONICAL_SCOPE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "star_ground_channel_impl": "simplified_leo_residual",
        **_single_observation_contract(),
        "canonical_inventory": "canonical.sqlite",
        "split_manifest": "split.json",
        "satellite_seed_by_scenario": {
            scenario: 713101 + index
            for index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
        },
        "out_npz_by_scenario": {
            scenario: f"cache/{scenario}.npz"
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "out_manifest": "cache/cache_set.json",
        "wisig_equalized": "1",
        "wisig_out_len": 8,
        "wisig_domain": "rx_day",
        "batch_size": 32,
    }


def _valid_cache_payload(*, canonical: bool) -> dict[str, np.ndarray]:
    scenario = FORMAL_LEO_WEAK_SCENARIOS[0]
    iq = np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8)
    tx_ids = np.asarray([OLD_TX_IDS[0], NEW_TX_ID])
    rx_ids = np.asarray(["rx-A", "rx-A"])
    day_ids = np.asarray(["day-A", "day-D"])
    eq_ids = np.asarray(["1", "1"])
    sig_ids = np.asarray(["0", "7"])
    dataset_hashes = np.asarray(["a" * 64, "b" * 64])
    record_indices = np.asarray([0, 7], dtype=np.int64)
    dataset_roles = np.asarray(["target_old", "target_new"])
    legacy_ids = [
        physical_sample_id_from_values(
            dataset_sha256=str(dataset_hashes[index]),
            source_record_index=int(record_indices[index]),
            role=str(dataset_roles[index]),
            tx_id=str(tx_ids[index]),
            rx_id=str(rx_ids[index]),
            day_id=str(day_ids[index]),
            eq_id=str(eq_ids[index]),
            sig_id=str(sig_ids[index]),
        )
        for index in range(2)
    ]
    canonical_ids = ["canonical-old", "canonical-new"]
    sample_ids = canonical_ids if canonical else legacy_ids
    seeds = np.asarray([713101, 713101], dtype=np.int64)
    channel_config = {"channel_model": "leo_residual", "scenario": scenario}
    channel_hash = canonical_json_sha256(channel_config)
    iq_hashes = [post_channel_iq_sha256(row) for row in iq]
    overlay_ids = [
        overlay_id(
            sample_id=sample_ids[index],
            scenario=scenario,
            satellite_seed=int(seeds[index]),
            channel_config_sha256=channel_hash,
            iq_sha256=iq_hashes[index],
        )
        for index in range(2)
    ]
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
        "overlay_role_policy": "all_roles",
        "star_ground_channel_impl": "simplified_leo_residual",
        "channel_model": "leo_residual",
        "channel_config": channel_config,
        "channel_config_sha256": channel_hash,
        "builder_sha256": "c" * 64,
        "output_roles": ["target_old", "target_new"],
        "row_count": 2,
        "physical_sample_ids_sha256": ids_sha256(sample_ids),
        "post_channel_iq_sha256_root": ids_sha256(iq_hashes),
        "overlay_ids_sha256": ids_sha256(overlay_ids),
        "sample_overlay_provenance_fields": [
            "sample_ids",
            "source_dataset_sha256",
            "source_record_indices",
            "sat_scenarios",
            "satellite_seeds",
            "post_channel_iq_sha256",
            "overlay_ids",
        ],
    }
    payload: dict[str, np.ndarray] = {
        "leo_weak_iq": iq,
        "raw_labels": np.asarray([0, 1], dtype=np.int64),
        "domain_labels": np.asarray([0, 0], dtype=np.int64),
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "eq_ids": eq_ids,
        "sig_ids": sig_ids,
        "source_dataset_sha256": dataset_hashes,
        "source_record_indices": record_indices,
        "dataset_role": dataset_roles,
        "channel_views": np.asarray(["rx_base", "rx_base"]),
        "sat_scenarios": np.asarray([scenario, scenario]),
        "satellite_seeds": seeds,
        "overlay_applied": np.asarray([True, True]),
        "sample_ids": np.asarray(sample_ids),
        "post_channel_iq_sha256": np.asarray(iq_hashes),
        "overlay_ids": np.asarray(overlay_ids),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }
    if canonical:
        payload.update(
            {
                "canonical_physical_sample_ids": np.asarray(canonical_ids),
                "split_roles": np.asarray(["support", "query"]),
                "split_ranks": np.asarray([0, 3], dtype=np.int64),
            }
        )
    return payload


def _write_cache(path: Path, *, canonical: bool = True, mutate=None) -> Path:
    payload = _valid_cache_payload(canonical=canonical)
    if mutate is not None:
        mutate(payload)
    np.savez(path, **payload)
    return path


def _load_cache(path: Path):
    return load_verified_leo_weak_cache(
        path,
        expected_scenario=FORMAL_LEO_WEAK_SCENARIOS[0],
        allowed_roles={"target_old", "target_new"},
    )


def test_canonical_physical_id_takes_precedence_and_empty_rejects():
    assert physical_sample_id(
        {"canonical_physical_sample_ids": np.asarray(["canonical-id"])}, 0
    ) == "canonical-id"
    with pytest.raises(ValueError, match="canonical physical sample ID must be nonempty"):
        physical_sample_id(
            {"canonical_physical_sample_ids": np.asarray([""])}, 0
        )


def test_legacy_physical_id_is_byte_for_byte_unchanged():
    arrays = {
        "source_dataset_sha256": np.asarray(["a" * 64]),
        "source_record_indices": np.asarray([99], dtype=np.int64),
        "dataset_role": np.asarray(["target_old"]),
        "tx_ids": np.asarray(["tx-1"]),
        "rx_ids": np.asarray(["rx-2"]),
        "day_ids": np.asarray(["day-3"]),
        "eq_ids": np.asarray(["1"]),
        "sig_ids": np.asarray(["4"]),
    }
    assert physical_sample_id(arrays, 0) == (
        f"{'a' * 64}|tx-1|rx-2|day-3|1|4"
    )


def test_legacy_cache_without_canonical_members_remains_accepted(tmp_path: Path):
    arrays, _manifest, _audit = _load_cache(
        _write_cache(tmp_path / "legacy.npz", canonical=False)
    )
    assert "canonical_physical_sample_ids" not in arrays


def test_canonical_cache_members_round_trip(tmp_path: Path):
    arrays, _manifest, _audit = _load_cache(
        _write_cache(tmp_path / "canonical.npz")
    )
    assert arrays["canonical_physical_sample_ids"].tolist() == [
        "canonical-old",
        "canonical-new",
    ]
    assert arrays["split_roles"].tolist() == ["support", "query"]
    assert arrays["split_ranks"].tolist() == [0, 3]


@pytest.mark.parametrize(
    "missing",
    ("canonical_physical_sample_ids", "split_roles", "split_ranks"),
)
def test_canonical_cache_requires_exact_member_trio(tmp_path: Path, missing: str):
    path = _write_cache(
        tmp_path / f"missing-{missing}.npz",
        mutate=lambda payload: payload.pop(missing),
    )
    with pytest.raises(ValueError, match="canonical split members must be present as an exact trio"):
        _load_cache(path)


def test_canonical_cache_rejects_empty_and_duplicate_ids(tmp_path: Path):
    for name, values, message in (
        ("empty", ["canonical-old", ""], "nonempty"),
        ("duplicate", ["same", "same"], "unique"),
    ):
        path = _write_cache(
            tmp_path / f"{name}.npz",
            mutate=lambda payload, values=values: payload.__setitem__(
                "canonical_physical_sample_ids", np.asarray(values)
            ),
        )
        with pytest.raises(ValueError, match=message):
            _load_cache(path)


def test_canonical_cache_rejects_bad_role_or_length(tmp_path: Path):
    bad_role = _write_cache(
        tmp_path / "bad-role.npz",
        mutate=lambda payload: payload.__setitem__(
            "split_roles", np.asarray(["support", "validation"])
        ),
    )
    with pytest.raises(ValueError, match="support or query"):
        _load_cache(bad_role)

    bad_length = _write_cache(
        tmp_path / "bad-length.npz",
        mutate=lambda payload: payload.__setitem__(
            "split_roles", np.asarray(["support"])
        ),
    )
    with pytest.raises(ValueError, match="row count drift"):
        _load_cache(bad_length)


def test_canonical_cache_rejects_scalar_split_member(tmp_path: Path):
    bad_scalar = _write_cache(
        tmp_path / "bad-scalar.npz",
        mutate=lambda payload: payload.__setitem__(
            "canonical_physical_sample_ids", np.asarray("scalar")
        ),
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        _load_cache(bad_scalar)


@pytest.mark.parametrize(
    "values",
    (
        np.asarray([False, True], dtype=bool),
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray(["0", "1"]),
        np.asarray([0, -1], dtype=np.int64),
    ),
)
def test_canonical_cache_rejects_non_exact_or_negative_split_ranks(
    tmp_path: Path, values: np.ndarray
):
    dtype_name = str(values.dtype).replace("<", "string-")
    path = _write_cache(
        tmp_path / f"bad-rank-{dtype_name}.npz",
        mutate=lambda payload: payload.__setitem__("split_ranks", values),
    )
    with pytest.raises(ValueError, match="nonnegative exact integers"):
        _load_cache(path)


def test_v3_canonical_spec_accepts_inventory_manifest_and_no_role_specs():
    checked = cache_builder.validate_build_spec(_canonical_spec())
    assert checked["schema"] == CANONICAL_SPEC_SCHEMA
    assert checked["cache_scope"] == CANONICAL_SCOPE
    assert "role_specs" not in checked


@pytest.mark.parametrize("missing", ("canonical_inventory", "split_manifest"))
def test_v3_canonical_spec_rejects_missing_path(missing: str):
    spec = _canonical_spec()
    spec.pop(missing)
    with pytest.raises(ValueError, match=missing):
        cache_builder.validate_build_spec(spec)


@pytest.mark.parametrize("role_specs", ([], [{"role": "target_old"}]))
def test_v3_canonical_spec_forbids_any_role_specs(role_specs: object):
    spec = _canonical_spec()
    spec["role_specs"] = role_specs
    with pytest.raises(ValueError, match="forbids role_specs"):
        cache_builder.validate_build_spec(spec)


def test_v3_canonical_spec_rejects_wrong_schema_and_protocol():
    for key, value, message in (
        ("schema", "cvs_leo_weak_iq_cache_build_spec_v2", "schema"),
        ("protocol_schema", "wrong", "protocol_schema"),
    ):
        spec = _canonical_spec()
        spec[key] = value
        with pytest.raises(ValueError, match=message):
            cache_builder.validate_build_spec(spec)


def _iq_value(tx_id: str, day_index: int, sig_index: int) -> float:
    return float(sum(ord(character) for character in tx_id) * 1000 + day_index * 100 + sig_index)


def _asset_payload(tx_counts: dict[str, int]) -> dict[str, object]:
    day_labels = ["day-A", "day-B", "day-C", "day-D"]
    data = []
    for tx_id, count in tx_counts.items():
        samples_by_day: list[list[np.ndarray]] = [[] for _ in day_labels]
        for serial in range(count):
            day_index = serial % len(day_labels)
            sig_index = len(samples_by_day[day_index])
            samples_by_day[day_index].append(
                np.full(
                    (8, 2),
                    _iq_value(tx_id, day_index, sig_index),
                    dtype=np.float32,
                )
            )
        day_rows = []
        for day_samples in samples_by_day:
            eq1 = (
                np.stack(day_samples)
                if day_samples
                else np.empty((0, 8, 2), dtype=np.float32)
            )
            eq0 = np.zeros_like(eq1)
            day_rows.append([eq0, eq1])
        data.append([day_rows])
    return {
        "data": data,
        "tx_list": list(tx_counts),
        "rx_list": ["rx-A"],
        "capture_date_list": day_labels,
        "equalized_list": [0, 1],
    }


def _write_pickle(path: Path, payload: dict[str, object]) -> Path:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _split_from_inventory(inventory: Path) -> dict[str, object]:
    connection = sqlite3.connect(inventory)
    try:
        columns = (
            "physical_sample_id",
            "tx_id",
            "rx_id",
            "day_id",
            "eq_id",
            "sig_id",
            "preferred_asset",
            "preferred_source_record_index",
        )
        records = [
            dict(zip(columns, row))
            for row in connection.execute(
                """
                SELECT physical_sample_id, tx_id, rx_id, day_id, eq_id, sig_id,
                       preferred_asset, preferred_source_record_index
                FROM canonical_records
                WHERE eligible = 1
                ORDER BY tx_id, day_id, CAST(sig_id AS INTEGER), physical_sample_id
                """
            )
        ]
    finally:
        connection.close()

    by_tx: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_tx[str(record["tx_id"])].append(record)
    rows: list[dict[str, object]] = []
    for tx_id in REGISTERED_TX_IDS:
        scene_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for index, record in enumerate(by_tx[tx_id]):
            scene_groups[FORMAL_LEO_WEAK_SCENARIOS[index % 3]].append(record)
        for scene in FORMAL_LEO_WEAK_SCENARIOS:
            for rank, record in enumerate(scene_groups[scene]):
                role = "support" if rank == 0 else "query"
                row = {
                    "physical_sample_id": record["physical_sample_id"],
                    "source_asset": record["preferred_asset"],
                    "source_record_index": record["preferred_source_record_index"],
                    "rx_id": record["rx_id"],
                    "day_id": record["day_id"],
                    "scene": scene,
                    "role": role,
                    "rank": rank,
                }
                if role == "support":
                    row["tx_id"] = tx_id
                rows.append(row)
    rows.reverse()
    support_count = sum(row["role"] == "support" for row in rows)
    return {
        "schema": SPLIT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "profile_id": "SYNTHETIC_CANONICAL",
        "query_policy": "MAXQ_ALL_UNIQUE",
        "k": 1,
        "registered_tx_ids": list(REGISTERED_TX_IDS),
        "eligible_receivers": ["rx-A"],
        "capsule_id": "1" * 64,
        "split_id": "2" * 64,
        "counts": {
            "registered_tx_count": len(REGISTERED_TX_IDS),
            "eligible_receiver_count": 1,
            "eligible_count": len(rows),
            "support_count": support_count,
            "query_count": len(rows) - support_count,
            "row_count": len(rows),
        },
        "rows": rows,
    }


def _canonical_fixture(tmp_path: Path) -> dict[str, object]:
    many_sig = _write_pickle(
        tmp_path / "ManySig.pkl",
        _asset_payload({OLD_TX_IDS[0]: 121, NEW_TX_ID: 3}),
    )
    many_tx = _write_pickle(
        tmp_path / "ManyTx.pkl",
        _asset_payload(
            {
                OLD_TX_IDS[0]: 121,
                **{tx_id: 3 for tx_id in OLD_TX_IDS[1:]},
            }
        ),
    )
    inventory = tmp_path / "canonical.sqlite"
    build_inventory(
        {"ManySig": many_sig, "ManyTx": many_tx},
        inventory,
        equalized=1,
    )
    split = _split_from_inventory(inventory)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    spec = _canonical_spec()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True), encoding="utf-8")
    return {
        "many_sig": many_sig,
        "many_tx": many_tx,
        "inventory": inventory,
        "split": split,
        "split_path": split_path,
        "spec": spec,
        "spec_path": spec_path,
    }


def _identity_overlay(
    x: torch.Tensor,
    scenario: str,
    _args,
    *,
    gen,
    return_meta: bool,
):
    assert gen is not None
    assert return_meta is True
    return x.clone(), {"channel_model": "leo_residual", "scenario": scenario}


def _assert_no_outputs(tmp_path: Path, spec: dict[str, object]) -> None:
    assert not (tmp_path / str(spec["out_manifest"])).exists()
    for raw_path in dict(spec["out_npz_by_scenario"]).values():
        assert not (tmp_path / str(raw_path)).exists()


def test_canonical_materialization_uses_preferred_sources_and_exact_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _canonical_fixture(tmp_path)
    inventory_before = Path(fixture["inventory"]).read_bytes()
    real_pickle_load = pickle.load
    load_counts: Counter[str] = Counter()

    def counting_load(handle, *args, **kwargs):
        load_counts[str(Path(handle.name).resolve())] += 1
        return real_pickle_load(handle, *args, **kwargs)

    monkeypatch.setattr(pickle, "load", counting_load)
    monkeypatch.setattr(
        cache_builder,
        "apply_sat_channel_for_scenario",
        _identity_overlay,
    )
    result = cache_builder.build_cache_set(
        fixture["spec_path"], device=torch.device("cpu")
    )

    assert Path(fixture["inventory"]).read_bytes() == inventory_before
    assert load_counts == Counter(
        {
            str(Path(fixture["many_sig"]).resolve()): 1,
            str(Path(fixture["many_tx"]).resolve()): 1,
        }
    )
    manifest_path = Path(result["cache_set_manifest"])
    arrays_by_scenario, set_manifest, audit = load_verified_leo_weak_cache_set(
        manifest_path,
        expected_scope=CANONICAL_SCOPE,
        allowed_roles={"target_old", "target_new"},
    )
    assert set_manifest["protocol_schema"] == PROTOCOL_SCHEMA
    assert set_manifest["capsule_id"] == fixture["split"]["capsule_id"]
    assert set_manifest["split_id"] == fixture["split"]["split_id"]
    assert audit["phase2_single_observation_compliant"] is True

    expected_by_scenario = {
        scenario: [
            row
            for row in fixture["split"]["rows"]
            if row["scene"] == scenario
        ]
        for scenario in FORMAL_LEO_WEAK_SCENARIOS
    }
    observed_sets: list[set[str]] = []
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        arrays = arrays_by_scenario[scenario]
        expected_rows = expected_by_scenario[scenario]
        expected_ids = [str(row["physical_sample_id"]) for row in expected_rows]
        assert arrays["canonical_physical_sample_ids"].tolist() == expected_ids
        assert arrays["sample_ids"].tolist() == expected_ids
        assert arrays["split_roles"].tolist() == [
            row["role"] for row in expected_rows
        ]
        assert arrays["split_ranks"].tolist() == [
            row["rank"] for row in expected_rows
        ]
        observed_sets.append(set(expected_ids))
    assert not any(
        left & right
        for index, left in enumerate(observed_sets)
        for right in observed_sets[index + 1 :]
    )
    assert sum(
        int(np.sum(arrays["tx_ids"] == OLD_TX_IDS[0]))
        for arrays in arrays_by_scenario.values()
    ) == 121

    many_sig_hash = sha256_file(fixture["many_sig"])
    many_tx_hash = sha256_file(fixture["many_tx"])
    hashes_by_tx: dict[str, set[str]] = defaultdict(set)
    for arrays in arrays_by_scenario.values():
        for tx_id, dataset_hash in zip(
            arrays["tx_ids"].tolist(),
            arrays["source_dataset_sha256"].tolist(),
        ):
            hashes_by_tx[str(tx_id)].add(str(dataset_hash))
    assert hashes_by_tx[OLD_TX_IDS[0]] == {many_sig_hash}
    assert hashes_by_tx[NEW_TX_ID] == {many_sig_hash}
    assert all(hashes_by_tx[tx_id] == {many_tx_hash} for tx_id in OLD_TX_IDS[1:])


def test_canonical_build_rejects_wrong_split_protocol_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _canonical_fixture(tmp_path)
    split = copy.deepcopy(fixture["split"])
    split["protocol_schema"] = "wrong"
    Path(fixture["split_path"]).write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr(cache_builder, "apply_sat_channel_for_scenario", _identity_overlay)
    with pytest.raises(ValueError, match="protocol_schema"):
        cache_builder.build_cache_set(fixture["spec_path"], device=torch.device("cpu"))
    _assert_no_outputs(tmp_path, fixture["spec"])


def test_canonical_build_rejects_support_query_overlap_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _canonical_fixture(tmp_path)
    split = copy.deepcopy(fixture["split"])
    duplicate = dict(split["rows"][0])
    duplicate["role"] = "query" if duplicate["role"] == "support" else "support"
    duplicate.pop("tx_id", None)
    split["rows"].append(duplicate)
    split["counts"]["row_count"] += 1
    split["counts"]["eligible_count"] += 1
    Path(fixture["split_path"]).write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr(cache_builder, "apply_sat_channel_for_scenario", _identity_overlay)
    with pytest.raises(ValueError, match="duplicate|overlap"):
        cache_builder.build_cache_set(fixture["spec_path"], device=torch.device("cpu"))
    _assert_no_outputs(tmp_path, fixture["spec"])


def test_canonical_build_rejects_preferred_reference_tamper_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _canonical_fixture(tmp_path)
    split = copy.deepcopy(fixture["split"])
    split["rows"][0]["source_record_index"] += 1
    Path(fixture["split_path"]).write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr(cache_builder, "apply_sat_channel_for_scenario", _identity_overlay)
    with pytest.raises(ValueError, match="preferred materialization reference"):
        cache_builder.build_cache_set(fixture["spec_path"], device=torch.device("cpu"))
    _assert_no_outputs(tmp_path, fixture["spec"])


@pytest.mark.parametrize("tamper", ("digest", "coordinate"))
def test_canonical_build_rejects_inventory_identity_tamper_before_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
):
    fixture = _canonical_fixture(tmp_path)
    connection = sqlite3.connect(fixture["inventory"])
    try:
        physical_id = connection.execute(
            "SELECT physical_sample_id FROM canonical_records ORDER BY physical_sample_id LIMIT 1"
        ).fetchone()[0]
        if tamper == "digest":
            connection.execute(
                "UPDATE canonical_records SET iq_sha256 = ? WHERE physical_sample_id = ?",
                ("f" * 64, physical_id),
            )
        else:
            connection.execute(
                "UPDATE canonical_records SET sig_id = ? WHERE physical_sample_id = ?",
                ("999999", physical_id),
            )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(cache_builder, "apply_sat_channel_for_scenario", _identity_overlay)
    with pytest.raises(ValueError, match="digest|canonical coordinate|physical sample ID"):
        cache_builder.build_cache_set(fixture["spec_path"], device=torch.device("cpu"))
    _assert_no_outputs(tmp_path, fixture["spec"])


def test_canonical_build_rejects_preexisting_output_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _canonical_fixture(tmp_path)
    first_output = tmp_path / dict(fixture["spec"]["out_npz_by_scenario"])[
        FORMAL_LEO_WEAK_SCENARIOS[0]
    ]
    first_output.parent.mkdir(parents=True)
    first_output.write_bytes(b"external-output")
    monkeypatch.setattr(cache_builder, "apply_sat_channel_for_scenario", _identity_overlay)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cache_builder.build_cache_set(fixture["spec_path"], device=torch.device("cpu"))
    assert first_output.read_bytes() == b"external-output"
    assert not (tmp_path / str(fixture["spec"]["out_manifest"])).exists()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS[1:]:
        assert not (
            tmp_path / dict(fixture["spec"]["out_npz_by_scenario"])[scenario]
        ).exists()
