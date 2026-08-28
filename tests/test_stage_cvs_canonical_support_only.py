from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cvsrffi.leo_weak_cache import (
    FORMAL_LEO_WEAK_SCENARIOS,
    LEO_WEAK_CACHE_SCHEMA,
    LEO_WEAK_CACHE_SET_SCHEMA,
    LEO_WEAK_CACHE_STAGE,
    PHASE2_SAMPLE_VIEW_POLICY,
    canonical_json_sha256,
    ids_sha256,
    load_verified_leo_weak_cache_set,
    overlay_id,
    post_channel_iq_sha256,
    sha256_file,
)
from cvsrffi.stage2_predictor_bundle import SUPPORT_NPZ_MEMBERS, SUPPORT_SCHEMA
from cvsrffi.stage2_target_row_export import export_target_row
from scripts import build_cvs_stage2_support_prototypes as prototype_bridge
from scripts import stage_cvs_canonical_support_only as subject


CAPSULE_ID = "536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2"
SPLIT_ID = "260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25"
CHECKPOINT = (
    "/home/szu2070436088/2510044040/CV-SincNet/runs/"
    "phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/"
    "best_joint_safe_ssdg.pth"
)
REGISTERED_TX_IDS = (
    "14-10",
    "14-7",
    "20-15",
    "20-19",
    "6-15",
    "8-20",
    "11-1",
    "7-11",
    "10-11",
    "10-7",
    "11-4",
    "11-7",
    "15-1",
    "16-16",
    "2-19",
    "20-12",
    "20-7",
    "3-13",
    "5-5",
    "6-1",
    "7-10",
    "8-18",
    "8-3",
    "13-3",
    "4-11",
    "3-18",
)


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


def _cache_payload(
    scenario: str,
    *,
    short_last_class: bool = False,
    duplicate_support_id: bool = False,
) -> dict[str, np.ndarray]:
    scenario_index = FORMAL_LEO_WEAK_SCENARIOS.index(scenario)
    rows: list[dict[str, object]] = []
    for class_id, tx_id in enumerate(REGISTERED_TX_IDS):
        support_count = 19 if short_last_class and class_id == 25 else 20
        for rank in range(support_count):
            rows.append(
                {
                    "class_id": class_id,
                    "tx_id": tx_id,
                    "rank": rank,
                    "split_role": "support",
                }
            )
        rows.append(
            {
                "class_id": class_id,
                "tx_id": tx_id,
                "rank": 20,
                "split_role": "query",
            }
        )
    rows.reverse()

    iq_rows: list[np.ndarray] = []
    canonical_ids: list[str] = []
    dataset_hashes: list[str] = []
    source_indices: list[int] = []
    tx_ids: list[str] = []
    dataset_roles: list[str] = []
    split_roles: list[str] = []
    split_ranks: list[int] = []
    for index, row in enumerate(rows):
        class_id = int(row["class_id"])
        rank = int(row["rank"])
        split_role = str(row["split_role"])
        marker = np.float32(
            scenario_index * 1000
            + class_id * 30
            + rank
            + (10000 if split_role == "query" else 0)
        )
        iq_rows.append(np.full((2, 4), marker + 1.0, dtype=np.float32))
        canonical_ids.append(
            f"{scenario}|1-1|{row['tx_id']}|{split_role}|{rank:02d}"
        )
        dataset_hashes.append(
            hashlib.sha256(str(row["tx_id"]).encode("ascii")).hexdigest()
        )
        source_indices.append(scenario_index * 10000 + index)
        tx_ids.append(str(row["tx_id"]))
        dataset_roles.append("target_old" if class_id < 6 else "target_new")
        split_roles.append(split_role)
        split_ranks.append(rank)
    if duplicate_support_id and scenario == "leo_clear_weak":
        support_positions = [
            index for index, role in enumerate(split_roles) if role == "support"
        ]
        canonical_ids[support_positions[1]] = canonical_ids[support_positions[0]]

    iq = np.stack(iq_rows).astype(np.float32)
    iq_hashes = [post_channel_iq_sha256(row) for row in iq]
    satellite_seed = 713101 + scenario_index
    channel_config = {"channel_model": "leo_residual", "scenario": scenario}
    channel_hash = canonical_json_sha256(channel_config)
    overlay_ids = [
        overlay_id(
            sample_id=sample_id,
            scenario=scenario,
            satellite_seed=satellite_seed,
            channel_config_sha256=channel_hash,
            iq_sha256=iq_hash,
        )
        for sample_id, iq_hash in zip(canonical_ids, iq_hashes)
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
        "row_count": len(rows),
        "physical_sample_ids_sha256": ids_sha256(canonical_ids),
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
    row_count = len(rows)
    return {
        "leo_weak_iq": iq,
        "raw_labels": np.asarray([row["class_id"] for row in rows], dtype=np.int64),
        "domain_labels": np.zeros(row_count, dtype=np.int64),
        "tx_ids": np.asarray(tx_ids),
        "rx_ids": np.asarray(["1-1"] * row_count),
        "day_ids": np.asarray(["2021_03_01"] * row_count),
        "eq_ids": np.asarray(["1"] * row_count),
        "sig_ids": np.asarray([str(index) for index in range(row_count)]),
        "source_dataset_sha256": np.asarray(dataset_hashes),
        "source_record_indices": np.asarray(source_indices, dtype=np.int64),
        "dataset_role": np.asarray(dataset_roles),
        "channel_views": np.asarray(["rx_base"] * row_count),
        "sat_scenarios": np.asarray([scenario] * row_count),
        "satellite_seeds": np.asarray([satellite_seed] * row_count, dtype=np.int64),
        "overlay_applied": np.ones(row_count, dtype=bool),
        "sample_ids": np.asarray(canonical_ids),
        "post_channel_iq_sha256": np.asarray(iq_hashes),
        "overlay_ids": np.asarray(overlay_ids),
        "canonical_physical_sample_ids": np.asarray(canonical_ids),
        "split_roles": np.asarray(split_roles),
        "split_ranks": np.asarray(split_ranks, dtype=np.int64),
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
    }


def _write_verified_cache_set(
    tmp_path: Path,
    *,
    short_last_class: bool = False,
    duplicate_support_id: bool = False,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache_paths: dict[str, str] = {}
    cache_hashes: dict[str, str] = {}
    ids_by_scenario: dict[str, list[str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        path = tmp_path / f"cache_{scenario}.npz"
        payload = _cache_payload(
            scenario,
            short_last_class=short_last_class,
            duplicate_support_id=duplicate_support_id,
        )
        np.savez(path, **payload)
        cache_paths[scenario] = path.name
        cache_hashes[scenario] = sha256_file(path)
        ids_by_scenario[scenario] = payload["sample_ids"].astype(str).tolist()
    set_manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "cache_set_id": "canonical-k20-real-shaped-fixture",
        "cache_scope": "stage2_canonical_registered",
        "protocol_schema": "p2_min_v1",
        "profile_id": "SRC5_MAXP2",
        "query_policy": "BALANCED_4DAY_CORE",
        "k": 20,
        "capsule_id": CAPSULE_ID,
        "split_id": SPLIT_ID,
        "registered_tx_ids": list(REGISTERED_TX_IDS),
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "target_channel_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "output_roles": ["target_old", "target_new"],
        "cache_npz_by_scenario": cache_paths,
        "cache_sha256_by_scenario": cache_hashes,
        **_single_observation_contract(),
        "physical_sample_ids_sha256_by_scenario": {
            scenario: ids_sha256(ids_by_scenario[scenario])
            for scenario in FORMAL_LEO_WEAK_SCENARIOS
        },
        "physical_sample_scenario_assignment_sha256": canonical_json_sha256(
            ids_by_scenario
        ),
    }
    manifest_path = tmp_path / "cache_set.json"
    manifest_path.write_text(json.dumps(set_manifest), encoding="utf-8")
    if not duplicate_support_id:
        load_verified_leo_weak_cache_set(
            manifest_path,
            expected_scope="stage2_canonical_registered",
            allowed_roles={"target_old", "target_new"},
        )
    return manifest_path


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": CAPSULE_ID,
        "split_id": SPLIT_ID,
        "checkpoint_path": CHECKPOINT,
        "support_path": str(tmp_path / "support_leo_clear_weak_rx1-1_k20.npz"),
        "prototype_path": str(
            tmp_path / "prototypes_leo_clear_weak_rx1-1_k20.npz"
        ),
        "candidate": "freq_f3_proj",
        "steps": 1,
        "learning_rate": 0.0005,
        "seed": 20260828,
        "k_shot": 20,
    }


class _ToyCheckpoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.eval()


def _patch_prototype_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prototype_bridge,
        "_load_frozen_checkpoint",
        lambda _path, *, device: _ToyCheckpoint().to(device),
    )
    monkeypatch.setattr(
        prototype_bridge,
        "_identity_features",
        lambda _model, rows: torch.stack(
            (rows[:, 0, 0], rows[:, 0, 1], rows[:, 1, 0] + 1.0), dim=1
        ),
    )


def test_verified_cache_to_export_to_prototype_bridge_closes_support_only_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_set = _write_verified_cache_set(tmp_path)
    config = _config(tmp_path)
    support_pool = tmp_path / "support_pool_leo_clear_weak_rx1-1_k20.npz"

    stage_audit = subject.stage_support_pool(
        config,
        cache_set_path=cache_set,
        output_path=support_pool,
        scene="leo_clear_weak",
        receiver="1-1",
    )

    with np.load(support_pool, allow_pickle=False) as staged:
        assert tuple(staged.files) == SUPPORT_NPZ_MEMBERS
        assert not any(
            "query" in name or "truth" in name or name == "role"
            for name in staged.files
        )
        assert staged["support_pool_leo_weak_iq"].shape[0] == 520
        assert staged["support_pool_class_indices"].tolist() == [
            class_id for class_id in range(26) for _ in range(20)
        ]
        assert staged["support_pool_rank_within_class"].tolist() == list(
            range(20)
        ) * 26
        tokens = staged["support_pool_tokens"].astype(str).tolist()
        assert len(tokens) == len(set(tokens)) == 520
        assert all("|support|" in token for token in tokens)
        support_manifest = json.loads(str(staged["manifest_json"].item()))
        assert support_manifest["schema"] == SUPPORT_SCHEMA
        assert support_manifest["capsule_id"] == CAPSULE_ID
        assert support_manifest["split_id"] == SPLIT_ID

    export_audit_path = tmp_path / "support_leo_clear_weak_rx1-1_k20.audit.json"
    export_audit = export_target_row(
        support_input=support_pool,
        support_output=config["support_path"],
        audit_output=export_audit_path,
        k_shot=20,
    )
    _patch_prototype_embedding(monkeypatch)
    prototype_audit = prototype_bridge.build_support_prototypes(
        config,
        support_audit_path=export_audit_path,
        scene="leo_clear_weak",
        receiver="1-1",
        device="cpu",
    )

    assert stage_audit["support_rows"] == 520
    assert stage_audit["canonical_physical_ids_unique"] is True
    assert stage_audit["query_artifact_opened"] is False
    assert export_audit["query_input_opened"] is False
    assert prototype_audit["target_new_class_ids"] == list(range(6, 26))
    with np.load(config["prototype_path"], allow_pickle=False) as artifact:
        assert set(artifact.files) == {"prototypes", "class_ids"}
        assert artifact["class_ids"].tolist() == list(range(26))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_schema", "wrong"),
        ("capsule_id", "f" * 64),
        ("split_id", "e" * 64),
        ("k", 19),
        ("registered_tx_ids", list(reversed(REGISTERED_TX_IDS))),
    ],
)
def test_parent_protocol_handle_k_and_registered_order_drift_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    cache_set = _write_verified_cache_set(tmp_path)
    manifest = json.loads(cache_set.read_text(encoding="utf-8"))
    manifest[field] = value
    cache_set.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        subject.stage_support_pool(
            _config(tmp_path),
            cache_set_path=cache_set,
            output_path=tmp_path / "support_pool_leo_clear_weak_rx1-1_k20.npz",
            scene="leo_clear_weak",
            receiver="1-1",
        )


def test_wrong_scene_receiver_or_incomplete_class_k_fails_without_output(
    tmp_path: Path,
) -> None:
    cache_set = _write_verified_cache_set(tmp_path)
    config = _config(tmp_path)
    output = tmp_path / "support_pool_leo_clear_weak_rx1-1_k20.npz"
    with pytest.raises(ValueError, match="scene"):
        subject.stage_support_pool(
            config,
            cache_set_path=cache_set,
            output_path=output,
            scene="leo_rain_weak",
            receiver="1-1",
        )
    with pytest.raises(ValueError, match="receiver"):
        subject.stage_support_pool(
            config,
            cache_set_path=cache_set,
            output_path=output,
            scene="leo_clear_weak",
            receiver="14-7",
        )
    short_cache_set = _write_verified_cache_set(
        tmp_path / "short",
        short_last_class=True,
    )
    with pytest.raises(ValueError, match="K-shot|rank"):
        subject.stage_support_pool(
            config,
            cache_set_path=short_cache_set,
            output_path=output,
            scene="leo_clear_weak",
            receiver="1-1",
        )
    assert not output.exists()


def test_duplicate_canonical_physical_id_and_preexisting_output_fail_closed(
    tmp_path: Path,
) -> None:
    duplicate_cache_set = _write_verified_cache_set(
        tmp_path / "duplicate",
        duplicate_support_id=True,
    )
    output = tmp_path / "support_pool_leo_clear_weak_rx1-1_k20.npz"
    with pytest.raises(ValueError, match="unique"):
        subject.stage_support_pool(
            _config(tmp_path),
            cache_set_path=duplicate_cache_set,
            output_path=output,
            scene="leo_clear_weak",
            receiver="1-1",
        )

    cache_set = _write_verified_cache_set(tmp_path / "valid")
    output.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="already exists"):
        subject.stage_support_pool(
            _config(tmp_path),
            cache_set_path=cache_set,
            output_path=output,
            scene="leo_clear_weak",
            receiver="1-1",
        )
    assert output.read_bytes() == b"preserve"


def test_staging_api_and_cli_cannot_accept_query_or_truth_artifacts() -> None:
    parameters = set(inspect.signature(subject.stage_support_pool).parameters)
    parser_destinations = {
        action.dest for action in subject._parser()._actions  # noqa: SLF001
    }
    forbidden = {
        "query",
        "query_path",
        "query_artifact",
        "query_truth",
        "truth_path",
        "query_role",
    }
    assert not parameters & forbidden
    assert not parser_destinations & forbidden
