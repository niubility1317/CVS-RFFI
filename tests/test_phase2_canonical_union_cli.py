import csv
import hashlib
import importlib
import json
import pickle
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvsrffi.leo_weak_cache import (
    PHASE2_SAMPLE_VIEW_POLICY,
    load_verified_leo_weak_cache_set,
)
from cvsrffi.phase2_canonical_summary import summarize_scored_rows
from cvsrffi.stage2_metric_scorer import score_prediction_arrays
from scripts.audit_wisig_canonical_union import main as audit_main
from scripts import build_cvs_leo_weak_iq_cache as cache_builder
from scripts import build_cvs_stage2_predictor_bundle as predictor_builder


FORMAL_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _write_asset(path: Path, value: float) -> Path:
    samples = np.full((1, 4, 2), value, dtype=np.float32)
    payload = {
        "data": [[[[[samples]]]]],
        "tx_list": ["tx-A"],
        "rx_list": ["rx-X"],
        "capture_date_list": ["day-0"],
        "equalized_list": [1],
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_cli_four_assets_writes_auditable_schemas_without_iq_bytes(tmp_path: Path):
    assets = {
        name: _write_asset(tmp_path / f"{name}.pkl", 7.25)
        for name in ("ManySig", "SingleDay", "ManyRx", "ManyTx")
    }
    sqlite_path = tmp_path / "canonical.sqlite"
    summary_path = tmp_path / "summary.json"
    coverage_path = tmp_path / "coverage.csv"
    conflicts_path = tmp_path / "conflicts.csv"
    argv = [
        "--sqlite-out",
        str(sqlite_path),
        "--summary-json",
        str(summary_path),
        "--coverage-csv",
        str(coverage_path),
        "--conflicts-csv",
        str(conflicts_path),
        "--equalized",
        "1",
    ]
    for name, path in assets.items():
        argv.extend(("--asset", f"{name}={path}"))

    assert audit_main(argv) == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with coverage_path.open(encoding="utf-8", newline="") as handle:
        coverage_rows = list(csv.DictReader(handle))
    with conflicts_path.open(encoding="utf-8", newline="") as handle:
        conflict_rows = list(csv.DictReader(handle))

    assert summary == {
        "canonical_record_count": 1,
        "conflict_count": 0,
        "eligible_record_count": 1,
        "equalized": "1",
        "merged_duplicate_count": 3,
        "protocol_schema": "p2_min_v1",
        "source_record_count": 4,
    }
    assert list(coverage_rows[0]) == ["tx_id", "rx_id", "day_id", "record_count", "asset_count"]
    assert coverage_rows[0] == {
        "tx_id": "tx-A",
        "rx_id": "rx-X",
        "day_id": "day-0",
        "record_count": "1",
        "asset_count": "4",
    }
    assert conflict_rows == []
    text_outputs = "\n".join(
        path.read_text(encoding="utf-8") for path in (summary_path, coverage_path, conflicts_path)
    )
    assert "7.25" not in text_outputs
    assert "[[" not in text_outputs


def test_cli_rejects_duplicate_assets_and_existing_output_paths(tmp_path: Path):
    asset = _write_asset(tmp_path / "ManySig.pkl", 7.25)
    sqlite_path = tmp_path / "canonical.sqlite"
    summary_path = tmp_path / "summary.json"
    coverage_path = tmp_path / "coverage.csv"
    conflicts_path = tmp_path / "conflicts.csv"
    base_argv = [
        "--sqlite-out",
        str(sqlite_path),
        "--summary-json",
        str(summary_path),
        "--coverage-csv",
        str(coverage_path),
        "--conflicts-csv",
        str(conflicts_path),
    ]

    assert audit_main(
        base_argv + ["--asset", f"ManySig={asset}", "--asset", f"ManySig={asset}"]
    ) == 2
    assert not sqlite_path.exists()
    summary_path.write_text("already exists", encoding="utf-8")
    assert audit_main(base_argv + ["--asset", f"ManySig={asset}"]) == 2


def _build_split_main():
    try:
        return importlib.import_module("scripts.build_phase2_canonical_splits").main
    except ModuleNotFoundError as error:
        pytest.fail(f"Task 4 split builder CLI is missing: {error}")


def _split_profile_payload():
    return {
        "schema": "cvs.phase2.canonical_union_profile.v1",
        "protocol_schema": "p2_min_v1",
        "source_profile_id": "TEST_MAXQ_BAL4D",
        "source_receivers": ["source-rx"],
        "receiver_tiers": {
            "dense": ["rx-a"],
            "single_day": [],
            "many_tx": [],
        },
        "old_tx_ids": [f"old-{index:02d}" for index in range(6)],
        "new_tx_candidates": [f"tx-{index:02d}" for index in range(22)],
        "new_class_sizes": [5, 10, 20],
        "k_values": [1, 5, 10, 20],
        "k_max": 20,
        "scenarios": list(FORMAL_SCENARIOS),
        "query_policies": ["MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE"],
    }


def _write_split_inventory(path: Path, *, sufficient: bool = True) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE canonical_records (
          physical_sample_id TEXT PRIMARY KEY,
          tx_id TEXT NOT NULL,
          rx_id TEXT NOT NULL,
          day_id TEXT NOT NULL,
          eq_id TEXT NOT NULL,
          sig_id TEXT NOT NULL,
          iq_sha256 TEXT NOT NULL,
          preferred_asset TEXT NOT NULL,
          preferred_source_record_index INTEGER NOT NULL,
          eligible INTEGER NOT NULL CHECK (eligible IN (0,1))
        )
        """
    )
    if sufficient:
        registered = tuple(f"old-{index:02d}" for index in range(6)) + tuple(
            f"tx-{index:02d}" for index in range(20)
        )
        source_index = 0
        rows = []
        for tx_id in registered:
            for day_index in range(4):
                for sample_index in range(63):
                    sample_id = f"cli-{tx_id}-d{day_index}-{sample_index:02d}"
                    rows.append(
                        (
                            sample_id,
                            tx_id,
                            "rx-a",
                            f"day-{day_index}",
                            "1",
                            str(sample_index),
                            "IQ_SECRET_DIGEST",
                            "ManyTx",
                            source_index,
                            1,
                        )
                    )
                    source_index += 1
        connection.executemany(
            """
            INSERT INTO canonical_records (
              physical_sample_id, tx_id, rx_id, day_id, eq_id, sig_id, iq_sha256,
              preferred_asset, preferred_source_record_index, eligible
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    connection.commit()
    connection.close()
    return path


def _write_split_profile(path: Path) -> Path:
    path.write_text(
        json.dumps(_split_profile_payload(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_split_builder_cli_writes_exact_tree_deterministically_without_leakage(
    tmp_path: Path,
):
    inventory = _write_split_inventory(tmp_path / "canonical.sqlite")
    profile = _write_split_profile(tmp_path / "profile.json")
    out_root = tmp_path / "splits"
    before_inventory = hashlib.sha256(inventory.read_bytes()).hexdigest()

    assert _build_split_main()(
        [
            "--inventory",
            str(inventory),
            "--profile",
            str(profile),
            "--out-root",
            str(out_root),
            "--seed",
            "713101",
        ]
    ) == 0

    expected_files = {
        "class_selection.json",
        *{
            f"{policy}/k{k}.json"
            for policy in ("MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE")
            for k in (1, 5, 10, 20)
        },
    }
    actual_files = {
        path.relative_to(out_root).as_posix() for path in out_root.rglob("*") if path.is_file()
    }
    assert actual_files == expected_files
    assert hashlib.sha256(inventory.read_bytes()).hexdigest() == before_inventory

    selection = json.loads((out_root / "class_selection.json").read_text(encoding="utf-8"))
    assert selection["protocol_schema"] == "p2_min_v1"
    assert selection["profile_id"] == "TEST_MAXQ_BAL4D"
    assert selection["seed"] == 713101
    assert selection["Y_new5"] == selection["Y_new10"][:5]
    assert selection["Y_new10"] == selection["Y_new20"][:10]
    assert len(selection["Y_new20"]) == 20

    balanced_query_ids = []
    maxq_eligible_counts = []
    for policy in ("MAXQ_ALL_UNIQUE", "BALANCED_4DAY_CORE"):
        for k in (1, 5, 10, 20):
            manifest_path = out_root / policy / f"k{k}.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            assert payload["protocol_schema"] == "p2_min_v1"
            assert payload["profile_id"] == "TEST_MAXQ_BAL4D"
            assert payload["query_policy"] == policy
            assert payload["k"] == k
            assert len(payload["registered_tx_ids"]) == 26
            assert payload["eligible_receivers"] == ["rx-a"]
            assert payload["counts"]["row_count"] == len(payload["rows"])
            assert payload["counts"]["support_count"] == sum(
                row["role"] == "support" for row in payload["rows"]
            )
            assert payload["counts"]["query_count"] == sum(
                row["role"] == "query" for row in payload["rows"]
            )
            support_rows = [
                row for row in payload["rows"] if row["role"] == "support"
            ]
            query_rows = [row for row in payload["rows"] if row["role"] == "query"]
            assert support_rows
            assert query_rows
            assert all(
                set(row)
                == {
                    "physical_sample_id",
                    "source_asset",
                    "source_record_index",
                    "tx_id",
                    "rx_id",
                    "day_id",
                    "scene",
                    "role",
                    "rank",
                }
                for row in support_rows
            )
            assert all(
                set(row)
                == {
                    "physical_sample_id",
                    "source_asset",
                    "source_record_index",
                    "rx_id",
                    "day_id",
                    "scene",
                    "role",
                    "rank",
                }
                for row in query_rows
            )
            query_truth_aliases = {
                "tx_id",
                "true_tx_id",
                "tx_label",
                "class_id",
                "class_label",
                "label",
                "truth",
                "query_truth",
            }
            assert all(not query_truth_aliases.intersection(row) for row in query_rows)
            serialized = manifest_path.read_text(encoding="utf-8").lower()
            for forbidden in (
                "iq_secret_digest",
                "iq_sha256",
                "dataset_path",
                "query_truth",
                "prediction",
                "class_quota",
            ):
                assert forbidden not in serialized
            if policy == "MAXQ_ALL_UNIQUE":
                maxq_eligible_counts.append(payload["counts"]["eligible_count"])
                assert payload["counts"]["eligible_count"] == payload["counts"]["row_count"]
            else:
                balanced_query_ids.append(
                    {
                        row["physical_sample_id"]
                        for row in payload["rows"]
                        if row["role"] == "query"
                    }
                )
    assert len(set(maxq_eligible_counts)) == 1
    assert balanced_query_ids[0] == balanced_query_ids[1] == balanced_query_ids[2] == balanced_query_ids[3]


def test_split_builder_cli_rejects_existing_root_without_mutating_inventory(tmp_path: Path):
    inventory = _write_split_inventory(tmp_path / "canonical.sqlite")
    profile = _write_split_profile(tmp_path / "profile.json")
    out_root = tmp_path / "existing"
    out_root.mkdir()
    marker = out_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    before_inventory = inventory.read_bytes()

    assert _build_split_main()(
        [
            "--inventory",
            str(inventory),
            "--profile",
            str(profile),
            "--out-root",
            str(out_root),
            "--seed",
            "713101",
        ]
    ) == 2
    assert marker.read_text(encoding="utf-8") == "keep"
    assert inventory.read_bytes() == before_inventory


def test_split_builder_cli_builds_all_manifests_before_creating_root(tmp_path: Path):
    inventory = _write_split_inventory(tmp_path / "empty.sqlite", sufficient=False)
    profile = _write_split_profile(tmp_path / "profile.json")
    out_root = tmp_path / "must-not-exist"

    assert _build_split_main()(
        [
            "--inventory",
            str(inventory),
            "--profile",
            str(profile),
            "--out-root",
            str(out_root),
            "--seed",
            "713101",
        ]
    ) == 2
    assert not out_root.exists()


TASK8_OLD_TX_IDS = tuple(f"old-{index:02d}" for index in range(6))
TASK8_NEW_TX_CANDIDATES = tuple(f"tx-{index:02d}" for index in range(22))
TASK8_REGISTERED_TX_IDS = TASK8_OLD_TX_IDS + TASK8_NEW_TX_CANDIDATES[:20]
TASK8_RX_ID = "rx-a"
TASK8_DAY_IDS = tuple(f"day-{index}" for index in range(4))
TASK8_SCENE_SEED = 713101


def _task8_asset_payload() -> dict[str, object]:
    data = []
    for tx_index, _tx_id in enumerate(
        TASK8_OLD_TX_IDS + TASK8_NEW_TX_CANDIDATES
    ):
        day_rows = []
        for day_index, _day_id in enumerate(TASK8_DAY_IDS):
            samples = np.empty((63, 8, 2), dtype=np.float32)
            template = np.arange(16, dtype=np.float32).reshape(8, 2)
            for signal_index in range(63):
                offset = tx_index * 100000 + day_index * 1000 + signal_index * 20
                samples[signal_index] = template + np.float32(offset)
            day_rows.append([samples])
        data.append([day_rows])
    return {
        "data": data,
        "tx_list": list(TASK8_OLD_TX_IDS + TASK8_NEW_TX_CANDIDATES),
        "rx_list": [TASK8_RX_ID],
        "capture_date_list": list(TASK8_DAY_IDS),
        "equalized_list": [1],
    }


def _write_task8_overlapping_assets(tmp_path: Path) -> dict[str, Path]:
    payload = _task8_asset_payload()
    assets = {}
    for name in ("ManySig", "ManyTx", "ManyRx", "SingleDay"):
        assets[name] = _write_pickle_payload(tmp_path / f"{name}.pkl", payload)
    return assets


def _write_pickle_payload(path: Path, payload: dict[str, object]) -> Path:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def _task8_cache_spec() -> dict[str, object]:
    return {
        "schema": "cvs_leo_weak_iq_cache_build_spec_v3",
        "protocol_schema": "p2_min_v1",
        "cache_scope": "stage2_canonical_registered",
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "star_ground_channel_impl": "simplified_leo_residual",
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
            "disjoint_preoverlay_tx_day_stratified_v1"
        ),
        "canonical_inventory": "canonical.sqlite",
        "split_manifest": "splits/MAXQ_ALL_UNIQUE/k1.json",
        "satellite_seed_by_scenario": {
            scenario: TASK8_SCENE_SEED + index
            for index, scenario in enumerate(FORMAL_SCENARIOS)
        },
        "out_npz_by_scenario": {
            scenario: f"cache/{scenario}.npz" for scenario in FORMAL_SCENARIOS
        },
        "out_manifest": "cache/cache_set.json",
        "wisig_equalized": "1",
        "wisig_out_len": 8,
        "wisig_domain": "rx_day",
        "batch_size": 512,
    }


def _task8_predictor_args(
    tmp_path: Path,
    cache_set_manifest: Path,
) -> SimpleNamespace:
    artifacts = tmp_path / "predictor-inputs"
    artifacts.mkdir()
    files = {}
    for name in ("candidate", "checkpoint", "adapter", "head"):
        path = artifacts / f"{name}.bin"
        path.write_bytes(f"task8-{name}".encode("ascii"))
        files[name] = path
    tta_policy = artifacts / "tta.json"
    tta_policy.write_text(
        json.dumps({"base_views": 1, "max_views": 5}),
        encoding="utf-8",
        newline="\n",
    )
    new_pool = list(TASK8_NEW_TX_CANDIDATES[:20])
    draw_order = np.random.default_rng(713131).permutation(len(new_pool))
    drawn_new = [new_pool[int(index)] for index in draw_order]
    return SimpleNamespace(
        target_cache_set=cache_set_manifest,
        predictor_out_root=tmp_path / "predictor",
        scorer_out_root=tmp_path / "scorer",
        detached_seal_path=None,
        stage="stage2c",
        receiver=TASK8_RX_ID,
        seed=TASK8_SCENE_SEED,
        support_seed=713111,
        query_seed=713121,
        new_class_draw_seed=713131,
        old_class_labels=",".join(TASK8_OLD_TX_IDS),
        new_class_labels=",".join(drawn_new),
        new_class_pool_labels=",".join(new_pool),
        stage2b_reference_new_class_labels="",
        new_class_count=len(new_pool),
        support_pool_max_k=1,
        query_per_tx=0,
        query_policy="manifest_all",
        candidate_lock=files["candidate"],
        checkpoint=files["checkpoint"],
        adapter=files["adapter"],
        head_artifact=files["head"],
        tta_policy_json=tta_policy,
    )


def _assert_task8_predictor_queries_are_opaque(
    predictor_root: Path,
    arrays_by_scenario: dict[str, dict[str, np.ndarray]],
) -> None:
    forbidden_values = {
        "target_old",
        "target_new",
        *TASK8_REGISTERED_TX_IDS,
        *{
            str(value)
            for arrays in arrays_by_scenario.values()
            for value in arrays["canonical_physical_sample_ids"].tolist()
        },
    }
    for scenario in FORMAL_SCENARIOS:
        with np.load(
            predictor_root / f"query_{scenario}.npz",
            allow_pickle=False,
        ) as archive:
            assert tuple(archive.files) == predictor_builder.QUERY_NPZ_MEMBERS
            manifest = json.loads(str(archive["manifest_json"]))
            assert manifest["query_truth_included"] is False
            assert manifest["query_role_included"] is False
            assert manifest["query_true_batch_class_count_included"] is False
            textual_values = []
            for member in archive.files:
                values = np.asarray(archive[member])
                if values.dtype.kind in {"S", "U"}:
                    textual_values.extend(
                        str(value) for value in values.reshape(-1).tolist()
                    )
            predictor_text = "\n".join([*archive.files, *textual_values])
            assert all(value not in predictor_text for value in forbidden_values)


def _task8_truth_blind_fake_prediction_arrays(
    predictor_root: Path,
) -> dict[str, np.ndarray]:
    package_manifest = json.loads(
        (predictor_root / "package_manifest.json").read_text(encoding="utf-8")
    )
    registered_classes = sorted(
        package_manifest["registered_classes"],
        key=lambda row: int(row["class_index"]),
    )
    fixed_prediction = str(registered_classes[0]["class_handle"])
    query_tokens = []
    scenarios = []
    for scenario in FORMAL_SCENARIOS:
        with np.load(
            predictor_root / f"query_{scenario}.npz",
            allow_pickle=False,
        ) as archive:
            current_tokens = np.asarray(archive["query_tokens"]).astype(str)
        query_tokens.extend(current_tokens.tolist())
        scenarios.extend([scenario] * len(current_tokens))
    fixed_predictions = np.asarray([fixed_prediction] * len(query_tokens))
    return {
        "query_tokens": np.asarray(query_tokens),
        "scenarios": np.asarray(scenarios),
        "candidate_after": fixed_predictions.copy(),
        "candidate_before": fixed_predictions.copy(),
        "identity_after": fixed_predictions.copy(),
        "identity_before": fixed_predictions.copy(),
        "direct": fixed_predictions.copy(),
        "shared_view_counts": np.ones(len(query_tokens), dtype=np.int64),
    }


def _run_task8_synthetic_chain(tmp_path: Path):
    assets = _write_task8_overlapping_assets(tmp_path)
    inventory = tmp_path / "canonical.sqlite"
    summary_path = tmp_path / "inventory-summary.json"
    coverage_path = tmp_path / "coverage.csv"
    conflicts_path = tmp_path / "conflicts.csv"
    audit_argv = [
        "--sqlite-out",
        str(inventory),
        "--summary-json",
        str(summary_path),
        "--coverage-csv",
        str(coverage_path),
        "--conflicts-csv",
        str(conflicts_path),
        "--equalized",
        "1",
    ]
    for name, path in assets.items():
        audit_argv.extend(("--asset", f"{name}={path}"))
    assert audit_main(audit_argv) == 0
    inventory_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    profile_path = _write_split_profile(tmp_path / "profile.json")
    split_root = tmp_path / "splits"
    assert _build_split_main()(
        [
            "--inventory",
            str(inventory),
            "--profile",
            str(profile_path),
            "--out-root",
            str(split_root),
            "--seed",
            str(TASK8_SCENE_SEED),
        ]
    ) == 0
    maxq_k1 = json.loads(
        (split_root / "MAXQ_ALL_UNIQUE" / "k1.json").read_text(
            encoding="utf-8"
        )
    )
    bal4d_k1 = json.loads(
        (split_root / "BALANCED_4DAY_CORE" / "k1.json").read_text(
            encoding="utf-8"
        )
    )

    cache_spec_path = tmp_path / "cache-spec.json"
    cache_spec_path.write_text(
        json.dumps(_task8_cache_spec(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    cache_result = cache_builder.build_cache_set(
        cache_spec_path,
        device=torch.device("cpu"),
    )
    cache_manifest_path = Path(cache_result["cache_set_manifest"])
    arrays_by_scenario, cache_manifest, cache_audit = (
        load_verified_leo_weak_cache_set(
            cache_manifest_path,
            expected_scope="stage2_canonical_registered",
            allowed_roles={"target_old", "target_new"},
        )
    )
    assert cache_manifest["capsule_id"] == maxq_k1["capsule_id"]
    assert cache_manifest["split_id"] == maxq_k1["split_id"]
    assert cache_audit["phase2_single_observation_compliant"] is True

    predictor_args = _task8_predictor_args(tmp_path, cache_manifest_path)
    predictor_result = predictor_builder.build(
        predictor_args,
        token_secret=b"t" * 32,
    )
    _assert_task8_predictor_queries_are_opaque(
        predictor_args.predictor_out_root,
        arrays_by_scenario,
    )

    prediction_arrays = _task8_truth_blind_fake_prediction_arrays(
        predictor_args.predictor_out_root
    )

    truth = json.loads(
        (predictor_args.scorer_out_root / "truth_sidecar.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_lock_sha256 = hashlib.sha256(
        Path(predictor_args.candidate_lock).read_bytes()
    ).hexdigest()
    scenario_rows, scored_predictions = score_prediction_arrays(
        binding={
            "row_id": "task8-synthetic-k1",
            "stage": "stage2c",
            "receiver": TASK8_RX_ID,
            "scenarios": list(FORMAL_SCENARIOS),
            "k_shot": 1,
            "candidate_lock_sha256": candidate_lock_sha256,
            "predictor_package_root_sha256": predictor_result[
                "predictor_package_root_sha256"
            ],
        },
        arrays=prediction_arrays,
        truth=truth,
    )
    canonical_summary = summarize_scored_rows(scored_predictions)

    cache_row_counts = {
        scenario: len(arrays["sample_ids"])
        for scenario, arrays in arrays_by_scenario.items()
    }
    cache_support_counts = {
        scenario: int(np.sum(arrays["split_roles"] == "support"))
        for scenario, arrays in arrays_by_scenario.items()
    }
    cache_query_counts = {
        scenario: int(np.sum(arrays["split_roles"] == "query"))
        for scenario, arrays in arrays_by_scenario.items()
    }
    return {
        "asset_count": len(assets),
        "source_record_count": inventory_summary["source_record_count"],
        "canonical_record_count": inventory_summary["canonical_record_count"],
        "eligible_record_count": inventory_summary["eligible_record_count"],
        "merged_duplicate_count": inventory_summary["merged_duplicate_count"],
        "conflict_count": inventory_summary["conflict_count"],
        "maxq_k1_row_count": maxq_k1["counts"]["row_count"],
        "maxq_k1_support_count": maxq_k1["counts"]["support_count"],
        "maxq_k1_query_count": maxq_k1["counts"]["query_count"],
        "bal4d_k1_row_count": bal4d_k1["counts"]["row_count"],
        "bal4d_k1_support_count": bal4d_k1["counts"]["support_count"],
        "bal4d_k1_query_count": bal4d_k1["counts"]["query_count"],
        "cache_row_count_by_scenario": cache_row_counts,
        "cache_support_count_by_scenario": cache_support_counts,
        "cache_query_count_by_scenario": cache_query_counts,
        "predictor_registered_class_count": predictor_result[
            "registered_class_count"
        ],
        "predictor_support_pool_count": predictor_result["support_pool_count"],
        "predictor_query_count_by_scenario": predictor_result[
            "query_count_by_scenario"
        ],
        "truth_row_count": len(truth["rows"]),
        "scenario_metric_row_count": len(scenario_rows),
        "scored_prediction_count": len(scored_predictions),
        "summary_sample_count": canonical_summary["sample_count"],
        "summary_correct_count": canonical_summary["correct_count"],
        "summary_class_group_count": canonical_summary["class_group_count"],
        "summary_receiver_group_count": canonical_summary[
            "receiver_group_count"
        ],
        "summary_day_group_count": canonical_summary["day_group_count"],
        "summary_scene_group_count": canonical_summary["scene_group_count"],
        "summary_observed_cell_count": canonical_summary["observed_cell_count"],
        "summary_sample_micro_accuracy": canonical_summary[
            "sample_micro_accuracy"
        ],
    }


def test_task8_four_asset_chain_closes_truth_last_summary_with_exact_counts(
    tmp_path: Path,
):
    observed = _run_task8_synthetic_chain(tmp_path)

    assert observed == {
        "asset_count": 4,
        "source_record_count": 28224,
        "canonical_record_count": 7056,
        "eligible_record_count": 7056,
        "merged_duplicate_count": 21168,
        "conflict_count": 0,
        "maxq_k1_row_count": 6552,
        "maxq_k1_support_count": 78,
        "maxq_k1_query_count": 6474,
        "bal4d_k1_row_count": 4134,
        "bal4d_k1_support_count": 78,
        "bal4d_k1_query_count": 4056,
        "cache_row_count_by_scenario": {
            "leo_clear_weak": 2184,
            "leo_low_elev_weak": 2184,
            "leo_rain_weak": 2184,
        },
        "cache_support_count_by_scenario": {
            "leo_clear_weak": 26,
            "leo_low_elev_weak": 26,
            "leo_rain_weak": 26,
        },
        "cache_query_count_by_scenario": {
            "leo_clear_weak": 2158,
            "leo_low_elev_weak": 2158,
            "leo_rain_weak": 2158,
        },
        "predictor_registered_class_count": 26,
        "predictor_support_pool_count": 26,
        "predictor_query_count_by_scenario": {
            "leo_clear_weak": 2158,
            "leo_low_elev_weak": 2158,
            "leo_rain_weak": 2158,
        },
        "truth_row_count": 6474,
        "scenario_metric_row_count": 3,
        "scored_prediction_count": 6474,
        "summary_sample_count": 6474,
        "summary_correct_count": 249,
        "summary_class_group_count": 26,
        "summary_receiver_group_count": 1,
        "summary_day_group_count": 4,
        "summary_scene_group_count": 3,
        "summary_observed_cell_count": 312,
        "summary_sample_micro_accuracy": 1.0 / 26.0,
    }


@pytest.mark.parametrize("forbidden_member", ("query_truth", "query_role"))
def test_task8_predictor_npz_rejects_query_truth_or_role_member(
    tmp_path: Path,
    forbidden_member: str,
):
    predictor_root = tmp_path / "predictor"
    predictor_root.mkdir()
    np.savez(
        predictor_root / "query_leo_clear_weak.npz",
        **{forbidden_member: np.asarray(["forbidden"])},
    )

    with pytest.raises(ValueError, match="forbidden truth/role token"):
        predictor_builder._reject_predictor_truth_leaks(
            predictor_root,
            ("query_truth", "query_role"),
        )
