from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
CODE_ROOT = ROOT / "code"
for value in (str(SCRIPT_ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_full_ablation_stage2_binding_registry as builder  # noqa: E402
from cvsrffi.full_ablation_spec import (  # noqa: E402
    PHASE2_T1_ARMS,
    SeedBundle,
    build_phase2_rows,
)


COMMIT = "f" * 40
CANDIDATE = "a" * 64


def _rows():
    arms = tuple(
        arm
        for arm in PHASE2_T1_ARMS
        if arm.ablation_id in {"P2-FULL", "P2-F3"}
    )
    return build_phase2_rows(
        stage="screening",
        arms=arms,
        seed_bundles=(
            SeedBundle(7282101, 7282201, 7282301),
            SeedBundle(7282102, 7282202, 7282302),
            SeedBundle(7282103, 7282203, 7282303),
        ),
        class_draw_seeds=(7282401,),
        git_commit=COMMIT,
    )


def _entry(identity: tuple, root: Path) -> dict:
    (
        stage_scope,
        receiver,
        method_seed,
        support_seed,
        query_seed,
        new_class_draw_seed,
        k_shot,
        new_class_count,
    ) = identity
    suffix = hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
    return {
        "stage_scope": stage_scope,
        "receiver": receiver,
        "method_seed": method_seed,
        "support_seed": support_seed,
        "query_seed": query_seed,
        "new_class_draw_seed": new_class_draw_seed,
        "k_shot": k_shot,
        "new_class_count": new_class_count,
        "feature_cache_payload": str((root / f"{suffix}.npz").resolve()),
        "feature_cache_manifest": str(
            (root / f"{suffix}.manifest.json").resolve()
        ),
        "predictor_package_root": str(
            (root / f"{suffix}.package").resolve()
        ),
        "predictor_detached_seal": str(
            (root / f"{suffix}.seal.json").resolve()
        ),
        "predictor_detached_seal_sha256": "b" * 64,
        "scoring_manifest": str(
            (root / f"{suffix}.scoring.json").resolve()
        ),
    }


def _fake_validate(raw, *, candidate_lock_sha256):
    assert candidate_lock_sha256 == CANDIDATE
    return {
        **raw,
        "_identity": builder._entry_identity(raw),
        "_feature_cache_payload_sha256": "1" * 64,
        "_feature_cache_manifest_sha256": "2" * 64,
        "_phase2_data_status": "VALIDATED_ONCE",
        "_capsule_id": "capsule-current-launch",
        "_split_id": "split-current-launch",
        "_phase1_bundle_sha256": "3" * 64,
        "_phase1_prototype_sha256": "4" * 64,
        "_scoring_manifest_sha256": "5" * 64,
    }


def test_registry_covers_every_row_and_reuses_physical_alias_binding(
    tmp_path: Path, monkeypatch
) -> None:
    rows = _rows()
    identities = sorted(
        {builder._row_identity(row) for row in rows},
        key=repr,
    )
    index = {
        "schema": builder.CACHE_BINDING_INDEX_SCHEMA,
        "candidate_lock_sha256": CANDIDATE,
        "entries": [_entry(identity, tmp_path) for identity in identities],
    }
    monkeypatch.setattr(builder, "_validate_entry", _fake_validate)
    registry = builder.build_registry(
        {
            "schema": "cvs.full_ablation.plan.v1",
            "phase": "phase2",
            "rows": rows,
        },
        index,
    )
    assert len(registry["bindings"]) == len(rows)
    by_key = {item["row_key"]: item for item in registry["bindings"]}
    pairs = {}
    for row in rows:
        key = (
            row["receiver_id"],
            row["k_shot"],
            row["new_class_count"],
            row["method_seed"],
            row["support_seed"],
            row["query_seed"],
            row["new_class_draw_seed"],
        )
        pairs.setdefault(key, []).append(by_key[row["row_key"]])
    assert all(
        len(items) == 2
        and items[0]["feature_cache_payload"]
        == items[1]["feature_cache_payload"]
        and items[0]["scoring_manifest"]
        == items[1]["scoring_manifest"]
        for items in pairs.values()
    )


def test_registry_rejects_incomplete_current_launch_index(
    tmp_path: Path, monkeypatch
) -> None:
    rows = _rows()
    identities = sorted(
        {builder._row_identity(row) for row in rows},
        key=repr,
    )
    monkeypatch.setattr(builder, "_validate_entry", _fake_validate)
    with pytest.raises(
        builder.Stage2BindingRegistryBuildError,
        match="coverage mismatch",
    ):
        builder.build_registry(
            {
                "schema": "cvs.full_ablation.plan.v1",
                "phase": "phase2",
                "rows": rows,
            },
            {
                "schema": builder.CACHE_BINDING_INDEX_SCHEMA,
                "candidate_lock_sha256": CANDIDATE,
                "entries": [
                    _entry(identity, tmp_path)
                    for identity in identities[:-1]
                ],
            },
        )


def test_stage2a_entry_rejects_nonzero_support_metadata(
    tmp_path: Path,
) -> None:
    stage2a = _entry(
        ("stage2a", "20-1", 7283101, 0, 7283301, 0, 0, 0),
        tmp_path,
    )
    for field, value in (
        ("support_seed", 99),
        ("new_class_draw_seed", 88),
        ("k_shot", 10),
        ("new_class_count", 20),
    ):
        malformed = {**stage2a, field: value}
        with pytest.raises(
            builder.Stage2BindingRegistryBuildError,
            match="zero support",
        ):
            builder._entry_identity(malformed)


def test_entry_verifies_package_cache_and_truth_sidecar_chain(
    tmp_path: Path, monkeypatch
) -> None:
    identity = (
        "stage2c",
        "20-1",
        7282101,
        7282201,
        7282301,
        7282401,
        5,
        10,
    )
    entry = _entry(identity, tmp_path)
    payload = Path(entry["feature_cache_payload"])
    manifest_path = Path(entry["feature_cache_manifest"])
    seal_path = Path(entry["predictor_detached_seal"])
    scoring_path = Path(entry["scoring_manifest"])
    package_root = Path(entry["predictor_package_root"])
    package_root.mkdir()
    payload.write_bytes(b"payload")
    manifest_path.write_text("{}\n", encoding="utf-8")
    seal_path.write_text("{}\n", encoding="utf-8")
    scoring_path.write_text("{}\n", encoding="utf-8")
    entry["predictor_detached_seal_sha256"] = hashlib.sha256(
        seal_path.read_bytes()
    ).hexdigest()
    payload_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    package_sha = "6" * 64
    feature_manifest = {
        "stage_scope": "stage2c",
        "receiver": "20-1",
        "method_seed": 7282101,
        "support_seed": 7282201,
        "query_seed": 7282301,
        "new_class_draw_seed": 7282401,
        "k_shot": 5,
        "new_classes": [f"cls-{index}" for index in range(10)],
        "package_root_sha256": package_sha,
        "package_seal_sha256": entry[
            "predictor_detached_seal_sha256"
        ],
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule",
        "split_id": "split",
        "phase1_bundle_sha256": "7" * 64,
        "phase1_prototype_sha256": "8" * 64,
    }
    monkeypatch.setattr(
        builder,
        "load_feature_cache",
        lambda *args, **kwargs: {
            "manifest": feature_manifest,
        },
    )
    monkeypatch.setattr(
        builder,
        "preflight_stage2_predictor_package",
        lambda *args, **kwargs: (
            {
                "stage": "stage2c",
                "receiver": "20-1",
                "candidate_lock_sha256": CANDIDATE,
                "package_root_sha256": package_sha,
            },
            {"package_root_sha256": package_sha},
            {},
        ),
    )
    monkeypatch.setattr(
        builder,
        "load_verified_scoring_sidecar",
        lambda *args, **kwargs: (
            {"stage": "stage2c", "receiver": "20-1"},
            {
                "predictor_package_root_sha256": package_sha,
                "predictor_package_seal_sha256": entry[
                    "predictor_detached_seal_sha256"
                ],
            },
            {
                "scoring_manifest_sha256": hashlib.sha256(
                    scoring_path.read_bytes()
                ).hexdigest()
            },
        ),
    )
    result = builder._validate_entry(
        entry,
        candidate_lock_sha256=CANDIDATE,
    )
    assert result["_feature_cache_payload_sha256"] == payload_sha

    monkeypatch.setattr(
        builder,
        "preflight_stage2_predictor_package",
        lambda *args, **kwargs: (
            {
                "stage": "stage2c",
                "receiver": "20-1",
                "candidate_lock_sha256": "9" * 64,
                "package_root_sha256": package_sha,
            },
            {"package_root_sha256": package_sha},
            {},
        ),
    )
    with pytest.raises(
        builder.Stage2BindingRegistryBuildError,
        match="predictor package",
    ):
        builder._validate_entry(
            entry,
            candidate_lock_sha256=CANDIDATE,
        )


def test_entry_rejects_truth_stage_or_receiver_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    identity = (
        "stage2a",
        "20-1",
        7283101,
        0,
        7283301,
        0,
        0,
        0,
    )
    entry = _entry(identity, tmp_path)
    payload = Path(entry["feature_cache_payload"])
    manifest_path = Path(entry["feature_cache_manifest"])
    seal_path = Path(entry["predictor_detached_seal"])
    scoring_path = Path(entry["scoring_manifest"])
    Path(entry["predictor_package_root"]).mkdir()
    payload.write_bytes(b"payload")
    manifest_path.write_text("{}\n", encoding="utf-8")
    seal_path.write_text("{}\n", encoding="utf-8")
    scoring_path.write_text("{}\n", encoding="utf-8")
    entry["predictor_detached_seal_sha256"] = hashlib.sha256(
        seal_path.read_bytes()
    ).hexdigest()
    package_sha = "6" * 64
    feature_manifest = {
        "stage_scope": "stage2a",
        "receiver": "20-1",
        "method_seed": 7283101,
        "support_seed": 0,
        "query_seed": 7283301,
        "new_class_draw_seed": 0,
        "k_shot": 0,
        "new_classes": [],
        "package_root_sha256": package_sha,
        "package_seal_sha256": entry[
            "predictor_detached_seal_sha256"
        ],
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule",
        "split_id": "split",
        "phase1_bundle_sha256": "7" * 64,
        "phase1_prototype_sha256": "8" * 64,
    }
    monkeypatch.setattr(
        builder,
        "load_feature_cache",
        lambda *args, **kwargs: {"manifest": feature_manifest},
    )
    monkeypatch.setattr(
        builder,
        "preflight_stage2_predictor_package",
        lambda *args, **kwargs: (
            {
                "stage": "stage2b",
                "receiver": "20-1",
                "candidate_lock_sha256": CANDIDATE,
                "package_root_sha256": package_sha,
            },
            {"package_root_sha256": package_sha},
            {},
        ),
    )
    wrong_truth = {"stage": "stage2b", "receiver": "wrong-rx"}
    monkeypatch.setattr(
        builder,
        "load_verified_scoring_sidecar",
        lambda *args, **kwargs: (
            wrong_truth,
            {
                "predictor_package_root_sha256": package_sha,
                "predictor_package_seal_sha256": entry[
                    "predictor_detached_seal_sha256"
                ],
            },
            {
                "scoring_manifest_sha256": hashlib.sha256(
                    scoring_path.read_bytes()
                ).hexdigest()
            },
        ),
    )
    with pytest.raises(
        builder.Stage2BindingRegistryBuildError,
        match="scoring manifest",
    ):
        builder._validate_entry(
            entry,
            candidate_lock_sha256=CANDIDATE,
        )
