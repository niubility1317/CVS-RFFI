from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cvsrffi.stage2_d106_matrix_protocol import (
    LEO_SCENARIOS,
    RECEIVERS,
    TARGET25_SEED,
    TARGET25_SLICES,
    canonical_sha256,
    freeze_d106_matrix_protocol,
)
from cvsrffi.stage2_d106_target25_inputs import (
    D106_INDEX_SCHEMA,
    D106_SPLIT_LOCATOR_SCHEMA,
    D106Target25InputError,
    prepare_d106_target25_inputs,
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return _sha_file(path)


def _asset(tmp_path: Path, name: str) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_bytes(f"sealed-{name}".encode())
    return path, _sha_file(path)


def _package_ref(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "package"
    root.mkdir()
    seal, seal_sha = _asset(tmp_path, "package.seal")
    policy, _ = _asset(tmp_path, "policy.json")
    authorization, _ = _asset(tmp_path, "authorization.json")
    envelope, envelope_sha = _asset(tmp_path, "envelope.json")
    return {
        "package_root": str(root.resolve()),
        "detached_seal_path": str(seal.resolve()),
        "expected_seal_sha256": seal_sha,
        "formal_policy_path": str(policy.resolve()),
        "formal_policy_authorization_path": str(authorization.resolve()),
        "signed_policy_authorization_envelope_path": str(envelope.resolve()),
        "expected_signed_policy_authorization_envelope_sha256": envelope_sha,
    }


def _matrix_locator(tmp_path: Path) -> dict[str, Any]:
    package_ref = _package_ref(tmp_path)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    cache, _ = _asset(tmp_path, "cache.json")
    authorities = [
        {
            "receiver": receiver,
            "authority_bundle_root": str(authority_root.resolve()),
            "expected_authority_commit_sha256": canonical_sha256(
                {"receiver": receiver}
            ),
            "cache_set_manifest_path": str(cache.resolve()),
        }
        for receiver in RECEIVERS
    ]
    rows = [
        {
            "receiver": receiver,
            "k_shot": k_shot,
            "new_count": new_count,
            "before_enrollment": package_ref,
            "before_apply": package_ref,
            "after_enrollment": package_ref,
            "after_apply": package_ref,
        }
        for receiver in RECEIVERS
        for k_shot, new_count in TARGET25_SLICES
    ]
    return {
        "schema": D106_INDEX_SCHEMA,
        "seed": TARGET25_SEED,
        "claim_scope": "development_screen",
        "formal_launch_authority": False,
        "authorities": authorities,
        "rows": rows,
    }


def _classes(new_count: int) -> tuple[list[str], list[str], list[str]]:
    old = ["old-0", "old-1"]
    new = [f"new-{index}" for index in range(new_count)]
    return old, new, old + new


def _physical_ids(
    *,
    receiver: str,
    scenario: str,
    state: str,
    k_shot: int,
    new_count: int,
    registered: list[str],
) -> tuple[list[str], list[str]]:
    support = [
        f"{receiver}/{scenario}/{state}/new{new_count}/{class_id}/shot{shot}"
        for class_id in registered
        for shot in range(k_shot)
    ]
    query_k = 10 if new_count == 20 and k_shot in (5, 10) else k_shot
    query = [
        f"{receiver}/{scenario}/{state}/new{new_count}/query-k{query_k}-{index}"
        for index in range(3)
    ]
    return support, query


def _state_locator(
    *,
    receiver: str,
    scenario: str,
    state: str,
    k_shot: int,
    new_count: int,
) -> dict[str, Any]:
    old, new, after_registry = _classes(new_count)
    registered = old if state == "before" else after_registry
    state_new = [] if state == "before" else new
    support, query = _physical_ids(
        receiver=receiver,
        scenario=scenario,
        state=state,
        k_shot=k_shot,
        new_count=new_count,
        registered=registered,
    )
    capsule_k = 10 if new_count == 20 and k_shot in (5, 10) else k_shot
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": canonical_sha256(
            {
                "receiver": receiver,
                "scenario": scenario,
                "state": state,
                "k": capsule_k,
                "new": new_count,
            }
        ),
        "split_id": canonical_sha256(
            {
                "receiver": receiver,
                "scenario": scenario,
                "state": state,
                "k": k_shot,
                "new": new_count,
            }
        ),
        "authority_receipt_sha256": canonical_sha256({"receiver": receiver}),
        "authority_envelope_sha256": canonical_sha256(
            {"receiver": receiver, "envelope": True}
        ),
        "support_physical_ids": support,
        "query_physical_ids": query,
        "registered_classes": registered,
        "old_classes": old,
        "new_classes": state_new,
    }


def _split_locator() -> dict[str, Any]:
    rows = []
    for receiver in RECEIVERS:
        for k_shot, new_count in TARGET25_SLICES:
            rows.append(
                {
                    "receiver": receiver,
                    "k_shot": k_shot,
                    "new_count": new_count,
                    "scenarios": [
                        {
                            "scenario": scenario,
                            "before": _state_locator(
                                receiver=receiver,
                                scenario=scenario,
                                state="before",
                                k_shot=k_shot,
                                new_count=new_count,
                            ),
                            "after": _state_locator(
                                receiver=receiver,
                                scenario=scenario,
                                state="after",
                                k_shot=k_shot,
                                new_count=new_count,
                            ),
                        }
                        for scenario in LEO_SCENARIOS
                    ],
                }
            )
    locator: dict[str, Any] = {
        "schema": D106_SPLIT_LOCATOR_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "seed": TARGET25_SEED,
        "rows": rows,
    }
    locator["locator_receipt_sha256"] = canonical_sha256(locator)
    return locator


def _inputs(tmp_path: Path) -> dict[str, Any]:
    matrix_path = tmp_path / "matrix.json"
    split_path = tmp_path / "splits.json"
    matrix_sha = _write_json(matrix_path, _matrix_locator(tmp_path))
    split_sha = _write_json(split_path, _split_locator())
    checkpoint, checkpoint_sha = _asset(tmp_path, "checkpoint.pth")
    rdce_wire, rdce_wire_sha = _asset(tmp_path, "rdce.wire")
    rdce_lock, rdce_lock_sha = _asset(tmp_path, "rdce.lock.json")
    rcmr_lock, rcmr_lock_sha = _asset(tmp_path, "rcmr.lock.json")
    return {
        "matrix_index_path": matrix_path,
        "expected_matrix_index_sha256": matrix_sha,
        "split_locator_path": split_path,
        "expected_split_locator_sha256": split_sha,
        "checkpoint_path": checkpoint,
        "expected_checkpoint_sha256": checkpoint_sha,
        "rdce_wire_path": rdce_wire,
        "expected_rdce_wire_sha256": rdce_wire_sha,
        "rdce_lock_path": rdce_lock,
        "expected_rdce_lock_sha256": rdce_lock_sha,
        "rcmr_lock_path": rcmr_lock,
        "expected_rcmr_lock_sha256": rcmr_lock_sha,
        "kcr_route_lock_sha256": "a" * 64,
        "output_dir": tmp_path / "prepared",
    }


def test_prepare_closes_25_rows_and_reuses_frozen_matrix_identity(tmp_path: Path) -> None:
    receipt = prepare_d106_target25_inputs(**_inputs(tmp_path))
    plan = json.loads(Path(receipt["plan_manifest"]).read_text(encoding="utf-8"))
    context = json.loads(
        Path(receipt["context_manifest"]).read_text(encoding="utf-8")
    )
    frozen = freeze_d106_matrix_protocol()
    assert receipt["outer_job_count"] == 25
    assert receipt["scenario_row_count"] == 75
    assert receipt["matched_arm_pair_count"] == 300
    assert receipt["state_surface_count"] == 600
    assert receipt["matrix_receipt_sha256"] == frozen.matrix_receipt_sha256
    assert plan["matrix_protocol"] == frozen.receipt_payload()
    assert len(plan["rows"]) == len(context["rows"]) == 25
    states = [
        state
        for row in context["rows"]
        for scenario in row["scenarios"]
        for state in scenario["states"]
    ]
    assert len(states) == 150
    assert all("support_received_iq_ref" in state for state in states)
    assert all("query_received_iq_ref" in state for state in states)
    assert all("support_physical_ids" not in state for state in states)
    assert all("query_physical_ids" not in state for state in states)
    assert all("d105_candidate" not in json.dumps(state) for state in states)
    assert plan["identity"]["assets"]["kcr_route_lock_sha256"] == "a" * 64


def test_prepare_rejects_d92_locator_sha_drift(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    path = kwargs["matrix_index_path"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(D106Target25InputError, match="SHA mismatch"):
        prepare_d106_target25_inputs(**kwargs)


def test_prepare_rejects_truth_field_in_split_locator(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    path = kwargs["split_locator_path"]
    locator = json.loads(path.read_text(encoding="utf-8"))
    locator["rows"][0]["scenarios"][0]["before"]["query_truth"] = ["old-0"]
    locator["locator_receipt_sha256"] = canonical_sha256(
        {key: value for key, value in locator.items() if key != "locator_receipt_sha256"}
    )
    kwargs["expected_split_locator_sha256"] = _write_json(path, locator)
    with pytest.raises(D106Target25InputError, match="forbidden truth/score"):
        prepare_d106_target25_inputs(**kwargs)


def test_prepare_rejects_k5_support_outside_k10_subset(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    path = kwargs["split_locator_path"]
    locator = json.loads(path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in locator["rows"]
        if item["receiver"] == RECEIVERS[0]
        and item["k_shot"] == 5
        and item["new_count"] == 20
    )
    row["scenarios"][0]["before"]["support_physical_ids"][0] = "outside-k10"
    locator["locator_receipt_sha256"] = canonical_sha256(
        {key: value for key, value in locator.items() if key != "locator_receipt_sha256"}
    )
    kwargs["expected_split_locator_sha256"] = _write_json(path, locator)
    with pytest.raises(D106Target25InputError, match="K5 support"):
        prepare_d106_target25_inputs(**kwargs)


def test_prepare_output_is_non_overwriting(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    prepare_d106_target25_inputs(**kwargs)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_d106_target25_inputs(**kwargs)
