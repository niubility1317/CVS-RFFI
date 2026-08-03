from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi import stage2_d127_s0_package_adapter as adapter
from cvsrffi.somph_predictor_bundle import QUERY_NPZ_MEMBERS, SUPPORT_NPZ_MEMBERS


SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
RECEIVERS = ("20-1", "3-19", "7-14")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    raw = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    return _sha(raw)


def _method_lock() -> dict:
    return {
        "schema": adapter.METHOD_LOCK_SCHEMA,
        "candidate_id": "D127-LIGHT-DA-X-D92-LITE-S0",
        "protocol_schema": "p2_min_v1",
        "claim_scope": "TARGET_DEVELOPMENT_S0_ONLY",
        "checkpoint": {"sha256": "1" * 64},
        "phase1_asset_build": {
            "source_received_iq_sha256": "9" * 64,
            "source_received_iq_receipt_sha256": "a" * 64,
            "source_label_join_archive_sha256": "b" * 64,
        },
        "student_t_qknn": {
            "active_k": [1, 5], "student_nu": 3, "kernel_effective_dim": 12,
            "kernel_volume_gamma": 1, "shared_h0": 0.35, "scale_prior_strength": 2,
            "scale_min_ratio": 0.5, "scale_max_ratio": 2, "temperature": 0.85,
            "phase1_lodo_receipt_sha256": "2" * 64,
            "quantization_margin_audit_sha256": "3" * 64,
        },
        "domain_adaptation": {"query_fit_count": 0, "query_update_count": 0, "query_selection_count": 0},
        "s0_matrix": {
            "seed": 713102, "receivers": list(RECEIVERS), "k_new_count": [[1, 20], [5, 20]],
            "scenes": list(SCENES), "row_pair_count": 18,
            "registration_states": ["before", "after"], "k5_source_pool_k": 10,
            "k5_is_ordered_k10_prefix": True,
            "d92_retry2_manifest_sha256": "b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c",
        },
    }


def _reference(name: str) -> dict:
    return {"package_root": f"/sealed/{name}", "detached_seal_path": f"/seal/{name}", "expected_seal_sha256": _sha(name.encode())}


def _context() -> dict:
    rows = []
    for receiver in RECEIVERS:
        for k in (1, 5):
            pool = 10 if k == 5 else 1
            prefix = f"{receiver}-k{k}"
            rows.append({
                "job_id": f"d106-{prefix}", "source_d92_job_id": f"rx_{receiver}__seed_713102__k_{pool}__new_20",
                "receiver": receiver, "seed": 713102, "k_shot": k, "source_pool_k": pool, "new_count": 20,
                "packages": {name: _reference(f"{prefix}-{name}") for name in (
                    "before_enrollment", "before_apply", "after_enrollment", "after_apply"
                )},
            })
    return {"schema": adapter.D106_CONTEXT_SCHEMA, "rows": rows}


def _payload(k: int, *, class_count: int, scene_index: int, query_prefix: str) -> tuple[dict, dict]:
    classes = [f"cls_{i:02d}" for i in range(class_count)]
    n = len(classes) * k
    support = {
        "support_leo_weak_iq": np.full((n, 2, 256), scene_index + k, dtype=np.float32),
        "support_class_indices": np.repeat(np.arange(len(classes), dtype=np.int64), k),
        "support_rank_within_class": np.tile(np.arange(k, dtype=np.int64), len(classes)),
        "support_tokens": np.asarray([f"sid-{scene_index}-{c}-{r}" for c in range(len(classes)) for r in range(k)]),
        "support_overlay_tokens": np.asarray([f"ov-{scene_index}-{x}" for x in range(n)]),
        "support_satellite_seeds": np.arange(n, dtype=np.int64),
        "support_post_channel_iq_sha256": np.asarray([f"h-{scene_index}-{x}" for x in range(n)]),
    }
    query = {
        "query_leo_weak_iq": np.full((5, 2, 256), scene_index + 0.5, dtype=np.float32),
        "query_tokens": np.asarray([f"qid-{query_prefix}-{scene_index}-{x}" for x in range(5)]),
        "query_overlay_tokens": np.asarray([f"qov-{scene_index}-{x}" for x in range(5)]),
        "query_satellite_seeds": np.arange(5, dtype=np.int64),
        "query_post_channel_iq_sha256": np.asarray([f"qh-{scene_index}-{x}" for x in range(5)]),
    }
    assert set(support) == set(SUPPORT_NPZ_MEMBERS) - {"manifest_json"}
    assert set(query) == set(QUERY_NPZ_MEMBERS) - {"manifest_json"}
    return support, query


def _loader(reference: dict):
    name = Path(reference["package_root"]).name
    parts = name.split("-")
    k = 10 if "k5" in name else 1
    state = "after" if "after" in name else "before"
    profile = "enrollment_only" if "enrollment" in name else "apply_only"
    # receiver is not material to fake package contents, but K and state are.
    class_count = 26 if state == "after" else 6
    classes = [{"class_handle": f"cls_{i:02d}"} for i in range(class_count)]
    scene_payloads = {}
    for index, scene in enumerate(SCENES):
        # Formal before old-query IDs must be an ordered subset of the after
        # state; the fake retains the same received query IDs for both.
        support, query = _payload(k, class_count=class_count, scene_index=index, query_prefix=name.split("-")[0])
        scene_payloads[scene] = support if profile == "enrollment_only" else query
    manifest = {
        "profile": profile, "stage": "stage2c", "registration_state": state,
        "receiver": name.split("-k")[0], "seed": 713102, "k_shot": k,
        "phase1_checkpoint_sha256": "4" * 64, "feature_runtime_sha256": "5" * 64,
        "method_lock_sha256": "6" * 64, "registered_classes": classes,
        "row_handle": None if profile == "enrollment_only" else "row-handle",
        "row_manifest_sha256": None if profile == "enrollment_only" else "7" * 64,
        "package_root_sha256": _sha(name.encode()),
    }
    return scene_payloads, manifest, {"sealed": True}


def _fake_pair_validator(left, right):
    for field in ("registration_state", "seed", "k_shot", "registered_classes"):
        if left[field] != right[field]:
            raise ValueError(field)


@pytest.fixture(autouse=True)
def _strict_pair(monkeypatch):
    monkeypatch.setattr(adapter, "_validate_matched_packages", _fake_pair_validator)


def _materialize(tmp_path: Path):
    lock = tmp_path / "lock.json"
    context = tmp_path / "context.json"
    lock_sha = _write_json(lock, _method_lock())
    context_sha = _write_json(context, _context())
    return adapter.materialize_d127_s0_package_rows(
        method_lock_path=lock, expected_method_lock_sha256=lock_sha,
        d106_context_path=context, expected_d106_context_sha256=context_sha,
        package_loader=_loader,
    )


def test_materializes_complete_paired_s0_rows_and_real_qknn_receipts(tmp_path):
    prepared = _materialize(tmp_path)
    assert len(prepared.before) == len(prepared.after) == 18
    assert tuple(prepared.qknn_locks) == (1, 5)
    assert prepared.qknn_locks[1].phase1_lodo_receipt_sha256 == "2" * 64
    assert prepared.qknn_locks[5].quantization_margin_audit_sha256 == "3" * 64
    assert [item.row.row_id for item in prepared.before] == [item.row.row_id for item in prepared.after]
    assert all(str(item.row.support_iq.dtype) == "torch.float32" for item in prepared.before)
    assert all(item.row.qknn_lock.active_k == item.row.k_shot for item in prepared.after)
    assert all(len(left.row.registered_classes) == 6 and len(right.row.registered_classes) == 26 for left, right in zip(prepared.before, prepared.after))
    assert all(left.row.registered_classes == right.row.registered_classes[:len(left.row.registered_classes)] for left, right in zip(prepared.before, prepared.after))
    assert prepared.prefix_receipt["record_count"] == 18
    assert len(prepared.prefix_receipt["pair_bindings"]) == 18
    assert all(pair["before_query_is_after_ordered_subset"] for pair in prepared.prefix_receipt["pair_bindings"])
    assert all("sid-" not in json.dumps(record) for record in prepared.prefix_receipt["records"])


def test_k5_is_strict_ordered_prefix_of_k10_package(tmp_path):
    prepared = _materialize(tmp_path)
    k5 = [item for item in prepared.after if item.row.k_shot == 5]
    assert len(k5) == 9
    assert all(item.source_pool_k == 10 for item in k5)
    assert all(len(item.row.support_labels) == 26 * 5 for item in k5)
    record = next(record for record in prepared.prefix_receipt["records"] if record["state"] == "after")
    assert record["source_pool_k"] == 10
    assert record["support_token_count"] == 26 * 5
    assert all(part["prefix_count"] == 5 for part in record["class_prefixes"])


def test_materialization_survives_disabled_numpy_torch_abi_bridge(tmp_path, monkeypatch):
    def _blocked(*_args, **_kwargs):
        raise TypeError("simulated NumPy2/Torch2.1 ABI mismatch")

    monkeypatch.setattr(torch, "from_numpy", _blocked)
    monkeypatch.setattr(torch, "as_tensor", _blocked)
    prepared = _materialize(tmp_path)
    assert len(prepared.before) == len(prepared.after) == 18
    assert all(item.row.support_iq.dtype == torch.float32 for item in prepared.before)
    assert all(item.row.query_iq.dtype == torch.float32 for item in prepared.after)


def test_rejects_truth_role_seed_and_prefix_drift(tmp_path):
    lock = tmp_path / "lock.json"
    context = _context()
    context["rows"][0]["query_truth"] = ["bad"]
    lock_sha = _write_json(lock, _method_lock())
    context_path = tmp_path / "context.json"
    context_sha = _write_json(context_path, context)
    with pytest.raises(adapter.D127S0PackageAdapterError, match="raw-row closure|forbidden"):
        adapter.materialize_d127_s0_package_rows(
            method_lock_path=lock, expected_method_lock_sha256=lock_sha,
            d106_context_path=context_path, expected_d106_context_sha256=context_sha,
            package_loader=_loader,
        )
    context = _context()
    context["rows"][0]["seed"] = 713103
    context_sha = _write_json(context_path, context)
    with pytest.raises(adapter.D127S0PackageAdapterError, match="seed"):
        adapter.materialize_d127_s0_package_rows(
            method_lock_path=lock, expected_method_lock_sha256=lock_sha,
            d106_context_path=context_path, expected_d106_context_sha256=context_sha,
            package_loader=_loader,
        )


def test_prefix_rank_and_exclusive_receipt_fail_closed(tmp_path, monkeypatch):
    def bad_loader(reference):
        payloads, manifest, audit = _loader(reference)
        if "k5" in reference["package_root"] and "enrollment" in reference["package_root"]:
            for payload in payloads.values():
                payload["support_rank_within_class"][4] = 8
        return payloads, manifest, audit

    lock = tmp_path / "lock.json"
    context = tmp_path / "context.json"
    lock_sha = _write_json(lock, _method_lock())
    context_sha = _write_json(context, _context())
    with pytest.raises(adapter.D127S0PackageAdapterError, match="prefix|rank"):
        adapter.materialize_d127_s0_package_rows(
            method_lock_path=lock, expected_method_lock_sha256=lock_sha,
            d106_context_path=context, expected_d106_context_sha256=context_sha,
            package_loader=bad_loader,
        )
    prepared = _materialize(tmp_path)
    receipt = tmp_path / "prefix.json"
    adapter.write_d127_s0_prefix_receipt_exclusive(receipt, prepared.prefix_receipt)
    with pytest.raises(adapter.D127S0PackageAdapterError, match="already exists"):
        adapter.write_d127_s0_prefix_receipt_exclusive(receipt, prepared.prefix_receipt)


def _fake_local_worker(*, model, candidate_id, asset, rows):
    del model, asset
    outputs = []
    for row in rows:
        classes = list(row.registered_classes)
        common = {
            "classes": classes,
            "predictions": [classes[0]] * len(row.opaque_query_ids),
            "receipt": {"common": True},
        }
        arms = {
            "M0": copy.deepcopy(common),
            "M_DA": {
                "classes": classes,
                "predictions": [classes[-1]] * len(row.opaque_query_ids),
                "receipt": {"candidate": candidate_id, "arm": "M_DA"},
            },
            "M_L92": copy.deepcopy(common),
            "M_JOINT": {
                "classes": classes,
                "predictions": [classes[-1]] * len(row.opaque_query_ids),
                "receipt": {"candidate": candidate_id, "arm": "M_JOINT"},
            },
        }
        outputs.append({
            "row_id": row.row_id,
            "receiver_id": row.receiver_id,
            "k_shot": row.k_shot,
            "scene": row.scene,
            "opaque_query_ids": list(row.opaque_query_ids),
            "arms": arms,
            "joint_receipt": {"candidate": candidate_id},
            "hook_receipt": {"candidate": candidate_id},
            "da_resource": {"candidate": candidate_id},
        })
    payload = {
        "schema": adapter.entry.LOCAL_WORKER_SCHEMA,
        "candidate_id": candidate_id,
        "evaluation_scope": "LOCAL_CANDIDATE_WORKER_NON_PUBLISHABLE",
        "truth_loaded": False,
        "row_count": len(outputs),
        "rows_complete": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "phase2_optimizer_steps": 0,
        "resource": {
            "asset_numeric_payload_bytes": 1,
            "adapter_macs_per_sample": 1,
            "total_adapter_macs_support_plus_query": len(outputs),
            "total_id_backbone_forwards": len(outputs),
            "total_query_rows": sum(len(row.opaque_query_ids) for row in rows),
        },
        "rows": outputs,
    }
    payload["prediction_sha256"] = adapter.entry._sha256(payload)
    return payload


def _phase1_manifest_receipt(plan, candidate_id, *, manifest_sha256="8" * 64):
    expected = plan["phase1_asset_expected_binding"]
    receipt = {
        "schema": adapter.PHASE1_MANIFEST_RECEIPT_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "candidate_id": candidate_id,
        "method_lock_sha256": expected["method_lock_sha256"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "source_binding": expected["source_binding"],
        "qknn_lock_binding": expected["qknn_lock_binding"],
        "episode_manifest_sha256": "c" * 64,
        "episode_contract_sha256": "d" * 64,
        "candidate_asset": {
            "candidate_id": candidate_id,
            "persistent_fp32_sidecar": False,
        },
    }
    receipt["receipt_sha256"] = adapter._canonical_sha256(receipt)
    return receipt


def test_candidate_workers_merge_to_one_common_pair_manifest(tmp_path, monkeypatch):
    prepared = _materialize(tmp_path)
    plan = adapter.build_d127_s0_prepared_plan(prepared)
    monkeypatch.setattr(adapter.entry, "_run_d127_s0_candidate_worker", _fake_local_worker)
    workers = [
        adapter.run_d127_s0_candidate_worker_pair(
            model=object(), candidate_id=candidate_id, asset=object(), prepared=prepared,
            prepared_plan=plan, phase1_asset_manifest_sha256="8" * 64,
            phase1_manifest_receipt=_phase1_manifest_receipt(plan, candidate_id),
            checkpoint_sha256="1" * 64,
        )
        for candidate_id in adapter.entry.CANDIDATE_IDS
    ]
    merged = adapter.merge_d127_s0_candidate_workers(
        prepared_plan=plan, workers=list(reversed(workers))
    )
    assert merged["candidate_ids"] == list(adapter.entry.CANDIDATE_IDS)
    assert merged["pair_manifest"]["method_lock_sha256"] == prepared.method_lock_sha256
    assert merged["pair_manifest"]["pair_bindings"] == plan["pair_bindings"]
    assert merged["physical_execution"]["physical_base_forwards_are_repeated_per_candidate"] is True
    assert all(
        set(row["common_arms"]) == {"M0", "M_L92"}
        and set(row["candidates"]) == set(adapter.entry.CANDIDATE_IDS)
        for state in merged["states"].values()
        for row in state["rows"]
    )
    adapter.validate_d127_s0_prediction_pairs(merged, prepared_plan=plan)

    broken = copy.deepcopy(workers)
    broken[1]["states"]["before"]["rows"][0]["arms"]["M0"]["predictions"][0] = "cls_01"
    broken[1]["states"]["before"]["prediction_sha256"] = adapter.entry._sha256(
        {key: value for key, value in broken[1]["states"]["before"].items() if key != "prediction_sha256"}
    )
    broken[1]["candidate_worker_sha256"] = adapter._canonical_sha256(
        {key: value for key, value in broken[1].items() if key != "candidate_worker_sha256"}
    )
    with pytest.raises(adapter.D127S0PackageAdapterError, match="common-arm"):
        adapter.merge_d127_s0_candidate_workers(prepared_plan=plan, workers=broken)


def test_candidate_and_paired_json_full_round_trip_is_order_independent(tmp_path, monkeypatch):
    prepared = _materialize(tmp_path)
    plan = adapter.build_d127_s0_prepared_plan(prepared)
    monkeypatch.setattr(adapter.entry, "_run_d127_s0_candidate_worker", _fake_local_worker)
    loaded_workers = []
    for candidate_id in adapter.entry.CANDIDATE_IDS:
        payload = adapter.run_d127_s0_candidate_worker_pair(
            model=object(), candidate_id=candidate_id, asset=object(), prepared=prepared,
            prepared_plan=plan, phase1_asset_manifest_sha256="8" * 64,
            phase1_manifest_receipt=_phase1_manifest_receipt(plan, candidate_id),
            checkpoint_sha256="1" * 64,
        )
        path = tmp_path / f"{candidate_id}.json"
        adapter.write_d127_s0_candidate_worker_exclusive(path, payload)
        loaded, _digest = adapter.load_d127_s0_candidate_worker(
            path, expected_sha256=_sha(path.read_bytes())
        )
        loaded_workers.append(loaded)
    paired = adapter.merge_d127_s0_candidate_workers(
        prepared_plan=plan, workers=list(reversed(loaded_workers))
    )
    paired_path = tmp_path / "paired.json"
    adapter.write_d127_s0_paired_prediction_exclusive(
        paired_path, paired, prepared_plan=plan
    )
    loaded_paired, _digest = adapter.load_d127_s0_paired_prediction(
        paired_path, expected_sha256=_sha(paired_path.read_bytes()), prepared_plan=plan
    )
    assert loaded_paired["paired_prediction_sha256"] == paired["paired_prediction_sha256"]
    assert set(loaded_paired["states"]) == {"before", "after"}


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("method_lock_sha256",), "e" * 64),
        (("checkpoint_sha256",), "e" * 64),
        (("qknn_lock_binding", "lock_digest_by_k", "1"), "e" * 64),
        (("source_binding", "selected_received_iq_sha256"), "e" * 64),
        (("source_binding", "selected_received_iq_receipt_sha256"), "e" * 64),
        (("source_binding", "source_label_join_archive_sha256"), "e" * 64),
    ),
)
def test_phase1_manifest_receipt_rejects_wrong_lineage(tmp_path, path, replacement):
    prepared = _materialize(tmp_path)
    plan = adapter.build_d127_s0_prepared_plan(prepared)
    receipt = copy.deepcopy(_phase1_manifest_receipt(plan, adapter.entry.CANDIDATE_IDS[0]))
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    receipt["receipt_sha256"] = adapter._canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(adapter.D127S0PackageAdapterError, match="lineage"):
        adapter.run_d127_s0_candidate_worker_pair(
            model=object(), candidate_id=adapter.entry.CANDIDATE_IDS[0], asset=object(), prepared=prepared,
            prepared_plan=plan, phase1_asset_manifest_sha256="8" * 64,
            phase1_manifest_receipt=receipt, checkpoint_sha256="1" * 64,
        )


def test_candidate_asset_loader_binds_full_phase1_manifest_lineage(tmp_path, monkeypatch):
    prepared = _materialize(tmp_path)
    plan = adapter.build_d127_s0_prepared_plan(prepared)
    candidate_id = adapter.entry.CANDIDATE_IDS[0]

    class FakeQuantizedAsset:
        def decode(self, *, device):
            return {"candidate_id": candidate_id, "device": str(device)}

    monkeypatch.setattr(
        adapter.phase1_release,
        "load_d127_phase1_asset_bundle",
        lambda bundle_dir, expected_manifest_sha256: {
            item: FakeQuantizedAsset() for item in adapter.entry.CANDIDATE_IDS
        },
    )
    expected = plan["phase1_asset_expected_binding"]
    manifest = {
        "bundle_kind": "merged_complete",
        "candidate_ids": list(adapter.entry.CANDIDATE_IDS),
        "method_lock_sha256": expected["method_lock_sha256"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "source_binding": expected["source_binding"],
        "qknn_lock_binding": expected["qknn_lock_binding"],
        "episode_manifest_sha256": "c" * 64,
        "episode_contract_sha256": "d" * 64,
        "candidate_assets": {
            item: {"candidate_id": item, "persistent_fp32_sidecar": False}
            for item in adapter.entry.CANDIDATE_IDS
        },
    }

    def write_bundle(name, value):
        root = tmp_path / name
        root.mkdir()
        raw = adapter._canonical_bytes(value)
        (root / adapter.phase1_release.MANIFEST_FILE_NAME).write_bytes(raw)
        return root, _sha(raw)

    root, manifest_sha = write_bundle("valid", manifest)
    asset, receipt = adapter.load_d127_s0_candidate_asset(
        bundle_dir=root, expected_manifest_sha256=manifest_sha, candidate_id=candidate_id,
        device="cpu", prepared_plan=plan,
    )
    assert asset["candidate_id"] == candidate_id
    assert receipt["source_binding"] == expected["source_binding"]

    mutations = (
        (("method_lock_sha256",), "e" * 64),
        (("checkpoint_sha256",), "e" * 64),
        (("qknn_lock_binding", "lock_digest_by_k", "1"), "e" * 64),
        (("source_binding", "selected_received_iq_sha256"), "e" * 64),
        (("source_binding", "selected_received_iq_receipt_sha256"), "e" * 64),
        (("source_binding", "source_label_join_archive_sha256"), "e" * 64),
    )
    for index, (path, replacement) in enumerate(mutations):
        broken = copy.deepcopy(manifest)
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        broken_root, broken_sha = write_bundle(f"broken-{index}", broken)
        with pytest.raises(adapter.D127S0PackageAdapterError, match="load/decode"):
            adapter.load_d127_s0_candidate_asset(
                bundle_dir=broken_root, expected_manifest_sha256=broken_sha,
                candidate_id=candidate_id, device="cpu", prepared_plan=plan,
            )
