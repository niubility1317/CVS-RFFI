from __future__ import annotations

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from cvsrffi import stage2_d127_da_candidates as da
from cvsrffi import stage2_d127_phase1_release as release
from cvsrffi import stage2_d127_s0_entry as entry
from cvsrffi import stage2_d127_s0_package_adapter as adapter
from cvsrffi import stage2_d128_a_one18 as one


RECEIVERS = ("20-1", "3-19", "7-14")
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha(index: int) -> str:
    return f"{index:064x}"


def _plan() -> dict:
    pairs: list[dict] = []
    index = 1
    for receiver in RECEIVERS:
        for k_shot in (1, 5):
            for scene in SCENES:
                row_id = f"d128-{receiver}-k{k_shot}-{scene}"
                old = [f"{row_id}-old0", f"{row_id}-old1"]
                after = [*old, f"{row_id}-new0"]
                before_classes = ["old0", "old1"]
                after_classes = [*before_classes, "new0"]
                def state(name: str, ids: list[str], classes: list[str]) -> dict:
                    nonlocal index
                    item = {
                        "row_id": row_id,
                        "receiver": receiver,
                        "k_shot": k_shot,
                        "scene": scene,
                        "state": name,
                        "capsule_id": _sha(index),
                        "split_id": _sha(index + 1),
                        "support_token_root_sha256": _sha(index + 2),
                        "query_token_root_sha256": adapter._opaque_root(ids),
                        "query_token_ordered_sha256": adapter._canonical_sha256(ids),
                        "registered_class_root_sha256": adapter._opaque_root(classes),
                        "qknn_lock_digest": _sha(701 if k_shot == 1 else 705),
                        "state_input_receipt_sha256": _sha(index + 3),
                        "query_token_count": len(ids),
                    }
                    index += 4
                    return item
                pairs.append(
                    {
                        "row_id": row_id,
                        "receiver": receiver,
                        "k_shot": k_shot,
                        "scene": scene,
                        "before": state("before", old, before_classes),
                        "after": state("after", after, after_classes),
                        "before_query_is_after_ordered_subset": True,
                        "formal_d92_reference": {
                            "source_d92_job_id": f"rx_{receiver.replace('-', '_')}__seed_713102__k_{10 if k_shot == 5 else k_shot}__new_20",
                            "pipeline_receipt_required": True,
                            "d92_retry2_manifest_sha256": _sha(900),
                        },
                    }
                )
    plan = {
        "schema": adapter.PREPARED_PLAN_SCHEMA,
        "method_lock_sha256": _sha(100),
        "checkpoint_sha256": _sha(101),
        "phase1_asset_expected_binding": {
            "method_lock_sha256": _sha(100),
            "checkpoint_sha256": _sha(101),
            "source_binding": {
                "checkpoint_sha256": _sha(101),
                "method_lock_sha256": _sha(100),
                "selected_received_iq_sha256": _sha(102),
                "selected_received_iq_receipt_sha256": _sha(103),
                "source_label_join_archive_sha256": _sha(104),
            },
            "qknn_lock_binding": {
                "phase1_lodo_receipt_sha256": _sha(105),
                "quantization_margin_audit_sha256": _sha(106),
                "lock_digest_by_k": {"1": _sha(701), "5": _sha(705)},
            },
        },
        "d106_context_sha256": _sha(107),
        "qknn_lock_digests": {"1": _sha(701), "5": _sha(705)},
        "row_pair_count": 18,
        "state_row_count": 36,
        "truth_loaded": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "prefix_receipt_sha256": _sha(108),
        "pair_bindings": pairs,
    }
    plan["prepared_plan_sha256"] = adapter._canonical_sha256(plan)
    return plan


def _arm(arm_id: str, classes: list[str], predictions: list[str]) -> dict:
    logits: list[list[float]] = []
    for prediction in predictions:
        logits.append([2.0 if candidate == prediction else 0.0 for candidate in classes])
    return {
        "arm_id": arm_id,
        "representation": "base_zid160" if arm_id in ("M0", "M_L92") else "adapted_zid160",
        "head": "phase1_locked_student_t_qknn" if arm_id in ("M0", "M_DA") else "d92_lite_dr_oas_lda",
        "classes": classes,
        "logits": logits,
        "predictions": predictions,
        "receipt": {
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        },
    }


def _worker(plan: dict, *, state: str) -> dict:
    rows: list[dict] = []
    for binding in plan["pair_bindings"]:
        k_shot = binding["k_shot"]
        classes = ["old0", "old1"] if state == "before" else ["old0", "old1", "new0"]
        query_ids = [f"{binding['row_id']}-old0", f"{binding['row_id']}-old1"]
        if state == "after":
            query_ids.append(f"{binding['row_id']}-new0")
        base = ["old0", "old1"] if state == "before" else ["old0", "old0", "old0"]
        if state == "before":
            adapted = ["old0", "old1"]
            joint = adapted
        elif k_shot == 1:
            adapted = ["old0", "old1", "new0"]
            joint = adapted
        else:
            adapted = ["old0", "old1", "old0"]
            joint = ["old0", "old1", "new0"]
        rows.append(
            {
                "row_id": binding["row_id"],
                "receiver_id": binding["receiver"],
                "k_shot": k_shot,
                "scene": binding["scene"],
                "opaque_query_ids": query_ids,
                "arms": {
                    "M0": _arm("M0", classes, base),
                    "M_DA": _arm("M_DA", classes, adapted),
                    "M_L92": _arm("M_L92", classes, base),
                    "M_JOINT": _arm("M_JOINT", classes, joint),
                },
                "joint_receipt": {},
                "hook_receipt": {},
                "da_resource": {},
            }
        )
    payload = {
        "schema": entry.LOCAL_WORKER_SCHEMA,
        "candidate_id": da.CANDIDATE_A,
        "evaluation_scope": "LOCAL_CANDIDATE_WORKER_NON_PUBLISHABLE",
        "truth_loaded": False,
        "row_count": 18,
        "rows_complete": True,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "query_selection_count": 0,
        "phase2_optimizer_steps": 0,
        "resource": {"total_id_backbone_forwards": 18, "total_query_rows": 36 if state == "before" else 54},
        "rows": rows,
    }
    payload["prediction_sha256"] = entry._sha256(payload)
    return payload


def _asset_receipt(plan: dict) -> dict:
    expected = plan["phase1_asset_expected_binding"]
    value = {
        "schema": one.PHASE1_ASSET_RECEIPT_SCHEMA,
        "manifest_sha256": _sha(110),
        "candidate_id": da.CANDIDATE_A,
        "method_lock_sha256": expected["method_lock_sha256"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "source_binding": expected["source_binding"],
        "qknn_lock_binding": expected["qknn_lock_binding"],
        "episode_manifest_sha256": _sha(111),
        "episode_contract_sha256": _sha(112),
        "candidate_asset": {"candidate_id": da.CANDIDATE_A, "persistent_fp32_sidecar": False},
    }
    value["asset_receipt_sha256"] = one.canonical_sha256(value)
    return value


def _prediction() -> tuple[dict, dict]:
    plan = _plan()
    prediction = one.build_d128_a_one18_prediction(
        prepared_plan=plan,
        before_worker=_worker(plan, state="before"),
        after_worker=_worker(plan, state="after"),
        phase1_asset_manifest_sha256=_sha(110),
        phase1_asset_receipt=_asset_receipt(plan),
    )
    return plan, prediction


def _resign_worker_and_prediction(prediction: dict, *, state: str) -> None:
    worker = prediction["states"][state]
    worker["prediction_sha256"] = entry._sha256({key: value for key, value in worker.items() if key != "prediction_sha256"})
    prediction["prediction_sha256"] = one.canonical_sha256({key: value for key, value in prediction.items() if key != "prediction_sha256"})


def test_a_only_prediction_closes_all_18_pairs_and_keeps_k1_alias() -> None:
    plan, prediction = _prediction()
    validated = one.validate_d128_a_one18_prediction(prediction, prepared_plan=plan)
    assert validated["candidate_id"] == da.CANDIDATE_A
    assert prediction["row_pair_count"] == 18 and prediction["state_row_count"] == 36
    assert prediction["physical_execution"]["candidate_workers"] == 1
    assert all(
        row["arms"]["M0"]["predictions"] == row["arms"]["M_L92"]["predictions"]
        and row["arms"]["M_DA"]["predictions"] == row["arms"]["M_JOINT"]["predictions"]
        for state in ("before", "after")
        for row in prediction["states"][state]["rows"]
        if row["k_shot"] == 1
    )


def test_prediction_rejects_forbidden_field_row_and_query_root_drift() -> None:
    plan, prediction = _prediction()
    forbidden = deepcopy(prediction)
    forbidden["states"]["before"]["rows"][0]["Query_Truth"] = ["bad"]
    _resign_worker_and_prediction(forbidden, state="before")
    with pytest.raises(one.D128AOne18Error, match="forbidden"):
        one.validate_d128_a_one18_prediction(forbidden, prepared_plan=plan)

    query_drift = deepcopy(prediction)
    query_drift["states"]["after"]["rows"][0]["opaque_query_ids"][0] = "different-opaque-id"
    _resign_worker_and_prediction(query_drift, state="after")
    with pytest.raises(one.D128AOne18Error, match="query-root"):
        one.validate_d128_a_one18_prediction(query_drift, prepared_plan=plan)

    row_drift = deepcopy(prediction)
    row_drift["states"]["before"]["rows"][0]["row_id"] = "wrong-row"
    _resign_worker_and_prediction(row_drift, state="before")
    with pytest.raises(one.D128AOne18Error, match="identity"):
        one.validate_d128_a_one18_prediction(row_drift, prepared_plan=plan)


def test_prediction_rejects_k1_alias_drift_and_exclusive_overwrite(tmp_path) -> None:
    plan, prediction = _prediction()
    alias_drift = deepcopy(prediction)
    row = next(row for row in alias_drift["states"]["after"]["rows"] if row["k_shot"] == 1)
    row["arms"]["M_L92"] = _arm("M_L92", row["arms"]["M0"]["classes"], ["old1", "old0", "new0"])
    _resign_worker_and_prediction(alias_drift, state="after")
    with pytest.raises(one.D128AOne18Error, match="K1 alias"):
        one.validate_d128_a_one18_prediction(alias_drift, prepared_plan=plan)
    target = tmp_path / "prediction.json"
    one.write_d128_a_one18_prediction_exclusive(target, prediction, prepared_plan=plan)
    loaded, observed = one.load_d128_a_one18_prediction(
        target,
        expected_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        prepared_plan=plan,
    )
    assert observed == hashlib.sha256(target.read_bytes()).hexdigest()
    assert loaded["prediction_sha256"] == prediction["prediction_sha256"]
    with pytest.raises(one.D128AOne18Error, match="already exists"):
        one.write_d128_a_one18_prediction_exclusive(target, prediction, prepared_plan=plan)


def test_single_a_loader_refuses_merged_or_non_a_bundle(monkeypatch) -> None:
    plan = _plan()
    manifest = {
        "bundle_kind": "merged_complete",
        "candidate_ids": [da.CANDIDATE_A, da.CANDIDATE_B, da.CANDIDATE_C],
        **plan["phase1_asset_expected_binding"],
    }
    monkeypatch.setattr(
        release,
        "_load_bundle_directory",
        lambda *_args, **_kwargs: (None, manifest, _sha(110), None, {}),
    )
    with pytest.raises(one.D128AOne18Error, match="exactly one A"):
        one.load_d128_a_single_candidate_asset(
            bundle_dir="unused", expected_manifest_sha256=_sha(110), prepared_plan=plan, device="cpu"
        )


def test_run_path_invokes_only_a_for_before_and_after(monkeypatch) -> None:
    plan = _plan()
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(adapter, "_assert_prepared_matches_plan", lambda *_args, **_kwargs: None)

    def fake_worker(*, candidate_id: str, rows, **_kwargs):
        calls.append((candidate_id, len(rows)))
        return _worker(plan, state="before" if len(calls) == 1 else "after")

    monkeypatch.setattr(entry, "_run_d127_s0_candidate_worker", fake_worker)
    prepared = SimpleNamespace(before=[SimpleNamespace(row=object())] * 18, after=[SimpleNamespace(row=object())] * 18)
    payload = one.run_d128_a_one18_prediction(
        model=object(),
        asset=object(),
        prepared=prepared,
        prepared_plan=plan,
        phase1_asset_manifest_sha256=_sha(110),
        phase1_asset_receipt=_asset_receipt(plan),
    )
    assert calls == [(da.CANDIDATE_A, 18), (da.CANDIDATE_A, 18)]
    assert payload["candidate_id"] == da.CANDIDATE_A
