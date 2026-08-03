from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvsrffi import stage2_d127_s0_entry as entry
from cvsrffi import stage2_zid_student_t_qknn as qknn


def _lock(k: int) -> qknn.Phase1ZIDStudentTLock:
    return qknn.Phase1ZIDStudentTLock(
        active_k=k,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256="1" * 64,
        quantization_margin_audit_sha256="2" * 64,
    )


def _rows() -> tuple[entry.D127S0Row, ...]:
    rows = []
    classes = ("c0", "c1")
    for receiver in ("r0", "r1", "r2"):
        for k in (1, 5):
            for scene in ("s0", "s1", "s2"):
                labels = tuple(label for label in classes for _ in range(k))
                rows.append(
                    entry.D127S0Row(
                        row_id=f"{receiver}.k{k}.{scene}",
                        receiver_id=receiver,
                        k_shot=k,
                        scene=scene,
                        support_iq=torch.zeros((2 * k, 2, 8), dtype=torch.float32),
                        query_iq=torch.zeros((3, 2, 8), dtype=torch.float32),
                        support_labels=labels,
                        registered_classes=classes,
                        opaque_query_ids=tuple(f"{receiver}.{k}.{scene}.q{i}" for i in range(3)),
                        qknn_lock=_lock(k),
                    )
                )
    return tuple(rows)


class _Receipt:
    protocol_closed = True
    adapter_macs_per_sample = 1280

    def as_dict(self):
        return {
            "protocol_closed": True,
            "adapter_macs_per_sample": self.adapter_macs_per_sample,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        }


class _HookReceipt:
    total_id_backbone_forwards = 5

    def as_dict(self):
        return {
            "total_id_backbone_forwards": 5,
            "query_rows_used_for_fit": 0,
            "query_state_updates": 0,
            "query_selection_count": 0,
        }


def test_complete_18_row_truth_free_entry_and_exclusive_writer(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_materialize(model, support_iq, support_labels, query_iq, *, asset):
        calls.append((len(support_iq), len(query_iq), tuple(support_labels.tolist())))
        base = SimpleNamespace(
            support_zid=np.zeros((len(support_iq), 160), dtype=np.float32),
            query_zid=np.zeros((len(query_iq), 160), dtype=np.float32),
        )
        adapted = SimpleNamespace(
            support_zid=np.ones((len(support_iq), 160), dtype=np.float32),
            query_zid=np.ones((len(query_iq), 160), dtype=np.float32),
        )
        return SimpleNamespace(
            base_cache=base,
            adapted_cache=adapted,
            state=SimpleNamespace(receipt=_Receipt()),
            hook_receipt=_HookReceipt(),
        )

    def fake_joint(**kwargs):
        arms = []
        for arm_id in entry.ARM_IDS:
            logits = np.asarray([[2.0, 1.0]] * len(kwargs["opaque_query_ids"]), dtype=np.float32)
            arms.append(
                SimpleNamespace(
                    arm_id=arm_id,
                    representation="base_zid160" if arm_id in ("M0", "M_L92") else "adapted_zid160",
                    head="fake",
                    classes=tuple(kwargs["registered_classes"]),
                    logits=logits,
                    predictions=tuple("c0" for _ in kwargs["opaque_query_ids"]),
                    receipt={"active_k": kwargs["qknn_lock"].active_k, "query_rows_used_for_fit": 0},
                )
            )
        return SimpleNamespace(
            arms=tuple(arms),
            receipt={"active_k": kwargs["qknn_lock"].active_k, "row_input_sha256": "a" * 64},
        )

    monkeypatch.setattr(entry.hooks, "materialize_d127_candidate", fake_materialize)
    monkeypatch.setattr(entry.joint, "run_d127_joint_four_arm", fake_joint)
    asset = SimpleNamespace(candidate_id="DA-C-RDHA-joint_proj")
    monkeypatch.setattr(entry, "_asset_candidate", lambda value: value.candidate_id)
    monkeypatch.setattr(entry, "_asset_payload_bytes", lambda value: 1328)
    payload = entry._run_d127_s0_candidate_worker(
        model=SimpleNamespace(),
        candidate_id="DA-C-RDHA-joint_proj",
        asset=asset,
        rows=_rows(),
    )
    assert len(calls) == 18
    assert payload["schema"] == entry.LOCAL_WORKER_SCHEMA
    assert payload["row_count"] == 18 and payload["rows_complete"] is True
    assert payload["truth_loaded"] is False
    assert payload["query_rows_used_for_fit"] == 0
    assert payload["resource"] == {
        "asset_numeric_payload_bytes": 1328,
        "adapter_macs_per_sample": 1280,
        "total_adapter_macs_support_plus_query": 1280 * (3 * (2 * 1 + 3) + 3 * (2 * 5 + 3)) * 3,
        "total_id_backbone_forwards": 90,
        "total_query_rows": 54,
    }
    assert [row["row_id"] for row in payload["rows"]] == [row.row_id for row in _rows()]
    assert all(tuple(row["arms"]) == entry.ARM_IDS for row in payload["rows"])
    with pytest.raises(entry.D127S0EntryError, match="invalid S0"):
        entry.write_d127_s0_predictions_exclusive(tmp_path / "worker.json", payload)


def test_matrix_and_public_field_fail_closed(tmp_path: Path) -> None:
    rows = _rows()
    with pytest.raises(entry.D127S0EntryError, match="exactly 18"):
        entry._validate_rows(rows[:-1])
    with pytest.raises(entry.D127S0EntryError, match="lexical order"):
        entry._validate_rows(tuple(reversed(rows)))
    bad = {
        "schema": entry.SCHEMA,
        "truth_loaded": False,
        "truth": [],
        "prediction_sha256": "0" * 64,
    }
    with pytest.raises(entry.D127S0EntryError, match="forbidden"):
        entry.write_d127_s0_predictions_exclusive(tmp_path / "bad.json", bad)

    tampered = {
        "schema": entry.SCHEMA,
        "truth_loaded": False,
        "prediction_sha256": "0" * 64,
    }
    with pytest.raises(entry.D127S0EntryError, match="digest drift"):
        entry.write_d127_s0_predictions_exclusive(tmp_path / "tampered.json", tampered)

    class HiddenTruth:
        def as_dict(self):
            return {"Query_Truth": ["c0"]}

    hidden = {
        "schema": entry.SCHEMA,
        "truth_loaded": False,
        "nested": HiddenTruth(),
    }
    hidden["prediction_sha256"] = entry._sha256(hidden)
    with pytest.raises(entry.D127S0EntryError, match="forbidden"):
        entry.write_d127_s0_predictions_exclusive(tmp_path / "hidden.json", hidden)


def test_row_has_no_truth_role_or_quota_fields() -> None:
    fields = set(entry.D127S0Row.__dataclass_fields__)
    assert not fields.intersection(entry.FORBIDDEN_PUBLIC_FIELDS)


def test_three_candidate_matrix_computes_common_pair_once(monkeypatch, tmp_path: Path) -> None:
    counts = {"materialize": 0, "common": 0, "adapted": 0}

    def fake_materialize(model, support_iq, support_labels, query_iq, *, asset):
        counts["materialize"] += 1
        base = SimpleNamespace(
            support_zid=np.zeros((len(support_iq), 160), dtype=np.float32),
            query_zid=np.zeros((len(query_iq), 160), dtype=np.float32),
        )
        adapted_value = float(entry.CANDIDATE_IDS.index(asset.candidate_id) + 1)
        adapted = SimpleNamespace(
            support_zid=np.full((len(support_iq), 160), adapted_value, dtype=np.float32),
            query_zid=np.full((len(query_iq), 160), adapted_value, dtype=np.float32),
        )
        return SimpleNamespace(
            base_cache=base,
            adapted_cache=adapted,
            state=SimpleNamespace(receipt=_Receipt()),
            hook_receipt=_HookReceipt(),
        )

    def arm(arm_id, query_ids, classes):
        logits = np.asarray([[2.0, 1.0]] * len(query_ids), dtype=np.float32)
        return SimpleNamespace(
            arm_id=arm_id,
            representation="base_zid160" if arm_id in ("M0", "M_L92") else "adapted_zid160",
            head="fake",
            classes=tuple(classes),
            logits=logits,
            predictions=tuple("c0" for _ in query_ids),
            receipt={"query_rows_used_for_fit": 0},
        )

    def fake_common(**kwargs):
        counts["common"] += 1
        arms = (
            arm("M0", kwargs["opaque_query_ids"], kwargs["registered_classes"]),
            arm("M_L92", kwargs["opaque_query_ids"], kwargs["registered_classes"]),
        )
        return SimpleNamespace(arms=arms, receipt={"pair_kind": "common"})

    def fake_adapted(**kwargs):
        counts["adapted"] += 1
        arms = (
            arm("M_DA", kwargs["opaque_query_ids"], kwargs["registered_classes"]),
            arm("M_JOINT", kwargs["opaque_query_ids"], kwargs["registered_classes"]),
        )
        return SimpleNamespace(arms=arms, receipt={"pair_kind": "adapted"})

    monkeypatch.setattr(entry.hooks, "materialize_d127_candidate", fake_materialize)
    monkeypatch.setattr(entry.joint, "run_d127_common_two_arm", fake_common)
    monkeypatch.setattr(entry.joint, "run_d127_adapted_two_arm", fake_adapted)
    monkeypatch.setattr(entry, "_asset_candidate", lambda value: value.candidate_id)
    monkeypatch.setattr(entry, "_asset_payload_bytes", lambda value: 100)
    assets = {
        candidate_id: SimpleNamespace(candidate_id=candidate_id)
        for candidate_id in entry.CANDIDATE_IDS
    }
    payload = entry.run_d127_s0_matrix(
        model=SimpleNamespace(), assets_by_candidate=assets, rows=_rows()
    )
    assert counts == {"materialize": 54, "common": 18, "adapted": 54}
    assert payload["execution_counts"] == {
        "candidate_materializations": 54,
        "common_two_arm_calls": 18,
        "adapted_two_arm_calls": 54,
    }
    assert all(tuple(row["common_arms"]) == ("M0", "M_L92") for row in payload["rows"])
    assert all(tuple(row["candidates"]) == entry.CANDIDATE_IDS for row in payload["rows"])
    class ChangingValue:
        def __init__(self):
            self.calls = 0

        def as_dict(self):
            self.calls += 1
            return {"safe": 1} if self.calls == 1 else {"Query_Truth": ["c0"]}

    changing = ChangingValue()
    safe_payload = dict(payload)
    safe_payload["dynamic"] = changing
    digest_input = dict(payload)
    digest_input.pop("prediction_sha256")
    digest_input["dynamic"] = {"safe": 1}
    safe_payload["prediction_sha256"] = entry._sha256(digest_input)
    output = tmp_path / "matrix.json"
    entry.write_d127_s0_predictions_exclusive(output, safe_payload)
    assert changing.calls == 1
    assert "Query_Truth" not in output.read_text(encoding="utf-8")
    with pytest.raises(entry.D127S0EntryError, match="already exists"):
        entry.write_d127_s0_predictions_exclusive(output, payload)

    def drifting_materialize(model, support_iq, support_labels, query_iq, *, asset):
        result = fake_materialize(model, support_iq, support_labels, query_iq, asset=asset)
        if asset.candidate_id == entry.CANDIDATE_IDS[-1]:
            result.base_cache.support_zid[0, 0] = np.float32(1.0)
        return result

    monkeypatch.setattr(entry.hooks, "materialize_d127_candidate", drifting_materialize)
    with pytest.raises(entry.D127S0EntryError, match="base z160 drifted"):
        entry.run_d127_s0_matrix(
            model=SimpleNamespace(), assets_by_candidate=assets, rows=_rows()
        )
