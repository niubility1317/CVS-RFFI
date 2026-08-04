from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import stage2_next_r1_real as next_r1_real
from cvsrffi import stage2_next_r2_matrix as matrix
from cvsrffi import stage2_next_r2_real as real


RECEIVERS = tuple(f"rx-{index}" for index in range(7))
CLASSES = tuple(f"class-{index}" for index in range(6))
SELECTED_SHA = "c" * 64
SELECTED_RECEIPT_SHA = "d" * 64
LABEL_SHA = "e" * 64
PHYSICAL_ROOT = "a" * 64


def _builder_rows() -> next_r1_real.NextR1RealRows:
    receiver_ids: list[str] = []
    labels: list[str] = []
    physical_ids: list[str] = []
    for receiver in RECEIVERS:
        for class_id in CLASSES:
            for physical_index in range(14):
                receiver_ids.append(receiver)
                labels.append(class_id)
                physical_ids.append(f"physical-{receiver}-{class_id}-{physical_index}")
    row_count = len(physical_ids)
    iq = np.zeros((row_count, 2, 8), dtype=np.float32)
    iq[:, 0, :] = np.arange(row_count, dtype=np.float32)[:, None] / np.float32(1000.0)
    iq[:, 1, :] = np.float32(0.25)
    return next_r1_real.NextR1RealRows(
        received_iq=np.ascontiguousarray(iq),
        receiver_ids=tuple(receiver_ids),
        day_ids=tuple("day-source" for _ in range(row_count)),
        tx_labels=tuple(labels),
        physical_ids=tuple(physical_ids),
        scenario_names=tuple("leo_source_weak" for _ in range(row_count)),
        observation_ids=tuple(f"observation-{index}" for index in range(row_count)),
        receiver_registry=RECEIVERS,
        class_registry=CLASSES,
        receipt={"physical_id_root_sha256": PHYSICAL_ROOT},
    )


def _prediction_rows(builder: next_r1_real.NextR1RealRows) -> real.NextR2PredictionRows:
    return real.NextR2PredictionRows(
        received_iq=builder.received_iq,
        receiver_ids=builder.receiver_ids,
        physical_ids=builder.physical_ids,
        receiver_registry=builder.receiver_registry,
        receipt={
            "selected_iq_archive_sha256": SELECTED_SHA,
            "selected_iq_receipt_sha256": SELECTED_RECEIPT_SHA,
            "physical_id_root_sha256": PHYSICAL_ROOT,
            "label_join_opened": False,
            "query_labels_present": False,
        },
    )


def _capsule(builder: next_r1_real.NextR1RealRows):
    return real.build_next_r2_prediction_capsule(
        builder,
        capsule_id="capsule-fixed",
        split_id="split-fixed",
        selected_iq_archive_sha256=SELECTED_SHA,
        selected_iq_receipt_sha256=SELECTED_RECEIPT_SHA,
        label_join_archive_sha256=LABEL_SHA,
    )


def _fake_bridge(rows: real.NextR2PredictionRows) -> real.NextR2RealModelBridge:
    bridge = object.__new__(real.NextR2RealModelBridge)
    bridge.rows = rows
    bridge.checkpoint_sha256 = "b" * 64

    def forward(indices, *, quarter_sign=0):
        result = np.zeros((len(indices), 160), dtype=np.float32)
        for row_index, source_index in enumerate(indices):
            receiver_offset = RECEIVERS.index(rows.receiver_ids[int(source_index)])
            result[row_index, int(source_index) % 6] = np.float32(1.0)
            result[row_index, 20 + receiver_offset] = np.float32(
                (int(source_index) % 14 + 1) / 1000.0
            )
            result[row_index, 159] = np.float32(quarter_sign * 0.01)
        return np.ascontiguousarray(result)

    bridge.forward_indices = forward
    return bridge


def test_phase_views_use_same_iq_and_preserve_sample_energy() -> None:
    rng = np.random.default_rng(101)
    iq = np.ascontiguousarray(rng.normal(size=(4, 2, 16)).astype(np.float32))
    canonical = real.phase_rotate_received_iq(iq, 0)
    plus = real.phase_rotate_received_iq(iq, 1)
    minus = real.phase_rotate_received_iq(iq, -1)
    assert np.array_equal(canonical, iq)
    assert np.allclose(np.sum(plus * plus, axis=1), np.sum(iq * iq, axis=1), atol=2e-6)
    assert np.allclose(np.sum(minus * minus, axis=1), np.sum(iq * iq, axis=1), atol=2e-6)
    with pytest.raises(real.NextR2RealError):
        real.phase_rotate_received_iq(iq, 2)


def test_capsule_is_canonical_bound_and_contains_no_query_labels() -> None:
    builder = _builder_rows()
    rows = _prediction_rows(builder)
    capsule = _capsule(builder)
    validated = real.validate_next_r2_prediction_capsule(capsule, rows=rows)
    encoded = real.capsule_bytes(validated)
    assert encoded == matrix.canonical_bytes(json.loads(encoded.decode("utf-8")))
    assert b'"query_labels"' not in encoded
    assert validated["capsule_id"] == "capsule-fixed"
    assert validated["split_id"] == "split-fixed"
    assert validated["selected_iq_archive_sha256"] == SELECTED_SHA
    assert validated["label_join_archive_sha256"] == LABEL_SHA
    assert validated["physical_id_root_sha256"] == PHYSICAL_ROOT
    assert validated["matrix_sha256"] == validated["plan"]["matrix_sha256"]


def test_prediction_rows_have_no_truth_or_class_field() -> None:
    rows = _prediction_rows(_builder_rows())
    fields = set(rows.__dataclass_fields__)
    assert "tx_labels" not in fields
    assert "query_labels" not in fields
    assert rows.receipt["label_join_opened"] is False


def test_capsule_receiver_drift_fails_closed() -> None:
    builder = _builder_rows()
    rows = _prediction_rows(builder)
    payload = json.loads(real.capsule_bytes(_capsule(builder)).decode("utf-8"))
    first = payload["keys"][0]
    held_receiver = first["held_receiver"]
    foreign_index = next(
        index for index, receiver in enumerate(rows.receiver_ids) if receiver != held_receiver
    )
    registration = first["registrations"]["REG1"]
    registration["support_indices"][0] = foreign_index
    registration["support_physical_ids"][0] = rows.physical_ids[foreign_index]
    content = dict(payload)
    content.pop("capsule_content_sha256")
    payload["capsule_content_sha256"] = matrix.canonical_sha256(content)
    with pytest.raises(real.NextR2RealError, match="physical IDs/receiver"):
        real.validate_next_r2_prediction_capsule(payload, rows=rows)


@pytest.mark.parametrize("k", [1, 5])
def test_four_state_builder_uses_capsule_only_and_retained_queries(k: int) -> None:
    builder = _builder_rows()
    rows = _prediction_rows(builder)
    capsule = _capsule(builder)
    bridge = _fake_bridge(rows)
    plan = capsule["plan"]
    key = next(
        matrix.outer_key_from_mapping(item)
        for item in plan["keys"]
        if item["active_k"] == k
    )
    inputs = real.build_next_r2_four_state_inputs(
        rows, bridge, key, capsule=capsule
    )
    reg0 = inputs["DA0_REG0"]
    reg1 = inputs["DA0_REG1"]
    assert reg0.capsule_id == "capsule-fixed"
    assert reg0.split_id == "split-fixed"
    assert len(reg0.registered_classes) == 5
    assert key.held_class not in reg0.registered_classes
    assert len(reg0.support_physical_ids) == 5 * k
    assert len(reg1.support_physical_ids) == 6 * k
    assert len(reg0.query_physical_ids) == 45
    assert len(reg1.query_physical_ids) == 54
    assert set(reg0.support_physical_ids).issubset(reg1.support_physical_ids)
    assert set(reg0.query_physical_ids).issubset(reg1.query_physical_ids)
    assert not set(reg1.support_physical_ids) & set(reg1.query_physical_ids)


def test_capsule_k1_support_is_subset_of_k5_and_query_indices_match() -> None:
    capsule = _capsule(_builder_rows())
    grouped = {}
    for key in capsule["keys"]:
        grouped.setdefault((key["held_receiver"], key["held_class"]), {})[
            key["active_k"]
        ] = key
    for pair in grouped.values():
        for registration in ("REG0", "REG1"):
            first = pair[1]["registrations"][registration]
            fifth = pair[5]["registrations"][registration]
            assert set(first["support_indices"]).issubset(fifth["support_indices"])
            assert first["query_indices"] == fifth["query_indices"]


def test_predict_cli_has_no_label_join_argument() -> None:
    script = Path(__file__).parents[1] / "code" / "scripts" / "run_next_r2_proxy24.py"
    spec = importlib.util.spec_from_file_location("run_next_r2_proxy24_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module._parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    predict = subparsers.choices["predict"]
    destinations = {action.dest for action in predict._actions}
    assert "ls_join" not in destinations
    assert "ls_join_sha256" not in destinations
    assert "capsule" in destinations
    assert "capsule_sha256" in destinations


def test_real_smoke_is_repeatable_and_truth_free() -> None:
    rows = _prediction_rows(_builder_rows())
    bridge = _fake_bridge(rows)
    receipt = real.verified_next_r2_real_smoke(bridge, (0, 1))
    assert receipt["canonical_repeat_exact"] is True
    assert receipt["query_truth_access"] is False
