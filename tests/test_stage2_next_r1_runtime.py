from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

from cvsrffi import stage2_next_r1_assets as assets
from cvsrffi import stage2_next_r1_fabr as fabr
from cvsrffi import stage2_next_r1_matrix as matrix
from cvsrffi import stage2_next_r1_runtime as runtime
from cvsrffi import stage2_next_r1_tsl as tsl


def _root(values: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _plan_rows():
    plan = matrix.build_next_r1_loco_plan(
        tuple(f"rx{index}" for index in range(matrix.RECEIVER_COUNT)),
        tuple(f"tx{index}" for index in range(matrix.CLASS_COUNT)),
    )
    rows = tuple(
        matrix.NextR1LocoRow(
            row_id=value["row_id"],
            held_receiver=value["held_receiver"],
            held_class=value["held_class"],
            active_k=value["active_k"],
            retained_classes=tuple(value["retained_classes"]),
            registered_classes=tuple(value["registered_classes"]),
        )
        for value in plan["rows"]
    )
    return plan, rows


def _binding(row_k1: matrix.NextR1LocoRow, row_k5: matrix.NextR1LocoRow):
    receivers = tuple(f"rx{index}" for index in range(matrix.RECEIVER_COUNT))
    classes = tuple(f"tx{index}" for index in range(matrix.CLASS_COUNT))
    cells = {
        (receiver, class_id): tuple(
            f"p-{receiver}-{class_id}-{index:02d}"
            for index in range(matrix.PHYSICAL_PER_CELL)
        )
        for receiver in receivers
        for class_id in classes
    }

    def ordered(receiver: str, class_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                cells[(receiver, class_id)],
                key=lambda physical_id: hashlib.sha256(
                    f"next-r1|{receiver}|{class_id}|{physical_id}".encode("utf-8")
                ).hexdigest(),
            )
        )

    phase1_fit = tuple(
        physical_id
        for receiver in receivers
        if receiver != row_k1.held_receiver
        for class_id in classes
        if class_id != row_k1.held_class
        for physical_id in ordered(receiver, class_id)
    )
    support5 = {
        class_id: ordered(row_k1.held_receiver, class_id)[:5]
        for class_id in row_k1.registered_classes
    }
    support1 = {class_id: values[:1] for class_id, values in support5.items()}
    query = {
        class_id: ordered(row_k1.held_receiver, class_id)[5:]
        for class_id in row_k1.registered_classes
    }
    support1_ordered = tuple(
        item for class_id in row_k1.registered_classes for item in support1[class_id]
    )
    support5_ordered = tuple(
        item for class_id in row_k1.registered_classes for item in support5[class_id]
    )
    query_ordered = tuple(
        item for class_id in row_k1.registered_classes for item in query[class_id]
    )
    receipt = {
        "held_receiver": row_k1.held_receiver,
        "held_class": row_k1.held_class,
        "phase1_fit_count": len(phase1_fit),
        "phase1_fit_physical_root_sha256": _root(phase1_fit),
        "support_k1_count": len(support1_ordered),
        "support_k1_physical_root_sha256": _root(support1_ordered),
        "support_k5_count": len(support5_ordered),
        "support_k5_physical_root_sha256": _root(support5_ordered),
        "outer_query_count": len(query_ordered),
        "outer_query_physical_root_sha256": _root(query_ordered),
        "k1_is_k5_prefix": True,
    }
    binding = matrix.bind_next_r1_physical_ids(
        row_k1=row_k1,
        row_k5=row_k5,
        loco_fold_receipt=receipt,
        phase1_fit_ids=phase1_fit,
        k1_support_ids_by_class=support1,
        k5_support_ids_by_class=support5,
        query_ids_by_class=query,
    )
    return binding, support1, support5, query


def _asset(phase1_seal_sha256: str) -> fabr.FABRAsset:
    basis = np.zeros((fabr.BLOCK_DIMENSIONS["t1_norm_affine"], fabr.RANK), dtype=np.int8)
    basis[0, 0] = 64
    basis[1, 1] = 64
    return fabr.FABRAsset(
        checkpoint_sha256="1" * 64,
        phase1_seal_sha256=phase1_seal_sha256,
        phase1_selection_sha256="2" * 64,
        block_id="t1_norm_affine",
        basis_qint8=basis,
        basis_scale_fp16=np.asarray((1.0 / 64.0, 1.0 / 64.0), dtype=np.float16),
        fisher_k_fp16=np.asarray(((1.0, 0.1), (0.1, 1.5)), dtype=np.float16),
        forward_jitter_tolerance_fp16=np.asarray((0.0,), dtype=np.float16),
    )


def _bundle(phase1_seal_sha256: str) -> assets.NextR1Phase1AssetBundle:
    asset = _asset(phase1_seal_sha256)
    prior = tsl.TSLPhase1Prior(
        q_logv0=np.zeros((tsl.Z_DIM,), dtype=np.int8),
        scale_logv0=np.float16(0.01),
        offset_logv0=np.float16(0.0),
        nu0=np.float16(2.0),
        rho_h=np.float16(1.0),
        checkpoint_sha256=asset.checkpoint_sha256,
        cell_physical_id_root_sha256="3" * 64,
        representation_rule_sha256=asset.representation_rule_sha256,
    )
    receipt = {
        "schema": assets.BUNDLE_SCHEMA,
        "selection_sha256": asset.phase1_selection_sha256,
        "fold_seal_sha256": "4" * 64,
        "checkpoint_sha256": asset.checkpoint_sha256,
        "representation_rule_sha256": asset.representation_rule_sha256,
        "row_phase1_seal_sha256": phase1_seal_sha256,
        "phase1_cell_physical_id_root_sha256": prior.cell_physical_id_root_sha256,
        "phase1_receiver_registry_sha256": "5" * 64,
        "phase1_class_registry_sha256": "6" * 64,
        "selected_block_id": asset.block_id,
        "tsl_prior_sha256": prior.prior_sha256,
    }
    return assets.NextR1Phase1AssetBundle(
        fabr_asset=asset,
        tsl_prior=prior,
        fold_seal_sha256=receipt["fold_seal_sha256"],
        phase1_receiver_registry_sha256=receipt["phase1_receiver_registry_sha256"],
        phase1_class_registry_sha256=receipt["phase1_class_registry_sha256"],
        receipt=receipt,
    )


def _external_smoke(
    bundle: assets.NextR1Phase1AssetBundle,
) -> runtime.NextR1VerifiedCheckpointSmoke:
    payload = {
        "schema": runtime.CHECKPOINT_SMOKE_SCHEMA,
        "completed": True,
        "actual_checkpoint_sha256": bundle.receipt["checkpoint_sha256"],
        "representation_rule_sha256": bundle.receipt["representation_rule_sha256"],
        "builder_bundle_sha256": bundle.bundle_sha256,
        "row_phase1_seal_sha256": bundle.receipt["row_phase1_seal_sha256"],
    }
    payload["checkpoint_smoke_receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return runtime.verify_next_r1_checkpoint_smoke(
        payload,
        bundle=bundle,
        row_phase1_seal_sha256=bundle.receipt["row_phase1_seal_sha256"],
    )


def _token(ids: tuple[str, ...], labels: tuple[str, ...] | None, classes: tuple[str, ...], seed: int):
    rng = np.random.default_rng(seed)
    count = len(ids)
    base = rng.normal(scale=0.04, size=(count, fabr.Z_DIM)).astype(np.float32)
    if labels is None:
        for index in range(count):
            base[index, index % len(classes)] += np.float32(1.0)
    else:
        index_of = {class_id: index for index, class_id in enumerate(classes)}
        for index, label in enumerate(labels):
            base[index, index_of[label]] += np.float32(1.0)
    direction = rng.normal(scale=0.03, size=(fabr.RANK, count, fabr.Z_DIM)).astype(np.float32)
    direction[0, :, 7] += np.linspace(-0.25, 0.25, count, dtype=np.float32)
    direction[1, :, 11] += np.arange(count, dtype=np.float32) / max(count, 1)
    return {"base": base, "direction": direction, "physical_ids": ids}


def _forward(token: dict[str, object], coefficient: np.ndarray) -> fabr.FABRForwardBatch:
    raw = np.asarray(token["base"], dtype=np.float32) + np.einsum(
        "r,rnd->nd", coefficient, np.asarray(token["direction"], dtype=np.float32)
    )
    return fabr.FABRForwardBatch(
        fabr.signed_pre_relu160(np.asarray(raw, dtype=np.float32)),
        tuple(token["physical_ids"]),
    )


def _q_logits(context: runtime.NextR1ArmContext) -> np.ndarray:
    logits = np.full(
        (len(context.query.physical_ids), len(context.registered_classes)),
        np.float32(-1.0),
        dtype=np.float32,
    )
    logits[np.arange(len(logits)), np.arange(len(logits)) % logits.shape[1]] = np.float32(2.0)
    return logits


def _runtime_inputs(active_k: int):
    plan, rows = _plan_rows()
    row_k1, row_k5 = rows[:2]
    row = row_k1 if active_k == 1 else row_k5
    binding, support1, support5, query = _binding(row_k1, row_k5)
    support_map = support1 if active_k == 1 else support5
    support_ids = tuple(
        item for class_id in row.registered_classes for item in support_map[class_id]
    )
    labels = tuple(
        class_id for class_id in row.registered_classes for _ in support_map[class_id]
    )
    query_ids = tuple(item for class_id in row.registered_classes for item in query[class_id])
    bundle = _bundle(binding["phase1_seal_sha256"])
    return {
        "plan": plan,
        "binding": binding,
        "row": row,
        "bundle": bundle,
        "verified_checkpoint_smoke": runtime.NextR1VerifiedCheckpointSmoke.for_test_only(
            bundle=bundle,
            row_phase1_seal_sha256=binding["phase1_seal_sha256"],
        ),
        "support_token": _token(support_ids, labels, row.registered_classes, 100 + active_k),
        "query_token": _token(query_ids, None, row.registered_classes, 200 + active_k),
        "support_labels": labels,
    }


def _execute(**kwargs: object) -> runtime.NextR1RuntimeResult:
    return runtime._execute_next_r1_row_impl(**kwargs, allow_test_smoke=True)


def test_k1_is_exact_qfl_alias_and_r0_is_common() -> None:
    values = _runtime_inputs(1)
    events: list[str] = []
    contexts: dict[str, runtime.NextR1ArmContext] = {}
    frozen_f_calls = 0

    def support_forward(token: dict[str, object], coefficient: np.ndarray) -> fabr.FABRForwardBatch:
        events.append("support")
        return _forward(token, coefficient)

    def query_forward(token: dict[str, object], coefficient: np.ndarray) -> fabr.FABRForwardBatch:
        events.append("query")
        return _forward(token, coefficient)

    def q_callback(context: runtime.NextR1ArmContext) -> np.ndarray:
        events.append("q")
        contexts[context.representation_id] = context
        return _q_logits(context)

    def frozen_f_callback(_context: runtime.NextR1ArmContext) -> np.ndarray:
        nonlocal frozen_f_calls
        frozen_f_calls += 1
        raise AssertionError("K1 F must remain an exact Q alias")

    result = _execute(
        **values,
        support_forward_with_coeff=support_forward,
        query_forward_with_coeff=query_forward,
        q_callback=q_callback,
        frozen_f_callback=frozen_f_callback,
        frozen_f_archive_sha256="4" * 64,
    )

    assert frozen_f_calls == 0
    assert events[:6] == ["support"] * 6
    assert events[6:8] == ["query", "query"]
    assert result.contexts["R0"] is contexts["R0"]
    assert result.contexts["R1"] is contexts["R1"]
    for representation in ("R0", "R1"):
        q_arm, f_arm, l_arm = (f"{representation}Q", f"{representation}F", f"{representation}L")
        assert result.arm_logits[q_arm] is result.arm_logits[f_arm]
        assert result.arm_logits[q_arm] is result.arm_logits[l_arm]
        assert result.arm_logits[q_arm].dtype == np.float32
    assert result.forward_receipt["support_base_forward_calls"] == 1
    assert result.forward_receipt["support_perturbation_forward_calls"] == 4
    assert result.forward_receipt["support_final_forward_calls"] == 1
    assert result.forward_receipt["r0_query_forward_calls"] == 1
    assert result.forward_receipt["r1_query_forward_calls"] == 1
    assert result.resource_receipt["query_rows_used_for_fit"] == 0
    assert result.resource_receipt["query_state_updates"] == 0
    assert result.resource_receipt["query_selection_count"] == 0
    assert result.prediction_receipt["k1_qfl_exact_alias"] is True
    assert result.smoke_receipt["actual_checkpoint_archive_smoke_required"] is True
    assert result.smoke_receipt["actual_checkpoint_archive_smoke_completed"] is True
    assert result.smoke_receipt["checkpoint_smoke_verification_mode"] == "test_only_synthetic"


def test_k5_uses_same_cache_for_qf_and_fails_closed_on_f_tie() -> None:
    values = _runtime_inputs(5)
    q_contexts: dict[str, runtime.NextR1ArmContext] = {}
    f_contexts: dict[str, runtime.NextR1ArmContext] = {}

    def q_callback(context: runtime.NextR1ArmContext) -> np.ndarray:
        q_contexts[context.representation_id] = context
        return _q_logits(context)

    def tie_f_callback(context: runtime.NextR1ArmContext) -> np.ndarray:
        f_contexts[context.representation_id] = context
        return np.zeros(
            (len(context.query.physical_ids), len(context.registered_classes)), dtype=np.float32
        )

    with pytest.raises(runtime.NextR1RuntimeError, match="R0F exact float32 top-tie"):
        _execute(
            **values,
            support_forward_with_coeff=_forward,
            query_forward_with_coeff=_forward,
            q_callback=q_callback,
            frozen_f_callback=tie_f_callback,
            frozen_f_archive_sha256="4" * 64,
        )
    assert q_contexts["R0"] is f_contexts["R0"]
    assert q_contexts["R1"] is f_contexts["R1"]


def test_all_six_arms_receive_strict_top_tie_closure_and_manifest_requires_84(monkeypatch) -> None:
    values = _runtime_inputs(1)
    calls: list[np.ndarray] = []
    original = fabr.strict_top1_predictions

    def strict(value: object) -> np.ndarray:
        calls.append(np.asarray(value))
        return original(value)

    monkeypatch.setattr(runtime.fabr, "strict_top1_predictions", strict)
    result = _execute(
        **values,
        support_forward_with_coeff=_forward,
        query_forward_with_coeff=_forward,
        q_callback=_q_logits,
        frozen_f_callback=_q_logits,
        frozen_f_archive_sha256="4" * 64,
    )
    assert len(calls) == len(matrix.ARM_IDS)
    with pytest.raises(runtime.NextR1RuntimeError, match="all 84 runtime results"):
        runtime.build_next_r1_sealed_manifest(values["plan"], (result,))

    values_k5 = _runtime_inputs(5)
    result_k5 = _execute(
        **values_k5,
        support_forward_with_coeff=_forward,
        query_forward_with_coeff=_forward,
        q_callback=_q_logits,
        frozen_f_callback=_q_logits,
        frozen_f_archive_sha256="4" * 64,
    )
    small_plan = dict(values["plan"])
    small_plan["rows"] = list(values["plan"]["rows"][:2])
    monkeypatch.setattr(runtime.matrix, "validate_next_r1_plan", lambda _value: small_plan)
    monkeypatch.setattr(runtime.matrix, "ROW_COUNT", 2)
    with pytest.raises(runtime.NextR1RuntimeError, match="exact runtime results"):
        runtime.build_next_r1_sealed_manifest(
            values["plan"], (result.row_seal, result_k5.row_seal)
        )
    manifest = runtime.build_next_r1_sealed_manifest(values["plan"], (result, result_k5))
    assert manifest["all_rows_sealed"] is True
    assert manifest["row_count"] == matrix.ROW_COUNT


def test_runtime_api_exposes_no_forbidden_query_side_inputs_and_rejects_seal_drift() -> None:
    names = set(inspect.signature(runtime.execute_next_r1_row).parameters)
    forbidden = ("query_label", "truth", "role", "quota", "source", "clean", "old_count")
    assert not any(token in name for name in names for token in forbidden)
    values = _runtime_inputs(1)
    wrong = _bundle("f" * 64)
    with pytest.raises(runtime.NextR1RuntimeError, match="row/Phase1 seal"):
        _execute(
            **{**values, "bundle": wrong},
            support_forward_with_coeff=_forward,
            query_forward_with_coeff=_forward,
            q_callback=_q_logits,
            frozen_f_callback=_q_logits,
            frozen_f_archive_sha256="4" * 64,
        )


def test_public_runtime_rejects_test_smoke_and_accepts_verified_external_receipt() -> None:
    values = _runtime_inputs(1)
    call = {
        "support_forward_with_coeff": _forward,
        "query_forward_with_coeff": _forward,
        "q_callback": _q_logits,
        "frozen_f_callback": _q_logits,
        "frozen_f_archive_sha256": "4" * 64,
    }
    with pytest.raises(runtime.NextR1RuntimeError, match="rejects test-only"):
        runtime.execute_next_r1_row(**values, **call)
    values["verified_checkpoint_smoke"] = _external_smoke(values["bundle"])
    result = runtime.execute_next_r1_row(**values, **call)
    assert result.smoke_receipt["checkpoint_smoke_verification_mode"] == "verified_external_receipt"
    assert result.smoke_receipt["actual_checkpoint_archive_smoke_completed"] is True
