from dataclasses import fields
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi import stage2_d100_ra_cgspr_lgf as d100
from cvsrffi import stage2_d99_d100_query_evaluation as evaluation


def _parameters() -> dict[str, float]:
    return {
        "eta": 0.25,
        "student_nu": 3.0,
        "kernel_volume_gamma": 0.1,
        "shared_h0": 0.5,
        "scale_prior_strength": 2.0,
        "scale_min_ratio": 0.5,
        "scale_max_ratio": 2.0,
        "d99_temperature": 2.0,
        "lambda0": 0.2,
        "ridge_temperature": 4.0,
        "alpha": 0.4,
    }


def _lodo() -> dict:
    selected_by_k = {}
    locked_by_k = {}
    for k in evaluation.ALLOWED_K:
        parameters = {**_parameters(), "eta": 0.1 + k / 100.0}
        locked_by_k[str(k)] = parameters
        selected_by_k[str(k)] = {
            "selected": {
                "effective_parameters": parameters,
                "alpha_forced_zero": False,
                "guard": {
                    "bidirectional_rescue_nonzero": True,
                    "every_receiver_pseudo_new_pair_floor_old_new_h_non_decreasing": True,
                    "degraded_pair_count": 0,
                },
            }
        }
    unsigned = {
        "schema": "cvs.phase1.d99_d100_lodo_lock.v1",
        "status": evaluation.lodo.STATUS_DIAGNOSTIC,
        "formal_authority_status": evaluation.lodo.STATUS_BLOCKED,
        "formal_phase1_lock": False,
        "canonical_lock_artifact_write_allowed": False,
        "locked_parameters_by_k": locked_by_k,
        "selected_by_k": selected_by_k,
        "protocol_audit": {
            "phase1_only": True,
            "target_rows_used": 0,
            "query_rows_used_for_selection": 0,
            "clean_or_raw_iq_used": False,
            "class_specific_hyperparameters": False,
        },
    }
    return {**unsigned, "receipt_sha256": evaluation.lodo.canonical_sha256(unsigned)}


def test_lodo_parameters_are_k_specific_phase1_only_and_immutable() -> None:
    selected = evaluation.locked_parameters_from_lodo(_lodo(), k_shot=10)
    assert selected["eta"] == pytest.approx(0.2)
    with pytest.raises(TypeError):
        selected["eta"] = 0.9
    broken = _lodo()
    broken["protocol_audit"]["query_rows_used_for_selection"] = 1
    unsigned = {key: value for key, value in broken.items() if key != "receipt_sha256"}
    broken["receipt_sha256"] = evaluation.lodo.canonical_sha256(unsigned)
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="Phase1-only"):
        evaluation.locked_parameters_from_lodo(broken, k_shot=10)
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="K-specific"):
        evaluation.locked_parameters_from_lodo(_lodo(), k_shot=3)
    unsafe = _lodo()
    unsafe["selected_by_k"]["10"]["selected"]["guard"][
        "bidirectional_rescue_nonzero"
    ] = False
    unsigned = {key: value for key, value in unsafe.items() if key != "receipt_sha256"}
    unsafe["receipt_sha256"] = evaluation.lodo.canonical_sha256(unsigned)
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="unsafe"):
        evaluation.locked_parameters_from_lodo(unsafe, k_shot=10)


def test_cross_state_accepts_active_k_and_new_count_grid() -> None:
    old = tuple(f"old-{index}" for index in range(6))
    all_classes = old + ("new-0", "new-1")

    def manifest(state: str, classes: tuple[str, ...]) -> dict:
        return {
            "registration_state": state,
            "receiver": "rx",
            "seed": 1,
            "k_shot": 20,
            "phase1_checkpoint_sha256": "1",
            "feature_runtime_sha256": "2",
            "method_lock_sha256": "3",
            "row_handle": "row",
            "row_manifest_sha256": "4",
            "registered_classes": [{"class_handle": value} for value in classes],
        }

    before = manifest("before", old)
    after = manifest("after", all_classes)
    got_old, got_all, k = evaluation._require_cross_state_lock(
        before, before, after, after
    )
    assert got_old == old
    assert got_all == all_classes
    assert k == 20


def test_receipt_tamper_is_rejected_before_parameter_read() -> None:
    receipt = _lodo()
    receipt["locked_parameters_by_k"]["10"]["eta"] = 0.99
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="schema"):
        evaluation.locked_parameters_from_lodo(receipt, k_shot=10)


@pytest.mark.parametrize(
    "updates",
    (
        {
            "status": evaluation.lodo.STATUS_FORMAL,
            "formal_authority_status": evaluation.lodo.STATUS_BLOCKED,
            "formal_phase1_lock": False,
            "canonical_lock_artifact_write_allowed": False,
        },
        {"formal_phase1_lock": True},
    ),
)
def test_lodo_status_fields_must_be_jointly_consistent(updates) -> None:
    receipt = _lodo()
    receipt.update(updates)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = evaluation.lodo.canonical_sha256(unsigned)
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="status contract"):
        evaluation.locked_parameters_from_lodo(receipt, k_shot=10)


def test_class_binding_bijection_and_order_are_exact() -> None:
    handles = tuple(f"cls_{index}" for index in range(6))
    payload = {
        "schema": "cvs.phase2.d20_adv3b02_class_binding.v2",
        "checkpoint_sha256": "c" * 64,
        "entries": [
            {
                "class_index": index,
                "direct_logit_index": index,
                "phase1_tx": f"tx-{index}",
                "registered_class_handle": handle,
            }
            for index, handle in enumerate(handles)
        ],
    }
    raw = evaluation._canonical_bytes(payload)
    sha = hashlib.sha256(raw).hexdigest()
    tx_to_handle, handle_to_tx = evaluation.class_binding_maps(
        payload,
        payload_sha256=sha,
        payload_bytes=raw,
        checkpoint_sha256="c" * 64,
        old_handles=handles,
    )
    assert tuple(handle_to_tx[handle] for handle in handles) == tuple(
        f"tx-{index}" for index in range(6)
    )
    permuted = handles[1:] + handles[:1]
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="bijection"):
        evaluation.class_binding_maps(
            payload,
            payload_sha256=sha,
            payload_bytes=raw,
            checkpoint_sha256="c" * 64,
            old_handles=permuted,
        )


def test_d100_config_is_complete_and_bound_to_d99_digest() -> None:
    signature_fields = {field.name for field in fields(d99.Phase1D99Lock)}
    values = {
        name: "b" * 64
        for name in signature_fields
        if name.endswith("sha256")
    }
    values.update(
        density_tau=0.1,
        max_ground_rank=2,
        max_target_rank=2,
        coverage_floor=0.1,
        ground_energy_scale=0.2,
        target_energy_scale=0.2,
        shrinkage_prior_strength=2.0,
        ground_weight_max=0.5,
        target_weight_max=0.5,
        student_nu=3.0,
        kernel_effective_dim=32,
        kernel_volume_gamma=0.1,
        shared_h0=0.5,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        z_weight=0.6,
        fft_weight=0.3,
        rf_weight=0.1,
        eta_k1=0.1,
        eta_k5=0.2,
        eta_k10=0.3,
        eta_k20=0.4,
        eta_k20_lodo_artifact_sha256=None,
        ground_old_registry=tuple(f"old-{index}" for index in range(6)),
    )
    base = d99.Phase1D99Lock(**values)
    receipt = _lodo()
    config99 = evaluation._d99_config(base, _parameters(), receipt)
    assert config99.eta_k10 == pytest.approx(0.2)
    config100 = evaluation._d100_config(config99, receipt, "a" * 64)
    assert config100.d99_phase1_lock_digest == config99.lock_digest
    assert config100.values_for_k(10) == pytest.approx((0.2, 4.0, 2.0, 0.4))


def test_lodo_binding_rejects_fourteen_domain_bundle_and_any_receipt_drift() -> None:
    base = SimpleNamespace(lock_digest="d" * 64)
    aggregation = SimpleNamespace(receipt_sha256="e" * 64)
    bundle = SimpleNamespace(
        domain_ids=tuple(f"domain-{index}" for index in range(7)),
        bundle_sha256="f" * 64,
        aggregation_receipt=aggregation,
    )
    unsigned = _lodo()
    unsigned = {key: value for key, value in unsigned.items() if key != "receipt_sha256"}
    unsigned.update(
        ground={
            "domain_ids": list(bundle.domain_ids),
            "bundle_sha256": bundle.bundle_sha256,
            "aggregation_receipt_sha256": aggregation.receipt_sha256,
            "release_manifest_sha256": "1" * 64,
            "receiver_domain_map": {
                f"rx-{index}": f"domain-{index}" for index in range(7)
            },
        },
        base_d99_lock_digest=base.lock_digest,
        checkpoint_sha256="2" * 64,
        d81_scorer={
            "receipt": {
                "phase1_checkpoint_sha256": "2" * 64,
                "ground_manifest_sha256": "3" * 64,
                "ground_component_npz_sha256": "4" * 64,
            }
        },
    )
    receipt = {**unsigned, "receipt_sha256": evaluation.lodo.canonical_sha256(unsigned)}
    evaluation.validate_lodo_input_binding(
        receipt,
        bundle=bundle,
        ground_manifest_sha256="1" * 64,
        base_d99_lock=base,
        d81_ground_manifest_sha256="3" * 64,
        d81_ground_component_sha256="4" * 64,
        checkpoint_sha256="2" * 64,
    )
    bundle14 = SimpleNamespace(
        domain_ids=tuple(f"domain-{index}" for index in range(14)),
        bundle_sha256=bundle.bundle_sha256,
        aggregation_receipt=aggregation,
    )
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="binding"):
        evaluation.validate_lodo_input_binding(
            receipt,
            bundle=bundle14,
            ground_manifest_sha256="1" * 64,
            base_d99_lock=base,
            d81_ground_manifest_sha256="3" * 64,
            d81_ground_component_sha256="4" * 64,
            checkpoint_sha256="2" * 64,
        )
