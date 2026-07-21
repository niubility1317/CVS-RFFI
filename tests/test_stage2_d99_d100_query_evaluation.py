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


def _support_payload(
    classes: tuple[str, ...], *, k_shot: int, order: tuple[int, ...] | None = None
) -> dict[str, np.ndarray]:
    pairs = [
        (class_index, rank)
        for class_index in range(len(classes))
        for rank in range(k_shot)
    ]
    if order is not None:
        pairs = [pairs[index] for index in order]
    tokens = np.asarray(
        [f"sample-{class_index}-{rank}" for class_index, rank in pairs], dtype=str
    )
    iq = np.stack(
        [
            np.full((2, 4), class_index * 10 + rank, dtype=np.float32)
            for class_index, rank in pairs
        ]
    )
    return {
        "support_rank_within_class": np.asarray([rank for _, rank in pairs], dtype=np.int64),
        "support_class_indices": np.asarray(
            [class_index for class_index, _ in pairs], dtype=np.int64
        ),
        "support_tokens": tokens,
        "support_leo_weak_iq": iq,
    }


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
        target_old_tx_labels=tuple(f"tx-{index}" for index in range(6)),
    )
    assert tuple(handle_to_tx[handle] for handle in handles) == tuple(
        f"tx-{index}" for index in range(6)
    )
    permuted = handles[1:] + handles[:1]
    remapped, _ = evaluation.class_binding_maps(
        payload,
        payload_sha256=sha,
        payload_bytes=raw,
        checkpoint_sha256="c" * 64,
        old_handles=permuted,
        target_old_tx_labels=tuple(f"tx-{index}" for index in range(6)),
    )
    assert tuple(remapped.values()) == permuted
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="TX order"):
        evaluation.class_binding_maps(
            payload,
            payload_sha256=sha,
            payload_bytes=raw,
            checkpoint_sha256="c" * 64,
            old_handles=handles,
            target_old_tx_labels=tuple(f"tx-{index}" for index in range(1, 6)) + ("tx-0",),
        )


def test_shared_support_features_use_one_forward_and_allow_before_permutation(
    monkeypatch,
) -> None:
    old_classes = ("old-0", "old-1")
    all_classes = old_classes + ("new-0",)
    before = _support_payload(old_classes, k_shot=2, order=(3, 0, 2, 1))
    after = _support_payload(all_classes, k_shot=2)
    calls = []

    def forward(_model, iq, *, device, batch_size):
        calls.append((np.asarray(iq).copy(), device, batch_size))
        return np.arange(len(iq), dtype=np.float32)[:, None]

    def register(iq, zid):
        return np.concatenate(
            [np.asarray(iq).reshape(len(iq), -1), np.asarray(zid)], axis=1
        ).astype(np.float32)

    monkeypatch.setattr(evaluation, "forward_zid160", forward)
    monkeypatch.setattr(evaluation, "registered_feature", register)
    result = evaluation._shared_before_after_support_features(
        before,
        after,
        model=object(),
        runtime_device="cuda",
        old_class_handles=old_classes,
        all_class_handles=all_classes,
        k_shot=2,
    )
    old_x, old_y, old_ids, old_indices, all_x, all_y, all_ids, all_indices, audit = result
    assert len(calls) == 1
    assert np.array_equal(calls[0][0], after["support_leo_weak_iq"])
    assert tuple(old_ids) == tuple(before["support_tokens"])
    positions = {token: index for index, token in enumerate(all_ids.tolist())}
    aligned = np.asarray([positions[token] for token in old_ids.tolist()])
    assert np.array_equal(old_x, all_x[aligned])
    assert np.array_equal(old_y, np.asarray(old_classes)[old_indices])
    assert np.array_equal(all_y, np.asarray(all_classes)[all_indices])
    assert audit == {
        "support_forward_count": 1,
        "old_support_feature_reused_from_after": True,
        "old_support_raw_iq_exact": True,
        "old_support_count": 4,
        "all_support_count": 6,
    }


@pytest.mark.parametrize(
    "mutation,error",
    (
        ("duplicate_token", "assignment/token"),
        ("missing_token", "token set"),
        ("index_drift", "exact reuse"),
        ("iq_drift", "exact reuse"),
        ("dtype_drift", "assignment/token"),
        ("selected_nonfinite", "assignment/token"),
        ("negative_rank", "assignment/token"),
        ("duplicate_rank", "assignment/token"),
    ),
)
def test_shared_support_features_reject_identity_and_kshot_drift(
    monkeypatch, mutation, error
) -> None:
    old_classes = ("old-0", "old-1")
    all_classes = old_classes + ("new-0",)
    before = _support_payload(old_classes, k_shot=2)
    after = _support_payload(all_classes, k_shot=2)
    if mutation == "duplicate_token":
        before["support_tokens"][1] = before["support_tokens"][0]
    elif mutation == "missing_token":
        before["support_tokens"][1] = "unknown-physical-sample"
    elif mutation == "index_drift":
        before["support_class_indices"] = before["support_class_indices"][::-1].copy()
        before["support_rank_within_class"] = before["support_rank_within_class"][::-1].copy()
    elif mutation == "iq_drift":
        before["support_leo_weak_iq"][0, 0, 0] += np.float32(1.0)
    elif mutation == "dtype_drift":
        before["support_leo_weak_iq"] = before["support_leo_weak_iq"].astype(np.float64)
    elif mutation == "selected_nonfinite":
        before["support_leo_weak_iq"][0, 0, 0] = np.float32(np.nan)
    elif mutation == "negative_rank":
        before["support_rank_within_class"][0] = -1
    elif mutation == "duplicate_rank":
        before["support_rank_within_class"][1] = before["support_rank_within_class"][0]
    monkeypatch.setattr(
        evaluation,
        "forward_zid160",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("drift must fail before support forward")
        ),
    )
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match=error):
        evaluation._shared_before_after_support_features(
            before,
            after,
            model=object(),
            runtime_device="cuda",
            old_class_handles=old_classes,
            all_class_handles=all_classes,
            k_shot=2,
        )


def test_inactive_high_rank_nonfinite_iq_does_not_affect_active_k(monkeypatch) -> None:
    old_classes = ("old-0", "old-1")
    all_classes = old_classes + ("new-0",)
    before = _support_payload(old_classes, k_shot=3)
    after = _support_payload(all_classes, k_shot=3)
    before["support_leo_weak_iq"][before["support_rank_within_class"] == 2] = np.nan
    after["support_leo_weak_iq"][after["support_rank_within_class"] == 2] = np.inf
    forwarded = []

    def forward(_model, iq, *, device, batch_size):
        forwarded.append(np.asarray(iq).copy())
        return np.zeros((len(iq), 1), dtype=np.float32)

    monkeypatch.setattr(evaluation, "forward_zid160", forward)
    monkeypatch.setattr(
        evaluation,
        "registered_feature",
        lambda iq, zid: np.concatenate(
            [np.asarray(iq).reshape(len(iq), -1), np.asarray(zid)], axis=1
        ).astype(np.float32),
    )
    result = evaluation._shared_before_after_support_features(
        before,
        after,
        model=object(),
        runtime_device="cuda",
        old_class_handles=old_classes,
        all_class_handles=all_classes,
        k_shot=2,
    )
    assert len(forwarded) == 1
    assert forwarded[0].shape[0] == len(all_classes) * 2
    assert np.isfinite(forwarded[0]).all()
    assert result[-1]["all_support_count"] == len(all_classes) * 2


def test_active_k_configs_do_not_require_or_claim_inactive_k_locks(monkeypatch) -> None:
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
    for inactive_k in (1, 5, 20):
        del receipt["locked_parameters_by_k"][str(inactive_k)]
        del receipt["selected_by_k"][str(inactive_k)]
    receipt["locked_parameters_by_k"]["10"]["alpha"] = 0.0
    receipt["selected_by_k"]["10"]["selected"]["effective_parameters"][
        "alpha"
    ] = 0.0
    receipt["selected_by_k"]["10"]["selected"]["alpha_forced_zero"] = True
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = evaluation.lodo.canonical_sha256(unsigned)

    calls = []
    original = evaluation.locked_parameters_from_lodo

    def observed(value, *, k_shot):
        calls.append(int(k_shot))
        return original(value, k_shot=k_shot)

    monkeypatch.setattr(evaluation, "locked_parameters_from_lodo", observed)
    parameters, config99, config100 = evaluation._active_k_configs(
        base, receipt, active_k=10, phase2_authority_sha256="a" * 64
    )
    assert calls == [10]
    assert config99.eta_k10 == pytest.approx(0.2)
    assert (config99.eta_k1, config99.eta_k5, config99.eta_k20) == pytest.approx(
        (base.eta_k1, base.eta_k5, base.eta_k20)
    )
    assert config100.d99_phase1_lock_digest == config99.lock_digest
    assert config100.values_for_k(10) == pytest.approx((0.2, 4.0, 2.0, 0.0))
    for inactive_k in (1, 5, 20):
        assert config100.values_for_k(inactive_k) == (1.0, 1.0, 1.0, 0.0)
    bank = SimpleNamespace(metric=SimpleNamespace(k_shot=10))
    evaluation._validate_active_k_state_binding(
        bank, config100, parameters, active_k=10
    )
    wrong_bank = SimpleNamespace(metric=SimpleNamespace(k_shot=5))
    with pytest.raises(evaluation.D99D100QueryEvaluationError, match="bank active-K"):
        evaluation._validate_active_k_state_binding(
            wrong_bank, config100, parameters, active_k=10
        )
    for missing_active_k in (1, 20):
        with pytest.raises(
            evaluation.D99D100QueryEvaluationError, match="K-specific"
        ):
            evaluation._active_k_configs(
                base,
                receipt,
                active_k=missing_active_k,
                phase2_authority_sha256="a" * 64,
            )


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
