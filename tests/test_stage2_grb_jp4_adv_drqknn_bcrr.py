from __future__ import annotations

import hashlib
import weakref

import numpy as np
import pytest
import torch

from cvsrffi import stage2_grb_jp4_adv_drqknn_bcrr as grb_module
from cvsrffi.stage2_adv3b02_ts_drqknn_bcrr import (
    append_stage2_c as append_parent_stage2_c,
    build_stage2_b_state as build_parent_stage2_b_state,
)
from cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr import (
    HIDDEN_DIM,
    OLD_CLASS_COUNT,
    RANK,
    Z_DIM,
    GRBJP4SpikeError,
    GroundReceiverBasis,
    StrictForward,
    _append_stage2_c_development_only,
    build_five_arm_state_view,
    _build_stage2_b_state_development_only,
    _fit_stage2_b_from_precomputed_jacobian_development_only,
    five_arm_state_receipt,
    merge_into_joint_proj,
    prepare_support_for_jp4_fit,
    resource_receipt,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _basis_and_weight(seed: int = 17):
    rng = np.random.default_rng(seed)
    weight = rng.normal(scale=.02, size=(Z_DIM, HIDDEN_DIM)).astype(np.float32)
    p = rng.normal(size=(OLD_CLASS_COUNT, Z_DIM)).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    left = rng.normal(size=(RANK, Z_DIM)).astype(np.float32)
    right = rng.normal(size=(RANK, HIDDEN_DIM)).astype(np.float32)
    checkpoint = "1" * 64
    basis = GroundReceiverBasis.from_decoded(
        prototypes=p, left=left, right=right, kappa_ground=2.5,
        old_class_order=tuple(f"old-{i}" for i in range(OLD_CLASS_COUNT)),
        checkpoint_sha256=checkpoint, joint_weight_sha256=_sha(weight.tobytes()),
        method_lock_sha256="2" * 64, generation_digest="3" * 64,
    )
    return rng, weight, basis, checkpoint


def test_k1_is_exact_identity_and_has_only_six_byte_theta_state():
    rng, weight, basis, checkpoint = _basis_and_weight()
    labels = tuple(f"old-{i}" for i in range(OLD_CLASS_COUNT))
    z = np.zeros((OLD_CLASS_COUNT, Z_DIM), np.float32)
    j = rng.normal(scale=.01, size=(OLD_CLASS_COUNT, RANK, Z_DIM)).astype(np.float32)
    state = _fit_stage2_b_from_precomputed_jacobian_development_only(support_zid=z, support_jacobian=j, support_labels=labels,
                         support_physical_tokens=tuple(f"p{i}" for i in range(OLD_CLASS_COUNT)),
                         ground=basis, checkpoint_weight=weight, checkpoint_sha256=checkpoint)
    assert np.array_equal(state.theta(), np.zeros(RANK, np.float32))
    assert state.theta_codes.nbytes + state.theta_scale.nbytes == 6
    assert state.fit_receipt["fallback"] == "K1_identity"
    assert state.fit_receipt["query_rows_used_for_fit"] == 0
    receipt = resource_receipt(state, basis)
    assert receipt["numeric_payload_bytes"] == 2914
    assert receipt["adapter_mac_per_query"] == 0


def test_k5_stream_fit_decodes_int8_for_semantic_inplace_merge():
    rng, weight, basis, checkpoint = _basis_and_weight(19)
    labels = tuple(label for label in basis.old_class_order for _ in range(5))
    proto = basis.prototypes()
    z = np.vstack([proto[i] + rng.normal(scale=.04, size=(5, Z_DIM)).astype(np.float32) for i in range(OLD_CLASS_COUNT)])
    z[2] = 0.0  # Existing K5 singleton-medoid repair is consumed before JP4 fit.
    j = rng.normal(scale=.015, size=(len(z), RANK, Z_DIM)).astype(np.float32)
    repaired, repaired_jac, repair = prepare_support_for_jp4_fit(
        support_zid=z, support_jacobian=j, support_labels=labels,
        support_physical_tokens=tuple(f"p{i}" for i in range(len(z))),
        registered_old_classes=basis.old_class_order,
    )
    donor_candidates = [row for row in (0, 1, 3, 4) if np.array_equal(repaired_jac[2], j[row])]
    assert len(donor_candidates) == 1 and not np.array_equal(repaired[2], z[2])
    assert repair["query_rows_used_for_fit"] == 0
    state = _fit_stage2_b_from_precomputed_jacobian_development_only(support_zid=z, support_jacobian=j, support_labels=labels,
                         support_physical_tokens=tuple(f"p{i}" for i in range(len(z))),
                         ground=basis, checkpoint_weight=weight, checkpoint_sha256=checkpoint)
    assert state.fit_receipt["optimizer_steps"] == 0
    assert state.fit_receipt["query_rows_used_for_fit"] == 0
    assert state.fit_receipt["lambda"] > 0.0
    assert state.fit_receipt["rank_diagnostic"] <= RANK
    linear = torch.nn.Linear(HIDDEN_DIM, Z_DIM, bias=False, dtype=torch.float32)
    with torch.no_grad(): linear.weight.copy_(torch.from_numpy(weight))
    merge_into_joint_proj(linear, state=state, ground=basis, checkpoint_sha256=checkpoint)
    assert _sha(linear.weight.detach().numpy().tobytes()) == state.joint_weight_semantic_sha256


def test_near_zero_trace_falls_back_without_failing_or_query_access():
    rng, weight, basis, checkpoint = _basis_and_weight(23)
    labels = tuple(label for label in basis.old_class_order for _ in range(5))
    z = np.vstack([np.repeat(basis.prototypes()[i:i + 1], 5, axis=0) for i in range(OLD_CLASS_COUNT)])
    j = np.zeros((len(z), RANK, Z_DIM), np.float32)
    state = _fit_stage2_b_from_precomputed_jacobian_development_only(support_zid=z, support_jacobian=j, support_labels=labels,
                         support_physical_tokens=tuple(f"p{i}" for i in range(len(z))),
                         ground=basis, checkpoint_weight=weight, checkpoint_sha256=checkpoint)
    assert np.array_equal(state.theta(), np.zeros(RANK, np.float32))
    assert state.fit_receipt["fallback"] == "trace_or_numeric_identity"
    assert state.fit_receipt["query_rows_used_for_fit"] == 0
    assert state.fit_receipt["condition"] == 0.0
    assert state.wire_bytes() and resource_receipt(state, basis)["numeric_payload_bytes"] == 2914


def test_nonfinite_and_solve_fallbacks_are_finite_and_serializable(monkeypatch):
    rng, weight, basis, checkpoint = _basis_and_weight(31)
    labels = tuple(label for label in basis.old_class_order for _ in range(5))
    z = np.vstack([basis.prototypes()[i:i + 1] + rng.normal(scale=.01, size=(5, Z_DIM)).astype(np.float32)
                   for i in range(OLD_CLASS_COUNT)])
    tokens = tuple(f"p{i}" for i in range(len(z)))
    nonfinite = np.ones((len(z), RANK, Z_DIM), np.float32); nonfinite[0, 0, 0] = np.nan
    first = _fit_stage2_b_from_precomputed_jacobian_development_only(
        support_zid=z, support_jacobian=nonfinite, support_labels=labels, support_physical_tokens=tokens,
        ground=basis, checkpoint_weight=weight, checkpoint_sha256=checkpoint,
    )
    assert np.array_equal(first.theta(), np.zeros(RANK, np.float32))
    assert first.fit_receipt["fallback"] == "nonfinite_jacobian_identity"
    assert first.fit_receipt["condition"] == 0.0 and first.wire_bytes()
    assert resource_receipt(first, basis)["numeric_payload_bytes"] == 2914

    jac = rng.normal(scale=.01, size=(len(z), RANK, Z_DIM)).astype(np.float32)
    def fail_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced solve failure")
    monkeypatch.setattr(np.linalg, "solve", fail_solve)
    second = _fit_stage2_b_from_precomputed_jacobian_development_only(
        support_zid=z, support_jacobian=jac, support_labels=labels, support_physical_tokens=tokens,
        ground=basis, checkpoint_weight=weight, checkpoint_sha256=checkpoint,
    )
    assert np.array_equal(second.theta(), np.zeros(RANK, np.float32))
    assert second.fit_receipt["fallback"] == "solve_nonfinite_identity"
    assert second.fit_receipt["condition"] == 0.0 and second.wire_bytes()
    assert resource_receipt(second, basis)["numeric_payload_bytes"] == 2914


def test_stage2_c_only_appends_to_the_frozen_jp4_and_old_r6_state(
    monkeypatch: pytest.MonkeyPatch,
):
    rng, weight, basis, checkpoint = _basis_and_weight(29)
    labels = tuple(f"old-{i}" for i in range(OLD_CLASS_COUNT))
    z = basis.prototypes() + rng.normal(scale=.01, size=(OLD_CLASS_COUNT, Z_DIM)).astype(np.float32)
    j = rng.normal(scale=.01, size=(OLD_CLASS_COUNT, RANK, Z_DIM)).astype(np.float32)
    jp4 = _fit_stage2_b_from_precomputed_jacobian_development_only(support_zid=z, support_jacobian=j, support_labels=labels,
                       support_physical_tokens=tuple(f"p{i}" for i in range(OLD_CLASS_COUNT)),
                       ground=basis, checkpoint_weight=weight, checkpoint_sha256=checkpoint)
    zdom = rng.normal(size=(OLD_CLASS_COUNT, Z_DIM)).astype(np.float32)
    old = _build_stage2_b_state_development_only(jp4=jp4, support_zid_after_merge=z, support_zdom=zdom,
                               support_labels=labels, registered_classes=labels,
                               support_physical_tokens=tuple(f"p{i}" for i in range(OLD_CLASS_COUNT)))
    no_ground = build_parent_stage2_b_state(support_zid=z, support_zdom=zdom, support_labels=labels,
                                            registered_classes=labels,
                                            support_physical_tokens=tuple(f"p{i}" for i in range(OLD_CLASS_COUNT)))
    arms = build_five_arm_state_view(no_ground_state=no_ground, adapted_state=old)
    arm_receipt = five_arm_state_receipt(arms, jp4=jp4)
    assert arm_receipt["k1_identity_r6_bytes"] is True
    assert arm_receipt["m_joint_reuses_m_da_state"] is True
    assert arm_receipt["base_r8_exact_masked_degenerate"] is False
    assert arm_receipt["adapted_r8_exact_masked_degenerate"] is False
    raw_directional_loo = grb_module._raw_directional_loo

    def base_masked_degenerate(state):
        qscore, bscore, _ = raw_directional_loo(state)
        return qscore, bscore, state is no_ground.id_bank

    monkeypatch.setattr(grb_module, "_raw_directional_loo", base_masked_degenerate)
    flagged_states = build_five_arm_state_view(
        no_ground_state=no_ground, adapted_state=old
    )
    flagged_receipt = five_arm_state_receipt(flagged_states, jp4=jp4)
    assert flagged_receipt["base_r8_exact_masked_degenerate"] is True
    assert flagged_receipt["adapted_r8_exact_masked_degenerate"] is False
    new_z = rng.normal(size=(2, Z_DIM)).astype(np.float32)
    new_dom = rng.normal(size=(2, Z_DIM)).astype(np.float32)
    after = _append_stage2_c_development_only(old, new_support_zid_after_merge=new_z, new_support_zdom=new_dom,
                            new_support_labels=("new-a", "new-b"), new_registered_classes=("new-a", "new-b"),
                            new_support_physical_tokens=("n0", "n1"),
                            after_full_teacher_zid=np.vstack((z, new_z)).astype(np.float32),
                            after_full_teacher_physical_tokens=tuple(f"p{i}" for i in range(OLD_CLASS_COUNT)) + ("n0", "n1"))
    assert after.jp4.wire_bytes() == old.jp4.wire_bytes()
    assert after.parent_state.domain.frozen_old_digest == old.parent_state.domain.digest
    assert after.append_receipt["old_int8_codes_preserved"] is True
    base_after, _ = append_parent_stage2_c(
        no_ground, new_support_zid=new_z, new_support_zdom=new_dom,
        new_support_labels=("new-a", "new-b"), new_registered_classes=("new-a", "new-b"),
        new_support_physical_tokens=("n0", "n1"),
        after_full_teacher_zid=np.vstack((z, new_z)).astype(np.float32),
        after_full_teacher_physical_tokens=tuple(f"p{i}" for i in range(OLD_CLASS_COUNT)) + ("n0", "n1"),
    )
    arms_c = build_five_arm_state_view(no_ground_state=base_after, adapted_state=after)
    receipt_c = five_arm_state_receipt(arms_c, jp4=after.jp4)
    assert arms_c["M_DA"].domain.stage == "S_C"
    assert arms_c["M_DA_NG"].domain.stage == "S_C"
    assert receipt_c["k1_identity_r6_bytes"] is True
    assert after.jp4.wire_bytes() == old.jp4.wire_bytes()


def test_support_int8_theta_decision_flip_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(240724)
    zid = rng.normal(size=(OLD_CLASS_COUNT, Z_DIM)).astype(np.float32)
    zid /= np.linalg.norm(zid, axis=1, keepdims=True)
    zdom = rng.normal(size=(OLD_CLASS_COUNT, Z_DIM)).astype(np.float32)
    zdom /= np.linalg.norm(zdom, axis=1, keepdims=True)
    teacher = StrictForward(
        zid,
        np.zeros((OLD_CLASS_COUNT, HIDDEN_DIM), np.float32),
        zid,
        True,
        zdom,
        "support_only_test",
    )
    deployed = StrictForward(
        np.roll(zid, 1, axis=1).copy(),
        np.zeros((OLD_CLASS_COUNT, HIDDEN_DIM), np.float32),
        zid,
        True,
        zdom,
        "support_only_test",
    )
    calls = 0

    def forced_flip(_state, _query):
        nonlocal calls
        logits = np.zeros((OLD_CLASS_COUNT, OLD_CLASS_COUNT), np.float32)
        columns = (
            np.arange(OLD_CLASS_COUNT)
            if calls == 0
            else np.roll(np.arange(OLD_CLASS_COUNT), 1)
        )
        logits[np.arange(OLD_CLASS_COUNT), columns] = np.float32(10.0)
        calls += 1
        return logits

    monkeypatch.setattr(grb_module, "qknn_logits", forced_flip)
    classes = tuple(f"old-{index}" for index in range(OLD_CLASS_COUNT))
    with pytest.raises(GRBJP4SpikeError, match="decision audit gate"):
        grb_module._support_int8_theta_audit(
            teacher=teacher,
            deployed=deployed,
            labels=classes,
            tokens=tuple(f"support-{index}" for index in range(OLD_CLASS_COUNT)),
            classes=classes,
        )


def test_runtime_ownership_rejects_second_live_materialization() -> None:
    class _Runtime:
        pass

    live_runtime = _Runtime()
    ownership = grb_module._ObservedRuntimeOwnership(
        runtime_refs=[weakref.ref(live_runtime)], observed_live_instances=[1]
    )
    with pytest.raises(GRBJP4SpikeError, match="before prior release"):
        grb_module._runtime_ownership_materialized(
            ownership, runtime=_Runtime()
        )
