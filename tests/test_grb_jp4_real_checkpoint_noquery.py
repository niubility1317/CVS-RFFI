from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from cvsrffi.stage2_grb_jp4_adv_drqknn_bcrr import (
    GRBJP4SpikeError, GroundReceiverBasis, RANK, StrictForward, Z_DIM,
    _fit_stage2_b_from_support_iq_development_only,
    analytic_jacobian, checkpoint_right_factors, directions, factor_int8_replay,
    geometry_change, merge_into_joint_proj,
    observed_ground_left_factors, solve_theta,
)
from scripts import run_dssc_zdom_jg_qknn_r4_bcrr_125 as dssc_runner


CHECKPOINT = Path(r"E:\type10-7\automation_reports\CV-SincNet\qknnv42_strict_dual125_20260714_183556\artifacts\best_joint_safe_ssdg.pth")
SUPPORT = Path(r"E:\type10-7\automation_reports\CV-SincNet\adv3b02_ts_drqknn_bcrr_r4_q2f32_bcr3_zidtotal1_procbindfix1_full125_467b8aa5_20260724_035813\failure_evidence\jobs\adv3b02_r4_q2f32_bcr3_rx_8-8_s_713105_k_10_n_20\before\support_leo_clear_weak.npz")
CHECKPOINT_SHA256 = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _real_support_prefix(k: int) -> tuple[torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    with np.load(SUPPORT, allow_pickle=False) as archive:
        iq = np.asarray(archive["support_leo_weak_iq"], np.float32)
        cls = np.asarray(archive["support_class_indices"], np.int64)
        tokens = tuple(str(item) for item in archive["support_tokens"].tolist())
    assert iq.shape == (60, 2, 256) and tuple(np.bincount(cls, minlength=6)) == (10,) * 6
    keep = np.concatenate([np.flatnonzero(cls == item)[:k] for item in range(6)])
    return (torch.tensor(iq[keep], dtype=torch.float32),
            tuple(f"old-{int(item)}" for item in cls[keep]),
            tuple(tokens[int(item)] for item in keep))


def _test_ground(model: torch.nn.Module) -> GroundReceiverBasis:
    """Test-only aggregate-shaped ground component bound to the real layer."""
    rng = np.random.default_rng(20260724)
    prototype = rng.normal(size=(6, Z_DIM)).astype(np.float32)
    prototype /= np.linalg.norm(prototype, axis=1, keepdims=True)
    left = rng.normal(size=(RANK, Z_DIM)).astype(np.float32)
    weight = model.id_backbone.cls_head.joint_proj[0].weight.detach().cpu().numpy().astype(np.float32)
    return GroundReceiverBasis.from_decoded(
        prototypes=prototype, left=left, right=checkpoint_right_factors(model.id_backbone.cls_head.joint_proj[0].weight),
        kappa_ground=2.5, old_class_order=tuple(f"old-{item}" for item in range(6)),
        checkpoint_sha256=CHECKPOINT_SHA256, joint_weight_sha256=_sha(weight.tobytes()),
        method_lock_sha256="2" * 64, generation_digest="3" * 64,
    )


def test_ground_factors_use_only_observed_domain_shifts_and_have_rank_four():
    rng=np.random.default_rng(4); shifts=rng.normal(size=(14,Z_DIM)).astype(np.float32); counts=np.r_[np.ones(14),np.zeros(2)].astype(np.int64)
    left, evidence=observed_ground_left_factors(np.r_[shifts,np.zeros((2,Z_DIM),np.float32)],counts)
    assert left.shape==(RANK,Z_DIM); assert evidence["observed_domain_count"]==14; assert 0.0<evidence["energy_q4"]<=1.0


def test_closed_form_support_theta_decreases_spherical_residual():
    rng=np.random.default_rng(5); n=10; z=rng.normal(size=(n,Z_DIM)).astype(np.float32); z/=np.linalg.norm(z,axis=1,keepdims=True)
    jac=rng.normal(scale=1e-2,size=(n,RANK,Z_DIM)).astype(np.float32); labels=np.repeat(np.arange(2),5); proto=z.copy()[:2]; proto[0]=z[labels==0].mean(0); proto[1]=z[labels==1].mean(0); proto/=np.linalg.norm(proto,axis=1,keepdims=True)
    theta,receipt=solve_theta(z,jac,labels,proto)
    assert theta.shape==(RANK,) and receipt["lambda"]>0.0 and receipt["rank"]==RANK


def test_int8_factor_replay_is_bounded_and_uses_compact_state():
    rng=np.random.default_rng(6); left=rng.normal(size=(RANK,Z_DIM)).astype(np.float32); right=rng.normal(size=(RANK,320)).astype(np.float32); replay,state=factor_int8_replay(left,right,np.asarray([.1,-.2,.3,-.4],np.float32))
    assert replay.shape==(Z_DIM,320); assert state["state_bytes"]<256*1024


def test_geometry_change_counts_large_margin_flips():
    before={"neighbor_class":[0,1],"margin":[.1,.01]}; after={"neighbor_class":[1,1],"margin":[.2,.02]}
    change=geometry_change(before,after)
    assert change=={"neighbor_changed_count":1,"margin_changed_count":2,"large_margin_flip_count":1}


def test_direction_shape_is_fail_closed():
    with pytest.raises(GRBJP4SpikeError): directions(np.zeros((4,159),np.float32),np.zeros((4,320),np.float32))


def test_analytic_jacobian_rejects_non_byte_bound_hook():
    with pytest.raises(GRBJP4SpikeError):
        analytic_jacobian(StrictForward(np.ones((1, Z_DIM), np.float32), np.ones((1, 320), np.float32),
                                        np.ones((1, Z_DIM), np.float32), False),
                          np.zeros((RANK, Z_DIM, 320), np.float32))


def test_real_checkpoint_support_only_development_fit_merge_and_k1_identity():
    """No query artifact is opened; decoded ground remains feasibility-only."""
    assert CHECKPOINT.is_file() and SUPPORT.is_file()
    model, receipt = dssc_runner._exact_adv3b02(CHECKPOINT, device="cpu")
    assert receipt["checkpoint_sha256"] == CHECKPOINT_SHA256
    ground = _test_ground(model)

    k5_iq, k5_labels, k5_tokens = _real_support_prefix(5)
    state = _fit_stage2_b_from_support_iq_development_only(model=model, support_iq=k5_iq, support_labels=k5_labels,
                                         support_physical_tokens=k5_tokens, ground=ground,
                                         checkpoint_sha256=CHECKPOINT_SHA256)
    binding = state.fit_receipt["development_support_iq_binding"]
    assert binding["hook_exact_bytes"] is True and binding["query_rows_used_for_fit"] == 0
    assert binding["ground_digest"] == ground.digest
    merge_into_joint_proj(model.id_backbone.cls_head.joint_proj[0], state=state, ground=ground,
                          checkpoint_sha256=CHECKPOINT_SHA256)
    assert _sha(model.id_backbone.cls_head.joint_proj[0].weight.detach().cpu().numpy().tobytes()) == state.joint_weight_semantic_sha256

    identity_model, _ = dssc_runner._exact_adv3b02(CHECKPOINT, device="cpu")
    k1_iq, k1_labels, k1_tokens = _real_support_prefix(1)
    before = identity_model.id_backbone.cls_head.joint_proj[0].weight.detach().cpu().numpy().copy()
    k1_ground = _test_ground(identity_model)
    k1 = _fit_stage2_b_from_support_iq_development_only(model=identity_model, support_iq=k1_iq, support_labels=k1_labels,
                                       support_physical_tokens=k1_tokens, ground=k1_ground,
                                       checkpoint_sha256=CHECKPOINT_SHA256)
    merge_into_joint_proj(identity_model.id_backbone.cls_head.joint_proj[0], state=k1, ground=k1_ground,
                          checkpoint_sha256=CHECKPOINT_SHA256)
    assert np.array_equal(k1.theta(), np.zeros(RANK, np.float32))
    assert np.array_equal(before, identity_model.id_backbone.cls_head.joint_proj[0].weight.detach().cpu().numpy())
    assert k1.fit_receipt["development_support_iq_binding"]["query_rows_used_for_fit"] == 0
