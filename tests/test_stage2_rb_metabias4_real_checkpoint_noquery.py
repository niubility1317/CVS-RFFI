from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch

from cvsrffi.stage2_rb_metabias4_qknn import (
    Phase1MetaBias4Lock,
    audit_d102_resources,
    build_phase1_metabias4_asset,
    fit_d102_stage2_state,
    serialize_d102_runtime_state,
)
from cvsrffi.stage2_zid_student_t_qknn import Phase1ZIDStudentTLock


CHECKPOINT = Path(
    "E:/type10-7/automation_reports/CV-SincNet/"
    "qknnv42_strict_dual125_20260714_183556/artifacts/"
    "best_joint_safe_ssdg.pth"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt(checkpoint_sha256: str, label: str) -> str:
    return hashlib.sha256(
        f"d102-real-checkpoint-noquery:{label}:{checkpoint_sha256}".encode("utf-8")
    ).hexdigest()


def _qknn_lock(checkpoint_sha256: str) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=1,
        student_nu=3.0,
        kernel_effective_dim=12,
        kernel_volume_gamma=1.0,
        shared_h0=0.35,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=0.85,
        phase1_lodo_receipt_sha256=_receipt(checkpoint_sha256, "qknn-lodo"),
        quantization_margin_audit_sha256=_receipt(
            checkpoint_sha256, "qknn-int8"
        ),
    )


def test_real_adv3b02_checkpoint_builds_support_only_d102_state_without_query() -> None:
    assert CHECKPOINT.is_file(), "the frozen local ADV3B02 checkpoint is required"
    checkpoint_sha256 = _sha256(CHECKPOINT)
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = payload["model"]
    weight = state_dict["id_backbone.cls_head.joint_proj.0.weight"]
    bias = state_dict["id_backbone.cls_head.joint_proj.0.bias"]
    domain_weight = state_dict["dom_backbone.cls_head.joint_proj.0.weight"]
    domain_bias = state_dict["dom_backbone.cls_head.joint_proj.0.bias"]
    assert tuple(weight.shape) == (160, 320)
    assert tuple(bias.shape) == (160,)
    assert tuple(domain_weight.shape) == (160, 480)
    assert tuple(domain_bias.shape) == (160,)

    generator = torch.Generator(device="cpu").manual_seed(102607)
    support_hidden = torch.randn(3, 320, generator=generator)
    support_domain_hidden = torch.randn(3, 480, generator=generator)
    with torch.no_grad():
        support_pre_relu = (
            support_hidden @ weight.detach().cpu().T + bias.detach().cpu()
        ).numpy().astype(np.float32)
        support_zdom = torch.relu(
            support_domain_hidden @ domain_weight.detach().cpu().T
            + domain_bias.detach().cpu()
        ).numpy().astype(np.float32)

    checkpoint_basis = np.linalg.svd(
        weight.detach().cpu().numpy().astype(np.float64),
        full_matrices=False,
    )[0][:, :4].astype(np.float32)
    domain_left = np.linalg.svd(
        domain_weight.detach().cpu().numpy().astype(np.float64),
        full_matrices=False,
    )[0]
    domain_u = domain_left[:, :32].T.astype(np.float32)
    # This is a reachability smoke, not a Phase1 performance claim.  Keep the
    # sealed-bank fixture target-independent: its values depend only on the
    # checkpoint hash, never on the support rows used below.
    bank_generator = np.random.default_rng(int(checkpoint_sha256[:16], 16))
    bank_g = bank_generator.normal(size=(4, 32)).astype(np.float32)
    bank_t = np.asarray(
        [
            [0.08, -0.04, 0.02, 0.05],
            [-0.03, 0.07, -0.05, 0.01],
            [0.02, 0.03, 0.06, -0.04],
            [0.04, -0.02, 0.03, 0.05],
        ],
        dtype=np.float32,
    )
    phase1_lock = Phase1MetaBias4Lock(
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=_receipt(checkpoint_sha256, "runtime"),
        bundle_sha256=_receipt(checkpoint_sha256, "bundle"),
        external_seal_sha256=_receipt(checkpoint_sha256, "seal"),
        method_lock_sha256=_receipt(checkpoint_sha256, "method"),
        receiver_held_receipt_sha256=_receipt(checkpoint_sha256, "receiver-held"),
        class_loco_receipt_sha256=_receipt(checkpoint_sha256, "class-loco"),
        tx_probe_receipt_sha256=_receipt(checkpoint_sha256, "tx-probe"),
        label_permutation_receipt_sha256=_receipt(
            checkpoint_sha256, "label-permutation"
        ),
        aggregation_receipt_sha256=_receipt(
            checkpoint_sha256, "aggregation"
        ),
        quantization_receipt_sha256=_receipt(
            checkpoint_sha256, "asset-int8"
        ),
        tx_probe_balanced_accuracy=0.20,
    )
    asset = build_phase1_metabias4_asset(
        np.float32(0.08) * checkpoint_basis,
        domain_u,
        bank_g,
        bank_t,
        np.ones((4, 4), dtype=np.float32),
        np.full(4, 0.9, dtype=np.float32),
        np.asarray([0.8, 0.9, 1.0, 1.1], dtype=np.float32),
        np.full(4, 0.5, dtype=np.float32),
        temperature=0.6,
        ellipsoid_radius=0.8,
        cell_min_physical_count=np.full(4, 2, dtype=np.int16),
        cell_class_count=np.full(4, 3, dtype=np.int16),
        lock=phase1_lock,
    )
    labels = np.asarray(["opaque-0", "opaque-1", "opaque-2"], dtype=str)
    state = fit_d102_stage2_state(
        asset,
        support_pre_relu,
        support_zdom,
        labels,
        tuple(labels.tolist()),
        qknn_config=_qknn_lock(checkpoint_sha256),
        stage="S_C",
        support_receipt_sha256=_receipt(checkpoint_sha256, "support-only"),
    )

    assert state.asset.lock.checkpoint_sha256 == checkpoint_sha256
    assert state.fit_audit["query_rows_used_for_fit"] == 0
    assert state.query_rows_used_for_fit == 0
    assert state.query_state_updates == 0
    assert state.fit_audit["data_information_rank"] == 4
    assert serialize_d102_runtime_state(state) == serialize_d102_runtime_state(
        state
    )
    resources = audit_d102_resources(state)
    assert resources["passes_state_gate"] is True
    assert resources["passes_query_mac_gate"] is True
    assert resources["fp32_persistent_sidecar_bytes"] == 0

    fit_parameters = set(inspect.signature(fit_d102_stage2_state).parameters)
    assert not fit_parameters.intersection(
        {
            "query",
            "query_pre_relu",
            "query_truth",
            "query_labels",
            "receiver",
            "tx",
            "class_quota",
        }
    )
