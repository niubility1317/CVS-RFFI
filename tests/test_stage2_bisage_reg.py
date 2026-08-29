from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVASupport
from cvsrffi.stage2_bisage_da import SAGEDConfig, SAGEDModule, SAGEDState
from cvsrffi.stage2_bisage_reg import (
    SAGERConfig,
    SAGERModule,
    apply_sage_r,
    boundary_gate,
    fit_sage_r,
    update_eta,
)


def _registered_support(shots: int = 5) -> BiNOVASupport:
    rng = np.random.default_rng(713103)
    class_count = 7
    labels = np.repeat(np.arange(class_count), shots)
    rows = len(labels)
    identity_centers = rng.normal(size=(class_count, 160)).astype(np.float32)
    fft_centers = rng.normal(size=(class_count, 96)).astype(np.float32)
    identity = identity_centers[labels] + 0.05 * rng.normal(size=(rows, 160)).astype(np.float32)
    features = BiNOVAFeatures(
        identity160=identity,
        late_time160=identity + 0.02 * np.sin(identity),
        domain160=rng.normal(size=(rows, 160)).astype(np.float32),
        fft96=fft_centers[labels] + 0.05 * rng.normal(size=(rows, 96)).astype(np.float32),
        physical6=rng.normal(size=(rows, 6)).astype(np.float32),
        physical_ids=tuple(f"registered-{index}" for index in range(rows)),
    )
    return BiNOVASupport(
        features=features,
        labels=labels,
        ranks=np.tile(np.arange(shots), class_count),
        context={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "cap-stage-b",
            "split_id": "split-stage-b",
        },
    )


def _stage_a() -> SAGEDState:
    module = SAGEDModule(late_rank=4, identity_rank=4, context_dim=4).double()
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return SAGEDState(
        module=module,
        config=SAGEDConfig(
            steps=1, late_rank=4, identity_rank=4, context_dim=4, covariance_rank=3
        ),
        domain_context166=np.zeros(166, dtype=np.float64),
        audit={
            "selected_mode": "S1_CANDIDATE",
            "nonaffine_energy": 0.2,
            "query_rows_used": 0,
        },
    )


def test_boundary_gate_suppresses_far_samples() -> None:
    gate = boundary_gate(torch.tensor([0.0, 10.0]), tau=1.0, gamma=5.0)
    assert float(gate[0]) > 0.99
    assert float(gate[1]) < 1.0e-6


def test_augmented_lagrangian_increases_eta_on_old_risk_violation() -> None:
    assert update_eta(eta=0.0, violation=0.2, rho=2.0) == pytest.approx(0.4)
    assert update_eta(eta=0.1, violation=-0.2, rho=2.0) == 0.0


def test_zero_initialized_stage_b_is_exact_identity() -> None:
    module = SAGERModule(rank=4)
    identity = torch.randn(3, 160)
    fft = torch.randn(3, 96)
    condition = torch.randn(3, 6)
    context = torch.randn(8)
    adapted_identity, adapted_fft = module(identity, fft, condition, context)
    torch.testing.assert_close(adapted_identity, identity, rtol=0, atol=0)
    torch.testing.assert_close(adapted_fft, fft, rtol=0, atol=0)


def test_sage_r_one_step_freezes_sage_d_and_produces_joint_state() -> None:
    stage_a = _stage_a()
    support = _registered_support()
    before = copy.deepcopy(stage_a.module.state_dict())
    state = fit_sage_r(
        stage_a,
        support,
        old_class_count=6,
        config=SAGERConfig(steps=1, rank=4, learning_rate=1.0e-3),
        device="cpu",
    )
    for name, value in stage_a.module.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in stage_a.module.parameters())
    identity, fft = apply_sage_r(state, support.features)
    assert identity.shape == (35, 160)
    assert fft.shape == (35, 96)
    assert np.isfinite(identity).all() and np.isfinite(fft).all()
    assert state.audit["stage_a_frozen"] is True
    assert state.audit["query_rows_used"] == 0
    assert state.audit["old_risk_constraint"] == "augmented_lagrangian"
