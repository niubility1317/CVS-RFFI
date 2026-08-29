from __future__ import annotations

import copy

import numpy as np
import torch

from cvsrffi.stage2_binova_da import NOVA_DA_Config, fit_nova_da
from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVASupport
from cvsrffi.stage2_binova_reg import (
    NOVA_REG_Config,
    NOVA_REG_Module,
    apply_nova_reg,
    fit_nova_reg,
    old_new_margin_loss,
    project_conflicting_gradient,
)


def _support() -> BiNOVASupport:
    rng = np.random.default_rng(23)
    class_count, shots = 7, 10
    labels = np.repeat(np.arange(class_count), shots)
    identity_centers = rng.normal(size=(class_count, 160)).astype(np.float32)
    fft_centers = rng.normal(size=(class_count, 96)).astype(np.float32)
    identity = identity_centers[labels] + 0.05 * rng.normal(size=(70, 160)).astype(np.float32)
    features = BiNOVAFeatures(
        identity160=identity,
        late_time160=identity + 0.02 * np.sin(identity),
        domain160=rng.normal(size=(70, 160)).astype(np.float32),
        fft96=fft_centers[labels] + 0.05 * rng.normal(size=(70, 96)).astype(np.float32),
        physical6=rng.normal(size=(70, 6)).astype(np.float32),
        physical_ids=tuple(f"reg-{index}" for index in range(70)),
    )
    return BiNOVASupport(
        features=features,
        labels=labels,
        ranks=np.tile(np.arange(10), class_count),
        context={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "cap-reg",
            "split_id": "split-reg",
        },
    )


def test_zero_initialized_stage_b_is_exact_identity_for_identity_and_fft() -> None:
    module = NOVA_REG_Module()
    identity = torch.randn(5, 160)
    fft = torch.randn(5, 96)
    condition = torch.randn(5, 6)
    adapted_identity, adapted_fft = module(identity, fft, condition)
    torch.testing.assert_close(adapted_identity, identity, rtol=0, atol=0)
    torch.testing.assert_close(adapted_fft, fft, rtol=0, atol=0)


def test_conflicting_gradient_projection_removes_negative_dot_product() -> None:
    primary = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 1.0])
    projected = project_conflicting_gradient(auxiliary, primary)
    assert float(torch.dot(projected, primary)) >= -1.0e-7
    torch.testing.assert_close(project_conflicting_gradient(torch.tensor([1.0, 1.0]), primary), torch.tensor([1.0, 1.0]))


def test_old_new_margin_penalizes_intrusion_on_both_sides() -> None:
    scores = torch.tensor(
        [[0.2, 0.1, 1.2], [1.1, 0.2, 0.3]], dtype=torch.float32
    )
    labels = torch.tensor([0, 2])
    loss, audit = old_new_margin_loss(scores, labels, old_class_count=2, margin=0.2)
    assert float(loss) > 0.0
    assert audit["old_intrusion_rows"] == 1
    assert audit["new_intrusion_rows"] == 1


def test_stage_b_fit_freezes_stage_a_and_returns_finite_joint_residual() -> None:
    support = _support()
    old_support = BiNOVASupport(
        features=BiNOVAFeatures(
            **{
                name: getattr(support.features, name)[:60]
                for name in ("identity160", "late_time160", "domain160", "fft96", "physical6")
            },
            physical_ids=support.features.physical_ids[:60],
        ),
        labels=support.labels[:60],
        ranks=support.ranks[:60],
        context=support.context,
    )
    stage_a = fit_nova_da(old_support, NOVA_DA_Config(steps=1, seed=3), device="cpu")
    before = copy.deepcopy(stage_a.module.state_dict())
    state = fit_nova_reg(
        stage_a,
        support,
        old_class_count=6,
        config=NOVA_REG_Config(steps=1, seed=4),
        device="cpu",
    )
    for name, value in stage_a.module.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in stage_a.module.parameters())
    identity, fft = apply_nova_reg(state, support.features)
    assert identity.shape == (70, 160) and fft.shape == (70, 96)
    assert np.isfinite(identity).all() and np.isfinite(fft).all()
    assert state.audit["query_rows_used"] == 0
    assert state.audit["stage_a_frozen"] is True
