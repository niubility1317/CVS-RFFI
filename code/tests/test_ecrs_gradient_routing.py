from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.schedule import configure_ecrs_for_epoch  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


def _model():
    return build_dual_model(
        num_classes=3,
        num_domains=2,
        model_size="M",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        branch_ablation="no_dac",
        domain_branch_ablation="no_stats",
        use_ecrs=True,
        fast_infer_when_no_aux=False,
    )


def test_stage2_and_stage4_parameter_routes_follow_report() -> None:
    model = _model()
    args = SimpleNamespace(
        use_ecrs=True,
        ecrs_enable_learnable_basis=False,
        ecrs_enable_fasttrust=False,
        ecrs_teacher_stable=False,
    )
    stage2 = configure_ecrs_for_epoch(model, 20, args)
    assert stage2["stage"] == 2
    assert all(p.requires_grad for p in model.ecrs.nuisance_estimator.parameters())
    assert all(p.requires_grad for p in model.ecrs.content_estimator.parameters())
    assert not any(p.requires_grad for p in model.ecrs.response_projection.parameters())

    stage4 = configure_ecrs_for_epoch(model, 120, args)
    assert stage4["stage"] == 4
    assert all(p.requires_grad for p in model.ecrs.response_projection.parameters())
    assert all(p.requires_grad for p in model.ecrs.fusion_gate.parameters())
    assert model.ecrs.detach_identification_for_identity is True


def test_tx_classification_gradient_cannot_update_content_estimator() -> None:
    torch.manual_seed(31)
    model = _model().train()
    args = SimpleNamespace(
        use_ecrs=True,
        ecrs_enable_learnable_basis=False,
        ecrs_enable_fasttrust=False,
        ecrs_teacher_stable=False,
    )
    configure_ecrs_for_epoch(model, 120, args)
    out = model(torch.randn(3, 2, 64), y_tx=torch.tensor([0, 1, 2]), return_aux=True)
    torch.nn.functional.cross_entropy(out["tx_logits"], torch.tensor([0, 1, 2])).backward()

    assert all(parameter.grad is None for parameter in model.ecrs.content_estimator.parameters())
    assert all(parameter.grad is None for parameter in model.ecrs.nuisance_estimator.parameters())
    assert any(parameter.grad is not None for parameter in model.ecrs.response_projection.parameters())
