from __future__ import annotations

import torch
import torch.nn as nn

from model_dual_cvsincnet import DualCVSincNetDisentangle

from cvsrffi.identity_only_forward import (
    can_use_identity_only_forward,
    identity_only_feature_forward,
)


class _IdentityBackbone(nn.Module):
    def forward(self, x, y=None, return_aux=False, domain_labels=None):
        logits = x.mean(dim=-1)
        if not return_aux:
            return logits
        return {"logits": logits, "feat_joint": x.flatten(1)}


class _DualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.id_backbone = _IdentityBackbone()

    @staticmethod
    def _pick_z_id(aux):
        return aux["feat_joint"]

    def forward(self, *args, **kwargs):
        raise AssertionError("full dual forward must not execute for qKNN z_id export")


def test_identity_only_forward_bypasses_domain_path() -> None:
    model = _DualModel()
    x = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    assert can_use_identity_only_forward(model, "z_id") is True
    result = identity_only_feature_forward(model, x, "z_id")
    assert result is not None
    features, logits = result
    torch.testing.assert_close(features, x.flatten(1))
    torch.testing.assert_close(logits, x.mean(dim=-1))


def test_identity_only_forward_falls_back_for_other_features() -> None:
    model = _DualModel()
    assert can_use_identity_only_forward(model, "z_dom") is False
    assert identity_only_feature_forward(model, torch.zeros(1, 2, 4), "z_dom") is None


def test_real_dual_model_is_bit_exact_and_skips_domain_backbone() -> None:
    torch.manual_seed(7)
    model = DualCVSincNetDisentangle(
        num_classes=4,
        num_domains=3,
        model_size="S",
        input_len=256,
        fast_infer_when_no_aux=False,
    ).eval()
    x = torch.randn(2, 2, 256)
    domain_calls = 0

    def count_domain_call(_module, _inputs, _output):
        nonlocal domain_calls
        domain_calls += 1

    hook = model.dom_backbone.register_forward_hook(count_domain_call)
    with torch.no_grad():
        full = model(x, return_aux=True)
        assert domain_calls == 1
        light = identity_only_feature_forward(model, x, "z_id")
    hook.remove()
    assert light is not None
    light_z_id, light_logits = light
    assert domain_calls == 1
    torch.testing.assert_close(light_z_id, full["z_id"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(light_logits, full["tx_logits"], rtol=0.0, atol=0.0)
