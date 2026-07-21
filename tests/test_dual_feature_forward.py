from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model_dual_cvsincnet import DualCVSincNetDisentangle

from cvsrffi.dual_feature_forward import (
    DualFeatureForwardError,
    dual_feature_forward,
)


class _Backbone(nn.Module):
    def __init__(self, key: str, offset: float) -> None:
        super().__init__()
        self.key = key
        self.offset = float(offset)
        self.calls = 0

    def forward(self, x, y=None, return_aux=True, domain_labels=None):
        del y, domain_labels
        self.calls += 1
        if not return_aux:
            raise AssertionError("dual feature path requires auxiliary outputs")
        feature = x.mean(dim=1)[:, :160] + self.offset
        logits = feature[:, :3]
        return {self.key: feature, "logits": logits}


class _Enhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_ptr = -1

    def forward(self, z_dom: torch.Tensor, x: torch.Tensor):
        self.calls += 1
        self.input_ptr = int(x.data_ptr())
        return z_dom + 3.0, None


class _ForbiddenDomHead(nn.Module):
    def forward(self, _value):
        raise AssertionError("dom_head must never execute")


class _Dual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_feature_key = "feat_joint"
        self.dom_feature_key = "feat_imp"
        self.id_backbone = _Backbone("feat_joint", 1.0)
        self.dom_backbone = _Backbone("feat_imp", 2.0)
        self.dom_enhancer = _Enhancer()
        self.dom_head = _ForbiddenDomHead()

    @staticmethod
    def _pick_z_id(aux):
        return aux["feat_joint"]

    @staticmethod
    def _pick_z_dom(aux):
        return aux["feat_imp"]

    def forward(self, *_args, **_kwargs):
        raise AssertionError("full dual forward must not execute")


def test_dual_feature_forward_uses_each_backbone_once_and_never_dom_head() -> None:
    model = _Dual().eval()
    rows = torch.randn(4, 2, 192, dtype=torch.float32, requires_grad=True)
    z_id, z_dom, logits = dual_feature_forward(model, rows)
    assert model.id_backbone.calls == 1
    assert model.dom_backbone.calls == 1
    assert model.dom_enhancer.calls == 1
    assert model.dom_enhancer.input_ptr == int(rows.data_ptr())
    assert z_id.shape == z_dom.shape == (4, 160)
    assert logits.shape == (4, 3)
    assert z_id.dtype == z_dom.dtype == logits.dtype == torch.float32
    assert z_id.requires_grad is False
    assert z_dom.requires_grad is False
    assert logits.requires_grad is False
    torch.testing.assert_close(z_dom, rows.mean(dim=1)[:, :160] + 5.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id_feature_key", "feat_cls", "feat_joint"),
        ("dom_feature_key", "feat_pa", "feat_imp"),
    ],
)
def test_dual_feature_forward_rejects_feature_key_drift(
    field: str, value: str, message: str
) -> None:
    model = _Dual().eval()
    setattr(model, field, value)
    with pytest.raises(DualFeatureForwardError, match=message):
        dual_feature_forward(model, torch.randn(2, 2, 160))


def test_dual_feature_forward_rejects_training_input_dtype_and_nonfinite() -> None:
    rows = torch.randn(2, 2, 160)
    with pytest.raises(DualFeatureForwardError, match="eval"):
        dual_feature_forward(_Dual().train(), rows)
    with pytest.raises(DualFeatureForwardError, match="float32"):
        dual_feature_forward(_Dual().eval(), rows.double())
    rows[0, 0, 0] = float("nan")
    with pytest.raises(DualFeatureForwardError, match="finite"):
        dual_feature_forward(_Dual().eval(), rows)


def test_dual_feature_forward_rejects_output_shape_dtype_and_nonfinite() -> None:
    model = _Dual().eval()
    model.id_backbone.offset = float("nan")
    with pytest.raises(DualFeatureForwardError, match="finite"):
        dual_feature_forward(model, torch.randn(2, 2, 160))

    class _WrongWidth(_Backbone):
        def forward(self, x, y=None, return_aux=True, domain_labels=None):
            result = super().forward(x, y=y, return_aux=return_aux, domain_labels=domain_labels)
            result[self.key] = result[self.key][:, :159]
            return result

    model = _Dual().eval()
    model.dom_backbone = _WrongWidth("feat_imp", 0.0)
    with pytest.raises(DualFeatureForwardError, match="z_dom"):
        dual_feature_forward(model, torch.randn(2, 2, 160))


def test_dual_feature_forward_rejects_selector_fallback_or_substitution() -> None:
    model = _Dual().eval()
    model.id_backbone.key = "feat_cls"
    model._pick_z_id = lambda aux: aux["feat_cls"]
    with pytest.raises(DualFeatureForwardError, match="feat_joint"):
        dual_feature_forward(model, torch.randn(2, 2, 160))

    model = _Dual().eval()
    model.dom_backbone.key = "feat_pa"
    model._pick_z_dom = lambda aux: aux["feat_pa"]
    with pytest.raises(DualFeatureForwardError, match="feat_imp"):
        dual_feature_forward(model, torch.randn(2, 2, 160))


def test_real_dual_model_matches_training_time_features_without_dom_head() -> None:
    torch.manual_seed(23)
    model = DualCVSincNetDisentangle(
        num_classes=4,
        num_domains=3,
        model_size="S",
        input_len=256,
        model_variant="lite_d",
        id_feature_key="feat_joint",
        dom_feature_key="feat_imp",
        fast_infer_when_no_aux=False,
    ).eval()
    rows = torch.randn(2, 2, 256)
    with torch.no_grad():
        expected = model(rows, return_aux=True)
    model.dom_head = _ForbiddenDomHead()
    calls = {"id": 0, "dom": 0, "enhancer": 0}

    def _count(name):
        def _hook(_module, _inputs, _output):
            calls[name] += 1

        return _hook

    hooks = [
        model.id_backbone.register_forward_hook(_count("id")),
        model.dom_backbone.register_forward_hook(_count("dom")),
        model.dom_enhancer.register_forward_hook(_count("enhancer")),
    ]
    actual = dual_feature_forward(model, rows)
    for hook in hooks:
        hook.remove()
    assert calls == {"id": 1, "dom": 1, "enhancer": 1}
    torch.testing.assert_close(actual[0], expected["z_id"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected["z_dom"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2], expected["tx_logits"], rtol=0.0, atol=0.0)
