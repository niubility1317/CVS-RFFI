from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

import cvsrffi.stage2_d105_feature_tap as feature_tap
from cvsrffi.stage2_d105_feature_tap import (
    D105FeatureTapError,
    extract_d105_feature_tap,
)


class _Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.joint_proj = nn.Sequential(
            nn.Linear(320, 160, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(0.25),
        )


class _IdentityBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls_head = _Head()
        self.calls = 0

    def forward(
        self,
        rows: torch.Tensor,
        y: torch.Tensor | None = None,
        return_aux: bool = True,
        domain_labels: torch.Tensor | None = None,
    ):
        del y, domain_labels
        self.calls += 1
        flat = rows.reshape(len(rows), -1)
        repeats = (320 + flat.shape[1] - 1) // flat.shape[1]
        hidden = flat.repeat(1, repeats)[:, :320]
        z_id = self.cls_head.joint_proj(hidden)
        logits = torch.stack((z_id[:, 0], z_id[:, 1]), dim=1)
        if not return_aux:
            return logits
        return {"feat_joint": z_id, "feat_imp": z_id, "logits": logits}


class _DomainBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        rows: torch.Tensor,
        y: torch.Tensor | None = None,
        return_aux: bool = True,
        domain_labels: torch.Tensor | None = None,
    ):
        del y, domain_labels
        self.calls += 1
        flat = rows.reshape(len(rows), -1)
        repeats = (160 + flat.shape[1] - 1) // flat.shape[1]
        feature = flat.repeat(1, repeats)[:, :160]
        logits = torch.stack((feature[:, 0], feature[:, 1]), dim=1)
        if not return_aux:
            return logits
        return {"feat_imp": feature, "logits": logits}


class _Enhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self, feature: torch.Tensor, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return feature + rows.mean(dim=(1, 2), keepdim=False)[:, None], feature


class _Model(nn.Module):
    id_feature_key = "feat_joint"
    dom_feature_key = "feat_imp"

    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _IdentityBackbone()
        self.dom_backbone = _DomainBackbone()
        self.dom_enhancer = _Enhancer()

    @staticmethod
    def _pick_z_id(aux):
        return aux["feat_joint"]

    @staticmethod
    def _pick_z_dom(aux):
        return aux["feat_imp"]


def test_d105_feature_tap_is_exact_one_pass_and_readonly() -> None:
    torch.manual_seed(7)
    model = _Model().eval()
    rows = torch.randn(5, 2, 12, dtype=torch.float32)
    result = extract_d105_feature_tap(model, rows)
    assert model.id_backbone.calls == 1
    assert model.dom_backbone.calls == 1
    assert model.dom_enhancer.calls == 1
    assert result.z_id.shape == (5, 160)
    assert result.z_dom.shape == (5, 160)
    assert result.hidden.shape == (5, 320)
    assert result.pre_relu.shape == (5, 160)
    hidden = (
        torch.frombuffer(
            bytearray(result.hidden.tobytes(order="C")), dtype=torch.float32
        )
        .reshape(result.hidden.shape)
        .clone()
    )
    expected_pre = np.asarray(
        model.id_backbone.cls_head.joint_proj[0](hidden).detach().cpu().tolist(),
        dtype=np.float32,
    )
    assert np.array_equal(result.pre_relu, expected_pre)
    assert np.array_equal(result.z_id, np.maximum(result.pre_relu, 0.0))
    assert all(
        not getattr(result, name).flags.writeable
        for name in ("z_id", "z_dom", "hidden", "pre_relu")
    )


def test_d105_feature_tap_output_bridge_bypasses_tensor_numpy_type_failure() -> None:
    class _NumpyRejectingTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value: torch.Tensor) -> torch.Tensor:
            return torch.Tensor._make_subclass(cls, value.detach(), require_grad=False)

        def numpy(self):
            raise TypeError("expected np.ndarray (got numpy.ndarray)")

    source = np.asarray(
        [[-0.0, 1.25, -3.5], [8.0, 2.0, -4.0]], dtype=np.float32
    )
    tensor = (
        torch.frombuffer(bytearray(source.tobytes(order="C")), dtype=torch.float32)
        .reshape(source.shape)
        .clone()
    )
    result = feature_tap._to_numpy(
        _NumpyRejectingTensor(tensor), width=3, name="bridge test"
    )
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert result.tobytes(order="C") == source.tobytes(order="C")


def test_d105_feature_tap_rejects_training_nonfinite_and_wrong_joint_width() -> None:
    rows = torch.zeros(2, 2, 8, dtype=torch.float32)
    with pytest.raises(D105FeatureTapError, match="eval model"):
        extract_d105_feature_tap(_Model().train(), rows)
    rows[0, 0, 0] = float("nan")
    with pytest.raises(D105FeatureTapError, match="finite"):
        extract_d105_feature_tap(_Model().eval(), rows)
    model = _Model().eval()
    model.id_backbone.cls_head.joint_proj[0] = nn.Linear(319, 160)
    with pytest.raises(D105FeatureTapError, match="contract drift"):
        extract_d105_feature_tap(model, torch.zeros(2, 2, 8))
