from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from model_dual_cvsincnet import build_dual_model
import post_stage_common


def _build(*, bicad_xr: bool = False):
    return build_dual_model(
        6,
        12,
        model_size="S",
        dataset="wisig",
        input_len=128,
        model_variant="lite_d",
        branch_ablation="no_dac",
        bicad_xr=bicad_xr,
    )


def test_default_dual_model_state_dict_is_unchanged_when_bicad_disabled() -> None:
    old = build_dual_model(
        6,
        12,
        model_size="S",
        dataset="wisig",
        input_len=128,
        model_variant="lite_d",
        branch_ablation="no_dac",
    )
    new = _build(bicad_xr=False)

    assert list(old.state_dict()) == list(new.state_dict())


def test_bicad_model_exports_training_features_without_changing_tx_logits() -> None:
    torch.manual_seed(20260830)
    reference = _build(bicad_xr=False).eval()
    torch.manual_seed(20260830)
    bicad = _build(bicad_xr=True).eval()
    x = torch.randn(2, 2, 128)
    y = torch.tensor([0, 1])

    with torch.no_grad():
        reference_out = reference(x, y_tx=y, return_aux=True)
        bicad_out = bicad(x, y_tx=y, return_aux=True)

    assert {
        "z_id",
        "z_dom",
        "shared_features",
        "identity_features",
        "domain_features",
    } <= set(bicad_out)
    torch.testing.assert_close(bicad_out["tx_logits"], reference_out["tx_logits"])
    torch.testing.assert_close(bicad_out["shared_features"], bicad_out["z_id"])
    torch.testing.assert_close(bicad_out["identity_features"], bicad_out["z_id"])
    torch.testing.assert_close(bicad_out["domain_features"], bicad_out["z_dom"])


def test_bicad_training_features_do_not_enter_return_aux_false_fast_path() -> None:
    model = _build(bicad_xr=True).eval()
    x = torch.randn(2, 2, 128)

    with torch.no_grad():
        fast = model(x, return_aux=False)

    assert torch.is_tensor(fast)
    assert not isinstance(fast, dict)


def test_bicad_public_tx_classifier_reproduces_cosface_logits() -> None:
    model = _build(bicad_xr=True).eval()
    x = torch.randn(3, 2, 128)
    labels = torch.tensor([0, 1, 2])

    with torch.no_grad():
        output = model(x, y_tx=labels, return_aux=True)
        rebuilt = model.classify_identity_features(output["z_id"], labels=labels)

    torch.testing.assert_close(rebuilt, output["tx_logits"])


def test_post_stage_common_only_passes_bicad_switch_for_explicit_method() -> None:
    captured: list[dict[str, object]] = []

    class _Model(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def _fake_builder(*args: object, **kwargs: object) -> _Model:
        del args
        captured.append(kwargs)
        return _Model()

    base_args = SimpleNamespace(num_classes=6, num_domains=12)
    with patch.object(post_stage_common, "build_dual_model", side_effect=_fake_builder):
        post_stage_common.build_baseline_model(
            SimpleNamespace(**vars(base_args), phase1_method="adv3b02"),
            torch.device("cpu"),
        )
        post_stage_common.build_baseline_model(
            SimpleNamespace(**vars(base_args), phase1_method="bicad_xr"),
            torch.device("cpu"),
        )

    assert "bicad_xr" not in captured[0]
    assert captured[1]["bicad_xr"] is True
