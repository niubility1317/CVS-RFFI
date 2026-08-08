from __future__ import annotations

import pytest
import torch

from SSDG.train_ssdg import _select_open_world_feature, build_arg_parser


def test_ow_feat_key_parser_allowlist_and_default():
    parser = build_arg_parser()

    default_args = parser.parse_args(["--output_dir", "unused"])
    geometry_args = parser.parse_args(
        ["--output_dir", "unused", "--ow_feat_key", "id_feat_cls"]
    )

    assert default_args.ow_feat_key == "z_id"
    assert geometry_args.ow_feat_key == "id_feat_cls"
    with pytest.raises(SystemExit):
        parser.parse_args(["--output_dir", "unused", "--ow_feat_key", "feat_joint"])


def test_default_open_world_feature_is_the_exact_z_id_object():
    z_id = torch.randn(3, 160, requires_grad=True)

    selected = _select_open_world_feature(
        {"id_feat_cls": torch.randn(3, 160, requires_grad=True)},
        z_id,
        key="z_id",
    )

    assert selected is z_id


def test_id_feat_cls_is_selected_and_only_it_receives_open_loss_gradient():
    z_id = torch.randn(3, 160, requires_grad=True)
    id_feat_cls = torch.randn(3, 160, requires_grad=True)

    selected = _select_open_world_feature(
        {"id_feat_cls": id_feat_cls},
        z_id,
        key="id_feat_cls",
    )
    selected.square().sum().backward()

    assert selected is id_feat_cls
    assert id_feat_cls.grad is not None
    assert float(id_feat_cls.grad.abs().sum()) > 0.0
    assert z_id.grad is None


def test_lite_d_top_level_id_feat_cls_is_160d_without_query_data():
    from model_dual_cvsincnet import build_dual_model

    model = build_dual_model(
        num_classes=4,
        num_domains=4,
        dataset="wisig",
        input_len=128,
        model_variant="lite_d",
    ).eval()
    x = torch.randn(2, 2, 128)
    y = torch.tensor([0, 1])
    domains = torch.tensor([0, 1])

    with torch.no_grad():
        out = model(x, y_tx=y, domain_labels=domains, return_aux=True)

    assert tuple(out["id_feat_cls"].shape) == (2, 160)
    assert tuple(out["z_id"].shape) == (2, 160)


@pytest.mark.parametrize(
    ("out", "match"),
    [
        ({}, "top-level tensor"),
        ({"id_feat_cls": "not-a-tensor"}, "top-level tensor"),
        ({"id_feat_cls": torch.full((2, 160), float("nan"))}, "non-finite"),
        ({"id_feat_cls": torch.randn(2, 160, 1)}, "2D"),
        ({"id_feat_cls": torch.randn(3, 160)}, "row mismatch"),
        ({"id_feat_cls": torch.randn(2, 159)}, "dimension mismatch"),
    ],
)
def test_id_feat_cls_selection_rejects_invalid_contract(out, match):
    z_id = torch.randn(2, 160)

    with pytest.raises(ValueError, match=match):
        _select_open_world_feature(out, z_id, key="id_feat_cls")
