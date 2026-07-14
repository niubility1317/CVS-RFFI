from __future__ import annotations

import torch

from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    LORA_TARGETS,
    LoRALinear,
    inject_feat_joint_lora,
)


class _Head(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_proj = torch.nn.Sequential(torch.nn.Linear(160, 160))
        self.pa_proj = torch.nn.Sequential(torch.nn.Linear(320, 160))
        self.id_gate = torch.nn.Sequential(torch.nn.Linear(160, 160))
        self.joint_proj = torch.nn.Sequential(torch.nn.Linear(320, 160))
        self.imp_merge = torch.nn.Sequential(torch.nn.Linear(160, 160))
        self.head = torch.nn.Linear(160, 6, bias=False)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t_proj = torch.nn.Linear(96, 160)
        self.f_proj = torch.nn.Linear(32, 160)
        self.pa_proj = torch.nn.Sequential(torch.nn.Linear(64, 160))
        self.fuse = torch.nn.Sequential(torch.nn.Linear(321, 160))
        self.cls_head = _Head()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = _Backbone()


def test_lora_linear_is_exact_identity_at_initialization() -> None:
    base = torch.nn.Linear(7, 5)
    rows = torch.randn(4, 7)
    expected = base(rows).detach()
    adapter = LoRALinear(base, rank=2, alpha=2.0)
    torch.testing.assert_close(adapter(rows), expected, rtol=0.0, atol=0.0)
    assert adapter.trainable_parameter_count == 24
    assert not any(parameter.requires_grad for parameter in adapter.base.parameters())


def test_feat_joint_lora_is_under_caps_and_freezes_original_checkpoint() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(model, rank=8, alpha=8.0)
    assert audit["trainable_parameters"] == 12_800
    assert audit["adapter_state_bytes_fp16"] == 25_600
    assert audit["adapter_macs_per_query"] == 12_800
    assert len(audit["target_modules"]) == len(LORA_TARGETS) == 4
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".lora_" in name for name in trainable)
    assert not model.id_backbone.cls_head.head.weight.requires_grad
    assert not model.id_backbone.cls_head.imp_merge[0].weight.requires_grad


def test_late_feat_joint_rank4_compresses_full_late_scope() -> None:
    model = _Model()
    audit = inject_feat_joint_lora(
        model, rank=4, alpha=4.0, scope="late_feat_joint"
    )
    assert audit["trainable_parameters"] == 11_012
    assert audit["adapter_state_bytes_fp16"] == 22_024
    assert audit["adapter_macs_per_query"] == 11_012
    assert audit["scope"] == "late_feat_joint"
    assert len(audit["target_modules"]) == 8


def test_cli_accepts_class_symmetric_dro_controls() -> None:
    from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import parse_args

    args = parse_args(
        [
            "--config",
            "config.json",
            "--ckpt",
            "model.pth",
            "--out_root",
            "out",
            "--receiver",
            "8-8",
            "--new_count",
            "20",
            "--seed",
            "713101",
            "--cosine_margin",
            "0.1",
            "--class_dro_temperature",
            "5",
        ]
    )
    assert args.cosine_margin == 0.1
    assert args.class_dro_temperature == 5.0
