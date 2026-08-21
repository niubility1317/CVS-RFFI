from types import SimpleNamespace

import torch
from torch import nn

from SSDG import train_ssdg


def test_fasttrust_defaults_to_independent_u_batch_256_without_changing_l_batch():
    args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true", "--batch_size", "128"]
    )

    assert args.batch_size == 128
    assert args.muse_unlabeled_batch_size == 256
    assert train_ssdg._resolve_unlabeled_batch_size(args) == 256


def test_non_muse_loader_keeps_the_legacy_batch_size():
    args = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "false", "--batch_size", "96"]
    )
    assert train_ssdg._resolve_unlabeled_batch_size(args) == 96


def test_muse_u_loader_keeps_tail_batch_and_m0_uses_fasttrust_lr():
    muse_m0 = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "true", "--muse_level", "M0"]
    )
    legacy = train_ssdg.build_arg_parser().parse_args(
        ["--output_dir", "out", "--use_muse_ssdg", "false"]
    )

    assert train_ssdg._unlabeled_drop_last(muse_m0) is False
    assert train_ssdg._unlabeled_drop_last(legacy) is True
    assert train_ssdg._fasttrust_lr_enabled(muse_m0) is True


def test_fasttrust_lr_schedule_has_warmup_cosine_and_backbone_tail_scales():
    assert train_ssdg._fasttrust_lr_scales(1) == (0.2, 1.0)
    assert train_ssdg._fasttrust_lr_scales(5) == (1.0, 1.0)
    global_160, backbone_160 = train_ssdg._fasttrust_lr_scales(160)
    assert abs(global_160 - 0.1) < 1e-12
    assert backbone_160 == 1.0
    assert train_ssdg._fasttrust_lr_scales(161) == (0.1, 0.2)
    assert train_ssdg._fasttrust_lr_scales(180) == (0.1, 0.2)
    assert train_ssdg._fasttrust_lr_scales(181) == (0.1, 0.05)
    assert train_ssdg._fasttrust_lr_scales(200) == (0.1, 0.05)


def test_fasttrust_epoch_resource_metrics_report_realized_throughput_and_peak_memory():
    metrics = train_ssdg._fasttrust_epoch_resource_metrics(
        u_samples_per_step=256.0,
        u_forward_samples_per_step=512.0,
        steps=10,
        elapsed_s=5.0,
        peak_memory_bytes=3 * 1024**3,
    )

    assert metrics == {
        "muse/u_samples_per_s": 512.0,
        "muse/u_forward_samples_per_s": 1024.0,
        "muse/peak_cuda_memory_mb": 3072.0,
    }


class _CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, **_kwargs):
        self.calls += 1
        flat = x.flatten(1)
        return {
            "tx_logits": flat[:, :3],
            "z_id": flat[:, :4],
            "z_dom": flat[:, -4:],
            "constant": "kept",
        }


def test_fused_student_forward_matches_two_deterministic_views_with_one_call():
    strong = torch.arange(32, dtype=torch.float32).reshape(2, 2, 8)
    nuisance = strong + 100.0
    domains = torch.tensor([0, 1])
    model = _CountingModel()

    outputs = train_ssdg._forward_muse_student_views(
        model,
        strong,
        nuisance,
        domains,
        grl_lambda=0.1,
        fused=True,
    )

    assert model.calls == 1
    assert torch.equal(outputs["strong"]["z_id"], strong.flatten(1)[:, :4])
    assert torch.equal(outputs["nuisance"]["z_id"], nuisance.flatten(1)[:, :4])
    assert outputs["strong"]["constant"] == "kept"


def test_disabled_nuisance_branch_does_not_add_a_forward():
    model = _CountingModel()
    strong = torch.zeros(2, 2, 8)

    outputs = train_ssdg._forward_muse_student_views(
        model,
        strong,
        None,
        torch.tensor([0, 1]),
        grl_lambda=0.1,
        fused=True,
    )

    assert model.calls == 1
    assert outputs["nuisance"] is None
