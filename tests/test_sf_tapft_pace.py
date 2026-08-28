from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from cvsrffi.target_only_progressive_adapt import (
    CompactTimeNormSuffix,
    SFTAPFTConfig,
    TargetPrototypeHead,
    TargetOnlyAdaptationDataset,
    encode_trainable_suffix_prefix,
    fit_sf_tapft_inplace,
    fit_sf_tapft_support_oof_head_bias,
    fit_zero_sum_class_bias,
    stable_preservation_kl,
    stable_support_weights,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dw = nn.Conv1d(4, 4, 3, padding=1, groups=4, bias=False)
        self.pw = nn.Conv1d(4, 4, 1, bias=False)
        self.norm = nn.GroupNorm(2, 4)
        self.act = nn.ReLU()
        self.pool = nn.Identity()
        self.drop = nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.act(self.norm(self.pw(self.dw(value))))))


class _SourceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 3, bias=False)

    def forward(
        self,
        base: torch.Tensor,
        *,
        dac_local: torch.Tensor,
        pa_local: torch.Tensor,
        labels=None,
        return_emb: bool = False,
        dac_delta: torch.Tensor,
        pa_delta: torch.Tensor,
    ):
        del labels
        embedding = base + 0.03 * dac_local + 0.02 * pa_local + dac_delta + pa_delta
        logits = self.head(embedding)
        zero = torch.zeros_like(embedding)
        if not return_emb:
            return logits, zero[:, 0]
        return logits, zero, zero, zero, zero, zero, zero, embedding


class _GeneralCacheableBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_dim = 4
        self.time_fuse = nn.Sequential(
            nn.Conv1d(4, 4, 1, bias=False), nn.GroupNorm(2, 4), nn.ReLU()
        )
        self.time_down = nn.Identity()
        self.t1 = _Block()
        self.t2 = _Block()
        self.t3 = _Block()
        self.t_pool = nn.AdaptiveAvgPool1d(1)
        self.t_proj = nn.Linear(4, 4)
        self.meta_adapter_time = nn.Identity()
        self.fuse = nn.Linear(8, 4)
        self.meta_adapter_fusion = nn.Identity()
        self.cls_head = _SourceHead()
        self.frozen_other = nn.Linear(4, 4)
        self.frozen_aux = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, y=None, return_aux: bool = False):
        del y
        time = self.time_down(self.time_fuse(x))
        time = self.t3(self.t2(self.t1(time)))
        time_embedding = self.meta_adapter_time(self.t_proj(self.t_pool(time).squeeze(-1)))
        pooled = x.mean(dim=-1)
        other = self.frozen_other(pooled)
        base = self.meta_adapter_fusion(self.fuse(torch.cat([time_embedding, other], dim=1)))
        aux = self.frozen_aux(pooled)
        output = self.cls_head(
            base,
            dac_local=aux,
            pa_local=0.5 * aux,
            labels=None,
            return_emb=True,
            dac_delta=0.01 * aux,
            pa_delta=0.02 * aux,
        )
        if return_aux:
            return {"feat_joint": output[-1], "logits": output[0]}
        return output[0]


def _values() -> torch.Tensor:
    torch.manual_seed(81)
    return torch.randn(6, 4, 8)


def _dataset() -> TargetOnlyAdaptationDataset:
    return TargetOnlyAdaptationDataset(
        received_iq=_values(),
        labels=torch.tensor([0, 0, 1, 1, 2, 2]),
        physical_ids=tuple(f"p{index}" for index in range(6)),
        groups=tuple(f"g{index}" for index in range(6)),
    )


def test_stable_support_weights_and_preservation_kl_are_support_only() -> None:
    teacher = torch.tensor(
        [[5.0, 0.0, -1.0], [0.1, 2.0, 1.8], [0.0, 0.2, 3.0], [2.0, 2.1, 0.0]]
    )
    labels = torch.tensor([0, 1, 2, 0])
    weights = stable_support_weights(teacher, labels)

    assert weights.shape == (4,)
    assert weights[0] > weights[1]
    assert weights[2] > weights[3]
    assert not weights.requires_grad

    student = teacher.detach().clone().requires_grad_(True)
    same = stable_preservation_kl(student, teacher, weights, temperature=2.0)
    shifted = stable_preservation_kl(
        student + torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        teacher,
        weights,
        temperature=2.0,
    )
    assert same.item() == pytest.approx(0.0, abs=1.0e-7)
    assert shifted > same


@pytest.mark.parametrize("boundary", ["t2.norm", "time_fuse.1"])
def test_generalized_time_norm_suffix_matches_full_logits_and_gradients(boundary: str) -> None:
    torch.manual_seed(83)
    reference = _GeneralCacheableBackbone().eval()
    cached_source = copy.deepcopy(reference).eval()
    values = _values()
    reference_head = TargetPrototypeHead(torch.randn(3, 4), (0, 1, 2), scale=8.0)
    compact_head = copy.deepcopy(reference_head)

    full_logits = reference_head(reference(values, return_aux=True)["feat_joint"])
    full_logits.square().mean().backward()
    cache = encode_trainable_suffix_prefix(
        cached_source, values, boundary, storage_dtype=torch.float32
    )
    compact = CompactTimeNormSuffix.from_model(cached_source, compact_head, cache)
    compact_logits = compact.logits()
    compact_logits.square().mean().backward()

    assert not hasattr(compact, "model")
    assert torch.allclose(full_logits, compact_logits, atol=1.0e-6, rtol=1.0e-6)
    expected_names = (
        {"t2.norm.weight", "t2.norm.bias", "t3.norm.weight", "t3.norm.bias"}
        if boundary == "t2.norm"
        else {
            "time_fuse.1.weight",
            "time_fuse.1.bias",
            "t1.norm.weight",
            "t1.norm.bias",
            "t2.norm.weight",
            "t2.norm.bias",
            "t3.norm.weight",
            "t3.norm.bias",
        }
    )
    reference_named = dict(reference.named_parameters())
    compact_named = dict(compact.named_parameters())
    for name in expected_names:
        assert torch.allclose(
            reference_named[name].grad,
            compact_named[name].grad,
            atol=1.0e-6,
            rtol=1.0e-6,
        )


def test_zero_sum_bias_improves_oof_nll_and_head_applies_it() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [2.5, 0.2, 0.0],
            [0.0, 1.0, 0.8],
            [0.0, 0.9, 0.7],
            [0.0, 0.8, 1.0],
            [0.0, 0.7, 0.9],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    calibration = fit_zero_sum_class_bias(logits, labels, steps=40, lr=0.1, l2=0.01)

    assert calibration.bias.shape == (3,)
    assert calibration.bias.sum().item() == pytest.approx(0.0, abs=1.0e-6)
    assert calibration.nll_after <= calibration.nll_before

    head = TargetPrototypeHead(torch.eye(3), (0, 1, 2), scale=1.0, bias=calibration.bias)
    assert torch.allclose(head(torch.eye(3)), torch.eye(3) + calibration.bias)


def test_pace_config_rejects_unsafe_or_incomplete_controls() -> None:
    config = SFTAPFTConfig(
        pace_expand_start_step=4,
        pace_norm_rules=(("t3", "weight_bias"), ("t2", "weight")),
        pace_tail_weight=0.03,
        pace_preserve_weight=0.10,
        pace_bias_steps=40,
    )
    assert config.pace_expand_start_step == 4
    with pytest.raises(ValueError, match="pace_expand_start_step"):
        SFTAPFTConfig(phase_steps=(2, 1, 1), pace_expand_start_step=999)
    with pytest.raises(ValueError, match="pace_preserve_weight"):
        SFTAPFTConfig(pace_preserve_weight=-0.1)
    with pytest.raises(ValueError, match="pace_bias_steps"):
        SFTAPFTConfig(pace_bias_steps=1, pace_expand_start_step=0)


def test_pace_fit_uses_d0_teacher_then_expands_only_registered_norms() -> None:
    torch.manual_seed(89)
    model = _GeneralCacheableBackbone()
    t1_before = model.t1.norm.weight.detach().clone()
    t2_before = model.t2.norm.weight.detach().clone()
    result = fit_sf_tapft_inplace(
        model,
        _dataset(),
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            pace_norm_rules=(("t3", "weight_bias"), ("t2", "weight")),
            phase_steps=(2, 2, 0),
            scheduler_reference_steps=4,
            pace_expand_start_step=2,
            pace_tail_weight=0.03,
            pace_preserve_weight=0.10,
            cache_storage_dtype="float32",
            suffix_compute_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
        ),
        checkpoint_selection_mode="final_step",
    )

    assert result.audit.trainable_names_by_phase["A"] == (
        "t3.norm.bias",
        "t3.norm.weight",
    )
    assert "t2.norm.weight" in result.audit.trainable_names_by_phase["B"]
    assert result.audit.pace_teacher_snapshot_count == 1
    assert result.audit.pace_expanded_optimizer_steps == 2
    assert len(result.audit.pace_tail_losses) == 2
    assert len(result.audit.pace_preserve_losses) == 2
    assert result.audit.backbone_train_forward_steps == 0
    assert result.audit.prefix_cache_build_forward_steps == 1
    assert torch.equal(model.t1.norm.weight, t1_before)
    assert not torch.equal(model.t2.norm.weight, t2_before)


def test_pace_expansion_can_start_inside_the_same_training_phase() -> None:
    torch.manual_seed(97)
    result = fit_sf_tapft_inplace(
        _GeneralCacheableBackbone(),
        _dataset(),
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            pace_norm_rules=(("t3", "weight_bias"), ("t2", "weight")),
            phase_steps=(2, 1, 3),
            scheduler_reference_steps=6,
            pace_expand_start_step=4,
            pace_tail_weight=0.03,
            pace_preserve_weight=0.10,
            cache_storage_dtype="float32",
            suffix_compute_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
            seed=97,
        ),
    )

    assert result.audit.pace_teacher_snapshot_count == 1
    assert result.audit.pace_expanded_optimizer_steps == 2
    assert len(result.audit.pace_tail_losses) == 2
    assert len(result.audit.pace_preserve_losses) == 2


def test_head_bias_oof_reuses_one_embedding_forward_and_updates_only_bias() -> None:
    torch.manual_seed(97)
    result = fit_sf_tapft_inplace(
        _GeneralCacheableBackbone(),
        _dataset(),
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=(2, 1, 1),
            scheduler_reference_steps=4,
            cache_storage_dtype="float32",
            suffix_compute_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
        ),
        checkpoint_selection_mode="final_step",
    )
    weight_before = result.head.weight.detach().clone()
    calibration = fit_sf_tapft_support_oof_head_bias(
        result,
        _dataset(),
        folds=2,
        head_only_steps=3,
        lr=0.05,
        l2=0.01,
        seed=101,
    )

    assert calibration.embedding_forward_steps == 1
    assert calibration.head_only_steps == 9
    assert result.head.bias.sum().item() == pytest.approx(0.0, abs=1.0e-6)
    assert torch.equal(result.head.weight, weight_before)
