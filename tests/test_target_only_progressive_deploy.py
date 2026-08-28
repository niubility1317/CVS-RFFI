from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from cvsrffi.target_only_progressive_adapt import (
    CompactH6Suffix,
    SFTAPFTConfig,
    H6SuffixTrainer,
    TargetOnlyAdaptationDataset,
    TargetPrototypeHead,
    audit_h6_support_safety,
    build_h6_prefix_cache,
    class_cvar_from_class_losses,
    encode_h6_prefix,
    fit_sf_tapft,
    fit_sf_tapft_inplace,
    forward_h6_suffix,
    forward_h6_prefix_cache,
    support_hard_pair_loss,
)


class _T3(nn.Module):
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
        feat_joint = base + 0.03 * dac_local + 0.02 * pa_local + dac_delta + pa_delta
        logits = self.head(feat_joint)
        zero = torch.zeros_like(feat_joint)
        if not return_emb:
            return logits, zero[:, 0]
        return logits, zero, zero, zero, zero, zero, zero, feat_joint


class _CacheableBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb_dim = 4
        self.meta_adapter_time = nn.Identity()
        self.t3 = _T3()
        self.t_pool = nn.AdaptiveAvgPool1d(1)
        self.t_proj = nn.Linear(4, 4)
        self.fuse = nn.Linear(8, 4)
        self.meta_adapter_fusion = nn.Identity()
        self.cls_head = _SourceHead()
        self.frozen_other = nn.Linear(4, 4)
        self.frozen_aux = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, y=None, return_aux: bool = False):
        del y
        t = self.t3(x)
        t_emb = self.meta_adapter_time(self.t_proj(self.t_pool(t).squeeze(-1)))
        pooled = x.mean(dim=-1)
        other = self.frozen_other(pooled)
        base = self.meta_adapter_fusion(self.fuse(torch.cat([t_emb, other], dim=1)))
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


def _dataset() -> TargetOnlyAdaptationDataset:
    torch.manual_seed(13)
    rows = torch.randn(6, 4, 5)
    rows[:2, 0] += 2.0
    rows[2:4, 1] += 2.0
    rows[4:, 2] += 2.0
    return TargetOnlyAdaptationDataset(
        received_iq=rows,
        labels=torch.tensor([0, 0, 1, 1, 2, 2]),
        physical_ids=tuple(f"p{index}" for index in range(6)),
        groups=tuple(f"g{index}" for index in range(6)),
    )


def test_support_hard_pair_loss_is_class_permutation_invariant() -> None:
    logits = torch.tensor(
        [
            [2.0, 1.7, -0.5],
            [1.6, 1.8, -0.2],
            [1.4, 2.1, 0.3],
            [1.7, 2.2, 0.1],
            [-0.2, 0.9, 1.8],
            [0.0, 1.1, 2.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    original = support_hard_pair_loss(logits, labels, class_count=3, margin=0.2)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(3)
    permuted = support_hard_pair_loss(
        logits[:, permutation],
        inverse[labels],
        class_count=3,
        margin=0.2,
    )
    assert original.item() > 0.0
    assert torch.allclose(original, permuted, atol=1.0e-7, rtol=0.0)


def test_support_hard_pair_loss_rejects_missing_support_class() -> None:
    with pytest.raises(ValueError, match="every registered class"):
        support_hard_pair_loss(
            torch.randn(4, 3),
            torch.tensor([0, 0, 1, 1]),
            class_count=3,
            margin=0.2,
        )


def test_h6_prefix_cache_matches_full_logits_and_norm_gradients() -> None:
    torch.manual_seed(7)
    full_model = _CacheableBackbone().eval()
    cached_model = copy.deepcopy(full_model).eval()
    values = _dataset().received_iq
    head = TargetPrototypeHead(torch.randn(3, 4), (0, 1, 2), scale=8.0)
    cached_head = copy.deepcopy(head)

    full_embedding = full_model(values, return_aux=True)["feat_joint"]
    full_logits = head(full_embedding)
    full_logits.square().mean().backward()
    full_grad = {
        name: parameter.grad.detach().clone()
        for name, parameter in full_model.named_parameters()
        if name in {"t3.norm.weight", "t3.norm.bias"}
    }

    cache = build_h6_prefix_cache(cached_model, values, storage_dtype=torch.float32)
    cached_embedding = forward_h6_prefix_cache(cached_model, cache)
    cached_logits = cached_head(cached_embedding)
    cached_logits.square().mean().backward()
    cached_grad = {
        name: parameter.grad.detach().clone()
        for name, parameter in cached_model.named_parameters()
        if name in {"t3.norm.weight", "t3.norm.bias"}
    }

    assert torch.max(torch.abs(full_logits - cached_logits)).item() < 1.0e-5
    assert set(full_grad) == set(cached_grad)
    assert max(
        torch.max(torch.abs(full_grad[name] - cached_grad[name])).item()
        for name in full_grad
    ) < 1.0e-5
    assert torch.equal(full_logits.argmax(dim=1), cached_logits.argmax(dim=1))


def test_h6_stable_suffix_api_reports_exact_cache_bytes_and_support_safety() -> None:
    torch.manual_seed(31)
    model = _CacheableBackbone().eval()
    values = _dataset().received_iq
    labels = _dataset().labels
    head = TargetPrototypeHead(torch.randn(3, 4), (0, 1, 2), scale=8.0)

    cache = encode_h6_prefix(model, values, storage_dtype=torch.float32)
    trainer = H6SuffixTrainer(model=model, head=head, cache=cache)
    embedding = forward_h6_suffix(model, cache)
    assert torch.allclose(trainer.embedding(), embedding, atol=0.0, rtol=0.0)
    expected_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            cache.pre_t3_norm,
            cache.frozen_fuse_tail,
            *cache.cls_head_kwargs.values(),
        )
    )
    assert cache.tensor_bytes == expected_bytes
    assert trainer.cache_tensor_bytes == expected_bytes

    safety = audit_h6_support_safety(model, head, cache, values, labels)
    assert safety.passed is True
    assert safety.prediction_mismatches == 0
    assert safety.per_class_recall_mismatches == 0
    assert safety.max_abs_logit_delta < 1.0e-5


def test_fit_h6_prefix_cache_removes_backbone_training_forwards() -> None:
    torch.manual_seed(17)
    result = fit_sf_tapft(
        _CacheableBackbone(),
        _dataset(),
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=(2, 1, 1),
            scheduler_reference_steps=4,
            fast_tail_start_step=2,
            fast_tail_steps=1,
            head_polish_steps=1,
            prefix_cache_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
        ),
        checkpoint_selection_mode="final_step",
    )
    assert result.audit.backbone_train_forward_steps == 0
    assert result.audit.prefix_cache_build_forward_steps == 1
    assert result.audit.cached_suffix_forward_steps == 3
    assert result.audit.head_polish_steps == 1


def _short_h6_config() -> SFTAPFTConfig:
    return SFTAPFTConfig(
        adapter_rank=2,
        trainability_profile="p1_head_norm",
        norm_rules=(("t3", "weight_bias"),),
        phase_steps=(2, 1, 1),
        scheduler_reference_steps=4,
        fast_tail_start_step=2,
        fast_tail_steps=1,
        head_polish_steps=1,
        prefix_cache_dtype="float32",
        mixed_precision=False,
        checkpoint_average_top_k=1,
    )


def test_fit_h6_inplace_reuses_owned_model_and_keeps_minimal_anchors() -> None:
    torch.manual_seed(23)
    model = _CacheableBackbone()
    frozen_before = model.t3.dw.weight.detach().clone()

    result = fit_sf_tapft_inplace(
        model,
        _dataset(),
        _short_h6_config(),
        checkpoint_selection_mode="final_step",
    )

    assert result.model is model
    assert set(result.base_parameter_anchors) == {
        "model.t3.norm.bias",
        "model.t3.norm.weight",
        "head.weight",
    }
    assert torch.equal(model.t3.dw.weight, frozen_before)
    assert result.audit.nonpermitted_changed_names == ()


def test_compact_h6_inplace_matches_reference_training_with_gradient_clipping() -> None:
    torch.manual_seed(47)
    checkpoint = _CacheableBackbone()
    reference_input = copy.deepcopy(checkpoint)
    compact_input = copy.deepcopy(checkpoint)
    config = replace(_short_h6_config(), gradient_clip_norm=0.01)

    torch.manual_seed(53)
    reference = fit_sf_tapft(
        reference_input,
        _dataset(),
        config,
        checkpoint_selection_mode="final_step",
    )
    torch.manual_seed(53)
    compact = fit_sf_tapft_inplace(
        compact_input,
        _dataset(),
        config,
        checkpoint_selection_mode="final_step",
    )

    assert torch.allclose(
        reference.model.t3.norm.weight,
        compact.model.t3.norm.weight,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        reference.model.t3.norm.bias,
        compact.model.t3.norm.bias,
        atol=1.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(
        reference.head.weight,
        compact.head.weight,
        atol=1.0e-7,
        rtol=0.0,
    )


def test_fit_h6_default_still_copies_checkpoint_model() -> None:
    model = _CacheableBackbone()
    result = fit_sf_tapft(
        model,
        _dataset(),
        _short_h6_config(),
        checkpoint_selection_mode="final_step",
    )
    assert result.model is not model


def test_hard_pair_config_is_strictly_validated() -> None:
    with pytest.raises(ValueError, match="hard_pair_weight"):
        SFTAPFTConfig(hard_pair_weight=-0.01)
    with pytest.raises(ValueError, match="hard_pair_margin"):
        SFTAPFTConfig(hard_pair_margin=-0.01)
    with pytest.raises(ValueError, match="prefix_cache_dtype"):
        SFTAPFTConfig(prefix_cache_dtype="bf16")


def test_legacy_prefix_cache_dtype_maps_to_storage_and_compute_dtypes() -> None:
    config = SFTAPFTConfig(prefix_cache_dtype="float16")

    assert config.cache_storage_dtype == "float16"
    assert config.suffix_compute_dtype == "float32"
    assert config.cache_device == "model"


def test_explicit_cache_precision_controls_are_strictly_validated() -> None:
    config = SFTAPFTConfig(
        cache_storage_dtype="bfloat16",
        suffix_compute_dtype="float32",
        cache_device="cpu",
    )
    assert config.cache_storage_dtype == "bfloat16"
    assert config.suffix_compute_dtype == "float32"
    with pytest.raises(ValueError, match="equivalence-qualified"):
        SFTAPFTConfig(
            cache_storage_dtype="bfloat16",
            suffix_compute_dtype="bfloat16",
        )
    with pytest.raises(ValueError, match="suffix_compute_dtype"):
        SFTAPFTConfig(suffix_compute_dtype="int8")
    with pytest.raises(ValueError, match="cache_device"):
        SFTAPFTConfig(cache_device="disk")


def test_prefix_cache_materializes_storage_to_compute_once() -> None:
    model = _CacheableBackbone().eval()
    cache = encode_h6_prefix(model, _dataset().received_iq, storage_dtype=torch.float16)

    compute_cache = cache.materialize_once(device=torch.device("cpu"), dtype=torch.float32)
    first_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            compute_cache.pre_t3_norm,
            compute_cache.frozen_fuse_tail,
            *compute_cache.cls_head_kwargs.values(),
        )
    )
    forward_h6_suffix(model, compute_cache)
    forward_h6_suffix(model, compute_cache)
    second_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            compute_cache.pre_t3_norm,
            compute_cache.frozen_fuse_tail,
            *compute_cache.cls_head_kwargs.values(),
        )
    )

    assert compute_cache.storage_dtype == torch.float32
    assert first_ptrs == second_ptrs


def test_compact_h6_suffix_has_no_full_model_reference_and_matches_gradients() -> None:
    torch.manual_seed(41)
    reference_model = _CacheableBackbone().eval()
    compact_source = copy.deepcopy(reference_model).eval()
    values = _dataset().received_iq
    reference_head = TargetPrototypeHead(torch.randn(3, 4), (0, 1, 2), scale=8.0)
    compact_head = copy.deepcopy(reference_head)
    cache = encode_h6_prefix(compact_source, values, storage_dtype=torch.float32)

    reference_logits = reference_head(forward_h6_suffix(reference_model, cache))
    reference_logits.square().mean().backward()
    compact = CompactH6Suffix.from_model(compact_source, compact_head, cache)
    compact_logits = compact.logits()
    compact_logits.square().mean().backward()

    assert not hasattr(compact, "model")
    assert torch.allclose(reference_logits, compact_logits, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(
        reference_model.t3.norm.weight.grad,
        compact.t3.norm.weight.grad,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.allclose(
        reference_head.weight.grad,
        compact.target_head.weight.grad,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert set(compact.export_permitted_state()) == {
        "model.t3.norm.bias",
        "model.t3.norm.weight",
        "head.weight",
    }


def test_class_cvar_uses_top2_class_mean_losses() -> None:
    losses = torch.tensor([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])

    value = class_cvar_from_class_losses(losses, top_k=2)

    assert value.item() == pytest.approx(0.85)


def test_head_cvar_config_is_strictly_validated() -> None:
    assert SFTAPFTConfig(
        head_cvar_weight=0.03,
        head_cvar_top_k=2,
        head_cvar_steps=30,
        head_polish_steps=30,
    ).head_cvar_steps == 30
    with pytest.raises(ValueError, match="head_cvar_weight"):
        SFTAPFTConfig(head_cvar_weight=-0.01)
    with pytest.raises(ValueError, match="head_cvar_top_k"):
        SFTAPFTConfig(head_cvar_top_k=0)
    with pytest.raises(ValueError, match="head_cvar_steps"):
        SFTAPFTConfig(head_cvar_steps=1, head_polish_steps=0)


def test_fit_h6_runs_head_only_class_cvar_after_base_adaptation() -> None:
    torch.manual_seed(43)
    result = fit_sf_tapft_inplace(
        _CacheableBackbone(),
        _dataset(),
        SFTAPFTConfig(
            adapter_rank=2,
            trainability_profile="p1_head_norm",
            norm_rules=(("t3", "weight_bias"),),
            phase_steps=(2, 1, 2),
            scheduler_reference_steps=5,
            head_polish_steps=2,
            head_cvar_steps=1,
            head_cvar_weight=0.03,
            head_cvar_top_k=2,
            prefix_cache_dtype="float32",
            mixed_precision=False,
            checkpoint_average_top_k=1,
        ),
        checkpoint_selection_mode="final_step",
    )

    assert result.audit.head_cvar_steps == 1
    assert result.audit.head_cvar_weight == pytest.approx(0.03)
    assert result.audit.head_cvar_top_k == 2
    assert len(result.audit.head_cvar_losses) == 1
    assert result.audit.head_cvar_losses[0] > 0.0
