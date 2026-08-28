from __future__ import annotations

import copy

import torch
from torch import nn

from cvsrffi.stage2_sf_t3_delta_bundle import (
    T3_DELTA_PARAMETER_NAMES,
    convert_sf_tapft_delta_bundle_to_t3_only,
    load_t3_only_delta_bundle_strict,
    write_t3_only_delta_bundle,
)


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.t3 = nn.Module()
        self.t3.norm = nn.LayerNorm(2)


class _RealPathToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = nn.Module()
        self.id_backbone.t3 = nn.Module()
        self.id_backbone.t3.norm = nn.LayerNorm(2)


def _full_delta_payload() -> dict:
    return {
        "schema": "cvs.sf_tapft.delta.v3",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule",
        "split_id": "split",
        "base_checkpoint_path": "/checkpoint.pth",
        "adapter_rank": 16,
        "class_ids": [0, 1],
        "head_weight": torch.ones(2, 2),
        "head_bias": torch.zeros(2),
        "head_scale": 8.0,
        "model_deltas": {
            "t3.norm.weight": torch.tensor([0.5, -0.25], dtype=torch.float16),
            "t3.norm.bias": torch.tensor([0.1, -0.2], dtype=torch.float16),
        },
        "updated_parameter_names": ["t3.norm.weight", "t3.norm.bias"],
        "support_count": 60,
        "da0_classifier_source_target_interpolation": 0.5,
        "da0_prototype_scale": 8.0,
    }


def test_conversion_removes_target_head_and_loader_applies_only_t3_norm(tmp_path) -> None:
    source = tmp_path / "full_delta.pt"
    target = tmp_path / "t3_only.pt"
    torch.save(_full_delta_payload(), source)

    receipt = convert_sf_tapft_delta_bundle_to_t3_only(
        source,
        target,
        candidate_id="D0_T3_D92",
        d92_method_lock="D92-E0-NORF32",
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)

    assert "head_weight" not in payload and "head_bias" not in payload
    assert payload["adapter_rank"] == 16
    assert tuple(payload["model_deltas"]) == T3_DELTA_PARAMETER_NAMES
    assert receipt["temporary_target_head_persisted"] is False
    assert receipt["query_rows_used"] == 0

    base = _ToyBackbone()
    original = copy.deepcopy(base.state_dict())
    loaded, head, audit = load_t3_only_delta_bundle_strict(
        target,
        device="cpu",
        expected_target_binding={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "capsule",
            "split_id": "split",
            "support_count": 60,
        },
        checkpoint_loader=lambda _path, *, device: _ToyBackbone().to(device),
        adapter_initializer=lambda model, *, rank: model,
    )

    assert head is None
    torch.testing.assert_close(
        loaded.t3.norm.weight,
        original["t3.norm.weight"] + torch.tensor([0.5, -0.25]),
    )
    torch.testing.assert_close(
        loaded.t3.norm.bias,
        original["t3.norm.bias"] + torch.tensor([0.1, -0.2]),
        atol=2.0e-4,
        rtol=0.0,
    )
    assert audit["d92_method_lock"] == "D92-E0-NORF32"
    assert audit["temporary_target_head_persisted"] is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_direct_writer_never_accepts_or_persists_a_target_head(tmp_path) -> None:
    target = tmp_path / "direct_t3_only.pt"
    receipt = write_t3_only_delta_bundle(
        target,
        model_deltas={
            "model.t3.norm.weight": torch.tensor([0.25, -0.50]),
            "model.t3.norm.bias": torch.tensor([0.10, -0.20]),
        },
        protocol_schema="p2_min_v1",
        phase2_data_status="VALIDATED_ONCE",
        capsule_id="capsule",
        split_id="split",
        base_checkpoint_path="/checkpoint.pth",
        candidate_id="R3_DUALDELTA_T3_D92_INLOOP",
        support_count=60,
        d92_method_lock="D92-E0-NORF32",
        adapter_rank=16,
    )

    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert set(payload["model_deltas"]) == set(T3_DELTA_PARAMETER_NAMES)
    assert "head_weight" not in payload and "head_bias" not in payload
    assert payload["adapter_rank"] == 16
    assert receipt["temporary_target_head_persisted"] is False


def test_real_checkpoint_id_backbone_t3_path_is_preserved_and_loaded(tmp_path) -> None:
    target = tmp_path / "real_path_t3_only.pt"
    write_t3_only_delta_bundle(
        target,
        model_deltas={
            "model.id_backbone.t3.norm.weight": torch.tensor([0.25, -0.50]),
            "model.id_backbone.t3.norm.bias": torch.tensor([0.10, -0.20]),
        },
        protocol_schema="p2_min_v1",
        phase2_data_status="VALIDATED_ONCE",
        capsule_id="capsule",
        split_id="split",
        base_checkpoint_path="/checkpoint.pth",
        candidate_id="D0_T3_D92",
        support_count=60,
        d92_method_lock="D92-E0-NORF32",
        adapter_rank=16,
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    assert tuple(payload["model_deltas"]) == (
        "id_backbone.t3.norm.weight",
        "id_backbone.t3.norm.bias",
    )

    model, _, audit = load_t3_only_delta_bundle_strict(
        target,
        device="cpu",
        expected_target_binding={"capsule_id": "capsule", "split_id": "split"},
        checkpoint_loader=lambda _path, *, device: _RealPathToyBackbone().to(device),
        adapter_initializer=lambda model, *, rank: model,
    )
    torch.testing.assert_close(
        model.id_backbone.t3.norm.weight,
        torch.tensor([1.25, 0.50]),
    )
    assert audit["updated_parameter_names"] == (
        "id_backbone.t3.norm.weight",
        "id_backbone.t3.norm.bias",
    )
