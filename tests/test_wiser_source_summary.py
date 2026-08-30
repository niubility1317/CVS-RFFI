from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import cvsrffi.wiser_source_summary as summary_module
from cvsrffi.wiser_source_summary import (
    classwise_sliced_wasserstein,
    load_quantized_source_summary,
)


def _write_summary(path: Path, *, forbidden_member: bool = False) -> Path:
    class_count, rank, feature_dim, domain_count = 2, 2, 4, 3
    members = {
        "schema": np.asarray(
            "int8_domain_class_center_lowrank_residual_radius_v2", dtype=np.str_
        ),
        "feature_schema": np.asarray("adv3b02_z_id160_fp32", dtype=np.str_),
        "residual_rank": np.asarray(rank, dtype=np.int16),
        "center_domain_handle": np.asarray("d0", dtype=np.str_),
        "domain_registry": np.asarray(["d0", "d1", "d2"], dtype=np.str_),
        "residual_domain_registry": np.asarray(["d1", "d2"], dtype=np.str_),
        "class_registry": np.asarray(["c0", "c1"], dtype=np.str_),
        "core_q": np.asarray([[127, 0, 0, 0], [0, 127, 0, 0]], dtype=np.int8),
        "core_scale": np.asarray([1 / 127, 1 / 127], dtype=np.float16),
        "residual_basis_q": np.asarray(
            [
                [[0, 0, 127, 0], [0, 0, 0, 127]],
                [[0, 0, 127, 0], [0, 0, 0, 127]],
            ],
            dtype=np.int8,
        ),
        "residual_basis_scale": np.full((class_count, rank), 1 / 127, np.float16),
        "residual_coeff_q": np.asarray(
            [
                [[10, 20], [15, 25]],
                [[-10, -20], [-15, -25]],
            ],
            dtype=np.int8,
        ),
        "residual_coeff_scale": np.full(
            (domain_count - 1, class_count), 0.01, np.float16
        ),
        "radius_q": np.asarray([[20, 30], [25, 35], [30, 40]], dtype=np.int8),
        "radius_scale": np.asarray([0.01, 0.01], dtype=np.float16),
    }
    if forbidden_member:
        members["source_sample_embeddings"] = np.zeros((1, feature_dim), np.float32)
    np.savez_compressed(path, **members)
    return path


def _write_dense_domain_summary(path: Path) -> Path:
    values = np.asarray(
        [
            [[127, 0, 0, 0], [0, 127, 0, 0]],
            [[120, 20, 0, 0], [20, 120, 0, 0]],
            [[120, -20, 0, 0], [-20, 120, 0, 0]],
        ],
        dtype=np.int8,
    )
    np.savez_compressed(
        path,
        domain_class_q=values,
        domain_class_scale=np.full((3, 2), 1 / 127, np.float16),
        domain_class_mask=np.ones((3, 2), np.uint8),
        domain_registry=np.asarray([10, 11, 12], np.int16),
        class_registry=np.asarray(["c0", "c1"], np.str_),
        feature_schema=np.asarray("adv3b02_z_id160_fp32", np.str_),
    )
    return path


def test_quantized_summary_builds_frozen_unit_virtual_points(tmp_path: Path) -> None:
    path = _write_summary(tmp_path / "summary.npz")

    summary = load_quantized_source_summary(path)
    points = summary.virtual_source_points()

    assert summary.class_registry == ("c0", "c1")
    assert points.shape == (2, 5, 4)
    assert points.requires_grad is False
    assert torch.allclose(torch.linalg.vector_norm(points, dim=-1), torch.ones(2, 5))
    assert torch.allclose(points[:, 0], summary.centers)
    assert not torch.equal(points[:, 1], points[:, 2])


def test_quantized_summary_rejects_sample_level_source_members(tmp_path: Path) -> None:
    path = _write_summary(tmp_path / "forbidden.npz", forbidden_member=True)

    with pytest.raises(ValueError, match="forbidden or unexpected"):
        load_quantized_source_summary(path)


def test_dense_domain_class_summary_uses_each_valid_domain_as_source_point(
    tmp_path: Path,
) -> None:
    summary = load_quantized_source_summary(
        _write_dense_domain_summary(tmp_path / "dense.npz")
    )

    points = summary.virtual_source_points()

    assert points.shape == (2, 3, 4)
    assert summary.class_registry == ("c0", "c1")
    assert torch.allclose(torch.linalg.vector_norm(points, dim=-1), torch.ones(2, 3))
    assert torch.allclose(summary.centers, F.normalize(points.mean(dim=1), dim=1))


def test_summary_loader_falls_back_when_n607_numpy_bridge_rejects_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        summary_module.torch,
        "from_numpy",
        lambda _value: (_ for _ in ()).throw(TypeError("N607 bridge")),
    )

    summary = load_quantized_source_summary(
        _write_dense_domain_summary(tmp_path / "dense.npz")
    )

    assert summary.virtual_source_points().shape == (2, 3, 4)


def test_classwise_vsw_is_differentiable_only_to_target_features(tmp_path: Path) -> None:
    summary = load_quantized_source_summary(_write_summary(tmp_path / "summary.npz"))
    target = torch.tensor(
        [
            [1.0, 0.0, 0.10, 0.00],
            [1.0, 0.0, -0.10, 0.00],
            [0.0, 1.0, 0.00, 0.10],
            [0.0, 1.0, 0.00, -0.10],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    first = classwise_sliced_wasserstein(
        target,
        labels,
        summary.virtual_source_points(),
        num_projections=8,
        seed=17,
    )
    second = classwise_sliced_wasserstein(
        target,
        labels,
        summary.virtual_source_points(),
        num_projections=8,
        seed=17,
    )
    first.backward()

    assert torch.equal(first.detach(), second.detach())
    assert torch.isfinite(first)
    assert target.grad is not None
    assert torch.isfinite(target.grad).all()
    assert summary.virtual_source_points().grad is None
