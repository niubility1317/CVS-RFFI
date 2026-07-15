from __future__ import annotations

import numpy as np
import pytest
import torch

from paper_reproduction.scripts.benchmark_cvs_adaptive_rxlight_tta import (
    apply_fp16_checkpoint_delta,
    build_view_prototypes,
    leave_one_out_support_scores,
    score_views,
)


def test_matching_view_prototypes_preserve_shape_and_class_order() -> None:
    rng = np.random.default_rng(7)
    support = rng.normal(size=(5, 6, 4)).astype(np.float32)
    labels = np.asarray(["b", "a", "b", "a", "b", "a"])
    classes = ["a", "b"]
    prototypes = build_view_prototypes(support, labels, classes)
    scores = score_views(support[:, :2], prototypes)
    assert prototypes.shape == (5, 2, 4)
    assert prototypes.dtype == np.float16
    assert scores.shape == (2, 5, 2)
    assert np.isfinite(scores).all()


def test_leave_one_out_scores_do_not_use_the_sample_in_its_class_mean() -> None:
    support = np.zeros((5, 4, 2), dtype=np.float32)
    support[:, 0] = [1.0, 0.0]
    support[:, 1] = [0.0, 1.0]
    support[:, 2] = [-1.0, 0.0]
    support[:, 3] = [0.0, -1.0]
    labels = np.asarray(["a", "a", "b", "b"])
    scores = leave_one_out_support_scores(support, labels, ["a", "b"])
    # Row 0's class-a LOO prototype is row 1, hence orthogonal rather than self-aligned.
    assert scores.shape == (4, 5, 2)
    assert scores[0, 0, 0] == pytest.approx(0.0, abs=1e-6)


class _TinyLateKey(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.id_backbone = torch.nn.Module()
        self.id_backbone.t_proj = torch.nn.Linear(1, 1)
        self.id_backbone.f_proj = torch.nn.Linear(1, 1)
        self.id_backbone.pa_proj = torch.nn.Sequential(torch.nn.Linear(1, 1))


def test_fp16_delta_rejects_wrong_element_budget() -> None:
    model = _TinyLateKey()
    state = {
        name: torch.zeros_like(parameter, dtype=torch.float16)
        for name, parameter in model.named_parameters()
    }
    with pytest.raises(ValueError, match="element budget drift"):
        apply_fp16_checkpoint_delta(model, state)
