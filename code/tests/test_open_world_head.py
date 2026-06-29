import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from cvsrffi.open_world_head import OpenWorldMultiPrototypeHead, UNKNOWN_LABEL  # noqa: E402


def test_open_world_head_accepts_equal_threshold_as_inlier():
    head = OpenWorldMultiPrototypeHead(feat_dim=2)
    head.add_old_classes(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0.25, 0.25]))
    details = head(torch.tensor([[1.0, 0.0]]))
    thresholds = {
        "min_cosine": float(details["best_proto_score"][0].item()),
        "min_radius_margin": float(details["radius_margin"][0].item()),
        "max_energy": float(details["energy"][0].item()),
    }

    decision = head.decide(details, thresholds)

    assert decision.accepted[0].item() is True
    assert int(decision.predicted_labels[0].item()) == 0
    assert decision.gate_reasons[0] == "multi_proto_accept"


def test_open_world_head_rejects_by_radius_and_uses_unknown_label():
    head = OpenWorldMultiPrototypeHead(feat_dim=2)
    head.add_old_classes(torch.tensor([[1.0, 0.0]]), torch.tensor([0.01]))

    decision = head.decide(torch.tensor([[0.0, 1.0]]), {"min_cosine": -1.0, "min_radius_margin": 0.0})

    assert decision.accepted[0].item() is False
    assert int(decision.predicted_labels[0].item()) == UNKNOWN_LABEL
    assert decision.gate_reasons[0] == "outside_class_radius"


def test_open_world_head_registers_seen_new_class_with_shrinkage_radius():
    head = OpenWorldMultiPrototypeHead(feat_dim=2)
    head.add_old_classes(torch.tensor([[1.0, 0.0]]), torch.tensor([0.30]))

    stats = head.register_new_class(10, torch.tensor([[0.0, 1.0], [0.1, 0.9]]), radius_prior=0.40)
    decision = head.decide(torch.tensor([[0.0, 1.0]]), {"min_cosine": 0.7, "min_radius_margin": -0.05})

    assert stats["support_count"] == 2.0
    assert int(decision.predicted_labels[0].item()) == 10
    assert decision.accepted[0].item() is True


def test_open_world_head_loads_old_classes_from_phase2_export_package():
    package = {
        "prototypes": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "radii": {"robust_max": torch.tensor([0.20, 0.30], dtype=torch.float32)},
    }

    head = OpenWorldMultiPrototypeHead.from_phase2_export(package, radius_key="robust_max")
    decision = head.decide(torch.tensor([[0.0, 1.0]]), {"min_cosine": 0.7, "min_radius_margin": -0.05})

    assert head.class_ids == [0, 1]
    assert int(decision.predicted_labels[0].item()) == 1


def test_open_world_head_rejects_overlapping_new_class_without_registering():
    head = OpenWorldMultiPrototypeHead(feat_dim=2)
    head.add_old_classes(torch.tensor([[1.0, 0.0]]), torch.tensor([0.30]))

    stats = head.register_new_class(
        10,
        torch.tensor([[1.0, 0.0], [0.99, 0.01]], dtype=torch.float32),
        radius_prior=0.30,
        overlap_margin=0.05,
    )

    assert stats["status"] == "rejected_overlap"
    assert 10 not in head.class_ids


def test_open_world_head_rejects_bad_shapes():
    head = OpenWorldMultiPrototypeHead(feat_dim=3)
    with pytest.raises(ValueError):
        head.add_old_classes(torch.randn(2, 2), torch.ones(2))
