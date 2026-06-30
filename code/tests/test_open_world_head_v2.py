import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.open_world_head import OpenWorldMultiPrototypeHead, open_world_energy_from_scores  # noqa: E402


def test_energy_helper_matches_head_forward_energy():
    head = OpenWorldMultiPrototypeHead(feat_dim=2, energy_temperature=0.7)
    head.add_target_prototypes(
        [0, 1],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        radii=torch.tensor([0.2, 0.2]),
    )
    details = head(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    helper_energy = open_world_energy_from_scores(details["class_scores"], energy_temperature=0.7)

    assert torch.allclose(details["energy"], helper_energy)


def test_fused_phase2_package_uses_per_component_accept_radius_not_class_max_radius():
    package = {
        "fused_tx_prototypes": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        "fused_tx_mask": torch.tensor([[True, True]]),
        "fused_tx_accept_radii": torch.tensor([[math.radians(5.0), math.radians(40.0)]], dtype=torch.float32),
        "fused_tx_radii": torch.tensor([[math.radians(25.0), math.radians(45.0)]], dtype=torch.float32),
    }
    head = OpenWorldMultiPrototypeHead.from_phase2_export(package)
    details = head(torch.tensor([[0.985, 0.174]], dtype=torch.float32))

    assert int(details["candidate_labels"][0].item()) == 0
    assert math.degrees(float(details["best_radius"][0].item())) < 6.0
    assert float(details["radius_margin"][0].item()) < 0.0
