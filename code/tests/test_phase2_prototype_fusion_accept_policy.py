import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase2_prototypes import PrototypeFusionConfig, fuse_tx_domain_prototypes  # noqa: E402


def test_fusion_exports_local_component_accept_radii_without_global_ball_acceptance():
    package = {
        "feature_key": "z_id",
        "prototypes": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "prototype_counts": torch.tensor([50]),
        "tx_domain_prototypes": torch.tensor(
            [[[1.0, 0.0], [0.996, 0.087], [0.0, 1.0]]],
            dtype=torch.float32,
        ),
        "tx_domain_counts": torch.tensor([[20, 20, 4]]),
        "radii": {
            "p95": torch.tensor([math.radians(5.0)]),
            "p99": torch.tensor([math.radians(12.0)]),
            "r_3sigma": torch.tensor([math.radians(18.0)]),
        },
        "metadata": {},
    }

    fused = fuse_tx_domain_prototypes(
        package,
        PrototypeFusionConfig(
            max_components_per_tx=2,
            merge_angle_deg=8.0,
            radius_cap_deg=25.0,
            tail_abs_deg=30.0,
            accept_policy="local_component",
            accept_radius_key="p95",
            max_p95_increase_deg=2.0,
            keep_tail_sentinel=True,
            global_ball_accept=False,
        ),
    )

    assert fused["fusion_accept_policy"] == "local_component"
    assert fused["global_fused_radius_is_accept_region"] is False
    assert fused["fused_tx_accept_radii"].shape == fused["fused_tx_radii"].shape
    assert fused["fused_tx_evidence_radii"].shape == fused["fused_tx_radii"].shape
    assert fused["fusion_components"][0][0]["accept_radius_deg"] <= 7.1
    assert fused["fusion_components"][0][0]["evidence_radius_deg"] >= fused["fusion_components"][0][0]["accept_radius_deg"]
    assert "component_p95_delta_deg" in fused["fusion_components"][0][0]
