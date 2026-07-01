import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402
from cvsrffi.source_episode_safe_gate import source_episode_safe_partition  # noqa: E402


def test_source_episode_safe_partition_does_not_force_outside_query_known():
    bank = VacuumGaussianPrototypeBank.from_phase2_package(
        {
            "fusion_components": [
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [1.0, 0.0],
                        "r_core_deg": 5.0,
                        "r_accept_deg": 10.0,
                        "r_tail_deg": 15.0,
                        "accept_enabled": True,
                    }
                ]
            ]
        }
    )
    z = torch.tensor([[1.0, 0.0], [0.985, 0.174], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 0])

    result = source_episode_safe_partition(z, labels, bank)

    assert result.known_query_mask.tolist() == [True, False, False]
    assert result.uncertain_query_mask.tolist() == [False, True, False]
    assert result.overflow_query_mask.tolist() == [False, False, True]

