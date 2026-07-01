import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402
from cvsrffi.unlabeled_risk_mining import UnlabeledRiskConfig, mine_unlabeled_risk  # noqa: E402


def test_unlabeled_risk_mining_splits_core_risk_and_ignore():
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
                ],
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [0.0, 1.0],
                        "r_core_deg": 5.0,
                        "r_accept_deg": 10.0,
                        "r_tail_deg": 15.0,
                        "accept_enabled": True,
                    }
                ],
            ]
        }
    )
    z = torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]], dtype=torch.float32)
    logits = torch.tensor([[5.0, 0.1], [4.0, 0.1], [0.1, 0.2]], dtype=torch.float32)

    result = mine_unlabeled_risk(z, logits, bank, UnlabeledRiskConfig(risk_maxprob_min=0.70))

    assert result.pseudo_known_mask.tolist() == [True, False, False]
    assert result.risk_mask.tolist() == [False, True, False]
    assert result.ignore_mask.tolist() == [False, False, True]

