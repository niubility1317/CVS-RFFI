import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.negative_sampling import (  # noqa: E402
    sample_interclass_slerp_negatives,
    sample_shell_negatives,
    sample_tail_outward_negatives,
)
from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402


def _bank():
    return VacuumGaussianPrototypeBank.from_phase2_package(
        {
            "fusion_components": [
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [1.0, 0.0],
                        "r_core_deg": 4.0,
                        "r_accept_deg": 8.0,
                        "r_tail_deg": 12.0,
                        "r_vac_deg": 18.0,
                        "accept_enabled": True,
                    }
                ],
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [0.0, 1.0],
                        "r_core_deg": 4.0,
                        "r_accept_deg": 8.0,
                        "r_tail_deg": 12.0,
                        "r_vac_deg": 18.0,
                        "accept_enabled": True,
                    }
                ],
            ]
        }
    )


def test_shell_negatives_are_normalized_and_outside_accept_radius():
    bank = _bank()
    batch = sample_shell_negatives(bank, n_per_component=4, gamma_deg=1.0, seed=7)

    assert batch.z.shape[0] == 8
    assert torch.isfinite(batch.z).all()
    assert torch.allclose(batch.z.norm(dim=1), torch.ones(batch.z.size(0)), atol=1e-5)
    for z, cls, comp_id in zip(batch.z, batch.source_class, batch.source_component):
        comp = bank.get_component(int(cls), int(comp_id))
        assert bank.angular_distance_deg(z, comp.mu).item() > comp.r_accept_deg


def test_tail_outward_moves_farther_from_component_center():
    bank = _bank()
    comp = bank.get_component(0, 0)
    tail = torch.tensor([[0.985, 0.174]], dtype=torch.float32)
    batch = sample_tail_outward_negatives(tail, comp.mu, alpha_range=(1.5, 1.5))

    assert batch.z.shape[0] == 1
    assert bank.angular_distance_deg(batch.z[0], comp.mu) > bank.angular_distance_deg(tail[0], comp.mu)


def test_interclass_slerp_negatives_are_finite():
    batch = sample_interclass_slerp_negatives(_bank(), n_per_pair=3, seed=3)

    assert batch.z.shape[0] >= 3
    assert torch.isfinite(batch.z).all()
    assert set(batch.kind) == {"inter_class"}

