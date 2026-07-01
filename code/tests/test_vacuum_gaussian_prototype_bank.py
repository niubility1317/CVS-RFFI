import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402


def _bank_dict():
    return {
        "feature_key": "z_id",
        "fusion_components": [
            [
                {
                    "component_id": 0,
                    "source_domains": [0],
                    "n_samples": 10,
                    "mu": [1.0, 0.0],
                    "r_core_deg": 5.0,
                    "r_accept_deg": 10.0,
                    "r_tail_deg": 15.0,
                    "r_vac_deg": 20.0,
                    "density_p05": None,
                    "density_p10": None,
                    "nll_p95": None,
                    "nearest_other_deg": 60.0,
                    "accept_enabled": True,
                },
                {
                    "component_id": 1,
                    "source_domains": [1],
                    "n_samples": 10,
                    "mu": [0.8, 0.6],
                    "r_core_deg": 4.0,
                    "r_accept_deg": 8.0,
                    "r_tail_deg": 12.0,
                    "r_vac_deg": 16.0,
                    "density_p05": None,
                    "density_p10": None,
                    "nll_p95": None,
                    "nearest_other_deg": 40.0,
                    "accept_enabled": True,
                },
            ],
            [
                {
                    "component_id": 0,
                    "source_domains": [0],
                    "n_samples": 10,
                    "mu": [0.0, 1.0],
                    "r_core_deg": 5.0,
                    "r_accept_deg": 10.0,
                    "r_tail_deg": 15.0,
                    "r_vac_deg": 20.0,
                    "density_p05": None,
                    "density_p10": None,
                    "nll_p95": None,
                    "nearest_other_deg": 60.0,
                    "accept_enabled": True,
                }
            ],
        ],
    }


def test_vacuum_gaussian_bank_loads_fusion_components_and_finds_neighbors():
    bank = VacuumGaussianPrototypeBank.from_phase2_package(_bank_dict())

    own = bank.nearest_own_component(torch.tensor([0.99, 0.05]), class_id=0)
    other = bank.nearest_other_component(torch.tensor([0.99, 0.05]), exclude_class_id=0)

    assert own.class_id == 0
    assert own.component_id == 0
    assert other.class_id == 1
    assert bank.angular_distance_deg(torch.tensor([1.0, 0.0]), own.mu).item() < 0.1


def test_vacuum_gaussian_bank_loads_old_json_without_density_and_serializes():
    old_pkg = {
        "feature_key": "z_id",
        "prototypes": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "prototype_counts": torch.tensor([4, 3]),
        "radii": {"p95": torch.tensor([0.1, 0.2])},
    }

    bank = VacuumGaussianPrototypeBank.from_phase2_package(old_pkg)
    own = bank.nearest_own_component(torch.tensor([1.0, 0.0]), class_id=0)
    data = bank.to_json_dict()

    assert own.density_core_min is None
    assert data["classes"]["0"]["components"][0]["r_accept_deg"] > 0.0

