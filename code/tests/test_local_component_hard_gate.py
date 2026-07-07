import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.hard_gate import GateThresholds, LocalComponentHardGate  # noqa: E402
from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402


def _gate():
    bank = VacuumGaussianPrototypeBank.from_phase2_package(
        {
            "feature_key": "z_id",
            "fusion_components": [
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [1.0, 0.0],
                        "r_core_deg": 6.0,
                        "r_accept_deg": 12.0,
                        "r_tail_deg": 18.0,
                        "r_vac_deg": 24.0,
                        "density_p05": None,
                        "density_p10": None,
                        "nll_p95": None,
                        "accept_enabled": True,
                    }
                ],
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [0.0, 1.0],
                        "r_core_deg": 6.0,
                        "r_accept_deg": 12.0,
                        "r_tail_deg": 18.0,
                        "r_vac_deg": 24.0,
                        "density_p05": None,
                        "density_p10": None,
                        "nll_p95": None,
                        "accept_enabled": True,
                    }
                ],
            ],
        }
    )
    return LocalComponentHardGate(
        bank,
        GateThresholds(
            logit_margin_core_min=0.5,
            logit_margin_tail_min=2.0,
            geo_margin_core_min_deg=2.0,
            geo_margin_tail_min_deg=4.0,
            use_density_gate=True,
            use_nll_gate=True,
            use_energy_gate=False,
        ),
    )


def test_hard_gate_accepts_only_core_and_reviews_tail():
    gate = _gate()

    core = gate.decide(torch.tensor([1.0, 0.0]), logits=torch.tensor([4.0, 1.0]))
    tail = gate.decide(torch.tensor([0.985, 0.174]), logits=torch.tensor([4.0, 1.0]))

    assert core["decision"] == "ACCEPT_KNOWN_CORE"
    assert core["debug"]["gates"]["density"] == "skipped"
    assert tail["decision"] == "REVIEW_KNOWN_TAIL"


def test_hard_gate_rejects_interclass_midpoint_and_nan():
    gate = _gate()

    midpoint = gate.decide(torch.tensor([1.0, 1.0]), logits=torch.tensor([4.0, 3.9]))
    bad = gate.decide(torch.tensor([float("nan"), 0.0]), logits=torch.tensor([4.0, 1.0]))

    assert midpoint["decision"].startswith("REJECT")
    assert bad["decision"] == "REJECT_NAN"


def test_hard_gate_accepts_core_with_exported_density_and_nll_thresholds():
    bank = VacuumGaussianPrototypeBank.from_phase2_package(
        {
            "feature_key": "z_id",
            "fusion_components": [
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [1.0, 0.0],
                        "r_core_deg": 6.0,
                        "r_accept_deg": 12.0,
                        "r_tail_deg": 18.0,
                        "r_vac_deg": 24.0,
                        "density_p05": 0.60,
                        "density_p10": 0.50,
                        "nll_p95": 0.90,
                        "nll_tail_p95": 1.20,
                        "accept_enabled": True,
                    }
                ],
                [
                    {
                        "component_id": 0,
                        "source_domains": [0],
                        "n_samples": 20,
                        "mu": [0.0, 1.0],
                        "r_core_deg": 6.0,
                        "r_accept_deg": 12.0,
                        "r_tail_deg": 18.0,
                        "r_vac_deg": 24.0,
                        "density_p05": 0.60,
                        "density_p10": 0.50,
                        "nll_p95": 0.90,
                        "nll_tail_p95": 1.20,
                        "accept_enabled": True,
                    }
                ],
            ],
        }
    )
    gate = LocalComponentHardGate(
        bank,
        GateThresholds(
            logit_margin_core_min=0.5,
            logit_margin_tail_min=2.0,
            geo_margin_core_min_deg=2.0,
            geo_margin_tail_min_deg=4.0,
            use_density_gate=True,
            use_nll_gate=True,
            use_energy_gate=False,
        ),
    )

    core = gate.decide(torch.tensor([1.0, 0.0]), logits=torch.tensor([4.0, 1.0]))

    assert core["decision"] == "ACCEPT_KNOWN_CORE"
    assert core["debug"]["gates"]["density"] is True
    assert core["debug"]["gates"]["nll"] is True
