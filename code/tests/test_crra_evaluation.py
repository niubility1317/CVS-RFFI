import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.crra_evaluation import (  # noqa: E402
    CRRATelemetryAccumulator,
    restore_crra_eval_epoch,
)


class _EpochAwareModel:
    def __init__(self):
        self.epochs = []

    def set_crra_epoch(self, epoch: int):
        self.epochs.append(int(epoch))


def _output(offset: float = 0.0):
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]) + offset
    return {
        "z_id": z,
        "crra_correction_energy": torch.tensor([0.1, 0.2, 0.3]) + offset,
        "crra_alpha": torch.tensor([[0.05, 0.10], [0.15, 0.20], [0.20, 0.25]]),
        "crra_gate": torch.tensor([0.2, 0.3, 0.4]),
        "crra_support_distance": torch.tensor([0.4, 0.5, 0.6]),
        "crra_branch_reliability": torch.tensor(
            [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.2, 0.5]]
        ),
        "crra_condition_tx_adv_logits": torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]
        ),
    }


def test_crra_telemetry_reports_satellite_diagnostics_and_paired_geometry():
    accumulator = CRRATelemetryAccumulator()
    labels = torch.tensor([0, 1, 1])
    accumulator.update(_output(), _output(offset=0.1), labels)
    summary = accumulator.summary()

    assert summary["clean"]["correction_energy"]["count"] == 3
    assert summary["satellite"]["alpha"]["count"] == 6
    assert summary["satellite"]["gate"]["p95"] == pytest.approx(0.39, abs=0.02)
    assert summary["satellite"]["q_tx_leakage_accuracy"]["accuracy"] == pytest.approx(2.0 / 3.0)
    assert summary["satellite"]["reliability_pa"]["mean"] == pytest.approx(0.3)
    assert summary["paired"]["view_cosine_distance"]["count"] == 3
    assert summary["paired"]["cross_domain_class_radius"]["count"] >= 2


def test_crra_telemetry_does_not_require_or_mutate_model_state():
    accumulator = CRRATelemetryAccumulator()
    clean = _output()
    satellite = _output(offset=0.1)
    clean_before = {key: value.clone() for key, value in clean.items() if torch.is_tensor(value)}
    accumulator.update(clean, satellite, torch.tensor([0, 1, 1]))
    for key, value in clean_before.items():
        assert torch.equal(clean[key], value)


def test_checkpoint_evaluation_restores_final_crra_epoch_after_model_rebuild():
    model = _EpochAwareModel()

    restored = restore_crra_eval_epoch(model, {"epoch": 200, "args": {"epochs": 200}})

    assert restored == 200
    assert model.epochs == [200]
