import pytest
import torch

from cvsrffi.ntrs_evaluation import (
    NTRSTelemetryAccumulator,
    restore_ntrs_eval_epoch,
)


class _EpochAwareModel:
    def __init__(self):
        self.epochs = []

    def set_ntrs_epoch(self, epoch: int):
        self.epochs.append(int(epoch))


def _output(role: str):
    if role == "clean":
        raw = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
        robust = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 0.0]])
    else:
        raw = torch.tensor([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
        robust = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]])
    fused = robust.clone()
    anchor = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    correction = torch.tensor([[-0.1, 0.0], [0.0, -0.1], [-0.1, 0.0], [0.0, -0.1]])
    return {
        "tx_logits": fused,
        "ntrs_raw_logits": raw,
        "ntrs_robust_logits": robust,
        "ntrs_z_anchor": anchor,
        "ntrs_correction": correction,
        "ntrs_gate": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "ntrs_safe_gate": torch.tensor([0.1, 0.2, 0.0, 0.4]),
        "ntrs_alpha": torch.tensor([0.05, 0.10, 0.15, 0.20]),
        "ntrs_correction_energy": torch.tensor([0.01, 0.02, 0.03, 0.04]),
        "ntrs_physical_correction_energy": torch.tensor([0.02, 0.03, 0.04, 0.05]),
        "ntrs_support_distance": torch.tensor([0.2, 0.3, 0.4, 0.5]),
        "ntrs_correctability": torch.tensor([0.8, 0.7, 0.6, 0.5]),
        "ntrs_uncertainty": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "ntrs_subspace_residual": torch.zeros(4),
        "ntrs_agreement": raw.argmax(dim=1) == robust.argmax(dim=1),
        "aux_id": {"ntrs_physical_view": False},
        "aux_phys": {
            "ntrs_physical_view": True,
            "ntrs_frequency_dual_view": True,
            "ntrs_pa_uses_original_iq": True,
        },
        "aux_dom": {"ntrs_physical_view": False},
    }


def test_ntrs_telemetry_reports_raw_robust_fused_transitions_and_safety():
    labels = torch.tensor([0, 1, 1, 1])
    prototypes = torch.eye(2)
    accumulator = NTRSTelemetryAccumulator(prototypes=prototypes, unknown_rescue=False)
    accumulator.update(_output("clean"), _output("satellite"), labels)
    summary = accumulator.summary()

    assert summary["satellite"]["raw_accuracy"] == pytest.approx(0.5)
    assert summary["satellite"]["robust_accuracy"] == pytest.approx(1.0)
    assert summary["satellite"]["fused_accuracy"] == pytest.approx(1.0)
    transitions = summary["satellite"]["transitions"]
    assert transitions["both_correct"] == 2
    assert transitions["rescued_correct"] == 2
    assert transitions["harmed_correct"] == 0
    assert transitions["both_wrong"] == 0
    assert summary["satellite"]["gate"]["count"] == 4
    assert summary["satellite"]["subspace_residual"]["mean"] == pytest.approx(0.0)
    assert summary["satellite"]["class_attraction_cosine"]["count"] == 4
    assert summary["safety"]["unknown_rescue_enabled"] is False
    assert summary["safety"]["unknown_transition_status"] == "N/A_NO_FROZEN_REJECTION_THRESHOLD"
    assert summary["paths"]["pa_original_iq_rate"] == pytest.approx(1.0)
    assert summary["paths"]["domain_raw_iq_rate"] == pytest.approx(1.0)


def test_ntrs_telemetry_reads_detached_outputs_without_mutation():
    clean = _output("clean")
    satellite = _output("satellite")
    before = {
        key: value.clone()
        for key, value in satellite.items()
        if torch.is_tensor(value)
    }
    accumulator = NTRSTelemetryAccumulator(prototypes=torch.eye(2), unknown_rescue=False)
    accumulator.update(clean, satellite, torch.tensor([0, 1, 1, 1]))
    for key, value in before.items():
        assert torch.equal(satellite[key], value)


def test_checkpoint_evaluation_restores_final_ntrs_epoch_after_rebuild():
    model = _EpochAwareModel()
    restored = restore_ntrs_eval_epoch(model, {"epoch": 200, "args": {"epochs": 200}})
    assert restored == 200
    assert model.epochs == [200]
