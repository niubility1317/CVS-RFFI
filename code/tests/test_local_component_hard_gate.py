import copy
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.hard_gate import GateThresholds, LocalComponentHardGate  # noqa: E402
from cvsrffi.phase2_prototypes import attach_endpoint_accept_v1_manifest  # noqa: E402
from cvsrffi.prototype_bank import VacuumGaussianPrototypeBank  # noqa: E402


def _artifact(*, density_core=0.0, density_tail=0.0, nll_core=10.0, nll_tail=10.0):
    rows = []
    for class_id, mu in enumerate(([1.0, 0.0], [0.0, 1.0])):
        rows.append(
            [
                {
                    "component_id": 0,
                    "source_domains": [0],
                    "n_samples": 20,
                    "mu": mu,
                    "r_core_deg": 6.0,
                    "r_accept_deg": 12.0,
                    "r_tail_deg": 18.0,
                    "r_vac_deg": 24.0,
                    "density_p05": density_core,
                    "density_p10": density_tail,
                    "nll_p95": nll_core,
                    "nll_tail_p95": nll_tail,
                    "nearest_other_deg": 90.0,
                    "accept_enabled": True,
                    "source_val_count": 20,
                }
            ]
        )
    package = {
        "feature_key": "z_id",
        "fused_tx_prototypes": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        "fused_tx_mask": torch.ones(2, 1, dtype=torch.bool),
        "fusion_components": rows,
        "fusion_accept_policy": "local_component",
        "global_fused_radius_is_accept_region": False,
        "metadata": {
            "source_checkpoint_sha256": "0" * 64,
            "run_id": "unit",
            "candidate_id": "gate",
            "known_class_count": 2,
            "class_id_to_tx": ["tx0", "tx1"],
            "logit_class_order": [0, 1],
            "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
            "checkpoint_load_strict": True,
            "endpoint_runtime_entry_parity_digest": "1" * 64,
            "endpoint_runtime_entry_parity_sample_count": 8,
        },
        "endpoint_gate_thresholds": {
            "energy_max_by_class": {"0": 0.0, "1": 0.0},
            "energy_temperature": 1.0,
            "energy_formula_id": "negative_logsumexp_temperature_v1",
            "density_formula_id": "exp_neg_sq_normalized_angle_v1",
            "nll_formula_id": "half_sq_normalized_angle_v1",
            "logit_margin_core_min": 0.5,
            "logit_margin_tail_min": 2.0,
            "geo_margin_core_min_deg": 2.0,
            "geo_margin_tail_min_deg": 4.0,
            "allow_tail_auto_accept": False,
            "use_density_gate": True,
            "use_nll_gate": True,
            "use_energy_gate": True,
            "use_geo_margin_gate": True,
            "reject_nan": True,
            "max_radius_to_inter_ratio": 0.50,
        },
        "endpoint_calibration": {
            "schema": "endpoint_accept_v1_source_val_calibration_v1",
            "threshold_source": "source_val_only",
            "calibration_split": "source_val",
            "num_samples": 40,
            "correct_samples": 40,
            "class_sample_counts": {"0": 20, "1": 20},
            "component_sample_counts": {"0:0": 20, "1:0": 20},
            "enabled_components_by_class": {"0": 1, "1": 1},
        },
    }
    return attach_endpoint_accept_v1_manifest(package)


def _gate(**kwargs):
    package = _artifact(**kwargs)
    return LocalComponentHardGate.from_runtime_inference(
        package, runtime_identity=package["endpoint_accept_v1"]["inference_identity"]
    )


def _runtime_gate(package):
    return LocalComponentHardGate.from_runtime_inference(
        package, runtime_identity=package["endpoint_accept_v1"]["inference_identity"]
    )


def test_hard_gate_accepts_only_core_and_reviews_tail():
    gate = _gate()

    core = gate.decide(torch.tensor([1.0, 0.0]), logits=torch.tensor([4.0, 1.0]))
    tail = gate.decide(torch.tensor([0.985, 0.174]), logits=torch.tensor([4.0, 1.0]))

    assert core["decision"] == "ACCEPT_KNOWN_CORE"
    assert core["debug"]["gates"]["density"] is True
    assert core["debug"]["endpoint_boundary_hash"]
    assert tail["decision"] == "REVIEW_KNOWN_TAIL"


def test_hard_gate_rejects_interclass_midpoint_and_nan():
    gate = _gate()

    midpoint = gate.decide(torch.tensor([1.0, 1.0]), logits=torch.tensor([4.0, 3.9]))
    bad = gate.decide(torch.tensor([float("nan"), 0.0]), logits=torch.tensor([4.0, 1.0]))

    assert midpoint["decision"].startswith("REJECT")
    assert bad["decision"] == "REJECT_NAN"


def test_hard_gate_rejects_wrong_feature_or_logit_shape():
    gate = _gate()

    bad_feature = gate.decide(torch.tensor([[1.0, 0.0]]), logits=torch.tensor([4.0, 1.0]))
    bad_logits = gate.decide(torch.tensor([1.0, 0.0]), logits=torch.tensor([[4.0, 1.0]]))
    missing_logits = gate.decide(torch.tensor([1.0, 0.0]), logits=None)

    assert bad_feature["decision"] == "REJECT_INVALID_FEATURE"
    assert bad_logits["decision"] == "REJECT_INVALID_LOGITS"
    assert missing_logits["decision"] == "REJECT_INVALID_LOGITS"


def test_hard_gate_accepts_core_with_exported_density_and_nll_thresholds():
    gate = _gate(density_core=0.60, density_tail=0.50, nll_core=0.90, nll_tail=1.20)

    core = gate.decide(torch.tensor([1.0, 0.0]), logits=torch.tensor([4.0, 1.0]))

    assert core["decision"] == "ACCEPT_KNOWN_CORE"
    assert core["debug"]["gates"]["density"] is True
    assert core["debug"]["gates"]["nll"] is True


def test_final_gate_rejects_unverified_bank_and_threshold_override():
    bank = VacuumGaussianPrototypeBank.from_phase2_package(_artifact())
    with pytest.raises(ValueError, match="verified endpoint"):
        LocalComponentHardGate(bank, GateThresholds())
    with pytest.raises(ValueError, match="thresholds differ"):
        LocalComponentHardGate.from_phase1_package(
            _artifact(), GateThresholds(logit_margin_core_min=9.0), entry_point="train_export"
        )

    with pytest.raises(ValueError, match="requires actual runtime identity"):
        LocalComponentHardGate.from_runtime_inference(_artifact(), runtime_identity=None)
    package = _artifact()
    wrong_identity = dict(package["endpoint_accept_v1"]["inference_identity"])
    wrong_identity["source_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="runtime identity mismatch"):
        LocalComponentHardGate.from_runtime_inference(package, runtime_identity=wrong_identity)


def test_three_endpoint_entry_points_share_one_boundary_and_decision():
    package = _artifact()
    gates = (
        LocalComponentHardGate.from_train_export(package),
        LocalComponentHardGate.from_offline_eval(
            package, runtime_identity=package["endpoint_accept_v1"]["inference_identity"]
        ),
        _runtime_gate(package),
    )
    decisions = [gate.decide(torch.tensor([1.0, 0.0]), logits=torch.tensor([4.0, 1.0])) for gate in gates]

    assert {row["decision"] for row in decisions} == {"ACCEPT_KNOWN_CORE"}
    assert len({row["debug"]["endpoint_boundary_hash"] for row in decisions}) == 1


def test_component_mu_tampering_breaks_boundary_hash():
    package = _artifact()
    tampered = copy.deepcopy(package)
    tampered["fusion_components"][0][0]["mu"] = [0.0, 1.0]

    with pytest.raises(ValueError, match="hash mismatch|radius-to-inter ratio unsafe"):
        _runtime_gate(tampered)


def test_runtime_rejects_energy_override_and_invalid_logit_shape():
    gate = _gate()

    mismatch = gate.decide(
        torch.tensor([1.0, 0.0]),
        logits=torch.tensor([4.0, 1.0]),
        energy=-999.0,
    )
    one_logit = gate.decide(torch.tensor([1.0, 0.0]), logits=torch.tensor([4.0]))

    assert mismatch["decision"] == "REJECT_ENERGY_MISMATCH"
    assert one_logit["decision"] == "REJECT_INVALID_LOGITS"


def test_endpoint_v11_rejects_disabled_gate_and_schema_tampering():
    package = _artifact()
    disabled = copy.deepcopy(package)
    disabled["endpoint_gate_thresholds"]["use_energy_gate"] = False
    schema = copy.deepcopy(package)
    schema["endpoint_accept_v1"]["schema_version"] = 999

    with pytest.raises(ValueError, match="requires use_energy_gate=true"):
        _runtime_gate(disabled)
    with pytest.raises(ValueError, match="schema version mismatch"):
        _runtime_gate(schema)


def test_endpoint_v11_rejects_inference_identity_and_unsafe_geometry_tampering():
    package = _artifact()
    identity = copy.deepcopy(package)
    identity["metadata"]["logit_class_order"] = [1, 0]
    nan_center = copy.deepcopy(package)
    nan_center["fusion_components"][0][0]["mu"] = [float("nan"), 0.0]
    oversized = copy.deepcopy(package)
    oversized["fusion_components"][0][0]["r_accept_deg"] = 60.0
    oversized["fusion_components"][0][0]["r_tail_deg"] = 65.0
    oversized["fusion_components"][0][0]["r_vac_deg"] = 70.0

    with pytest.raises(ValueError, match="logit class order"):
        _runtime_gate(identity)
    with pytest.raises(ValueError, match="component center invalid"):
        _runtime_gate(nan_center)
    with pytest.raises(ValueError, match="radius-to-inter ratio unsafe"):
        _runtime_gate(oversized)
