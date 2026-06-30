import math
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.opgac_net import (  # noqa: E402
    DECISION_AMBIGUOUS,
    DECISION_NEW,
    DECISION_OLD,
    DECISION_UNKNOWN,
    FixedFeatureTransform,
    GaussianClassState,
    OPGACConfig,
    OPGACMemory,
    OPGACNet,
    RFConditionBranch,
    build_old_memory_from_prototypes,
    drift_alarm,
    fit_fixed_feature_transform,
    predict_with_opgac_head,
    register_new_classes_opgac,
    register_old_classes_opgac,
    rollback_memory,
)
from cvsrffi.spaceborne_fewshot import PrototypeSet  # noqa: E402


def _state(label: int, group: str, vector: list[float], *, threshold: float = 3.0) -> GaussianClassState:
    mean = torch.tensor([vector], dtype=torch.float32)
    dim = mean.size(1)
    return GaussianClassState(
        class_id=label,
        group=group,
        means=mean,
        diag_vars=torch.full((1, dim), 0.04, dtype=torch.float32),
        weights=torch.ones(1, dtype=torch.float32),
        component_thresholds=torch.tensor([threshold], dtype=torch.float32),
        class_threshold=8.0,
        support_count=10,
    )


def _memory() -> OPGACMemory:
    return OPGACMemory(
        old_states={
            0: _state(0, "old", [1.0, 0.0, 0.0, 0.0]),
            1: _state(1, "old", [0.0, 1.0, 0.0, 0.0]),
        }
    )


def test_build_old_memory_from_prototypes_uses_metadata_gaussians():
    proto = PrototypeSet(
        labels=torch.tensor([3, 4], dtype=torch.long),
        vectors=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        counts=torch.tensor([20, 30], dtype=torch.long),
        metadata={
            "diag_var": torch.tensor([[0.02, 0.03], [0.04, 0.05]], dtype=torch.float32),
            "mahalanobis_thresholds": torch.tensor([2.5, 3.5], dtype=torch.float32),
        },
    )
    memory = build_old_memory_from_prototypes(proto, config=OPGACConfig(feature_dim=2, context_dim=8, hidden_dim=16))

    assert set(memory.old_states) == {3, 4}
    assert memory.old_states[3].group == "old"
    assert torch.allclose(memory.old_states[3].diag_vars[0], torch.tensor([0.02, 0.03]))
    assert math.isclose(float(memory.old_states[4].component_thresholds[0]), 3.5, rel_tol=1e-6)
    assert set(memory.ground_old_states) == {3, 4}


def test_fixed_feature_transform_fits_ground_pca_whitening_without_trainable_params():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.1],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.1],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float32,
    )
    transform = fit_fixed_feature_transform(features, output_dim=2, whitening=True)
    assert isinstance(transform, FixedFeatureTransform)
    assert transform.input_dim == 3
    assert transform.output_dim == 2
    assert list(transform.parameters()) == []
    out = transform(features)
    assert out.shape == (4, 2)
    assert torch.allclose(out.norm(dim=1), torch.ones(4), atol=1e-5)


def test_stage2_b_rejects_target_new_support_before_calibration():
    config = OPGACConfig(feature_dim=4, context_dim=8, hidden_dim=16, low_rank=2)
    model = OPGACNet(config)
    support_old = torch.tensor([[0.98, 0.02, 0.0, 0.0]], dtype=torch.float32)
    support_new = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="Stage2-B cannot use target-new support"):
        model.initialize_memory(
            _memory(),
            stage="Stage2-B",
            target_old_support=support_old,
            target_old_labels=torch.tensor([0]),
            target_new_support=support_new,
            target_new_labels=torch.tensor([10]),
        )


def test_opgac_stage2c_registers_new_class_without_query_and_predicts_old_new_unknown():
    torch.manual_seed(0)
    config = OPGACConfig(
        feature_dim=4,
        context_dim=8,
        hidden_dim=16,
        low_rank=2,
        top2_margin=-1.0,
        old_new_margin=0.05,
        default_class_threshold=8.0,
    )
    model = OPGACNet(config)
    ground = _memory()
    old_support = torch.tensor(
        [
            [0.99, 0.01, 0.0, 0.0],
            [0.02, 0.98, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.01, 0.99, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    old_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    new_support = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.01],
            [0.0, 0.0, 0.98, 0.02],
            [0.0, 0.01, 0.99, 0.0],
        ],
        dtype=torch.float32,
    )
    new_labels = torch.tensor([10, 10, 10], dtype=torch.long)

    memory = model.initialize_memory(
        ground,
        stage="Stage2-C",
        target_old_support=old_support,
        target_old_labels=old_labels,
        target_new_support=new_support,
        target_new_labels=new_labels,
    )
    assert set(memory.new_states) == {10}
    assert memory.new_states[10].group == "seen_new"
    assert memory.new_states[10].support_count == 3
    assert ground.new_states == {}

    query = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    prediction = model.predict(query, memory)

    assert prediction.decisions[0] == DECISION_OLD
    assert int(prediction.predicted_labels[0]) == 0
    assert prediction.decisions[1] == DECISION_NEW
    assert int(prediction.predicted_labels[1]) == 10
    assert prediction.decisions[2] in {DECISION_UNKNOWN, DECISION_AMBIGUOUS}
    assert not bool(prediction.accepted[2])


def test_new_class_overlap_stays_provisional_and_does_not_steal_old_samples():
    config = OPGACConfig(feature_dim=4, context_dim=8, hidden_dim=16, low_rank=2, top2_margin=-1.0)
    model = OPGACNet(config)
    new_near_old = torch.tensor([[0.99, 0.01, 0.0, 0.0], [0.98, 0.02, 0.0, 0.0]], dtype=torch.float32)

    memory = model.initialize_memory(
        _memory(),
        stage="Stage2-C",
        target_new_support=new_near_old,
        target_new_labels=torch.tensor([20, 20], dtype=torch.long),
    )

    assert memory.new_states[20].lifecycle == "provisional"
    prediction = model.predict(torch.tensor([[0.99, 0.01, 0.0, 0.0]], dtype=torch.float32), memory)
    assert prediction.decisions[0] == DECISION_OLD
    assert int(prediction.predicted_labels[0]) == 0
    assert "NEW_PROVISIONAL" in prediction.reject_reasons[0]


def test_old_class_calibration_updates_only_targeted_multi_proto_component():
    config = OPGACConfig(feature_dim=4, context_dim=8, hidden_dim=16, low_rank=2, top2_margin=-1.0)
    model = OPGACNet(config)
    state0 = GaussianClassState(
        class_id=0,
        group="old",
        means=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.65, 0.76, 0.0, 0.0]], dtype=torch.float32),
        diag_vars=torch.full((2, 4), 0.04, dtype=torch.float32),
        weights=torch.tensor([0.5, 0.5], dtype=torch.float32),
        component_thresholds=torch.tensor([3.0, 3.0], dtype=torch.float32),
        class_threshold=8.0,
        support_count=20,
        metadata={"component_count": 2},
    )
    memory = OPGACMemory(old_states={0: state0, 1: _state(1, "old", [0.0, 1.0, 0.0, 0.0])})
    support = torch.tensor([[0.70, 0.72, 0.0, 0.0], [0.68, 0.74, 0.0, 0.0]], dtype=torch.float32)

    calibrated = model.initialize_memory(
        memory,
        stage="Stage2-B",
        target_old_support=support,
        target_old_labels=torch.tensor([0, 0], dtype=torch.long),
    )

    updates = calibrated.old_states[0].metadata["component_updates"]
    assert [row["component"] for row in updates] == [1]
    assert torch.allclose(calibrated.old_states[0].means[0], memory.old_states[0].means[0], atol=1e-5)
    assert not torch.allclose(calibrated.old_states[0].means[1], memory.old_states[0].means[1])


def test_new_class_labels_must_not_overlap_old_memory():
    config = OPGACConfig(feature_dim=4, context_dim=8, hidden_dim=16, low_rank=2)
    model = OPGACNet(config)

    with pytest.raises(ValueError, match="overlap old-class memory|overlap old"):
        model.initialize_memory(
            _memory(),
            stage="Stage2-C",
            target_new_support=torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            target_new_labels=torch.tensor([0], dtype=torch.long),
        )


def test_thin_opgac_interfaces_return_existing_prediction_result_contract():
    config = OPGACConfig(feature_dim=4, context_dim=8, hidden_dim=16, low_rank=2, top2_margin=-1.0)
    memory, model = register_old_classes_opgac(
        _memory(),
        torch.tensor([[0.99, 0.01, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([0], dtype=torch.long),
        config=config,
        stage="Stage2-B",
    )
    memory, model = register_new_classes_opgac(
        memory,
        torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.01, 0.99, 0.0]], dtype=torch.float32),
        torch.tensor([30, 30], dtype=torch.long),
        config=config,
        model=model,
        stage="Stage2-C",
    )
    result = predict_with_opgac_head(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        memory,
        config=config,
        model=model,
    )

    assert result.predicted_labels.shape == (2,)
    assert result.accepted.dtype == torch.bool
    assert result.decisions[0] in {"accept", "reject", "uncertain"}
    assert "opgac_old_score" in result.diagnostics
    assert "opgac_new_score" in result.diagnostics


def test_rollback_restores_ground_old_memory_and_drift_alarm_reports_shift():
    config = OPGACConfig(feature_dim=4, context_dim=8, hidden_dim=16, low_rank=2, drift_alarm_shift=0.1)
    memory = _memory()
    shifted = memory.clone()
    shifted.old_states[0].means = torch.tensor([[0.7, 0.7, 0.0, 0.0]], dtype=torch.float32)
    shifted.version = 5

    alarm = drift_alarm(shifted, config=config)
    assert "old_class_0_mean_shift" in alarm["alarms"]

    restored = rollback_memory(shifted)
    assert restored.new_states == {}
    assert torch.allclose(restored.old_states[0].means, memory.ground_old_states[0].means)
    assert restored.update_log[-1]["type"] == "rollback_to_ground_old_memory"


def test_rf_condition_branch_accepts_iq_segments():
    branch = RFConditionBranch(q_dim=6)
    output = branch(torch.randn(3, 2, 32))
    assert output.shape == (3, 6)
