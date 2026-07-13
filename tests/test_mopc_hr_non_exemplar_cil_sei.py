import pytest
import torch
import torch.nn.functional as F

from paper_reproduction.mopc_hr_non_exemplar_cil_sei import (
    compute_class_prototypes,
    correct_old_prototypes,
    hierarchical_regularization,
    mopc_hr_incremental_objective,
    prototype_augmentation,
    validate_mopc_hr_config,
)


def test_class_prototypes_are_classwise_feature_means():
    features = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 4.0]])
    labels = torch.tensor([7, 7, 9])
    prototypes, class_ids = compute_class_prototypes(features, labels)
    assert class_ids.tolist() == [7, 9]
    assert torch.allclose(prototypes, torch.tensor([[2.0, 1.0], [0.0, 4.0]]))


def test_paper_momentum_correction_matches_equations_10_to_14():
    old = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    new_previous = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    new_current = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    corrected = correct_old_prototypes(old, new_previous, new_current, alpha=0.5)
    expected = 0.5 * old + 0.5 * torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert torch.allclose(corrected, expected)


def test_official_code_compatibility_mode_uses_dot_softmax():
    old = torch.tensor([[1.0, 0.0]])
    new_previous = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    new_current = new_previous + torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    corrected = correct_old_prototypes(
        old,
        new_previous,
        new_current,
        alpha=0.97,
        similarity_mode="official_code_dot_softmax",
    )
    similarity = torch.softmax(old @ new_previous.t(), dim=1)
    expected = 0.97 * old + 0.03 * (similarity @ (new_current - new_previous))
    assert torch.allclose(corrected, expected)


def test_hierarchical_regularization_decreases_with_layer_depth_and_is_squared_l2():
    current = {"early": torch.tensor([2.0]), "late": torch.tensor([2.0])}
    previous = {"early": torch.tensor([0.0]), "late": torch.tensor([0.0])}
    penalty = hierarchical_regularization(current, previous, lambda_max=1.0)
    assert penalty.item() == pytest.approx(1.0 * 4.0 + 0.5 * 4.0)


def test_incremental_objective_is_equation_22_without_distillation():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    proto_logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    proto_labels = torch.tensor([0])
    current = {"w": torch.tensor([2.0], requires_grad=True)}
    previous = {"w": torch.tensor([1.0])}
    result = mopc_hr_incremental_objective(
        logits,
        labels,
        proto_logits,
        proto_labels,
        current,
        previous,
        beta=0.4,
    )
    expected = F.cross_entropy(logits, labels) + F.cross_entropy(proto_logits, proto_labels) + 0.4
    assert torch.allclose(result.total, expected)
    result.total.backward()
    assert current["w"].grad is not None


def test_prototype_augmentation_and_protocol_defaults_match_paper():
    generator = torch.Generator().manual_seed(4)
    augmented, labels = prototype_augmentation(
        torch.eye(2),
        torch.tensor([10, 11]),
        num_samples=5,
        noise_std=0.0,
        generator=generator,
    )
    assert augmented.shape == (5, 2)
    assert set(labels.tolist()) <= {10, 11}

    checked = validate_mopc_hr_config(
        {"total_classes": 100, "base_classes": 50, "classes_per_increment": 10}
    )
    assert checked["base_epochs"] == 20
    assert checked["incremental_epochs"] == 20
    assert checked["batch_size"] == 16
    assert checked["optimizer"] == "SGD"
    assert checked["learning_rate"] == 0.01
    assert checked["prototype_noise_std"] == 0.05
    assert checked["prototype_momentum"] == 0.97
    assert checked["distillation_in_total_loss"] is False

    with pytest.raises(ValueError, match="non-exemplar"):
        validate_mopc_hr_config(
            {
                "total_classes": 100,
                "base_classes": 50,
                "classes_per_increment": 10,
                "replay_raw_samples": True,
            }
        )
