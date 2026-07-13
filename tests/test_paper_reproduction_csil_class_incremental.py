import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from paper_reproduction.CSIL.losses import compute_csil_loss, compute_ewc_penalty
from paper_reproduction.CSIL.metrics import degree_of_conflict, stage_accuracy_breakdown
from paper_reproduction.CSIL.model import CSILClassifier, ZeroBiasCosineClassifier, csil_masked_sgd_step
from paper_reproduction.CSIL.protocol import (
    build_stage_plan,
    validate_paper_faithful_config,
)


def test_stage_plan_matches_paper_five_batches_without_replay():
    plan = build_stage_plan(total_classes=100, initial_classes=20, classes_per_increment=20)

    assert len(plan) == 5
    assert plan[0]["stage"] == 0
    assert plan[0]["train_class_ids"] == list(range(20))
    assert plan[1]["train_class_ids"] == list(range(20, 40))
    assert plan[-1]["known_class_ids"] == list(range(100))
    assert all(stage["uses_historical_samples"] is False for stage in plan[1:])


def test_csil_classifier_expands_channels_and_keeps_old_fingerprints_fixed():
    model = CSILClassifier(input_dim=8, embedding_dim=4, num_classes=3, stage_id=0)
    old_embedding = model.embedding.weight.detach().clone()
    old_fingerprints = model.classifier.weight.detach().clone()

    model.expand_for_stage(new_classes=2, added_embedding_dim=2, stage_id=1)

    assert model.embedding.out_features == 6
    assert model.classifier.out_features == 5
    assert torch.allclose(model.embedding.weight[:4, :], old_embedding)
    assert torch.allclose(model.classifier.weight[:3, :4], old_fingerprints)
    assert torch.count_nonzero(model.classifier.weight[:3, 4:]).item() == 0
    assert torch.count_nonzero(model.classifier.weight[3:, :4]).item() == 0

    features = torch.randn(7, 8)
    logits = model(features)
    assert logits.shape == (7, 5)


def test_zero_bias_classifier_matches_official_normmag_shifted_cosine():
    classifier = ZeroBiasCosineClassifier(2, 2)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))

    logits = classifier(torch.tensor([[3.0, 0.0], [0.0, 4.0], [-1.0, 0.0]]))

    assert torch.allclose(logits, torch.tensor([[10.0, 5.0], [5.0, 10.0], [0.0, 5.0]]), atol=1e-6)


def test_csil_gradient_masks_lock_old_embedding_bias_and_weights():
    model = CSILClassifier(input_dim=4, embedding_dim=3, num_classes=2, stage_id=0)
    model.expand_for_stage(new_classes=1, added_embedding_dim=2, stage_id=1)

    loss = model(torch.randn(5, 4)).sum()
    loss.backward()
    model.apply_gradient_masks()

    assert torch.count_nonzero(model.embedding.weight.grad[:3, :]).item() == 0
    assert torch.count_nonzero(model.embedding.bias.grad[:3]).item() == 0
    assert torch.count_nonzero(model.classifier.weight.grad[:2, :]).item() == 0


def test_csil_expansion_preserves_existing_device_and_dtype():
    model = CSILClassifier(input_dim=4, embedding_dim=3, num_classes=2, stage_id=0).to(dtype=torch.float64)

    model.expand_for_stage(new_classes=1, added_embedding_dim=2, stage_id=1)

    assert model.embedding.weight.dtype == torch.float64
    assert model.embedding.bias.dtype == torch.float64
    assert model.classifier.weight.dtype == torch.float64
    assert model.embedding_train_mask.dtype == torch.float64
    assert model.classifier_train_mask.dtype == torch.float64


def test_ewc_penalty_slices_expanded_current_parameters_to_previous_shape():
    current = torch.zeros(5, 6)
    current[:3, :4] = 2.0
    previous = torch.ones(3, 4)
    fisher = torch.full((3, 4), 0.5)

    penalty = compute_ewc_penalty(
        params={"classifier.weight": current},
        previous_params={"classifier.weight": previous},
        fisher={"classifier.weight": fisher},
        reference=current,
    )

    assert torch.isclose(penalty, torch.tensor(3.0))


def test_kd_loss_rejects_mismatched_shapes_and_detaches_previous_response():
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.7]], requires_grad=True)
    labels = torch.tensor([0, 1])
    current_old_response = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    previous_old_response = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    result = compute_csil_loss(
        logits=logits,
        labels=labels,
        current_old_response=current_old_response,
        previous_old_response=previous_old_response,
        kd_weight=1.0,
    )
    result.total.backward()

    assert current_old_response.grad is not None
    assert previous_old_response.grad is None

    bad_previous = torch.ones(2, 3)
    try:
        compute_csil_loss(
            logits=logits.detach(),
            labels=labels,
            current_old_response=current_old_response.detach(),
            previous_old_response=bad_previous,
        )
    except ValueError as exc:
        assert "KD responses must have the same shape" in str(exc)
    else:
        raise AssertionError("mismatched KD responses should be rejected")


def test_masked_sgd_step_locks_old_parameters_against_weight_decay_and_momentum():
    model = CSILClassifier(input_dim=4, embedding_dim=3, num_classes=2, stage_id=0)
    model.expand_for_stage(new_classes=1, added_embedding_dim=2, stage_id=1)
    old_embedding = model.embedding.weight[:3, :].detach().clone()
    old_classifier = model.classifier.weight[:2, :].detach().clone()

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    state = {
        "embedding.weight": torch.full_like(model.embedding.weight, 5.0),
        "embedding.bias": torch.full_like(model.embedding.bias, 5.0),
        "classifier.weight": torch.full_like(model.classifier.weight, 5.0),
    }

    next_state = csil_masked_sgd_step(model, lr=0.1, momentum=0.9, weight_decay=0.01, state=state)

    assert torch.allclose(model.embedding.weight[:3, :], old_embedding)
    assert torch.allclose(model.classifier.weight[:2, :], old_classifier)
    assert not torch.allclose(next_state["embedding.weight"][3:, :], torch.full_like(next_state["embedding.weight"][3:, :], 5.0))
    assert not torch.allclose(model.embedding.weight[3:, :], torch.zeros_like(model.embedding.weight[3:, :]))


def test_degree_of_conflict_is_zero_for_simplex_fingerprints_and_positive_for_collision():
    simplex = torch.tensor(
        [
            [1.0, 0.0],
            [-0.5, math.sqrt(3.0) / 2.0],
            [-0.5, -math.sqrt(3.0) / 2.0],
        ]
    )
    collision = torch.tensor(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [-0.5, -math.sqrt(3.0) / 2.0],
        ]
    )

    assert degree_of_conflict(simplex) == 0.0
    assert degree_of_conflict(collision) > 0.5


def test_csil_loss_combines_ce_kd_and_ewc_components():
    logits = torch.tensor([[4.0, 0.5, -1.0], [0.2, 3.0, 0.1]], requires_grad=True)
    labels = torch.tensor([0, 1])
    current_old_response = torch.tensor([[0.9, 0.1], [0.2, 0.8]], requires_grad=True)
    previous_old_response = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    params = {"w": torch.tensor([1.5, -0.5], requires_grad=True)}
    previous_params = {"w": torch.tensor([1.0, -1.0])}
    fisher = {"w": torch.tensor([2.0, 0.5])}

    result = compute_csil_loss(
        logits=logits,
        labels=labels,
        current_old_response=current_old_response,
        previous_old_response=previous_old_response,
        params=params,
        previous_params=previous_params,
        fisher=fisher,
        kd_weight=0.7,
        ewc_weight=0.3,
    )

    assert result.total.requires_grad
    assert result.cross_entropy.item() > 0
    assert result.knowledge_distillation.item() > 0
    assert result.ewc.item() > 0
    assert torch.isclose(
        result.total,
        result.cross_entropy + 0.7 * result.knowledge_distillation + 0.3 * result.ewc,
    )


def test_stage_accuracy_breakdown_reports_old_new_and_overall():
    y_true = torch.tensor([0, 1, 20, 21, 22])
    y_pred = torch.tensor([0, 2, 20, 0, 22])

    metrics = stage_accuracy_breakdown(y_true, y_pred, old_class_ids={0, 1}, new_class_ids={20, 21, 22})

    assert metrics["old_device_accuracy"] == 0.5
    assert metrics["new_device_accuracy"] == 2 / 3
    assert metrics["overall_accuracy"] == 0.6


def test_csil_dry_run_reports_paper_faithful_boundary(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "method_id": "csil_class_incremental_iot",
                "dataset": "ADS-B",
                "total_classes": 100,
                "initial_classes": 20,
                "classes_per_increment": 20,
                "train_ratio": 0.6,
                "validation_ratio": 0.4,
                "batch_size": 64,
                "incremental_epochs": 10,
                "optimizer": "SGD",
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.01,
            }
        ),
        encoding="utf-8",
    )

    checked = validate_paper_faithful_config(json.loads(config.read_text(encoding="utf-8")))
    assert checked["claim_boundary"] == "paper_faithful_adsb_class_incremental_only"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "paper_reproduction.CSIL.train",
            "--config",
            str(config),
            "--dry-run",
            "--formal",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"paper": "Class-Incremental Learning for Wireless Device Identification in IoT"' in result.stdout
    assert '"claim_boundary": "paper_faithful_adsb_class_incremental_only"' in result.stdout
    assert '"not_cvs_stage2": true' in result.stdout
