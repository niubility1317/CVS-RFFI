import torch

from paper_reproduction.hu_feature_separation_2024.augmentation import augment_iq
from paper_reproduction.hu_feature_separation_2024.evaluation import evaluate_matrix, evaluation_matrix_names
from paper_reproduction.hu_feature_separation_2024.finetune import fine_tune_tx_step, tx_finetune_parameters
from paper_reproduction.hu_feature_separation_2024.model import FeatureSeparationNet
from paper_reproduction.hu_feature_separation_2024.training import fit_feature_separation


def test_channel_augmentation_is_seeded_and_preserves_iq_shape():
    iq = torch.randn(2, 2, 256)
    first = augment_iq(iq, generator=torch.Generator().manual_seed(17))
    second = augment_iq(iq, generator=torch.Generator().manual_seed(17))
    assert first.shape == iq.shape
    assert torch.allclose(first, second)
    assert not torch.allclose(first, iq)


def test_training_selects_validation_checkpoint_and_carries_method_metadata():
    model = FeatureSeparationNet(num_tx=2, num_rx=2)
    batch = (torch.randn(4, 2, 256), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 0, 1]))
    result = fit_feature_separation(model, [batch], [batch], max_epochs=2, early_stopping_patience=2)
    assert result.best_epoch in {1, 2}
    assert 0.0 <= result.best_validation_accuracy <= 1.0
    assert result.method_metadata["unpublished_defaults"]
    assert result.best_state_dict


def test_tx_only_finetune_freezes_rx_weights_and_batchnorm_state():
    model = FeatureSeparationNet(num_tx=3, num_rx=2)
    optimizer = torch.optim.SGD(tx_finetune_parameters(model), lr=0.01)
    rx_weight = model.rx_classifier.weight.detach().clone()
    rx_running_mean = model.rx_branch[1].running_mean.detach().clone()
    fine_tune_tx_step(model, torch.randn(3, 2, 256), torch.tensor([0, 1, 2]), optimizer)
    assert torch.equal(model.rx_classifier.weight.detach(), rx_weight)
    assert torch.equal(model.rx_branch[1].running_mean.detach(), rx_running_mean)
    assert not any(parameter.requires_grad for parameter in model.rx_branch.parameters())


def test_evaluation_matrix_contains_every_paper_experiment_family():
    assert evaluation_matrix_names() == (
        "no_new_receiver",
        "cross_receiver",
        "cross_date",
        "channel_augmentation",
        "loss_ablation",
        "few_shot_25_per_tx",
    )


def test_evaluation_matrix_runs_a_transmitter_accuracy_for_each_paper_family():
    model = FeatureSeparationNet(num_tx=2, num_rx=2)
    batch = (torch.randn(2, 2, 256), torch.tensor([0, 1]), torch.tensor([0, 1]))
    result = evaluate_matrix(model, {name: [batch] for name in evaluation_matrix_names()})
    assert tuple(result) == evaluation_matrix_names()
    assert all(0.0 <= row["tx_accuracy"] <= 1.0 for row in result.values())
