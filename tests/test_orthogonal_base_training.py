import torch
from experiments.orthogonal_paper_repro.base import train_base


def test_base_training_only_uses_provided_train_and_validation_tensors():
    torch.manual_seed(190)
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 8))
    train_x = torch.randn(24, 2, 4)
    train_y = torch.arange(3).repeat(8)
    validation_x = torch.randn(12, 2, 4)
    validation_y = torch.arange(3).repeat(4)
    state, history = train_base(encoder, train_x, train_y, validation_x, validation_y,
        num_classes=3, num_targets=5, feature_dim=8, epochs=2, batch_size=24, seed=77)
    assert state['targets'].shape == (5, 8)
    assert state['base_weights'].shape == (3, 8)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(item['training_loss'])) for item in history)
    assert all(torch.isfinite(torch.tensor(item['validation_loss'])) for item in history)
    assert 1 <= state['best_epoch'] <= 2
