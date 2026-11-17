"""Checks against RadioFingerprinting's published Homegrown implementation."""
import pytest
import torch
from experiments.poster_paper_repro.model import PosterHomegrownAuthor, prepare_finetune


def test_published_keras_pool_preserves_time_dimension():
    # In Keras 2.2.4 GlobalAveragePooling1D() defaults to channels_last.
    # The author's Conv1D outputs (batch, 50, length), so it averages axis 1.
    model = PosterHomegrownAuthor(5, input_len=288)
    assert model.dense1.in_features == 288
    x = torch.randn(3, 2, 288)
    model.eval()
    with torch.no_grad():
        first = torch.relu(model.conv1(x))
        second = torch.relu(model.conv2(first))
        expected = model.head(torch.relu(model.dense2(torch.relu(model.dense1(second.mean(1))))))
        torch.testing.assert_close(model(x), expected)


def test_repository_behavior_updates_convolutions_because_compile_precedes_freeze():
    torch.manual_seed(7)
    source = PosterHomegrownAuthor(5, input_len=288)
    old_conv = source.conv1.weight.detach().clone()
    old_dense = source.dense1.weight.detach().clone()
    model = prepare_finetune(source, mode='repository')
    torch.testing.assert_close(model.conv1.weight, old_conv)
    assert not torch.equal(model.dense1.weight, old_dense)
    assert all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-7)
    torch.nn.functional.cross_entropy(model(torch.randn(12, 2, 288)), torch.arange(12) % 5).backward()
    optimizer.step()
    assert not torch.equal(model.conv1.weight, old_conv)


def test_paper_last_layer_only_preserves_other_weights():
    torch.manual_seed(8)
    source = PosterHomegrownAuthor(5, input_len=288)
    model = prepare_finetune(source, mode='paper_last_layer')
    original = {name:p.detach().clone() for name,p in model.named_parameters()}
    assert {name for name,p in model.named_parameters() if p.requires_grad} == {'head.weight','head.bias'}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
    torch.nn.functional.cross_entropy(model(torch.randn(12, 2, 288)), torch.arange(12) % 5).backward()
    optimizer.step()
    assert not torch.equal(model.head.weight, original['head.weight'])
    for name,p in model.named_parameters():
        if not name.startswith('head.'):
            torch.testing.assert_close(p, original[name])


def test_author_pooling_with_reset_dense_freezes_convolutions():
    source = PosterHomegrownAuthor(5, input_len=288)
    model = prepare_finetune(source, mode='reset_dense_freeze_conv')
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        'dense1.weight', 'dense1.bias', 'dense2.weight', 'dense2.bias', 'head.weight', 'head.bias'}
    torch.testing.assert_close(model.conv1.weight, source.conv1.weight)
    assert not torch.equal(model.dense1.weight, source.dense1.weight)


def test_reject_unknown_behavior():
    with pytest.raises(ValueError):
        prepare_finetune(PosterHomegrownAuthor(5), mode='unknown')
