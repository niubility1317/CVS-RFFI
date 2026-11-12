import copy

import pytest
import torch
from torch import nn

from experiments.pamf_phase2_practical.group2_lora_bridge import build_gallery, register_lora


class Shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.tensor([[2., 0.], [0., 1.]]))

    def forward(self, x):
        return self.layer(x)


def extract(model, x):
    return model(x)


def test_lora_uses_frozen_b_gallery_and_adds_only_new_support():
    model = Shift()
    old_x = torch.tensor([[0., 0.], [10., 0.]])
    old_y = torch.tensor([0, 1])
    gallery = build_gallery(model, extract, old_x, old_y, old_count=2)
    original = gallery.features.clone()
    a = register_lora(gallery, torch.tensor([[0., 20.]]), torch.tensor([2]), k=1)
    b = register_lora(gallery, torch.tensor([[0., 30.]]), torch.tensor([2]), k=1)
    assert torch.equal(gallery.features, original)
    assert torch.equal(a.features[:2], original)
    assert not torch.equal(a.features[2:], b.features[2:])
    before = copy.deepcopy(a.model.state_dict())
    assert a.predict(torch.tensor([[0., 0.], [0., 20.]])).tolist() == [0, 2]
    assert all(torch.equal(before[key], value) for key, value in a.model.state_dict().items())
    assert all(not param.requires_grad for param in a.model.parameters())


def test_lora_ties_use_lowest_class_and_rejects_invalid_support():
    model = Shift()
    old = build_gallery(model, extract, torch.tensor([[0., 0.], [1., 0.]]), torch.tensor([0, 1]), old_count=2)
    state = register_lora(old, torch.tensor([[0., 1.]]), torch.tensor([2]), k=2)
    assert state.predict(torch.tensor([[.5, 0.]])).tolist() == [0]
    with pytest.raises(ValueError):
        register_lora(old, torch.tensor([[0., 1.]]), torch.tensor([1]), k=2)
    with pytest.raises(ValueError):
        build_gallery(model, extract, torch.tensor([[0., 0.]]), torch.tensor([0]), old_count=2)
    with pytest.raises(ValueError):
        register_lora(old, torch.tensor([[0., 1.], [0., 2.]]), torch.tensor([2, 2]), k=2)


def test_default_neighbours_follow_balanced_shots():
    model = Shift()
    old = build_gallery(model, extract, torch.tensor([[0., 0.], [10., 0.]]), torch.tensor([0, 1]), old_count=2)
    state = register_lora(old, torch.tensor([[0., 20.]]), torch.tensor([2]))
    assert state.k == 1
    assert state.predict(torch.tensor([[0., 20.]])).tolist() == [2]
    old_x = torch.tensor([[float(i), 0.] for i in range(5)] + [[float(i), 10.] for i in range(5)])
    old_y = torch.tensor([0]*5 + [1]*5)
    old = build_gallery(model, extract, old_x, old_y, old_count=2)
    state = register_lora(old, torch.tensor([[float(i), 20.] for i in range(5)]), torch.tensor([2]*5))
    assert state.k == 9
