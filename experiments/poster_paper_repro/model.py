"""POSTER Homegrown behavior in the published Keras 2.2.4 repository.

The original combines channels-first convolution with a default channels-last
GlobalAveragePooling1D. Its pool reduces the 50-filter axis, leaving signal
length features. This intentionally differs from the earlier Phase2 port.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F


def _keras_glorot(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class PosterHomegrownAuthor(nn.Module):
    def __init__(self, num_classes: int, input_len: int = 288):
        super().__init__()
        if num_classes < 2 or input_len < 1:
            raise ValueError('positive input length and at least two classes required')
        self.input_len = input_len
        self.conv1 = nn.Conv1d(2, 50, 7, padding='same')
        self.conv2 = nn.Conv1d(50, 50, 7, padding='same')
        self.drop1, self.drop2 = nn.Dropout(.5), nn.Dropout(.5)
        self.dense1, self.dense2 = nn.Linear(input_len, 256), nn.Linear(256, 80)
        self.head = nn.Linear(80, num_classes)
        self.apply(_keras_glorot)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (2, self.input_len):
            raise ValueError('expected input shape [batch, 2, input_len]')
        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))
        x = x.mean(dim=1)
        return F.relu(self.dense2(F.relu(self.dense1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Logits plus cross entropy are equivalent to Keras softmax + CE.
        return self.head(self.embedding(x))


def prepare_finetune(source: PosterHomegrownAuthor, *, mode: str) -> PosterHomegrownAuthor:
    """Create independent B state with an explicitly named reference behavior.

    repository: Keras 2.2.4 compiles before freezing and hence updates the
    compile-time parameter list, even after layers are marked non-trainable.
    The author resets all three dense layers before that training.
    paper_last_layer: paper text says only the last layer is tuned; retain the
    learned head and freeze all other parameters.
    """
    model = copy.deepcopy(source)
    if mode == 'repository':
        for layer in (model.dense1, model.dense2, model.head):
            _keras_glorot(layer)
        model.requires_grad_(True)
    elif mode == 'paper_last_layer':
        model.requires_grad_(False)
        model.head.requires_grad_(True)
    else:
        raise ValueError('unknown POSTER fine-tuning mode')
    return model
