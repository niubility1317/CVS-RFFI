"""PyTorch ports of POSTER Homegrown and RadioNet DF author architectures.

Sources: SmartHomePrivacyProject/RadioFingerprinting@23d1fd4 and
UCdasec/RadioNet@64f4b0a. Layer shapes/activations/dropout/pooling are preserved;
source training uses this project's explicitly matched residual augmentation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SameMaxPool(nn.Module):
    def forward(self, x):
        length = x.shape[-1]
        padding = max((math.ceil(length / 4) - 1) * 4 + 8 - length, 0)
        return F.max_pool1d(F.pad(x, (padding // 2, padding - padding // 2), value=-float('inf')), 8, 4)


def keras_initialize(module):
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class PosterHomegrown(nn.Module):
    def __init__(self, num_classes, input_len=256, **unused):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 50, 7, padding='same')
        self.conv2 = nn.Conv1d(50, 50, 7, padding='same')
        self.drop1, self.drop2 = nn.Dropout(.5), nn.Dropout(.5)
        self.dense1, self.dense2 = nn.Linear(50, 256), nn.Linear(256, 80)
        self.head = nn.Linear(80, num_classes)
        self.apply(keras_initialize)

    def embedding(self, x):
        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))
        x = x.mean(-1)
        return F.relu(self.dense2(F.relu(self.dense1(x))))

    def forward(self, x):
        return self.head(self.embedding(x))

    def author_finetune(self, num_classes):
        for p in self.parameters():
            p.requires_grad_(False)
        # Author code copies layers[:-3] and initializes all three Dense layers.
        self.dense1 = nn.Linear(50, 256).to(self.head.weight.device)
        self.dense2 = nn.Linear(256, 80).to(self.head.weight.device)
        self.head = nn.Linear(80, num_classes).to(self.head.weight.device)
        for layer in (self.dense1, self.dense2, self.head):
            keras_initialize(layer)


class RadioNetDF(nn.Module):
    def __init__(self, num_classes, input_len=256, **unused):
        super().__init__()
        self.blocks = nn.ModuleList()
        channels = [2, 32, 64, 128, 256]
        for i in range(4):
            act = nn.ELU if i == 0 else nn.ReLU
            layers = [nn.Conv1d(channels[i], channels[i + 1], 8, padding='same'), act(),
                      nn.Conv1d(channels[i + 1], channels[i + 1], 8, padding='same'), act(), SameMaxPool()]
            if i < 3:
                layers.append(nn.Dropout(.1))
            self.blocks.append(nn.Sequential(*layers))
        width = input_len
        for _ in range(4):
            width = math.ceil(width / 4)
        self.head = nn.Linear(width * 256, num_classes)
        self.apply(keras_initialize)

    def embedding(self, x):
        for block in self.blocks:
            x = block(x)
        # Keras channels-last Flatten orders time before channel.
        return x.transpose(1, 2).flatten(1)

    def forward(self, x):
        return self.head(self.embedding(x))

    def author_finetune(self, num_classes):
        for p in self.parameters():
            p.requires_grad_(False)
        self.head = nn.Linear(self.head.in_features, num_classes).to(self.head.weight.device)
        keras_initialize(self.head)
