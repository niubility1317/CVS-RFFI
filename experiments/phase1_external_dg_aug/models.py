"""Existing identity models and default heads; no post-fit head replacement."""
import torch
from experiments.phase1_external_dg import source as original
from experiments.phase1_external_dg.models import RoutedResNet, mix_statistics
from experiments.standard_ce_baselines.model import ResNet


class AugmentedResNet(RoutedResNet):
    def encode(self, x):
        self.mixstyle_counts = [0, 0]
        for index, module in enumerate(self.features):
            x = module(x)
            if (index in (5, 7) and self.training and self.context is not None
                    and self.route in ('mixstyle', 'mixstyle_id')):
                y, rx = self.context
                # Method-specific style mixing is applied to clean source rows;
                # the common satellite branch only carries identity CE.
                mixed, count = mix_statistics(x[:len(y)], y, rx,
                    identity_protected=self.route == 'mixstyle_id')
                x = torch.cat([mixed, x[len(y):]], dim=0)
                self.mixstyle_counts[0 if index == 5 else 1] = count
        return x.mean(-1)


class StrongResNet50(ResNet):
    def __init__(self):
        super().__init__((3, 4, 6, 3), bottleneck=True)

    def forward(self, x):
        z = self.features(x).mean(-1)
        return z, self.classifier(z)


def build(c):
    if c['family'] == 'resnet18':
        return AugmentedResNet(c['num_classes'], c['route'])
    if c['family'] == 'resnet50':
        if c['num_classes'] != 6:
            raise ValueError('Strong ResNet50 registered only for six-class F0')
        return StrongResNet50()
    return original.build(c)


def identity(net, c, x):
    return original.forward(net, c, x)
