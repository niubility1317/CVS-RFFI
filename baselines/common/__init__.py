"""Shared components used by the paper baseline implementations."""

from .grl import GradientReversalLayer, gradient_reverse
from .resnet1d import ResNet1DEncoder
from .resnet2d import ResNet2DEncoder

__all__ = [
    "GradientReversalLayer",
    "gradient_reverse",
    "ResNet1DEncoder",
    "ResNet2DEncoder",
]
