"""Shared components used by the paper baseline implementations."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CODE_DIR = _PROJECT_ROOT / "code"
if _CODE_DIR.exists() and str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from .grl import GradientReversalLayer, gradient_reverse
from .resnet1d import ResNet1DEncoder
from .resnet2d import ResNet2DEncoder

__all__ = [
    "GradientReversalLayer",
    "gradient_reverse",
    "ResNet1DEncoder",
    "ResNet2DEncoder",
]
