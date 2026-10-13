"""Deployment feature boundary; class-specific Phase1 rows are excluded."""
from .model import TransferFeatures, from_phase1

__all__ = ["TransferFeatures", "from_phase1"]
