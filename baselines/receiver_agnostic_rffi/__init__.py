"""Receiver-agnostic and collaborative RFFI baseline."""

from .collaborative_inference import adaptive_soft_fusion, soft_fusion
from .model import ReceiverAgnosticRFFI

__all__ = ["ReceiverAgnosticRFFI", "soft_fusion", "adaptive_soft_fusion"]
