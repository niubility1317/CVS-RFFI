from __future__ import annotations

"""Compatibility shim for FUCL contrastive-learning primitives."""

from Fedbase.FUCL import (
    TDLChannelConfig,
    apply_tdl_channel,
    channel_independent_spectrogram,
    encoder_only_state_dict,
    make_two_channel_views,
    nt_xent_loss,
)

__all__ = [
    "TDLChannelConfig",
    "apply_tdl_channel",
    "channel_independent_spectrogram",
    "encoder_only_state_dict",
    "make_two_channel_views",
    "nt_xent_loss",
]
