"""Prevent nonfinite waveforms or reader scores from becoming valid class IDs."""

import torch


def checked_logits(reader, signal):
    if not torch.isfinite(signal).all():
        raise FloatingPointError('Nonfinite input or restored IQ waveform')
    logits = reader.classify_features(reader.features(signal))
    if logits.ndim != 2 or logits.shape[0] != len(signal) or not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite or invalid frozen-reader scores')
    return logits
