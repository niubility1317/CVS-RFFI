"""One waveform-only decision per observation, before the frozen classifier.

Small proposed changes are suppressed; this is not a clean/scene oracle or an
identity guarantee. A damaged sample can also be preserved. Source diagnostics
must measure that failure rather than infer quality from the gate alone.
"""
import math
import torch
from torch import nn

POLICY = dict(schema='waveform_change_preserve_v1', calibration_samples=8192,
              calibration_seed=2026101933, threshold_cap=.02, denominator_floor=1e-4,
              rule='threshold=min(.02,max relative change over fixed L/U clean calibration)',
              decision='return received if change<=threshold, otherwise restored',
              decision_inputs=['received_IQ', 'restored_IQ', 'frozen_threshold'],
              TX_labels=False, F_outputs=False, scene_labels=False, target_fit=False)


def relative_change(received, restored):
    if received.ndim != 3 or received.shape[1:] != (2, 256) or restored.shape != received.shape:
        raise ValueError('Matching packet IQ[B,2,256] required')
    if not torch.isfinite(received).all() or not torch.isfinite(restored).all():
        raise FloatingPointError('Nonfinite observed/restored IQ')
    return ((restored-received).square().mean((1, 2)) /
            received.square().mean((1, 2)).clamp_min(POLICY['denominator_floor']))


def calibrated_threshold(changes):
    if (changes.ndim != 1 or len(changes) != POLICY['calibration_samples']
            or not torch.isfinite(changes).all() or (changes < 0).any()):
        raise ValueError('All fixed finite nonnegative source-calibration changes required')
    return min(POLICY['threshold_cap'], float(changes.max()))


def preserve(received, restored, threshold):
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= POLICY['threshold_cap']:
        raise ValueError('Frozen source threshold outside the registered interval')
    changes = relative_change(received, restored)
    mask = changes <= threshold
    return torch.where(mask[:, None, None], received, restored), mask, changes


class PreserveSmallChanges(nn.Module):
    def __init__(self, restorer, threshold):
        super().__init__()
        if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= POLICY['threshold_cap']:
            raise ValueError('One source-calibrated threshold required')
        self.restorer = restorer.eval().requires_grad_(False)
        self.register_buffer('threshold', torch.tensor(float(threshold), dtype=torch.float64))
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def from_restored(self, received, restored):
        return preserve(received, restored, self.threshold)

    def forward(self, received):
        restored = self.restorer(received)
        return self.from_restored(received, restored)[0]
