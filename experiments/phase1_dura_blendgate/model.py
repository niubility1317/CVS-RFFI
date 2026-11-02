"""DURA anchored, input-only temporal fusion with a frozen spectral expert."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from experiments.phase1_spectral_coupling.model import SpectralCoupling

ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
SOURCE = '20261102-phase1-spectral-aligned-source-s2026092701-r01'
EXPERT_COMMIT = 'b326a21a54ae7b9b198e3d077c01a3919d8bd135'


@torch.no_grad()
def align_to_anchor(wave, anchor, preserve_edges=False):
    padded = F.pad(wave, (64, 64), mode='reflect')
    windows = torch.complex(padded[:, 0], padded[:, 1]).unfold(-1, 256, 1)
    center = windows[:, :, 64:-64]
    target = torch.complex(anchor[:, 0, 64:-64], anchor[:, 1, 64:-64])[:, None]
    gain = (center.conj() * target).sum(-1) / center.abs().square().sum(-1).clamp_min(1e-20)
    error = (gain[:, :, None] * center - target).abs().square().mean(-1)
    choice = error.argmin(1)
    row = torch.arange(len(wave), device=wave.device)
    value = gain[row, choice, None] * windows[row, choice]
    aligned = torch.stack((value.real, value.imag), 1)
    if preserve_edges:
        indices = torch.arange(256, device=wave.device)[None, :] + (choice - 64)[:, None]
        valid = (indices >= 0) & (indices < 256)
        aligned = torch.where(valid[:, None, :], aligned, anchor)
    return aligned


class AlignedBlend(nn.Module):
    def __init__(self, expert, preserve_edges=False):
        super().__init__()
        self.expert = expert.eval().requires_grad_(False)
        self.preserve_edges = preserve_edges
        self.entry = nn.Conv1d(6, 24, 7, padding=3)
        self.layers = nn.ModuleList(nn.Conv1d(24, 24, 5, dilation=d, padding=2*d) for d in (1, 2, 4))
        self.gate_output = nn.Conv1d(24, 1, 1)
        nn.init.zeros_(self.gate_output.weight)
        nn.init.zeros_(self.gate_output.bias)

    def train(self, mode=True):
        super().train(mode)
        self.expert.eval()
        return self

    def forward(self, received, initial, log_variance):
        shape = received.shape
        if (received.ndim != 3 or shape[1:] != (2, 256) or initial.shape != shape
                or log_variance.shape != (shape[0],)):
            raise ValueError('Matching IQ windows and one blind log variance are required')
        if not all(torch.isfinite(t).all() for t in (received, initial, log_variance)):
            raise FloatingPointError('Finite waveform evidence is required')
        with torch.no_grad():
            candidate = self.expert(received, received, log_variance)
            aligned = align_to_anchor(candidate, initial, preserve_edges=self.preserve_edges)
        if not torch.isfinite(aligned).all():
            raise FloatingPointError('Finite aligned expert output is required')
        power = received.square().mean((1, 2), keepdim=True).clamp_min(1e-8)
        scale = power.sqrt()
        x = torch.complex(received[:, 0], received[:, 1]) / scale[:, 0]
        b = torch.complex(initial[:, 0], initial[:, 1]) / scale[:, 0]
        z = torch.complex(aligned[:, 0], aligned[:, 1]) / scale[:, 0]
        cross = b.conj() * z
        noise = ((log_variance - power[:, 0, 0].log()).clamp(-16, 8) / 8)[:, None, None]
        features = torch.stack((torch.log1p(x.abs().square()), torch.log1p(b.abs().square()),
                                torch.log1p(z.abs().square()), cross.real.clamp(-16, 16),
                                cross.imag.clamp(-16, 16)), dim=1)
        h = F.silu(self.entry(torch.cat((features, noise.expand(-1, 1, 256)), dim=1)))
        for layer in self.layers:
            h = h + 0.2 * F.silu(layer(h))
        alpha = 0.5 * self.gate_output(h).tanh()
        return initial + alpha * (aligned - initial)


def make_model():
    folder = ROOT / 'runs' / SOURCE
    manifest = json.loads((folder / 'prediction_complete.json').read_text(encoding='utf8'))
    expected = dict(status='SOURCE_PREDICTIONS_FIXED', count=672, commit=EXPERT_COMMIT,
                    fit_ids=8192, F_frozen=True, TX_labels_read=0, U_s_truth_reads=0,
                    V_reads=0, test_reads=0, model_updates_after_prediction=0)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen source-only spectral expert provenance is required')
    state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    if (state.get('schema') != 'phase1_spectral_aligned_v1' or state.get('commit') != EXPERT_COMMIT
            or state.get('epochs') != 5 or len(state.get('fit_ids', [])) != 8192):
        raise ValueError('Exact source-trained spectral expert checkpoint is required')
    expert = SpectralCoupling()
    expert.load_state_dict(state['model'], strict=True)
    return AlignedBlend(expert)
