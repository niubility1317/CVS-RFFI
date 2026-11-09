"""Source-only fixed FIR tap-window decomposition, not physical-path labeling."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.oracle_action_source_probe import true_action
from experiments.pamf_action_restore.operators import Action


def tap_windows(kernel):
    """Partition matched complex FIR into four nonoverlapping 16-tap windows."""
    if kernel.ndim != 2 or kernel.shape[1] != 64 or not torch.is_complex(kernel):
        raise ValueError('Matched complex 64-tap source kernels required')
    mask = torch.arange(64, device=kernel.device)[None, :] // 16
    return torch.stack([kernel * (mask == k) for k in range(4)], dim=1)


def component_metrics(components, observed):
    if (components.ndim != 3 or components.shape[1:] != (4, 256) or
            observed.shape != (len(components), 256) or not torch.is_complex(components)
            or not torch.is_complex(observed)):
        raise ValueError('Matching complex source components and observed IQ required')
    if not torch.isfinite(components).all() or not torch.isfinite(observed).all():
        raise FloatingPointError('Finite source components required')
    total = components.sum(1)
    energy = components.abs().square().sum(-1)
    total_energy = total.abs().square().sum(-1).clamp_min(1e-20)
    ratios = energy / total_energy[:, None]
    coherence = []
    for i in range(4):
        for j in range(i + 1, 4):
            numerator = (components[:, i].conj() * components[:, j]).sum(-1).abs()
            denominator = (energy[:, i] * energy[:, j]).clamp_min(1e-40).sqrt()
            coherence.append(numerator / denominator)
    residual = (observed - total).abs().square().sum(-1) / total_energy
    return ratios, torch.stack(coherence, dim=1), residual, energy.argmax(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    if a.out.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Unique output and CPU-only execution required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, clean = source_inputs(a.root)
    folder = a.root / 'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/epochs/19'
    metadata = core.read(folder / 'complete.json')
    if (metadata.get('count') != core.COUNT or metadata.get('namespace') != 'learned-action/train/epoch19'
            or any(metadata.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
        raise ValueError('Matched legal source simulator cache required')
    kernel = np.load(folder / 'kernel.npy', mmap_mode='r', allow_pickle=False)
    coefficients = np.load(folder / 'coefficients.npy', mmap_mode='r', allow_pickle=False)
    if (kernel.shape != (core.COUNT, 2, 64) or coefficients.shape != (core.COUNT, 4, 256)
            or kernel.dtype != np.float32 or coefficients.dtype != np.float32):
        raise ValueError('Complete matched simulator action required')
    dura = frozen_dura(a.root, 'cpu')
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen source checkpoint required')
    chunks = [[], [], [], []]
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 16):
            selected = ids[start:start+16]
            x = core.tensor(clean[selected], 'cpu')
            y = core.tensor(received[selected], 'cpu')
            h = core.tensor(kernel[selected], 'cpu')
            c = core.tensor(coefficients[selected], 'cpu')
            action = true_action(Action, h, c)
            if start == 0:
                smoke = dura(y[:2])
                if smoke.shape != (2, 2, 256) or not torch.isfinite(smoke).all():
                    raise FloatingPointError('Finite real-checkpoint source smoke required')
                a.out.mkdir(parents=True)
                print('SOURCE_DELAY_COMPONENTS_SMOKE_PASS 2', flush=True)
            z = torch.complex(x[:, 0], x[:, 1])
            observed = torch.complex(y[:, 0], y[:, 1])
            components = torch.stack([Action(part, action.main, action.image)(z)
                                      for part in tap_windows(action.kernel).unbind(1)], dim=1)
            torch.testing.assert_close(components.sum(1), action(z), rtol=2e-5, atol=1e-5)
            for dest, value in zip(chunks, component_metrics(components, observed)):
                if not torch.isfinite(value).all():
                    raise FloatingPointError('Finite component diagnostics required')
                dest.append(value.cpu())
            if (start + len(selected)) % 168 == 0:
                print('SOURCE_DELAY_COMPONENTS_PROGRESS', start + len(selected), flush=True)
    ratios, coherence, residual, dominant = (torch.cat(parts, 0) for parts in chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            indices = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(indices) != 48:
                raise ValueError('Complete source cells required')
            idx = torch.tensor(indices.tolist(), dtype=torch.long)
            groups[group+'/'+view] = dict(count=len(idx), mean_energy_ratio=ratios[idx].mean(0).tolist(),
                median_energy_ratio=ratios[idx].median(0).values.tolist(),
                mean_complex_coherence=coherence[idx].mean(0).tolist(),
                mean_observed_residual=float(residual[idx].mean()),
                dominant_counts=[int((dominant[idx] == k).sum()) for k in range(4)])
    core.write(a.out/'result.json', dict(status='SOURCE_DELAY_COMPONENTS_COMPLETE', commit=a.commit,
        records=len(ids), tap_windows=[[0,15],[16,31],[32,47],[48,63]],
        coherence_pairs=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], groups=groups,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, model_updates=0, DURA_frozen=True,
        caveat='Tap windows are not physical paths; source simulator and clean are unavailable at target inference; none of these statistics measure TX identity.'))
    print('SOURCE_DELAY_COMPONENTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
