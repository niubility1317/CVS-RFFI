"""Source-only synthetic perturbation energy relative to observation residual."""
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
from experiments.phase1_dura_orthogonal.source_dura_perturbation_probe import DIRECTIONS, EPSILON, perturbations
from experiments.pamf_action_restore.operators import Action


def relative_energy_db(effects, residual):
    if (effects.ndim != 3 or effects.shape[1:] != (4, 256) or
            residual.shape != (len(effects), 256) or not torch.is_complex(effects)
            or not torch.is_complex(residual)):
        raise ValueError('Matching complex perturbation effects and residual required')
    if not torch.isfinite(effects).all() or not torch.isfinite(residual).all():
        raise FloatingPointError('Finite source effects and residual required')
    effect_energy = effects.abs().square().sum(-1)
    residual_energy = residual.abs().square().sum(-1)
    if (effect_energy <= 1e-20).any() or (residual_energy <= 1e-20).any():
        raise ValueError('Nonzero source effect and residual energies required')
    return 10 * torch.log10(effect_energy / residual_energy[:, None])


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
    mask = scene != 'clean'
    ids, rx, scene = ids[mask], rx[mask], scene[mask]
    if len(ids) != 576:
        raise ValueError('All fixed source practical observations required')
    folder = a.root/'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache/epochs/19'
    metadata = core.read(folder/'complete.json')
    if (metadata.get('count') != core.COUNT or metadata.get('namespace') != 'learned-action/train/epoch19'
            or any(metadata.get(k) != 0 for k in ('TX_labels_read', 'F_outputs_read', 'target_reads'))):
        raise ValueError('Matched legal source simulator cache required')
    kernel = np.load(folder/'kernel.npy', mmap_mode='r', allow_pickle=False)
    coefficients = np.load(folder/'coefficients.npy', mmap_mode='r', allow_pickle=False)
    if (kernel.shape != (core.COUNT, 2, 64) or coefficients.shape != (core.COUNT, 4, 256)
            or kernel.dtype != np.float32 or coefficients.dtype != np.float32):
        raise ValueError('Complete matched simulator action required')
    dura = frozen_dura(a.root, 'cpu')
    if any(p.requires_grad for p in dura.parameters()):
        raise ValueError('Frozen source checkpoint required')
    chunks = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(ids), 16):
            selected = ids[start:start+16]
            x, y = core.tensor(clean[selected], 'cpu'), core.tensor(received[selected], 'cpu')
            h = core.tensor(kernel[selected], 'cpu')
            c = core.tensor(coefficients[selected], 'cpu')
            action = true_action(Action, h, c)
            if start == 0:
                smoke = dura(y[:2])
                if smoke.shape != (2, 2, 256) or not torch.isfinite(smoke).all():
                    raise FloatingPointError('Finite real-checkpoint source smoke required')
                a.out.mkdir(parents=True)
                print('SOURCE_PERTURBATION_VISIBILITY_SMOKE_PASS 2', flush=True)
            original = torch.complex(x[:, 0], x[:, 1])
            observed = torch.complex(y[:, 0], y[:, 1])
            residual = observed - action(original)
            effects = torch.stack([action(delta) for delta in perturbations(x)], dim=1)
            chunks.append(relative_energy_db(effects, residual).cpu())
            if (start + len(selected)) % 192 == 0:
                print('SOURCE_PERTURBATION_VISIBILITY_PROGRESS', start + len(selected), flush=True)
    values = torch.cat(chunks)
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS[1:]:
            positions = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(positions) != 48:
                raise ValueError('Complete source receiver and practical cells required')
            cell = values[torch.tensor(positions.tolist(), dtype=torch.long)]
            groups[group+'/'+view] = dict(count=len(cell), median_db=torch.quantile(cell, .5, dim=0).tolist(),
                p10_db=torch.quantile(cell, .1, dim=0).tolist(), p90_db=torch.quantile(cell, .9, dim=0).tolist(),
                below_minus20db=(cell < -20).sum(0).tolist())
    core.write(a.out/'result.json', dict(status='SOURCE_PERTURBATION_VISIBILITY_COMPLETE', commit=a.commit,
        records=len(ids), directions=list(DIRECTIONS), epsilon_relative_rms=EPSILON, groups=groups,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, model_updates=0, DURA_frozen=True,
        caveat='Residual can include noise and simulator mismatch; artificial source perturbations are not established TX identity features and this ratio is not SNR or an information-theoretic bound.'))
    print('SOURCE_PERTURBATION_VISIBILITY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
