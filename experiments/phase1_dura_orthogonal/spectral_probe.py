"""Source-only frequency decomposition of two frozen DURA refinements."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.model import OrthogonalRecovery
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, TRAIN_EPOCHS, FIT_SIZE, extract_context

MODELS = (
    ('iq', '20261106-phase1-dura-orthogonal-source-s2026092701-r01',
     '3e6b3c0722af52efd5c812f574a2b3d75af65f88'),
    ('guard', '20261106-phase1-dura-orthogonal-guard-source-s2026092701-r01',
     '49706d9acaa7a4f8085b5b48d3eb65e331a7c373'),
)


def band_energy(signal, clean):
    """Four absolute-frequency bands, normalized by total clean energy per record."""
    if signal.shape != clean.shape or signal.ndim != 3 or signal.shape[1:] != (2, 256):
        raise ValueError('Matching complex IQ windows required')
    z = torch.complex(signal[:, 0], signal[:, 1])
    y = torch.complex(clean[:, 0], clean[:, 1])
    spectrum = torch.fft.fft(z, dim=-1, norm='ortho').abs().square()
    denominator = y.abs().square().sum(-1).clamp_min(1e-8)
    frequency = torch.fft.fftfreq(256, device=signal.device).abs()
    masks = [(frequency >= lo) & (frequency < hi) for lo, hi in
             ((0., .125), (.125, .25), (.25, .375))]
    masks.append(frequency >= .375)
    return torch.stack([spectrum[:, m].sum(-1) / denominator for m in masks], dim=-1)


def load_model(root, name, run, commit, device):
    folder = root/'runs'/run
    summary = core.read(folder/'source_summary.json')
    checkpoint = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    required = dict(commit=commit, epochs=TRAIN_EPOCHS, TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if (summary.get('status') != 'SOURCE_TRAINED' or summary.get('fit_ids') != FIT_SIZE or
            any(summary.get(k) != v for k, v in required.items()) or
            checkpoint.get('schema') != SCHEMA or len(checkpoint.get('fit_ids', [])) != FIT_SIZE or
            any(checkpoint.get(k) != v for k, v in required.items())):
        raise ValueError('Frozen source-only checkpoint with matching provenance required')
    expected_objective = 'iq' if name == 'iq' else 'dura_guard'
    if summary.get('objective', 'iq') != expected_objective or checkpoint.get('objective', 'iq') != expected_objective:
        raise ValueError('Frozen objective mismatch')
    model = OrthogonalRecovery(width=checkpoint['width'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def measure(x, clean, dura, models):
    anchor, direction, variance = extract_context(dura, x)
    outputs = {'dura': anchor}
    outputs.update({name: model(x, anchor, direction, variance) for name, model in models.items()})
    result = {name+'_error': band_energy(value-clean, clean) for name, value in outputs.items()}
    result.update({name+'_correction': band_energy(value-anchor, clean)
                   for name, value in outputs.items() if name != 'dura'})
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Finite source-only band energies required')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('A unique diagnostic output is required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or not torch.cuda.is_available():
        raise RuntimeError('An explicitly selected GPU is required')
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    device = 'cuda:0'
    dura = frozen_dura(args.root, device)
    models = {name: load_model(args.root, name, run, commit, device) for name, run, commit in MODELS}
    x = core.tensor(received[ids[:2]], device)
    y = core.tensor(clean[ids[:2]], device)
    trial = measure(x, y, dura, models)
    if any(v.shape != (2, 4) for v in trial.values()):
        raise ValueError('Real checkpoint source smoke failed')
    args.out.mkdir(parents=True)
    print('SPECTRAL_SOURCE_SMOKE_PASS', flush=True)
    started = time.monotonic()
    chunks = {}
    for start in range(0, len(ids), 32):
        selected = ids[start:start+32]
        values = measure(core.tensor(received[selected], device), core.tensor(clean[selected], device), dura, models)
        for key, value in values.items():
            chunks.setdefault(key, []).append(value.cpu())
    arrays = {key: torch.cat(value).numpy() for key, value in chunks.items()}
    groups = {}
    for group, receivers in (('RX4_6', (4, 6)), ('RX8', (8,))):
        for view in core.VIEWS:
            mask = np.isin(rx, receivers) & (scene == view)
            if mask.sum() != 48:
                raise ValueError('Complete source receiver and view cells required')
            groups[group+'/'+view] = {'count': int(mask.sum()),
                **{name: values[mask].mean(0).tolist() for name, values in arrays.items()}}
    core.write(args.out/'result.json', dict(status='SOURCE_SPECTRAL_DIAGNOSTIC_COMPLETE', commit=args.commit,
        records=len(ids), groups=groups, bands_cycles_per_sample=['0-.125', '.125-.25', '.25-.375', '.375-.5'],
        normalizer='total per-record clean IQ energy; band sums equal NMSE', elapsed_seconds=time.monotonic()-started,
        source_only=True, source_pilot_predictions_read=False, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0, model_updates=0,
        caveat='Exploratory source IQ, previously inspected and not independent of DURA; not identity accuracy.'))
    print('SPECTRAL_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
