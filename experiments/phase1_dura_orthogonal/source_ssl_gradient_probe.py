"""Compare source IQ and frozen-encoder gradients on recovered waveforms."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.joint_ssl_iq import SourceIQEncoder, SSL_FEATURE_WEIGHT
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.source_ssl_representation_probe import SOURCE, SOURCE_COMMIT
from experiments.phase1_dura_orthogonal.train_source import extract_context


def alignment(feature_gradient, iq_gradient):
    if feature_gradient.shape != iq_gradient.shape or feature_gradient.ndim != 3:
        raise ValueError('Matching per-record waveform gradients required')
    f, q = feature_gradient.flatten(1), iq_gradient.flatten(1)
    fn, qn = f.norm(dim=1), q.norm(dim=1)
    valid = (fn > 1e-12) & (qn > 1e-12)
    cosine = (f*q).sum(1)/(fn*qn).clamp_min(1e-24)
    ratio = SSL_FEATURE_WEIGHT*fn/qn.clamp_min(1e-12)
    return cosine, ratio, valid


def measure(encoder, restored, clean):
    with torch.no_grad():
        target = encoder(clean)
        energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    value = restored.detach().requires_grad_(True)
    feature_loss = (encoder(value)-target).square().sum(1)
    feature_gradient, = torch.autograd.grad(feature_loss.sum(), value)
    iq_gradient = 2*(value.detach()-clean)/(512*energy[:, None, None])
    cosine, ratio, valid = alignment(feature_gradient, iq_gradient)
    if not torch.isfinite(cosine).all() or not torch.isfinite(ratio).all():
        raise FloatingPointError('Finite source output-space gradients required')
    return torch.stack((cosine, ratio, valid.to(cosine.dtype)), 1).detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    training_environments('cuda:0', os.environ)
    if args.out.exists():
        raise FileExistsError('Unique source gradient diagnostic output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    source = args.root/'runs'/SOURCE
    summary = core.read(source/'source_summary.json')
    expected = dict(status='SOURCE_TRAINED', commit=SOURCE_COMMIT, objective='ssl_joint_guard',
                    architecture='dura_multiscale', TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0, DURA_frozen=True)
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed source-only joint SSL training required')
    state = torch.load(source/'final.pt', map_location='cpu', weights_only=False)
    if (state.get('commit') != SOURCE_COMMIT or state.get('objective') != 'ssl_joint_guard' or
            np.intersect1d(ids, state['fit_ids']).size):
        raise ValueError('Source pilot must be outside the fixed candidate fitting IDs')
    model = DuraMultiscaleSpectral(width=32, taps=9)
    model.load_state_dict(state['model'], strict=True)
    model = model.to('cuda:0').eval().requires_grad_(False)
    encoder = SourceIQEncoder()
    encoder.load_state_dict(state['source_ssl_aux_state'], strict=True)
    encoder = encoder.to('cuda:0').eval().requires_grad_(False)
    dura = frozen_dura(args.root, 'cuda:0')
    with torch.no_grad():
        x = core.tensor(received[ids[:2]], 'cuda:0')
        y = core.tensor(clean[ids[:2]], 'cuda:0')
        anchor, direction, variance = extract_context(dura, x)
        if not torch.equal(anchor, dura(x)):
            raise ValueError('DURA context must match its native output')
        output = model(x, anchor, direction, variance)
    if not torch.isfinite(measure(encoder, output, y)).all():
        raise FloatingPointError('Real-checkpoint source gradient smoke failed')
    print('SOURCE_SSL_GRADIENT_SMOKE_PASS', flush=True)
    start = time.monotonic()
    all_values = {'dura': [], 'candidate': []}
    for offset in range(0, len(ids), 32):
        chosen = ids[offset:offset+32]
        with torch.no_grad():
            x, y = core.tensor(received[chosen], 'cuda:0'), core.tensor(clean[chosen], 'cuda:0')
            anchor, direction, variance = extract_context(dura, x)
            output = model(x, anchor, direction, variance)
        all_values['dura'].append(measure(encoder, anchor, y))
        all_values['candidate'].append(measure(encoder, output, y))
    all_values = {key: torch.cat(value) for key, value in all_values.items()}
    groups = {}
    for r, s in sorted({(int(r), str(s)) for r, s in zip(rx, scene)}):
        selected = np.flatnonzero((rx == r) & (scene == s)).tolist()
        for position, values in all_values.items():
            subset = values[selected]
            valid = subset[:, 2] > 0
            c, q = subset[valid, 0], subset[valid, 1]
            groups[f'RX{r}/{s}/{position}'] = dict(count=len(selected), valid_count=int(valid.sum()),
                mean_cosine=float(c.mean()) if len(c) else None,
                opposed_fraction=float((c < 0).float().mean()) if len(c) else None,
                mean_weighted_norm_ratio=float(q.mean()) if len(q) else None)
    core.write(args.out/'result.json', dict(status='SOURCE_SSL_GRADIENT_DIAGNOSTIC_COMPLETE',
        commit=args.commit, source_checkpoint=SOURCE, source_checkpoint_commit=SOURCE_COMMIT,
        records=len(ids), groups=groups, source_pilot_not_candidate_fit=True,
        feature_weight=SSL_FEATURE_WEIGHT, elapsed_seconds=time.monotonic()-start,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0, model_updates=0,
        caveat='Gradients are per-record waveform-output gradients for feature and normalized IQ losses, not network parameter gradients or TX identification. Source pilot previously explored.'))
    print('SOURCE_SSL_GRADIENT_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
