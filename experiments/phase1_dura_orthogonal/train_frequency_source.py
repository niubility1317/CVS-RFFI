"""One GPU closed-form fit of an F-free, source-only DURA spectral correction."""
import argparse
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.dura_frequency_recovery import DuraFrequencyRecovery
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.source_frequency_regression import BATCH, RIDGE, spectra, solve
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE, load_source_training, extract_context, evaluate_source


def fit(data):
    received, anchor, _, variance, clean = data
    ratio = variance/received.square().mean((1, 2)).clamp_min(1e-12)
    limits = torch.quantile(ratio, torch.tensor([.25, .5, .75], device=received.device))
    bins = torch.bucketize(ratio, limits)
    gram = torch.zeros((4, 256, 4, 4), dtype=torch.complex64, device=received.device)
    rhs = torch.zeros((4, 256, 4, 1), dtype=torch.complex64, device=received.device)
    counts = [0]*4
    for start in range(0, len(received), BATCH):
        end = min(start+BATCH, len(received))
        features, a, _, y = spectra(received[start:end], anchor[start:end], clean[start:end])
        energy = y.abs().square().mean(-1).clamp_min(1e-12).sqrt()
        features, residual = features/energy[:, None, None], (y-a)/energy[:, None]
        for group in range(4):
            mask = bins[start:end] == group
            if not mask.any():
                continue
            z = features[mask].permute(1, 0, 2)
            target = residual[mask].permute(1, 0)
            gram[group] += z.conj().transpose(1, 2) @ z
            rhs[group] += z.conj().transpose(1, 2) @ target.unsqueeze(-1)
            counts[group] += int(mask.sum())
    weights = solve(gram, rhs)
    if not torch.isfinite(weights).all() or not all(counts):
        raise ValueError('Finite fitted weights and nonempty source bins required')
    model = DuraFrequencyRecovery().to(received.device)
    model.weights.copy_(weights)
    model.limits.copy_(limits)
    return model.eval(), counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prepare-batch', type=int, default=32)
    p.add_argument('--objective', required=True)
    p.add_argument('--architecture', required=True)
    p.add_argument('--source-context-cache', type=Path, required=True)
    p.add_argument('--deterministic', action='store_true')
    args = p.parse_args()
    if (args.objective != 'wide_frequency_regression' or args.architecture != 'dura_frequency_regression'
            or args.device != 'cuda:0' or args.prepare_batch != 32 or not args.deterministic):
        raise ValueError('Registered deterministic GPU frequency fit required')
    training_environments(args.device, os.environ)
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    ids, rx, scene, _, source_received, source_clean = source_inputs(args.root)
    _, _, _, source_fit = load_source_training(args.root, ids, fit_size=WIDE_FIT_SIZE)
    raw, counts = load_cache(args.source_context_cache, source_fit, 32, args.device, deterministic=True)
    data = tuple(v.to(args.device) for v in raw)
    dura = frozen_dura(args.root, args.device)
    with torch.no_grad():
        actual = extract_context(dura, data[0][:8])
        for got, cached in zip(actual, data[1:4]):
            torch.testing.assert_close(got, cached[:8], rtol=1e-4, atol=1e-5)
    if args.smoke:
        data = tuple(torch.cat([v[i:i+64] for i in (0, WIDE_FIT_SIZE, 2*WIDE_FIT_SIZE)]) for v in data)
    started = time.monotonic()
    model, bin_counts = fit(data)
    with torch.no_grad():
        result = model(*(v[:8] for v in data[:4]))
        if not torch.isfinite(result).all():
            raise FloatingPointError('Finite source recovery required')
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    args.out.mkdir(parents=True)
    metadata = dict(commit=args.commit, objective=args.objective, architecture=args.architecture,
        source_fit_size=WIDE_FIT_SIZE, DURA_training_reads=1, DURA_frozen=True,
        source_frequency_ridge=RIDGE, source_noise_bins=4,
        source_bin_metric='variance_over_received_power', source_fit_method='closed_form_complex_ridge',
        source_bin_counts=bin_counts, source_fit_seconds=elapsed,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    if args.smoke:
        core.write(args.out/'smoke.json', dict(status='SOURCE_TRAINING_SMOKE_PASS', **metadata))
        print('FREQUENCY_SOURCE_TRAINING_SMOKE_PASS', flush=True)
        return
    state = dict(schema=SCHEMA, model=model.state_dict(), width=0, fit_ids=source_fit, epochs=0, **metadata)
    with (args.out/'final.pt').open('xb') as stream:
        torch.save(state, stream)
    _, groups = evaluate_source(model, dura, ids, rx, scene, source_received, source_clean, args.device, 32)
    core.write(args.out/'source_summary.json', dict(status='SOURCE_TRAINED', fit_ids=len(source_fit),
        epochs=0, counts=counts, source_groups=groups, source_IQ_metrics_not_identity_accuracy=True, **metadata))
    print('FREQUENCY_SOURCE_TRAINED', flush=True)


if __name__ == '__main__':
    main()
