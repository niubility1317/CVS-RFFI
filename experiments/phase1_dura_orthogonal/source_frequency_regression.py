"""Source-only held-out diagnostic of a shared widely-linear IQ correction."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

FIT_SIZE = 32768
VIEW_COUNT = 3
BATCH = 512
RIDGE = .001


def spectra(received, anchor, clean):
    def transform(v):
        return torch.fft.fft(torch.complex(v[:, 0], v[:, 1]), dim=-1, norm='ortho')
    a, x, y = (transform(v) for v in (anchor, received, clean))
    reverse = (-torch.arange(a.shape[-1], device=a.device)) % a.shape[-1]
    features = torch.stack((a, x, a[:, reverse].conj(), x[:, reverse].conj()), -1)
    return features, a, x, y


def solve(gram, rhs):
    scale = gram.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(1e-12)
    eye = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + RIDGE*scale[..., None, None]*eye, rhs).squeeze(-1)


def load_context(path):
    cache = torch.load(path, map_location='cpu', weights_only=True)
    contract = cache['contract']
    required = dict(schema='phase1_fixed_dura_source_context_v1', source_epochs=[21, 22, 23],
        batch_size=32, precision='FP32_TF32off', dura='dura_v1_R65_E20_4940',
        deterministic_algorithms=True, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0)
    if (any(contract.get(k) != v for k, v in required.items()) or
            len(contract['fit_ids']) != FIT_SIZE or len(set(contract['fit_ids'])) != FIT_SIZE):
        raise ValueError('Valid unique source-only fit records required')
    data = cache['data']
    shapes = ((98304, 2, 256), (98304, 2, 256), (98304, 2, 256), (98304,), (98304, 2, 256))
    if (len(data) != 5 or any(v.dtype != torch.float32 or tuple(v.shape) != shape
                             for v, shape in zip(data, shapes)) or sum(cache['counts'].values()) != 98304):
        raise ValueError('Complete matched three-view source IQ required')
    return tuple(data[i].to('cuda:0') for i in (0, 1, 3, 4))


def diagnostic(data):
    received, anchor, variance, clean = data
    n = len(received)
    if n != VIEW_COUNT*FIT_SIZE or (variance <= 0).any():
        raise ValueError('Three complete source views and positive physical variance required')
    ids = torch.arange(n, device=received.device) % FIT_SIZE
    train = ids.remainder(8) != 0
    held = ~train
    ratio = variance/received.square().mean((1, 2)).clamp_min(1e-12)
    limits = torch.quantile(ratio[train], torch.tensor([.25, .5, .75], device=received.device))
    bins = torch.bucketize(ratio, limits)
    gram = torch.zeros((5, 256, 4, 4), dtype=torch.complex64, device=received.device)
    rhs = torch.zeros((5, 256, 4, 1), dtype=torch.complex64, device=received.device)
    counts = [0]*5
    indexes = torch.nonzero(train).flatten()
    for offset in range(0, len(indexes), BATCH):
        ix = indexes[offset:offset+BATCH]
        features, a, _, y = spectra(received[ix], anchor[ix], clean[ix])
        # Normalize by clean energy, but learn a linear residual y-a.
        energy = y.abs().square().mean(-1).clamp_min(1e-12).sqrt()
        z = features/energy[:, None, None]
        residual = (y-a)/energy[:, None]
        for group in range(5):
            mask = torch.ones(len(ix), dtype=torch.bool, device=ix.device) if group == 0 else bins[ix] == group-1
            if not mask.any():
                continue
            f = z[mask].permute(1, 0, 2)
            r = residual[mask].permute(1, 0)
            gram[group] += f.conj().transpose(1, 2) @ f
            rhs[group] += f.conj().transpose(1, 2) @ r.unsqueeze(-1)
            counts[group] += int(mask.sum())
    weights = solve(gram, rhs)
    indexes = torch.nonzero(held).flatten()
    records = []
    for offset in range(0, len(indexes), BATCH):
        ix = indexes[offset:offset+BATCH]
        features, a, x, y = spectra(received[ix], anchor[ix], clean[ix])
        global_out = a + (features*weights[0][None]).sum(-1)
        conditioned = a + (features*weights[1:][bins[ix]]).sum(-1)
        energy = y.abs().square().sum(-1).clamp_min(1e-12)
        errors = torch.stack([((v-y).abs().square().sum(-1)/energy) for v in (x, a, global_out, conditioned)], -1)
        if not torch.isfinite(errors).all():
            raise FloatingPointError('Finite held-out source errors required')
        records.append(torch.cat((ix[:, None].float(), bins[ix, None].float(), errors), 1).cpu())
    rows = torch.cat(records)
    results = {}
    for view in range(3):
        for group in range(5):
            mask = ((rows[:, 0].long() // FIT_SIZE) == view)
            if group:
                mask &= rows[:, 1].long() == group-1
            value = rows[mask, 2:]
            results[f'view{view+21}/'+('all' if group == 0 else f'noise_q{group}')] = dict(
                count=len(value), mean_nmse=value.mean(0).tolist(),
                global_better_than_dura=int((value[:, 2] < value[:, 1]).sum()),
                binned_better_than_dura=int((value[:, 3] < value[:, 1]).sum()))
    return dict(train_records=28672, heldout_records=4096, train_views=len(train.nonzero()),
                heldout_views=len(indexes), train_counts=counts, noise_cutoffs=limits.cpu().tolist(), groups=results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data = load_context(args.cache)
    print('SOURCE_FREQUENCY_CONTEXT_SMOKE_PASS', flush=True)
    args.out.mkdir(parents=True)
    start = time.monotonic()
    result = diagnostic(data)
    torch.cuda.synchronize()
    result.update(status='SOURCE_FREQUENCY_DIAGNOSTIC_COMPLETE', commit=args.commit,
        metric_order=['received', 'dura', 'global_regression', 'four_noise_bin_regressions'],
        ridge=RIDGE, bins='training-source observable variance/received-IQ power quartiles',
        split='source physical fit ID index modulo 8; same ID kept together across all three views',
        elapsed_seconds=time.monotonic()-start, device=torch.cuda.get_device_name(),
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        caveat='Source IQ diagnostic only; fitted filters are not saved or used on target. IQ NMSE is not TX identity accuracy.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('SOURCE_FREQUENCY_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
