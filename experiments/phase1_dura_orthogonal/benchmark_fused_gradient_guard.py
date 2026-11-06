"""Paired source-IQ benchmark for one-sync finite gradient checking."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch

from experiments.phase1_dura_orthogonal.benchmark_masked_training_full import load_inputs
from experiments.phase1_dura_orthogonal.fused_gradient_guard import checked_backward_and_clip
from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss, MASK_SEED, WEIGHT
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.train_source import source_loss


def train(fields, prior, initial, fused, epochs):
    torch.manual_seed(2026110612)
    model = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0')
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    order_gen = torch.Generator().manual_seed(2026110613)
    mask_gen = torch.Generator().manual_seed(MASK_SEED)
    losses = []
    torch.cuda.synchronize()
    started = time.monotonic()
    for epoch in range(epochs):
        order = torch.randperm(len(fields[0]), generator=order_gen).to('cuda:0')
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            x, anchor, direction, variance, clean = (v[index] for v in fields)
            output = model(x, anchor, direction, variance, validate=False)
            starts = torch.randint(1, 224, (len(output),), generator=mask_gen).to('cuda:0')
            loss = source_loss(output, anchor, clean, objective='dura_guard') + WEIGHT*masked_prior_loss(prior, output, clean, starts)
            optimizer.zero_grad(set_to_none=True)
            if fused:
                checked_backward_and_clip(loss, model, (model.wave.weight, model.filter_head.weight))
            else:
                if not torch.isfinite(loss):
                    raise FloatingPointError('Finite source loss required')
                loss.backward()
                if (model.wave.weight.grad is None or model.filter_head.weight.grad is None or
                        not torch.isfinite(model.wave.weight.grad).all() or not torch.isfinite(model.filter_head.weight.grad).all()):
                    raise ValueError('Both learned correction paths require finite gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        print('FUSED_GUARD_EPOCH', 'fused' if fused else 'original', epoch+1, flush=True)
    torch.cuda.synchronize()
    return model, torch.stack(losses).cpu(), time.monotonic()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--initial-state', type=Path, required=True)
    parser.add_argument('--initial-commit', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-count', type=int, choices=(512, 32768), default=512)
    parser.add_argument('--pairs', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data, prior, initial = load_inputs(args.cache, args.teacher, args.initial_state, args.initial_commit)
    small = tuple(torch.cat([v[i:i+64] for i in (0,32768,65536)]) for v in data)
    selected = tuple(torch.cat([v[i:i+args.source_count] for i in (0,32768,65536)]) for v in data)
    smoke = DuraMultiscaleSpectral(width=32, taps=9).to('cuda:0').eval()
    smoke.load_state_dict(initial, strict=True)
    with torch.no_grad():
        if not torch.equal(smoke(*(v[:8] for v in selected[:4])), selected[1][:8]):
            raise ValueError('Real source context and initialization smoke required')
    print('FUSED_GUARD_REAL_SOURCE_SMOKE_PASS', flush=True)
    for fused in (False, True):
        warm, _, _ = train(small, prior, initial, fused, 1)
        del warm
    results = []
    for order in ((False, True), (True, False))[:args.pairs]:
        outcomes = {}
        for fused in order:
            outcomes[fused] = train(selected, prior, initial, fused, 5)
        a, b = outcomes[False], outcomes[True]
        if not torch.equal(a[1], b[1]) or any(not torch.equal(a[0].state_dict()[k], b[0].state_dict()[k]) for k in a[0].state_dict()):
            raise ValueError('Source training trajectories differ')
        results.append(dict(original_seconds=a[2], fused_seconds=b[2], steps=len(a[1]),
                            step_losses_exact=True, parameters_exact=True))
        del outcomes, a, b
    args.out.mkdir(parents=True)
    result = dict(status='FUSED_GUARD_BENCHMARK_COMPLETE', commit=args.commit,
                  source_ids=args.source_count, views_per_id=3, epochs=5, batch=32, results=results,
                  gpu=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0,
                  F_outputs_read=0, V_reads=0, target_reads=0, weights_saved=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('FUSED_GUARD_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
