"""Synthetic-input, real-source-checkpoint paired loss/gradient microbenchmark."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss, SOURCE_COMMIT
from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior, masked_inputs, SPAN


def original(prior, output, clean, starts):
    observed, mask, scale, _ = masked_inputs(output, starts)
    clean_observed, clean_mask, clean_scale, _ = masked_inputs(clean, starts)
    with torch.no_grad():
        target = prior(clean_observed, clean_mask, clean_scale)
    prediction = prior(observed, mask, scale)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    return (((prediction-target).square()*mask[:, None]).sum((1, 2))/(2*SPAN*energy)).mean()


def measure(fn, prior, output, clean, starts):
    torch.cuda.synchronize()
    begin = time.perf_counter()
    loss = fn(prior, output, clean, starts)
    gradient = torch.autograd.grad(loss, output)[0]
    torch.cuda.synchronize()
    return (time.perf_counter()-begin)*1000, loss.detach(), gradient.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and CUDA GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('commit') != SOURCE_COMMIT:
        raise ValueError('Matching source-only masked prior checkpoint required')
    prior = MaskedIQPrior().to('cuda:0').eval().requires_grad_(False)
    prior.load_state_dict(state['model'], strict=True)
    torch.manual_seed(2026110712)
    clean = torch.randn(32, 2, 256, device='cuda:0')
    output = (clean+.1*torch.randn_like(clean)).requires_grad_()
    starts = torch.randint(1, 224, (32,), device='cuda:0')
    old, new = original, masked_prior_loss
    for _ in range(4):
        measure(old, prior, output, clean, starts)
        measure(new, prior, output, clean, starts)
    print('MASKED_LOSS_REAL_CHECKPOINT_SMOKE_PASS', flush=True)
    timings = {'original': [], 'optimized': []}
    max_difference = 0.
    equal = True
    for index in range(20):
        order = [('original', old), ('optimized', new)]
        if index % 2:
            order.reverse()
        result = {}
        for name, fn in order:
            duration, loss, gradient = measure(fn, prior, output, clean, starts)
            timings[name].append(duration)
            result[name] = (loss, gradient)
        a, b = result['original'], result['optimized']
        equal &= torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        max_difference = max(max_difference, float((a[1]-b[1]).abs().max().cpu()))
    args.out.mkdir(parents=True)
    result = dict(status='MASKED_LOSS_BENCHMARK_COMPLETE', commit=args.commit, source_checkpoint_commit=SOURCE_COMMIT,
                  device=torch.cuda.get_device_name(), batch=32, repetitions=20, warmup=4,
                  loss_and_gradient_exact=equal, gradient_max_abs_difference=max_difference,
                  median_ms={name: statistics.median(values) for name, values in timings.items()},
                  measurements_ms=timings, synthetic_inputs=True, source_or_target_IQ_reads=0,
                  F_weights_read=0, F_outputs_read=0, TX_labels_read=0, target_reads=0, model_updates=0,
                  caveat='One fixed synthetic batch on local GPU with real source checkpoint; loss/gradient only, not full training or test predictions.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('MASKED_LOSS_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
