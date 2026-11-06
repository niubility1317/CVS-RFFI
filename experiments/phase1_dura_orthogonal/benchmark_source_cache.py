"""Build and compare full fixed source DURA contexts; never reads target IQ."""
import argparse
from pathlib import Path
import time

import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.controller import training_environments
from experiments.phase1_dura_orthogonal.source_context_cache import FIELDS, load_cache, save_cache
from experiments.phase1_dura_orthogonal.train_source import load_source_training, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    import os
    training_environments('cuda:0', os.environ)
    if args.out.exists():
        raise FileExistsError('Unique source cache benchmark output required')
    args.out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, _, _, _, _, _ = source_inputs(args.root)
    _, fresh, clean, fit = load_source_training(args.root, ids)
    dura = frozen_dura(args.root, 'cuda:0')
    torch.manual_seed(2026110612)
    start = time.monotonic()
    first, counts = prepare(args.root, dura, clean, fresh, fit, 'cuda:0', 32, 'dura_multiscale')
    fresh_seconds = time.monotonic()-start
    print('SOURCE_CONTEXT_FRESH_COMPLETE', round(fresh_seconds, 3), flush=True)
    start = time.monotonic()
    save_cache(args.out/'source_context.pt', first, counts, fit, 32, 'cuda:0')
    write_seconds = time.monotonic()-start
    start = time.monotonic()
    cached, cached_counts = load_cache(args.out/'source_context.pt', fit, 32, 'cuda:0')
    load_seconds = time.monotonic()-start
    if cached_counts != counts or not all(torch.equal(x, y) for x, y in zip(first, cached)):
        raise ValueError('Cached source tensors must equal the saved fresh tensors exactly')
    del first
    start = time.monotonic()
    repeated, repeated_counts = prepare(args.root, dura, clean, fresh, fit, 'cuda:0', 32, 'dura_multiscale')
    repeat_seconds = time.monotonic()-start
    comparisons = {name: bool(torch.equal(a, b)) for name, a, b in zip(FIELDS, cached, repeated)}
    if repeated_counts != counts or not all(comparisons.values()):
        raise ValueError('Full recomputed source tensors must match cached tensors exactly')
    core.write(args.out/'result.json', dict(status='SOURCE_CACHE_BENCHMARK_COMPLETE', commit=args.commit,
        fit_ids=len(fit), exposure=len(cached[0]), fields_equal=comparisons, counts=counts,
        fresh_prepare_seconds=fresh_seconds, repeat_prepare_seconds=repeat_seconds,
        cache_write_seconds=write_seconds, cache_load_seconds=load_seconds,
        cache_bytes=(args.out/'source_context.pt').stat().st_size,
        device='cuda:0', batch_size=32, source_only=True, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0))
    print('SOURCE_CONTEXT_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
