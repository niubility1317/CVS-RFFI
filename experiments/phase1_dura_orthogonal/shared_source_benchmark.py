"""Verify and time shared frozen inference on source IQ, never target records."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.train_source import extract_context

MODELS = (
    ('iq', '20261106-phase1-dura-orthogonal-source-s2026092701-r01',
     '3e6b3c0722af52efd5c812f574a2b3d75af65f88'),
    ('guard', '20261106-phase1-dura-orthogonal-guard-source-s2026092701-r01',
     '49706d9acaa7a4f8085b5b48d3eb65e331a7c373'),
    ('band', '20261106-phase1-dura-band-guard-source-s2026092701-r01',
     'ddf3bc659aed6d78695e80580720ac6da3322cc5'),
)


def classify(reader, signal):
    return reader.classify_features(reader.features(signal))


@torch.no_grad()
def independent(x, triples):
    result = []
    for model, dura, reader in triples:
        anchor, direction, variance = extract_context(dura, x)
        restored = model(x, anchor, direction, variance)
        result.append((anchor, direction, variance, restored,
                       classify(reader, x), classify(reader, anchor), classify(reader, restored)))
    return result


@torch.no_grad()
def shared(x, triples):
    _, dura, reader = triples[0]
    anchor, direction, variance = extract_context(dura, x)
    raw_logits = classify(reader, x)
    anchor_logits = classify(reader, anchor)
    result = []
    for model, _, _ in triples:
        restored = model(x, anchor, direction, variance)
        result.append((anchor, direction, variance, restored, raw_logits, anchor_logits,
                       classify(reader, restored)))
    return result


def same_bits(a, b):
    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


def compare(first, second):
    if len(first) != len(second):
        raise ValueError('Matching model outputs required')
    for a, b in zip(first, second):
        if (len(a) != len(b) or any(not torch.isfinite(x).all() or not same_bits(x, y)
                                    for x, y in zip(a, b))):
            raise ValueError('Shared and independent outputs differ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique output root required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    positions = []
    for receivers in ((4, 6), (8,)):
        for view in core.VIEWS:
            cell = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(cell) != 48:
                raise ValueError('Complete source cells required')
            positions.extend(cell[:8].tolist())
    selected = ids[np.asarray(positions)]
    triples = [load_fixed(args.root, run, commit) for _, run, commit in MODELS]
    for index in (1, 2):
        for left, right in ((triples[0][1], triples[index][1]), (triples[0][2], triples[index][2])):
            a, b = left.state_dict(), right.state_dict()
            if a.keys() != b.keys() or any(not same_bits(a[k], b[k]) for k in a):
                raise ValueError('Identical frozen DURA and classifier weights required')
    batches = [core.tensor(received[selected[start:start+8]], 'cpu') for start in range(0, len(selected), 8)]
    compare(independent(batches[0], triples), shared(batches[0], triples))
    args.out.mkdir(parents=True)
    print('SHARED_SOURCE_SMOKE_PASS', flush=True)
    independent_seconds = shared_seconds = 0.
    for x in batches:
        start = time.perf_counter()
        original = independent(x, triples)
        independent_seconds += time.perf_counter()-start
        start = time.perf_counter()
        common = shared(x, triples)
        shared_seconds += time.perf_counter()-start
        compare(original, common)
    if shared_seconds <= 0:
        raise ValueError('Positive measured elapsed time required')
    core.write(args.out/'result.json', dict(status='SHARED_SOURCE_BENCHMARK_COMPLETE', commit=args.commit,
        records=len(selected), groups=14, batch_size=8, models=[name for name, _, _ in MODELS],
        independent_seconds=independent_seconds, shared_seconds=shared_seconds,
        speedup=independent_seconds/shared_seconds, exact_tensor_comparisons='all batches, all three models: anchor, innovation, variance, restored IQ, raw/anchor/restored logits',
        exact_tensor_mismatches=0, source_only=True, TX_truth_reads=0, V_reads=0,
        target_IQ_reads=0, target_truth_reads=0, training_updates=0,
        caveat='Source-only CPU experiment; target-wide exactness and parallel throughput are not established.'))
    print('SHARED_SOURCE_BENCHMARK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
