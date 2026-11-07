"""Compare frozen source inference with and without an unused anchor classification."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.ground import load_fixed
from experiments.phase1_dura_orthogonal.train_source import extract_context

CANDIDATE = '20261107-phase1-dura-monotone-envelope-source-s2026092701-r01'
CANDIDATE_COMMIT = 'ba3a01a6267cec4caa7d4d52e0d5c640f7ddee9c'


@torch.no_grad()
def logits(x, model, dura, reader, *, include_anchor):
    anchor, direction, variance = extract_context(dura, x)
    restored = model(x, anchor, direction, variance)
    signals = (x, anchor, restored) if include_anchor else (x, restored)
    outputs = []
    for signal in signals:
        value = reader.classify_features(reader.features(signal))
        if value.shape != (len(x), 6) or not torch.isfinite(value).all():
            raise ValueError('Finite frozen reader logits required')
        outputs.append(value)
    return outputs[0], outputs[-1]


def statistics(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(count=len(array), mean_ms=float(array.mean()*1000),
                median_ms=float(np.median(array)*1000), p90_ms=float(np.percentile(array, 90)*1000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source benchmark output required')
    torch.set_num_threads(2)
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    selected = []
    for receivers in ((4, 6), (8,)):
        for view in core.VIEWS:
            found = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(found) < 8:
                raise ValueError('Eight source physical records per group required')
            selected.extend(found[:8].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) != 112 or len(np.unique(selected)) != 112:
        raise ValueError('112 distinct source records required')
    model, dura, reader = load_fixed(args.root, CANDIDATE, CANDIDATE_COMMIT)
    models = (model, dura, reader)
    snapshots = [{key: value.detach().clone() for key, value in obj.state_dict().items()} for obj in models]
    if any(param.requires_grad for obj in models for param in obj.parameters()):
        raise ValueError('Frozen source-evaluation models required')
    x = core.tensor(received[ids[selected]], 'cpu')
    with torch.no_grad():
        first = logits(x[:8], model, dura, reader, include_anchor=True)
        second = logits(x[:8], model, dura, reader, include_anchor=False)
        if any(not torch.equal(a, b) for a, b in zip(first, second)):
            raise ValueError('Source real-checkpoint logits smoke mismatch')
        print('PAIR_INFERENCE_REAL_CHECKPOINT_SMOKE_PASS', flush=True)
        results = {}
        for batch_size in (1, 8):
            batches = [x[i:i+batch_size] for i in range(0, len(x), batch_size)]
            for _ in range(3):
                logits(batches[0], model, dura, reader, include_anchor=True)
                logits(batches[0], model, dura, reader, include_anchor=False)
            timings = {True: [], False: []}
            comparisons = 0
            for repeat in range(2):
                for i, batch in enumerate(batches):
                    outputs = {}
                    order = (True, False) if (i+repeat) % 2 == 0 else (False, True)
                    for original in order:
                        started = time.perf_counter()
                        outputs[original] = logits(batch, model, dura, reader, include_anchor=original)
                        timings[original].append(time.perf_counter()-started)
                    if any(not torch.equal(a, b) for a, b in zip(outputs[True], outputs[False])):
                        raise ValueError('Full raw or recovered source logits differ')
                    comparisons += len(batch)
            results[str(batch_size)] = dict(original=statistics(timings[True]), pair=statistics(timings[False]),
                                            paired_records=comparisons, raw_and_restored_logits_exact=True)
            print('PAIR_INFERENCE_BATCH_COMPLETE', batch_size, comparisons, flush=True)
    for obj, snapshot in zip(models, snapshots):
        if any(not torch.equal(value, snapshot[key]) for key, value in obj.state_dict().items()):
            raise ValueError('Frozen model state changed during source evaluation')
    args.out.mkdir(parents=True)
    result = dict(status='PAIR_INFERENCE_SOURCE_COMPLETE', commit=args.commit, candidate=CANDIDATE,
                  candidate_commit=CANDIDATE_COMMIT, F_run=core.F_RUN, F_row=core.F_ROW,
                  source_records=112, cpu_threads=2, TX_labels_read=0, F_weights_read=1,
                  F_outputs_read=1, V_reads=0, target_reads=0, weights_saved=False, results=results)
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('PAIR_INFERENCE_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
