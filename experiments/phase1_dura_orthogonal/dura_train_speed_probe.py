"""Source-only transient DURA full-gradient throughput at two batch sizes."""
import argparse
import gc
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs


def measure(root, received, clean, selected):
    model = frozen_dura(root, 'cuda:0').requires_grad_(True).eval()
    x, y = core.tensor(received[selected], 'cuda:0'), core.tensor(clean[selected], 'cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
    torch.cuda.reset_peak_memory_stats()
    times, losses = [], []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        energy = y.square().mean((1, 2)).clamp_min(1e-8)
        loss = ((output-y).square().mean((1, 2))/energy).mean()
        if output.shape != x.shape or not torch.isfinite(loss) or not torch.isfinite(output).all():
            raise FloatingPointError('Finite source-only output and IQ loss required')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not torch.isfinite(norm) or not norm > 0:
            raise ValueError('Finite nonzero original DURA gradient required')
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
        losses.append(float(loss.detach()))
    return dict(batch=len(selected), step_seconds=times, source_iq_losses=losses,
                steady_samples_per_second=len(selected)*2/sum(times[1:]),
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source technical output required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != 1 or not visible[0].strip() or not torch.cuda.is_available():
        raise ValueError('Exactly one GPU must be selected')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, clean = source_inputs(args.root)
    cells = []
    for view in core.VIEWS[:4]:
        cell = np.flatnonzero(np.isin(rx, (4, 6)) & (scene == view))
        if len(cell) != 48:
            raise ValueError('Complete source cells required')
        cells.append(cell[:8])
    args.out.mkdir(parents=True)
    measurements = []
    for per_view in (4, 8):
        selected = ids[np.concatenate([cell[:per_view] for cell in cells])]
        measurements.append(measure(args.root, received, clean, selected))
        gc.collect()
        print('DURA_GRADIENT_SPEED_BATCH_COMPLETE', len(selected), flush=True)
    core.write(args.out/'result.json', dict(status='DURA_GRADIENT_SPEED_SOURCE_COMPLETE', commit=args.commit,
        measurements=measurements, views=list(core.VIEWS[:4]), steps_per_batch=3,
        warmup_steps=1, weights_saved=False, original_checkpoint_modified=False, TX_labels_read=0,
        F_weights_read=0, F_outputs_read=0, V_reads=0, target_IQ_reads=0, target_truth_reads=0,
        caveat='Transient same-source training timing; steady throughput averages last two steps, excludes loading and data preparation; no performance or identity result.'))
    print('DURA_GRADIENT_SPEED_SOURCE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
