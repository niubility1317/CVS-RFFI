"""Generate a fixed source-only mixture-IQ teacher for single-expert distillation."""
import argparse
import os
import time
from pathlib import Path

import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.expert_teacher_contract import (
    EXPERT_SOURCE, EXPERT_COMMIT, FIT_SIZE, TEACHER_SCHEMA, load_expert_teacher)
from experiments.phase1_dura_orthogonal.source_context_cache import load_cache
from experiments.phase1_dura_orthogonal.train_source import load_source_training


@torch.no_grad()
def forward(model, data, indices, device):
    inputs = [value[indices].to(device) for value in data[:4]]
    result = model(*inputs)
    if result.shape != inputs[0].shape or not torch.isfinite(result).all():
        raise FloatingPointError('Finite source expert waveform required')
    return result.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--context-cache', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique source teacher output required')
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        raise ValueError('Registered CUDA device required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, _, _, _, _, _ = source_inputs(args.root)
    _, _, _, fit = load_source_training(args.root, ids, fit_size=FIT_SIZE)
    model = load_expert_teacher(args.root, fit, args.device)
    data, counts = load_cache(args.context_cache, fit, 32, args.device, deterministic=True)
    smoke_indices = torch.cat([torch.arange(i*FIT_SIZE, i*FIT_SIZE+64) for i in range(3)])
    for start in range(0, len(smoke_indices), 32):
        forward(model, data, smoke_indices[start:start+32], args.device)
    print('EXPERT_TEACHER_SOURCE_SMOKE_PASS', len(smoke_indices), flush=True)
    args.out.mkdir(parents=True)
    started = time.monotonic()
    teacher = torch.empty((3*FIT_SIZE, 2, 256), dtype=torch.float32)
    for start in range(0, len(teacher), 32):
        end = min(start+32, len(teacher))
        teacher[start:end] = forward(model, data, slice(start, end), args.device)
        if end % 8192 == 0:
            print('EXPERT_TEACHER_PROGRESS', end, len(teacher), round(time.monotonic()-started, 1), flush=True)
    contract = dict(schema=TEACHER_SCHEMA, source=EXPERT_SOURCE, source_commit=EXPERT_COMMIT,
        fit_ids=[int(x) for x in fit], epochs=[21, 22, 23], context_cache=str(args.context_cache),
        deterministic_algorithms=True, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0)
    with (args.out/'teacher.pt').open('xb') as handle:
        torch.save(dict(contract=contract, teacher=teacher), handle)
    core.write(args.out/'result.json', dict(status='SOURCE_EXPERT_TEACHER_COMPLETE', commit=args.commit,
        exposure=len(teacher), fit_ids=len(fit), counts=counts, model_updates=0,
        elapsed_seconds=time.monotonic()-started, TX_labels_read=0, F_weights_read=0,
        F_outputs_read=0, V_reads=0, target_reads=0))
    print('EXPERT_TEACHER_COMPLETE', len(teacher), flush=True)


if __name__ == '__main__':
    main()
